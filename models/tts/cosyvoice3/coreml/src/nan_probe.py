"""Stage 1 of the Flow → ANE port: localize the fp16 blowup inside the DiT.

Stage 0 shipped fp16 unfused LN (see `ane_layernorm.py`) and confirmed the
CoreML output is NaN even on CPU_ONLY. That means the precision loss is not
localized to `layer_norm`; it is distributed across the DiT forward. Before
committing to a full ml-ane-transformers rewrite (Stage 2) we want to know
*which* block / sub-op first overflows in fp16 so the rewrite targets the
actual culprit (e.g. attention softmax logits, AdaLN `(1+scale)*norm`, FFN
GELU).

Two complementary probes
------------------------

1. `trace_fp32_peaks(wrapper, inputs)` — runs the wrapper in fp32 with
   forward hooks on every DiTBlock sub-op. Records peak |activation| per
   (block, op) across all Euler steps. Any op whose peak exceeds ~65504
   will overflow to +inf once that value passes through an fp16 accumulator.

2. `linear_scan_fp16(wrapper, inputs)` — sweeps `k` from 1..depth. For each
   `k`, wraps the first `k` DiTBlocks with an fp16 round-trip on input and
   output (`x.half().float()`), leaving weights in fp32. Reports the
   smallest `k` at which the final wrapper output becomes NaN. That tells
   us the first block at which fp16 storage is lethal.

Both probes run in PyTorch (not CoreML), so they are fast, deterministic,
and produce directly actionable output. The probes expect a `FlowCoreML`
wrapper (`src/flow_coreml.py`) whose DiT has already been patched via
`ane_layernorm.patch_dit_norms` to match the shipping graph.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "verify" / "CosyVoice"))
sys.path.insert(0, str(HERE / "verify" / "CosyVoice" / "third_party" / "Matcha-TTS"))

from src.flow_coreml import FlowCoreML  # noqa: E402

FP16_MAX = 65504.0


# --------------------------------------------------------------------------- #
# Utility: peak tracker
# --------------------------------------------------------------------------- #

@dataclass
class PeakRecord:
    block_idx: int
    site: str
    peak_abs: float
    n_calls: int

    def overflows_fp16(self) -> bool:
        return self.peak_abs > FP16_MAX


def _finite_peak(x: Any) -> float:
    if not torch.is_tensor(x):
        return 0.0
    with torch.no_grad():
        finite = torch.isfinite(x)
        if not finite.any():
            return float("inf")
        return float(x[finite].abs().max().item())


# --------------------------------------------------------------------------- #
# Probe 1: fp32 peak trace
# --------------------------------------------------------------------------- #

def trace_fp32_peaks(wrapper: FlowCoreML, inputs: tuple[torch.Tensor, ...]) -> list[PeakRecord]:
    """Run the wrapper in fp32 and record per-(block, site) peak |activation|.

    Sites traced per block:
        pre_attn_norm   : input to AdaLayerNormZero
        post_attn_norm  : normalized+modulated output of AdaLayerNormZero
        post_attn       : attention output (before gated residual)
        post_attn_res   : block state after attention residual add
        post_ff_norm    : LayerNorm + AdaLN modulation before FFN
        post_ff         : FFN output (before gated residual)
        post_ff_res     : block output after FFN residual add
    """
    dit = wrapper.flow.decoder.estimator
    blocks: list[nn.Module] = list(dit.transformer_blocks)

    # peaks[(block_idx, site)] -> (peak, count)
    peaks: dict[tuple[int, str], list[float]] = {}

    def record(block_idx: int, site: str, x: Any) -> None:
        p = _finite_peak(x)
        key = (block_idx, site)
        if key not in peaks:
            peaks[key] = [p, 1]
        else:
            peaks[key][0] = max(peaks[key][0], p)
            peaks[key][1] += 1

    hooks: list[Any] = []

    for idx, block in enumerate(blocks):
        # attn_norm returns a 5-tuple (norm, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        # We trace input and the `norm` component (norm output that feeds attention).
        def make_attn_norm_hook(i: int):
            def hook(module, args, kwargs, out):
                # Pre: args[0] or kwargs['x']
                x_in = args[0] if len(args) > 0 else kwargs.get("x")
                record(i, "pre_attn_norm", x_in)
                # `out` is tuple; element 0 is the modulated LN result fed into attention.
                if isinstance(out, tuple) and len(out) >= 1:
                    record(i, "post_attn_norm", out[0])
            return hook

        def make_attn_hook(i: int):
            def hook(module, args, kwargs, out):
                record(i, "post_attn", out)
            return hook

        def make_ff_hook(i: int):
            def hook(module, args, kwargs, out):
                x_in = args[0] if len(args) > 0 else kwargs.get("x")
                record(i, "pre_ff", x_in)
                record(i, "post_ff", out)
            return hook

        def make_block_hook(i: int):
            def hook(module, args, kwargs, out):
                record(i, "post_block", out)
            return hook

        hooks.append(block.attn_norm.register_forward_hook(
            make_attn_norm_hook(idx), with_kwargs=True))
        hooks.append(block.attn.register_forward_hook(
            make_attn_hook(idx), with_kwargs=True))
        hooks.append(block.ff.register_forward_hook(
            make_ff_hook(idx), with_kwargs=True))
        hooks.append(block.register_forward_hook(
            make_block_hook(idx), with_kwargs=True))

    try:
        with torch.no_grad():
            _ = wrapper(*inputs)
    finally:
        for h in hooks:
            h.remove()

    records: list[PeakRecord] = []
    for (idx, site), (peak, count) in peaks.items():
        records.append(PeakRecord(block_idx=idx, site=site, peak_abs=peak, n_calls=count))
    records.sort(key=lambda r: (r.block_idx, r.site))
    return records


# --------------------------------------------------------------------------- #
# Probe 1b: attention softmax-logit magnitudes
# --------------------------------------------------------------------------- #

def trace_attention_logits(
    wrapper: FlowCoreML,
    inputs: tuple[torch.Tensor, ...],
) -> list[PeakRecord]:
    """Monkey-patch `F.scaled_dot_product_attention` to measure the peak
    magnitude of the raw `Q @ K.T / sqrt(d_head)` logits tensor, which is
    the specific intermediate that CoreML materializes when lowering SDPA
    into `matmul → mul(scale) → add(mask) → softmax → matmul`.

    When this peak exceeds ~65504, CoreML's fp16 decomposed attention
    produces +inf in the logits, softmax of +inf / +inf yields NaN, and
    the NaN propagates to the final mel output — matching Stage 0's
    observation.

    We rely on call ordering: for each DiT forward, the 22 blocks invoke
    `F.scaled_dot_product_attention` exactly once in order. Across Euler
    steps, the same ordering repeats. We track per-block peak by
    (call_index mod 22).
    """
    import torch.nn.functional as F

    depth = len(wrapper.flow.decoder.estimator.transformer_blocks)
    orig_sdpa = F.scaled_dot_product_attention

    # peaks[block_idx] = [logit_peak, value_peak, q_peak, k_peak, count]
    peaks: list[list[float]] = [[0.0, 0.0, 0.0, 0.0, 0] for _ in range(depth)]
    call_counter = {"n": 0}

    def instrumented_sdpa(q, k, v, attn_mask=None, dropout_p=0.0,
                          is_causal=False, scale=None):
        block_idx = call_counter["n"] % depth
        call_counter["n"] += 1
        with torch.no_grad():
            head_dim = q.shape[-1]
            s = scale if scale is not None else (1.0 / (head_dim ** 0.5))
            # Q @ K.T then scale — exactly what coremltools emits.
            logits = torch.matmul(q, k.transpose(-1, -2)) * s
            lp = _finite_peak(logits)
            peaks[block_idx][0] = max(peaks[block_idx][0], lp)
            peaks[block_idx][1] = max(peaks[block_idx][1], _finite_peak(v))
            peaks[block_idx][2] = max(peaks[block_idx][2], _finite_peak(q))
            peaks[block_idx][3] = max(peaks[block_idx][3], _finite_peak(k))
            peaks[block_idx][4] += 1
        return orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
                         is_causal=is_causal, scale=scale)

    F.scaled_dot_product_attention = instrumented_sdpa
    try:
        with torch.no_grad():
            _ = wrapper(*inputs)
    finally:
        F.scaled_dot_product_attention = orig_sdpa

    records: list[PeakRecord] = []
    for i, (lp, vp, qp, kp, cnt) in enumerate(peaks):
        records.append(PeakRecord(i, "sdpa_logits", lp, cnt))
        records.append(PeakRecord(i, "sdpa_q", qp, cnt))
        records.append(PeakRecord(i, "sdpa_k", kp, cnt))
        records.append(PeakRecord(i, "sdpa_v", vp, cnt))
    return records


# --------------------------------------------------------------------------- #
# Probe 2: fp16 round-trip wrapper on a subset of blocks
# --------------------------------------------------------------------------- #

class _Fp16RoundTrip(nn.Module):
    """Wraps a DiTBlock so its input and output are cast fp32→fp16→fp32.

    This simulates fp16 *storage* across block boundaries while leaving the
    block's internal compute in fp32. Any value that saturates fp16 at the
    boundary becomes +inf, turning subsequent ops into NaN — which is the
    exact failure mode we see in the CoreML conversion.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x, *args, **kwargs):
        x = x.to(torch.float16).to(torch.float32)
        out = self.inner(x, *args, **kwargs)
        if isinstance(out, torch.Tensor):
            return out.to(torch.float16).to(torch.float32)
        if isinstance(out, tuple):
            return tuple(
                o.to(torch.float16).to(torch.float32) if torch.is_tensor(o) else o
                for o in out
            )
        return out


def _swap_blocks(dit: nn.Module, indices: Iterable[int], wrap: bool) -> dict[int, nn.Module]:
    """In-place swap for blocks at `indices`. Returns the originals so caller
    can restore. When `wrap=False` this is a no-op and returns {}."""
    originals: dict[int, nn.Module] = {}
    if not wrap:
        return originals
    for i in indices:
        orig = dit.transformer_blocks[i]
        originals[i] = orig
        dit.transformer_blocks[i] = _Fp16RoundTrip(orig)
    return originals


def _restore_blocks(dit: nn.Module, originals: dict[int, nn.Module]) -> None:
    for i, orig in originals.items():
        dit.transformer_blocks[i] = orig


def fp16_cast_probe(
    wrapper: FlowCoreML,
    inputs: tuple[torch.Tensor, ...],
    k: int,
) -> dict[str, Any]:
    """Run wrapper with blocks [0..k-1] wrapped in fp16 round-trip.

    Returns a dict with output, nan/inf counts, and output peak magnitude.
    """
    dit = wrapper.flow.decoder.estimator
    originals = _swap_blocks(dit, range(k), wrap=(k > 0))
    try:
        with torch.no_grad():
            out, _ = wrapper(*inputs)
    finally:
        _restore_blocks(dit, originals)

    nan_count = int(torch.isnan(out).sum().item())
    inf_count = int(torch.isinf(out).sum().item())
    finite = out[torch.isfinite(out)]
    peak = float(finite.abs().max().item()) if finite.numel() > 0 else float("nan")
    return {
        "k": k,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_peak": peak,
    }


def linear_scan_fp16(
    wrapper: FlowCoreML,
    inputs: tuple[torch.Tensor, ...],
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Sweep k=0..depth and report when NaN/Inf first appears in output."""
    depth = len(wrapper.flow.decoder.estimator.transformer_blocks)
    results: list[dict[str, Any]] = []
    for k in range(depth + 1):
        r = fp16_cast_probe(wrapper, inputs, k)
        results.append(r)
        if verbose:
            print(
                f"  k={k:>2}  nan={r['nan_count']:>6}  inf={r['inf_count']:>6}  "
                f"finite_peak={r['finite_peak']:>10.3f}"
            )
    return results


def bisect_fp16(
    wrapper: FlowCoreML,
    inputs: tuple[torch.Tensor, ...],
    *,
    verbose: bool = True,
) -> tuple[int | None, list[dict[str, Any]]]:
    """Binary-search the smallest k in 1..depth where fp16 round-trip on
    blocks [0..k-1] yields non-finite output. Returns (k_first_bad, probes)
    where probes are the (k, result) pairs actually evaluated.

    k=0 means no blocks wrapped (pure fp32 baseline). If k=depth still
    produces finite output, returns (None, probes) meaning block-boundary
    fp16 storage is not sufficient to break the DiT — the blowup is
    internal to a block (e.g. softmax, GELU).
    """
    dit = wrapper.flow.decoder.estimator
    depth = len(dit.transformer_blocks)
    probes: list[dict[str, Any]] = []

    def probe(k: int) -> dict[str, Any]:
        r = fp16_cast_probe(wrapper, inputs, k)
        probes.append(r)
        if verbose:
            print(
                f"  probe k={k:>2}  nan={r['nan_count']:>6}  inf={r['inf_count']:>6}  "
                f"finite_peak={r['finite_peak']:>10.3f}"
            )
        return r

    def is_bad(r: dict[str, Any]) -> bool:
        return r["nan_count"] > 0 or r["inf_count"] > 0

    base = probe(0)
    if is_bad(base):
        # fp32 baseline already bad — probe is broken; bail.
        return 0, probes

    top = probe(depth)
    if not is_bad(top):
        return None, probes

    lo, hi = 0, depth  # invariant: probe(lo) good, probe(hi) bad
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if is_bad(probe(mid)):
            hi = mid
        else:
            lo = mid
    return hi, probes


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

def _load_flow_with_patch() -> nn.Module:
    """Load flow + apply the same patch_dit_norms used in the conversion
    pipeline so the probe targets exactly the graph we convert."""
    from hyperpyyaml import load_hyperpyyaml
    from src.ane_layernorm import patch_dit_norms

    yaml_path = HERE / "cosyvoice3_dl" / "cosyvoice3.yaml"
    pt_path = HERE / "cosyvoice3_dl" / "flow.pt"
    with open(yaml_path) as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})
    flow = cfg["flow"]
    sd = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    flow.load_state_dict(sd, strict=False)
    flow.eval()
    patch_dit_norms(flow.decoder.estimator)
    return flow


def _load_inputs(ref_path: Path) -> tuple[torch.Tensor, ...]:
    ref = torch.load(str(ref_path), map_location="cpu", weights_only=False)
    return (
        ref["token_total"],
        ref["num_prompt_tokens"],
        ref["prompt_feat"],
        ref["embedding"],
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="Path to ref-N{N}.pt from convert-flow.py")
    p.add_argument("--n-tokens", type=int, default=125)
    p.add_argument("--n-timesteps", type=int, default=1,
                   help="Euler steps during probe. 1 is enough to surface per-block NaN "
                        "and is ~10x faster than the shipping default of 10.")
    p.add_argument("--skip-peaks", action="store_true",
                   help="Skip fp32 peak trace (one full wrapper forward with hooks)")
    p.add_argument("--skip-logits", action="store_true",
                   help="Skip attention softmax-logit peak trace")
    p.add_argument("--skip-scan", action="store_true",
                   help="Skip fp16 cast scan")
    p.add_argument("--linear-scan", action="store_true",
                   help="Use full linear scan (depth+1 runs) instead of bisect (log2(depth)+2)")
    args = p.parse_args()

    print(f"[load] flow + patch_dit_norms")
    flow = _load_flow_with_patch()

    print(f"[load] ref inputs from {args.ref}")
    inputs = _load_inputs(Path(args.ref))

    print(f"[wrap] FlowCoreML(N={args.n_tokens}, n_timesteps={args.n_timesteps})")
    wrapper = FlowCoreML(flow, n_total_tokens=args.n_tokens,
                         n_timesteps=args.n_timesteps).eval()

    if not args.skip_peaks:
        print("\n[probe 1] fp32 peak trace (across all Euler steps)")
        records = trace_fp32_peaks(wrapper, inputs)
        print(f"  {'block':>5}  {'site':<15}  {'peak':>12}  calls   fp16_risk")
        for r in records:
            risk = "OVERFLOW" if r.overflows_fp16() else ("warn" if r.peak_abs > 10_000 else "")
            print(f"  {r.block_idx:>5}  {r.site:<15}  {r.peak_abs:>12.3f}  {r.n_calls:>5}   {risk}")

        overflow_sites = [r for r in records if r.overflows_fp16()]
        if overflow_sites:
            print(f"\n  !! {len(overflow_sites)} site(s) exceed fp16 max ({FP16_MAX}):")
            for r in overflow_sites:
                print(f"     block {r.block_idx}  {r.site}  peak={r.peak_abs:.1f}")
        else:
            print("\n  (no site exceeds fp16 max in fp32 trace; blowup is precision, not overflow)")

    if not args.skip_logits:
        print("\n[probe 1b] attention SDPA Q/K/V/logits peaks (across all Euler steps)")
        records_att = trace_attention_logits(wrapper, inputs)
        print(f"  {'block':>5}  {'site':<14}  {'peak':>12}  calls   fp16_risk")
        for r in records_att:
            risk = "OVERFLOW" if r.overflows_fp16() else ("warn" if r.peak_abs > 10_000 else "")
            print(f"  {r.block_idx:>5}  {r.site:<14}  {r.peak_abs:>12.3f}  {r.n_calls:>5}   {risk}")
        overflow_att = [r for r in records_att if r.overflows_fp16()]
        if overflow_att:
            print(f"\n  !! {len(overflow_att)} attention site(s) exceed fp16 max:")
            for r in overflow_att:
                print(f"     block {r.block_idx}  {r.site}  peak={r.peak_abs:.1f}")
        else:
            print("\n  (no attention logit overflow in fp32; if CoreML still NaN's, cause"
                  " is matmul accumulation or an op not traced here)")

    if not args.skip_scan:
        print(f"\n[probe 2] fp16 round-trip scan over blocks 0..k-1 "
              f"({'linear' if args.linear_scan else 'bisect'})")
        if args.linear_scan:
            results = linear_scan_fp16(wrapper, inputs)
            first_bad_k = next(
                (r["k"] for r in results if r["nan_count"] > 0 or r["inf_count"] > 0),
                None,
            )
        else:
            first_bad_k, _ = bisect_fp16(wrapper, inputs)

        if first_bad_k is not None and first_bad_k > 0:
            print(f"\n  First non-finite output at k={first_bad_k} "
                  f"(blocks 0..{first_bad_k-1} in fp16 round-trip)")
            print(f"    → first block where fp16 storage breaks the DiT: block {first_bad_k-1}")
        elif first_bad_k == 0:
            print("\n  fp32 baseline already produced non-finite output; probe setup is broken.")
        else:
            print("\n  No NaN/Inf detected across any k. fp16 round-trip alone is not lethal;"
                  " the real blowup is inside a block (softmax/AdaLN/GELU/attn proj),"
                  " not at block boundaries.")


if __name__ == "__main__":
    main()
