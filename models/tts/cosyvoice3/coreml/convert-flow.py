"""Convert CosyVoice3 CausalMaskedDiffWithDiT to CoreML mlpackage.

Pipeline mirrors convert-coreml.py (HiFT):
  1. Load flow + state_dict via hyperpyyaml
  2. Wrap with src.flow_coreml.FlowCoreML (static shapes, traceable euler+CFG)
  3. Sanity-check PyTorch wrapper output
  4. torch.jit.trace
  5. coremltools convert → mlpackage (FP32 default, FP16 optional)

Bucket sizing
-------------
The wrapper uses a single static N_total = N_prompt + N_new tokens, producing
M = N_total * token_mel_ratio mel frames. Default is N=125 → 250 mel frames,
which matches the existing HiFT mlpackage (T=250).

Usage:
    uv run python convert-flow.py --output-dir ./build/flow-fp32
    uv run python convert-flow.py --output-dir ./build/flow-fp16 --fp16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "verify" / "CosyVoice"))
sys.path.insert(0, str(HERE / "verify" / "CosyVoice" / "third_party" / "Matcha-TTS"))

from src.ane_attention import patch_dit_attention  # noqa: E402
from src.ane_layernorm import patch_dit_norms  # noqa: E402
from src.flow_coreml import FlowCoreML  # noqa: E402
from src.flow_coreml_ane import FlowCoreMLANE  # noqa: E402


def _make_precision(fp16: bool, unfuse_ln: bool, fp32_sdpa: bool = False, ane_port: bool = False):
    """FP16 everywhere except numerically sensitive ops.

    Flow has RMSNorm / LayerNorm + softmax attention inside the DiT blocks.
    Keep those in fp32 (same recipe used for the LLM) to avoid overflow on
    activation outliers and precision loss in long-key softmax.

    When `unfuse_ln=True`, every `nn.LayerNorm` has been replaced by primitive
    ops (`reduce_mean/sub/mul/rsqrt`) and the fused `layer_norm` MIL op no
    longer appears in the graph, so pinning it is a no-op. The primitives are
    left in fp16 to let ANE/GPU pick optimal placement.
    """
    if not fp16:
        return ct.precision.FLOAT32
    # Flow DiT has AdaLN (layer_norm * (1+scale) + shift) stacked 22 blocks deep x
    # 2 CFG branches x N CFM steps. With only {pow, reduce_mean, rsqrt, softmax}
    # in fp32, fp16 accumulation overflows to NaN. Pin gelu as well.
    if unfuse_ln:
        # layer_norm op does not exist in the unfused graph; no point pinning.
        FP32_OPS = {"pow", "reduce_mean", "rsqrt", "softmax", "gelu"}
    else:
        FP32_OPS = {"pow", "reduce_mean", "rsqrt", "softmax", "layer_norm", "gelu"}
    if fp32_sdpa:
        # Stage 2a: ANEAttnProcessor decomposes SDPA so the QK^T matmul
        # output (peaks ~1.6M in 9 blocks) doesn't saturate in fp16.
        # `_sdpa_fp32` pre-scales Q in fp16 (safe: q*0.125 ~25), then casts
        # to fp32 for matmul -> where(mask) -> softmax -> matmul. Pinning
        # {matmul, select, where} keeps that subgraph in fp32 while every
        # other mul/add/linear/... stays fp16 for ANE.
        # nn.Linear lowers to MIL `linear` (distinct from `matmul`) so the
        # 6 linear projections per block (to_q/k/v/out + 2 FFN) stay fp16.
        FP32_OPS = FP32_OPS | {"matmul", "select", "where"}
    if ane_port:
        # Stage 3 iter 3: collapse most fp16↔fp32 cast boundaries.
        # Fully unpinned fp16 loaded on ANE but parity MAE=2.6 vs fp32 ref
        # (precision drift accumulated across 22 blocks × 10 timesteps).
        # Iter 4: re-pin ONLY the two LN-internal reductions (`reduce_mean`
        # for mean+var, `rsqrt` for inv_std) — these are the single-pass
        # precision-critical ops where fp16 var computation of ~1024-wide
        # channel reductions can lose significant digits. softmax / gelu
        # stay fp16 (those only drift smoothly, the LN drift is the one
        # that compounds across blocks). `where` stays fp32 so the
        # attention mask's -inf doesn't underflow.
        FP32_OPS = {"where", "reduce_mean", "rsqrt"}
    return ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type not in FP32_OPS
    )


def _make_pass_pipeline(unfuse_ln: bool):
    """Build a pass pipeline. When `unfuse_ln=True`, remove the
    `fuse_layernorm_or_instancenorm` pass so coremltools keeps our primitive
    ops instead of re-fusing them back into a single `layer_norm` MIL op.
    """
    pipeline = ct.PassPipeline.DEFAULT
    if unfuse_ln:
        pipeline.remove_passes(
            ["common::fuse_layernorm_or_instancenorm"]
        )
    return pipeline


def load_flow() -> torch.nn.Module:
    from hyperpyyaml import load_hyperpyyaml

    yaml = HERE / "cosyvoice3_dl" / "cosyvoice3.yaml"
    pt = HERE / "cosyvoice3_dl" / "flow.pt"
    with open(yaml) as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})
    flow = cfg["flow"]
    sd = torch.load(str(pt), map_location="cpu", weights_only=False)
    missing, unexpected = flow.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected: {len(unexpected)} (first 3: {unexpected[:3]})")
    flow.eval()
    return flow


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-tokens", type=int, default=125,
                   help="Total token slots = N_prompt + N_new (mel frames = 2x)")
    p.add_argument("--n-timesteps", type=int, default=10)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--unfuse-ln", action="store_true",
                   help="Replace nn.LayerNorm with primitive-op decomposition "
                        "to unblock ANE compilation; also removes coremltools' "
                        "fuse_layernorm_or_instancenorm pass from the pipeline.")
    p.add_argument("--fp32-sdpa", action="store_true",
                   help="Replace F.scaled_dot_product_attention in each DiT "
                        "block with a manually decomposed fp32-cast equivalent "
                        "(matmul/softmax/matmul in fp32). Fixes QK^T fp16 "
                        "overflow localized in Stage 1 probe.")
    p.add_argument("--ane-port", action="store_true",
                   help="Use the BC1S ml-ane-transformers port: wraps flow "
                        "with FlowCoreMLANE which replaces the DiT estimator "
                        "with ANEDiT (channels-first 4D layout, Conv2d-linear, "
                        "manual per-head einsum attention, axis=1 LayerNorm). "
                        "Mutually exclusive with --unfuse-ln / --fp32-sdpa — "
                        "the port obsoletes both.")
    p.add_argument("--min-deployment", default="macOS14",
                   choices=["macOS13", "macOS14", "macOS15"])
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N = args.n_tokens
    M = N * 2  # token_mel_ratio=2

    if args.ane_port and (args.unfuse_ln or args.fp32_sdpa):
        raise SystemExit(
            "--ane-port is mutually exclusive with --unfuse-ln and --fp32-sdpa; "
            "the BC1S port replaces the DiT graph entirely (ANELayerNormBC1S + "
            "einsum-attention), so both workarounds are already unnecessary."
        )

    print("[1/5] Loading flow...")
    flow = load_flow()
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"      params: {n_params:,}")

    if args.unfuse_ln:
        from src.ane_layernorm import ANEUnfusedLayerNorm  # noqa: F401
        # Sanity-check LN parity on a dummy input before swapping in-place, so
        # any arithmetic divergence shows up here rather than after tracing.
        import torch.nn as _nn
        _probe = torch.randn(2, 8, 1024)
        _ref_ln = None
        for _m in flow.decoder.estimator.modules():
            if isinstance(_m, _nn.LayerNorm) and _m.normalized_shape == (1024,):
                _ref_ln = _m
                break
        if _ref_ln is not None:
            with torch.no_grad():
                _ref_out = _ref_ln(_probe)
            _alt = ANEUnfusedLayerNorm(1024, eps=_ref_ln.eps,
                                       elementwise_affine=_ref_ln.elementwise_affine)
            if _ref_ln.elementwise_affine:
                _alt.weight.data.copy_(_ref_ln.weight.data)
                _alt.bias.data.copy_(_ref_ln.bias.data)
            with torch.no_grad():
                _alt_out = _alt(_probe)
            _mae = (_ref_out - _alt_out).abs().max().item()
            print(f"      LN primitive-op parity: max|Δ|={_mae:.2e}")
            if _mae > 1e-5:
                raise RuntimeError(
                    f"ANEUnfusedLayerNorm parity check failed: max|Δ|={_mae}"
                )
        n_replaced = patch_dit_norms(flow.decoder.estimator)
        print(f"      patched {n_replaced} LayerNorm instances with ANEUnfusedLayerNorm")

    if args.fp32_sdpa:
        # Verify fp32 parity of the fp32-cast SDPA processor vs the original
        # F.scaled_dot_product_attention on a dummy input, BEFORE the swap.
        from src.ane_attention import _sdpa_fp32
        import torch.nn.functional as _F
        torch.manual_seed(0)
        _q = torch.randn(2, 8, 16, 64)
        _k = torch.randn(2, 8, 16, 64)
        _v = torch.randn(2, 8, 16, 64)
        _m = torch.ones(2, 8, 16, 16, dtype=torch.bool)
        with torch.no_grad():
            _ref = _F.scaled_dot_product_attention(
                _q, _k, _v, attn_mask=_m, dropout_p=0.0, is_causal=False
            )
            _alt = _sdpa_fp32(_q, _k, _v, attn_mask=_m)
        _mae = (_ref - _alt).abs().max().item()
        print(f"      SDPA fp32-core parity: max|Δ|={_mae:.2e}")
        if _mae > 1e-5:
            raise RuntimeError(
                f"_sdpa_fp32 parity check failed: max|Δ|={_mae}"
            )
        n_attn = patch_dit_attention(flow.decoder.estimator)
        print(f"      patched {n_attn} DiTBlock attention processors with ANEAttnProcessor")

    if args.ane_port:
        print(f"[2/5] Wrapping with FlowCoreMLANE (BC1S port, N_total={N}, M={M})...")
        wrapper = FlowCoreMLANE.build_from_flow(
            flow, n_total_tokens=N, n_timesteps=args.n_timesteps
        ).eval()
        print(f"      ANEDiT: depth={wrapper.ane_dit.depth}, dim={wrapper.ane_dit.dim}")
    else:
        print(f"[2/5] Wrapping with FlowCoreML (N_total={N}, M={M})...")
        wrapper = FlowCoreML(flow, n_total_tokens=N, n_timesteps=args.n_timesteps).eval()

    # Dummy inputs
    torch.manual_seed(0)
    N_prompt = N // 2
    token_total = torch.randint(0, 6561, (1, N), dtype=torch.int64)
    num_prompt_tokens = torch.tensor([N_prompt], dtype=torch.int32)
    prompt_feat = torch.randn(1, M, 80)
    embedding = torch.randn(1, 192)

    print("[3/5] Verifying PyTorch wrapper output...")
    with torch.no_grad():
        full_mel, num_prompt_mel = wrapper(token_total, num_prompt_tokens, prompt_feat, embedding)
    print(f"      mel shape: {tuple(full_mel.shape)}  range=[{full_mel.min().item():.3f}, {full_mel.max().item():.3f}]")
    print(f"      num_prompt_mel: {int(num_prompt_mel.item())}")

    print("[4/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (token_total, num_prompt_tokens, prompt_feat, embedding),
            strict=False,
        )

    print("[5/5] Converting to CoreML mlpackage...")
    precision = _make_precision(args.fp16, args.unfuse_ln, args.fp32_sdpa, args.ane_port)
    pipeline = _make_pass_pipeline(args.unfuse_ln)
    min_dep = {
        "macOS13": ct.target.macOS13,
        "macOS14": ct.target.macOS14,
        "macOS15": ct.target.macOS15,
    }[args.min_deployment]

    # For the ANE port we emit `mel` as fp16. Forcing fp32 on the output
    # inserts a trailing cast that cascades casts back through the final
    # CFG-combine math and `proj_out` conv, evicting ~20 ops to CPU. The
    # Swift consumer can upcast on receive — mel magnitudes (std≈1.2, mean
    # ≈-5.6) are well within fp16 range.
    mel_dtype = np.float16 if args.ane_port else np.float32
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="token_total", shape=(1, N), dtype=np.int32),
            ct.TensorType(name="num_prompt_tokens", shape=(1,), dtype=np.int32),
            ct.TensorType(name="prompt_feat", shape=(1, M, 80), dtype=np.float32),
            ct.TensorType(name="embedding", shape=(1, 192), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="mel", dtype=mel_dtype),
            ct.TensorType(name="num_prompt_mel", dtype=np.int32),
        ],
        compute_precision=precision,
        pass_pipeline=pipeline,
        minimum_deployment_target=min_dep,
        convert_to="mlprogram",
        # Skip the post-convert MLModel load; for large fp16 Flow this triggers
        # anecompilerservice which can hang for >1 h on the DiT graph.
        skip_model_load=True,
    )

    tag = "fp16" if args.fp16 else "fp32"
    if args.unfuse_ln:
        tag += "-unfused"
    if args.fp32_sdpa:
        tag += "-sdpafp32"
    if args.ane_port:
        tag += "-ane"
    mlp = out_dir / f"Flow-N{N}-{tag}.mlpackage"
    mlmodel.save(str(mlp))
    print(f"      saved: {mlp}")

    # Save reference IO for parity
    ref_pt = out_dir / f"ref-N{N}.pt"
    torch.save(
        {
            "token_total": token_total,
            "num_prompt_tokens": num_prompt_tokens,
            "prompt_feat": prompt_feat,
            "embedding": embedding,
            "mel": full_mel,
            "num_prompt_mel": num_prompt_mel,
        },
        str(ref_pt),
    )
    print(f"      ref  : {ref_pt}")


if __name__ == "__main__":
    main()
