"""Trial 10b: decoder_upsample, Conv1d → Conv2d rewrite, fp32, ANE probe.

Why
---
Trial 10 confirmed: ANE refuses `decoder_upsample` even at fixed shape
fp32. `CPU_AND_NE` ran *slower* than `CPU_ONLY` (ANE-attempt-then-
fallback signature). Conclusion: dynamic shape was not the blocker;
ConvTranspose1d itself is off-limits to ANE.

ANE has tuned Conv2d / ConvTranspose2d kernels but no 1D variants.
This trial substitutes every `nn.Conv1d` and `nn.ConvTranspose1d` in
the HiFi-GAN generator with drop-in 2D replacements that internally
`unsqueeze(H=1) → conv2d → squeeze(H)`. The 1D weight `[C_out, C_in/G,
K]` becomes `[C_out, C_in/G, 1, K]` (just an `unsqueeze(2)` of the
same parameter). All other generator code (resblocks, AdaIN, source
filter, residual adds) is untouched — the (B, C, T) signature is
preserved at every replacement boundary, so the per-op unsqueeze/
squeeze pairs become candidates for MIL's redundant-op folding pass.

If ANE accepts the rewritten graph, the win is potentially large
(decoder_upsample is 84 % of pipeline). If ANE still rejects, the
diagnosis is structural — some non-conv op in the subgraph (likely
AdaIN, source filter add, or the leaky ReLU pattern) is the blocker —
and we move on to vocoder swap.

Output
------
* `coreml/packages/decoder_upsample_trial10b_fp32_conv2d.mlpackage`

Bench
-----
Same protocol as Trial 10: load + warm × 8 under CPU_ONLY, CPU_AND_NE,
ALL; parity vs eager rewritten wrapper (and vs eager *original* wrapper
to confirm the 1D→2D swap is mathematically a no-op).

Run
---
    cd models/tts/styletts2
    uv run python coreml/trial10b_decoder_upsample_conv2d.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coreml import convert as _convert  # noqa: F401  (installs MIL patches)
from coreml._runtime import HERE, build_runtime, stage_example_inputs
from coreml.wrappers import build_wrapper

PACKAGES_DIR = HERE / "coreml" / "packages"
PACKAGES_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = PACKAGES_DIR / "decoder_upsample_trial10b_fp32_conv2d.mlpackage"


# ---------------------------------------------------------------------------
# Drop-in 2D replacements
# ---------------------------------------------------------------------------


class Conv1dAs2d(nn.Module):
    """Behaviourally equivalent to nn.Conv1d, internally a Conv2d on H=1."""

    def __init__(self, src: nn.Conv1d):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=src.in_channels,
            out_channels=src.out_channels,
            kernel_size=(1, src.kernel_size[0]),
            stride=(1, src.stride[0]),
            padding=(0, src.padding[0]),
            dilation=(1, src.dilation[0]),
            groups=src.groups,
            bias=src.bias is not None,
            padding_mode=src.padding_mode,
        )
        with torch.no_grad():
            # 1D weight [C_out, C_in/G, K]  ->  2D weight [C_out, C_in/G, 1, K]
            self.conv.weight.copy_(src.weight.detach().unsqueeze(2))
            if src.bias is not None:
                self.conv.bias.copy_(src.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]  ->  [B, C, 1, T]  -> conv -> [B, C', 1, T'] -> [B, C', T']
        return self.conv(x.unsqueeze(2)).squeeze(2)


class ConvTranspose1dAs2d(nn.Module):
    """Behaviourally equivalent to nn.ConvTranspose1d, internally
    ConvTranspose2d on H=1."""

    def __init__(self, src: nn.ConvTranspose1d):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=src.in_channels,
            out_channels=src.out_channels,
            kernel_size=(1, src.kernel_size[0]),
            stride=(1, src.stride[0]),
            padding=(0, src.padding[0]),
            output_padding=(0, src.output_padding[0]),
            dilation=(1, src.dilation[0]),
            groups=src.groups,
            bias=src.bias is not None,
            padding_mode=src.padding_mode,
        )
        with torch.no_grad():
            # ConvTranspose1d weight [C_in, C_out/G, K] ->
            # ConvTranspose2d weight [C_in, C_out/G, 1, K]
            self.conv.weight.copy_(src.weight.detach().unsqueeze(2))
            if src.bias is not None:
                self.conv.bias.copy_(src.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.unsqueeze(2)).squeeze(2)


def _swap_convs_inplace(root: nn.Module) -> dict[str, int]:
    """Recursively replace every Conv1d / ConvTranspose1d under `root`
    with the 2D-backed analog. Returns a count of substitutions."""
    counts = {"Conv1d": 0, "ConvTranspose1d": 0}
    for name, child in list(root.named_children()):
        if isinstance(child, nn.ConvTranspose1d):
            setattr(root, name, ConvTranspose1dAs2d(child))
            counts["ConvTranspose1d"] += 1
        elif isinstance(child, nn.Conv1d):
            setattr(root, name, Conv1dAs2d(child))
            counts["Conv1d"] += 1
        else:
            sub = _swap_convs_inplace(child)
            for k, v in sub.items():
                counts[k] += v
    return counts


# ---------------------------------------------------------------------------
# Convert + bench
# ---------------------------------------------------------------------------


def _metric(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    af, bf = a.flatten(), b.flatten()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))
    return {
        "shape": tuple(a.shape),
        "mse": float(np.mean(diff * diff)),
        "max_abs_delta": float(np.max(np.abs(diff))),
        "rms_a": float(np.sqrt(np.mean(a * a))),
        "rms_b": float(np.sqrt(np.mean(b * b))),
        "cos": cos,
    }


def convert_conv2d():
    import coremltools as ct

    print("=== Trial 10b convert: decoder_upsample fp32 + Conv1d→Conv2d rewrite ===")
    rt = build_runtime()
    wrapper = build_wrapper("decoder_upsample", rt.model)
    example_inputs = stage_example_inputs("decoder_upsample", rt)

    x_pre, ref, har = example_inputs
    print(f"  x_pre      = {tuple(x_pre.shape)} {x_pre.dtype}")
    print(f"  ref        = {tuple(ref.shape)} {ref.dtype}")
    print(f"  har_source = {tuple(har.shape)} {har.dtype}")

    # Eager output BEFORE swap — ground truth.
    with torch.no_grad():
        eager_out_orig = wrapper(*example_inputs)
    print(f"  eager (1D) wav = {tuple(eager_out_orig.shape)} {eager_out_orig.dtype}")

    # Swap every Conv1d / ConvTranspose1d under the wrapper's generator.
    # Wrapper's __init__ has already stripped weight_norm so the convs have
    # plain weight/bias.
    counts = _swap_convs_inplace(wrapper)
    print(f"  swapped: Conv1d×{counts['Conv1d']}  ConvTranspose1d×{counts['ConvTranspose1d']}")

    # Eager output AFTER swap — must match the 1D version bit-equivalently
    # (the swap is mathematically a no-op).
    with torch.no_grad():
        eager_out_2d = wrapper(*example_inputs)
    swap_metric = _metric(eager_out_orig.numpy(), eager_out_2d.numpy())
    print(
        f"  swap parity (1D vs 2D): cos={swap_metric['cos']:.6f} "
        f"max|d|={swap_metric['max_abs_delta']:.3e}"
    )
    if swap_metric["max_abs_delta"] > 1e-4:
        raise SystemExit(
            "ABORT: 1D→2D swap is not a no-op. Investigate before converting."
        )

    print("  tracing rewritten wrapper ...")
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, example_inputs, check_trace=False, strict=False
        )

    print("  ct.convert (fp32, fixed shapes) ...")
    inputs = [
        ct.TensorType(name="x_pre", shape=tuple(x_pre.shape), dtype=np.float32),
        ct.TensorType(name="ref", shape=tuple(ref.shape), dtype=np.float32),
        ct.TensorType(name="har_source", shape=tuple(har.shape), dtype=np.float32),
    ]
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"  ct.convert: {time.time() - t0:.1f}s")

    if OUT_PATH.exists():
        import shutil

        shutil.rmtree(OUT_PATH)
    mlmodel.save(str(OUT_PATH))
    print(f"  saved {OUT_PATH.relative_to(HERE)}")
    return example_inputs, eager_out_orig


def bench(example_inputs, eager_out):
    import coremltools as ct

    feed = {
        "x_pre": example_inputs[0].detach().numpy().astype(np.float32),
        "ref": example_inputs[1].detach().numpy().astype(np.float32),
        "har_source": example_inputs[2].detach().numpy().astype(np.float32),
    }
    eager_np = eager_out.detach().numpy().astype(np.float32)

    placements = [
        ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("ALL", ct.ComputeUnit.ALL),
    ]

    print("\n=== Trial 10b bench (fp32 fixed-shape Conv2d-rewritten) ===")
    for name, units in placements:
        print(f"\n  --- {name} ---")
        t0 = time.time()
        m = ct.models.MLModel(str(OUT_PATH), compute_units=units)
        load_ms = (time.time() - t0) * 1000.0

        for _ in range(3):
            m.predict(feed)

        timings = []
        for _ in range(8):
            t1 = time.time()
            out = m.predict(feed)
            timings.append((time.time() - t1) * 1000.0)
        timings.sort()
        out_arr = np.asarray(list(out.values())[0])
        met = _metric(eager_np, out_arr)
        med = timings[len(timings) // 2]
        avg = sum(timings) / len(timings)
        spread = timings[-1] - timings[0]
        print(
            f"  load={load_ms:6.0f}ms  warm: min={timings[0]:6.1f} med={med:6.1f} "
            f"avg={avg:6.1f} max={timings[-1]:6.1f}  spread={spread:5.1f} ms"
        )
        print(
            f"  parity vs eager (1D): cos={met['cos']:.6f}  max|d|={met['max_abs_delta']:.3e}"
        )


def main():
    example_inputs, eager_out = convert_conv2d()
    bench(example_inputs, eager_out)


if __name__ == "__main__":
    main()
