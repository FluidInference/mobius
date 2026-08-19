"""Convert the NeuCodec decoder (codes → 24 kHz audio) to a CoreML mlpackage.

The code-length axis is flexible (RangeDim up to --max-codes); the upstream
RoPE quirk rotates by head index only (see src.codec_coreml.AttentionRope),
so no time-dependent tables are needed. Flexible shapes keep the model off ANE, but the decoder runs once per
utterance on GPU/CPU, where exact-length decode gives bit-comparable parity
with the PyTorch reference.

Pipeline: load neucodec fp32 → wrap (src.codec_coreml, which self-verifies
FSQ dequant + ISTFT against the library) → PyTorch parity on the reference
codes → trace → convert → CoreML parity at multiple lengths.

Usage:
    uv run python convert-codec.py --output-dir ./build/codec --fp16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from src.codec_coreml import HOP, NeuCodecDecoder  # noqa: E402

CODEC_REPO = "neuphonic/neucodec"


def _make_precision(fp16: bool):
    if not fp16:
        return ct.precision.FLOAT32
    # exp/cos/sin feed the ISTFT; softmax for range; norms for stability.
    FP32_OPS = {"pow", "reduce_mean", "rsqrt", "softmax", "exp", "cos", "sin", "layer_norm"}
    return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in FP32_OPS)


def load_codes(default_len: int = 300) -> list[int]:
    ref = HERE / "build" / "ref" / "ref_codes.json"
    if ref.exists():
        return json.loads(ref.read_text())
    rng = np.random.default_rng(0)
    return rng.integers(0, 65_536, default_len).tolist()


def snr_db(got: np.ndarray, want: np.ndarray) -> float:
    noise = got - want
    return 10.0 * np.log10((want**2).sum() / max((noise**2).sum(), 1e-12))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max-codes", type=int, default=2000, help="RangeDim upper bound (40s)")
    p.add_argument("--trace-codes", type=int, default=500)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "fp16" if args.fp16 else "fp32"

    print(f"[1/4] Loading {CODEC_REPO} (fp32, cpu)...")
    from neucodec import NeuCodec

    codec = NeuCodec.from_pretrained(CODEC_REPO)
    codec.eval()

    print("[2/4] Building wrapper (self-verifies FSQ + ISTFT vs library)...")
    wrapper = NeuCodecDecoder(codec).eval()

    codes = load_codes()
    codes_t = torch.tensor(codes, dtype=torch.int32)[None, :]
    with torch.no_grad():
        got = wrapper(codes_t)
        want = codec.decode_code(torch.tensor(codes, dtype=torch.long)[None, None, :])
    got_np = got[0].numpy()
    want_np = want[0, 0, :].numpy()
    assert got_np.shape == want_np.shape, (got_np.shape, want_np.shape)
    print(f"      PyTorch wrapper vs neucodec: max|Δ|={np.abs(got_np - want_np).max():.4e}  "
          f"SNR={snr_db(got_np, want_np):.1f} dB")
    if snr_db(got_np, want_np) < 40:
        raise SystemExit("wrapper parity too low")

    print("[3/4] Tracing + converting...")
    t0 = args.trace_codes
    trace_codes = torch.randint(0, 65_536, (1, t0), dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (trace_codes,), strict=False)

    T = ct.RangeDim(lower_bound=2, upper_bound=args.max_codes, default=t0)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="codes", shape=(1, T), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="audio", dtype=np.float32)],
        compute_precision=_make_precision(args.fp16),
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    mlp = out_dir / f"NeuCodec-Decoder-{tag}.mlpackage"
    mlmodel.save(str(mlp))
    print(f"      saved: {mlp}")

    print("[4/4] CoreML parity at multiple lengths...")
    for t in (125, len(codes), 800):
        if t == len(codes):
            test_codes = codes_t
            ref_audio = want_np
        else:
            test_codes = torch.randint(0, 65_536, (1, t), dtype=torch.int32)
            with torch.no_grad():
                ref_audio = codec.decode_code(test_codes.to(torch.long).unsqueeze(1))[0, 0].numpy()
        pred = mlmodel.predict({"codes": test_codes.numpy().astype(np.int32)})["audio"][0]
        print(f"      T={t}: max|Δ|={np.abs(pred - ref_audio).max():.4e}  "
              f"SNR={snr_db(pred, ref_audio):.1f} dB")

    print("[done]")


if __name__ == "__main__":
    main()
