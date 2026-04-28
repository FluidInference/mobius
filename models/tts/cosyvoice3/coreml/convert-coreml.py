"""Convert CosyVoice3 CausalHiFTGenerator to CoreML mlpackage.

Pipeline:
  1. Load upstream CausalHiFTGenerator from `cosyvoice3_dl/`
  2. Wrap with `src.hift_coreml.HiFTCoreML` (patches SineGen, folds weight_norm,
     swaps STFT/iSTFT for matmul variants)
  3. torch.jit.trace on CPU
  4. coremltools convert → mlpackage (FP32 by default; FP16 optional)
  5. Save a PyTorch reference output alongside the mlpackage for parity tests.

Usage:
    uv run python convert-coreml.py --output-dir ./build/hift-fp32
    uv run python convert-coreml.py --output-dir ./build/hift-fp16 --fp16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "verify" / "CosyVoice"))
sys.path.insert(0, str(HERE / "verify" / "CosyVoice" / "third_party" / "Matcha-TTS"))

from src.hift_coreml import HiFTCoreML  # noqa: E402


def _make_precision(fp16: bool):
    """FP16 everywhere except RMSNorm-style / softmax ops.

    HiFT's vocoder stack is mostly conv/upsample (fp16-safe) but the
    SineGen front-end does a long phase cumsum that accumulates error in
    fp16 — the cumsum already CPU-falls back on ANE, so keeping it fp32
    is free.  pow/reduce_mean/rsqrt/softmax listed for parity with the
    LLM recipe; rarely present in HiFT but harmless if they are.
    """
    if not fp16:
        return ct.precision.FLOAT32
    FP32_OPS = {"pow", "reduce_mean", "rsqrt", "softmax", "cumsum"}
    return ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type not in FP32_OPS
    )


def load_generator() -> torch.nn.Module:
    from hyperpyyaml import load_hyperpyyaml

    yaml_path = HERE / "cosyvoice3_dl" / "cosyvoice3.yaml"
    hift_pt = HERE / "cosyvoice3_dl" / "hift.pt"
    with open(yaml_path, "r") as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
    gen = cfg["hift"]
    sd = torch.load(str(hift_pt), map_location="cpu", weights_only=False)
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    gen.eval()
    return gen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True, help="Directory for mlpackage + reference outputs")
    p.add_argument("--mel-frames", type=int, default=250, help="Fixed mel length for conversion (T)")
    p.add_argument("--fp16", action="store_true", help="Emit FP16 weights (default: FP32)")
    p.add_argument("--min-deployment", default="macOS14", choices=["macOS13", "macOS14", "macOS15"])
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading CausalHiFTGenerator...")
    gen = load_generator()
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"      params: {n_params:,}")

    print("[2/5] Wrapping with HiFTCoreML (SineGen patch, weight_norm fold, matmul STFT)...")
    wrapper = HiFTCoreML(gen).eval()

    T = args.mel_frames
    mel = torch.randn(1, 80, T)
    num_valid_frames = torch.tensor([T], dtype=torch.int32)

    print("[3/5] Verifying PyTorch wrapper output...")
    with torch.no_grad():
        audio_torch, alen_torch = wrapper(mel, num_valid_frames)
    print(f"      mel   : {tuple(mel.shape)}")
    print(f"      audio : {tuple(audio_torch.shape)}  range=[{audio_torch.min().item():.3f}, {audio_torch.max().item():.3f}]")
    print(f"      alen  : {int(alen_torch.item())} samples")

    print("[4/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (mel, num_valid_frames), strict=False)

    print("[5/5] Converting to CoreML mlpackage...")
    precision = _make_precision(args.fp16)
    min_dep = {
        "macOS13": ct.target.macOS13,
        "macOS14": ct.target.macOS14,
        "macOS15": ct.target.macOS15,
    }[args.min_deployment]

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=(1, 80, T), dtype=np.float32),
            ct.TensorType(name="num_valid_frames", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="audio", dtype=np.float32),
            ct.TensorType(name="audio_length_samples", dtype=np.int32),
        ],
        compute_precision=precision,
        minimum_deployment_target=min_dep,
        convert_to="mlprogram",
    )

    tag = "fp16" if args.fp16 else "fp32"
    mlp = out_dir / f"HiFT-T{T}-{tag}.mlpackage"
    mlmodel.save(str(mlp))
    print(f"      saved: {mlp}")

    # Save reference IO for parity test
    ref_pt = out_dir / f"ref-T{T}.pt"
    torch.save(
        {"mel": mel, "num_valid_frames": num_valid_frames, "audio": audio_torch, "audio_length_samples": alen_torch},
        str(ref_pt),
    )
    print(f"      ref  : {ref_pt}")


if __name__ == "__main__":
    main()
