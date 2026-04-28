"""Export Package D: HiFi-GAN Decoder.

Frame-rate convention (matches upstream demo):
  T_mel here = asr / DurationEncoder frame rate (≈80 fps at 24 kHz × hop=300).
  F0Ntrain emits F0/N at 2× that rate (its AdainResBlk1d stack upsamples ×2),
  and `decoder.F0_conv` / `decoder.N_conv` downsample by 2 with their stride.

Inputs:
  asr:      (1, 512, T_mel)
  F0_curve: (1, 2*T_mel)
  N:        (1, 2*T_mel)
  s:        (1, 128)            (the `ref` half of the predicted style)

Outputs:
  waveform: (1, 1, T_mel * 600) at 24 kHz
            (decoder.decode's last AdainResBlk1d has upsample=True (×2),
             then Generator upsamples by 300 = ∏[10,5,3,2])

Compute units: ALL (ANE-eligible). Falls back to CPU/GPU if any
ConvTranspose1d in `Generator` rejects ANE placement — verify post-export.

Bucketing: CoreML mlprogram only allows one EnumeratedShapes input per model
and disallows mixing EnumeratedShapes with RangeDim. The decoder has three
variable-T inputs (asr, F0_curve, N), so a single multi-bucket mlpackage
fails to compile at runtime. The script emits one fixed-shape mlpackage per
bucket: `styletts2_decoder_<T_mel>.mlpackage`. Callers route by length.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _styletts2_lib import (  # noqa: E402
    COREML_DIR,
    DEFAULT_CHECKPOINT,
    HifiGanDecoderTraceable,
    LibriTTSConfig,
    load_inference_modules,
    register_coreml_op_shims,
)

register_coreml_op_shims()

DEFAULT_MEL_BUCKETS = (256, 512, 1024, 2048, 4096)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out-dir", type=Path, default=COREML_DIR,
        help="Directory; per-bucket mlpackages are saved as styletts2_decoder_<T_mel>.mlpackage.",
    )
    parser.add_argument("--mel-buckets", type=int, nargs="+", default=list(DEFAULT_MEL_BUCKETS))
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument(
        "--compute-units",
        choices=["all", "cpu_and_gpu", "cpu_only"],
        default="all",
        help="If post-export ANE placement check fails, re-run with cpu_and_gpu.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[04] loading inference modules …")
    modules, cfg = load_inference_modules(args.checkpoint)

    print("[04] building HifiGanDecoderTraceable …")
    wrapper = HifiGanDecoderTraceable(modules["decoder"]).eval()

    if args.trace_only:
        largest = max(args.mel_buckets)
        example_asr = torch.zeros(1, cfg.hidden_dim, largest, dtype=torch.float32)
        example_F0 = torch.zeros(1, largest * 2, dtype=torch.float32)
        example_N = torch.zeros(1, largest * 2, dtype=torch.float32)
        example_s = torch.zeros(1, cfg.style_dim, dtype=torch.float32)
        print(f"[04] sanity forward at T_mel={largest} …")
        with torch.no_grad():
            wav = wrapper(example_asr, example_F0, example_N, example_s)
        expected = largest * 2 * cfg.hop_factor
        print(f"[04]   waveform: {tuple(wav.shape)}  (expected last dim = {expected})")
        print("[04] tracing (largest bucket only) …")
        with torch.no_grad():
            torch.jit.trace(
                wrapper, (example_asr, example_F0, example_N, example_s), strict=False
            )
        print("[04] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct

    cu = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }[args.compute_units]

    # CoreML mlprogram allows only one EnumeratedShapes input per model and
    # disallows mixing EnumeratedShapes with RangeDim. With three variable-T
    # inputs (asr, F0, N) any combined-flexibility export fails to compile.
    # Emit one fixed-shape mlpackage per bucket; routing happens at runtime.
    for t_mel in sorted(args.mel_buckets):
        out_path = args.out_dir / f"styletts2_decoder_{t_mel}.mlpackage"
        print(f"[04] === T_mel={t_mel}  →  {out_path.name} ===")

        example_asr = torch.zeros(1, cfg.hidden_dim, t_mel, dtype=torch.float32)
        example_F0 = torch.zeros(1, t_mel * 2, dtype=torch.float32)
        example_N = torch.zeros(1, t_mel * 2, dtype=torch.float32)
        example_s = torch.zeros(1, cfg.style_dim, dtype=torch.float32)

        with torch.no_grad():
            wav = wrapper(example_asr, example_F0, example_N, example_s)
        expected = t_mel * 2 * cfg.hop_factor
        print(f"[04]   sanity waveform: {tuple(wav.shape)}  (expected last dim = {expected})")

        print("[04]   tracing …")
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper, (example_asr, example_F0, example_N, example_s), strict=False
            )

        print(f"[04]   converting (compute_units={args.compute_units}) …", flush=True)
        # skip_model_load avoids the post-convert MLModel instantiation which
        # otherwise hangs Apple's anecompilerservice for several minutes per
        # bucket. The mlpackage is still saved correctly; first runtime load
        # in the consumer pays the ANE compile cost once, then it's cached.
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="asr", shape=(1, cfg.hidden_dim, t_mel), dtype=np.float32),
                ct.TensorType(name="F0_curve", shape=(1, 2 * t_mel), dtype=np.float32),
                ct.TensorType(name="N", shape=(1, 2 * t_mel), dtype=np.float32),
                ct.TensorType(name="s", shape=(1, cfg.style_dim), dtype=np.float32),
            ],
            outputs=[ct.TensorType(name="waveform")],
            minimum_deployment_target=ct.target.iOS17,
            compute_units=cu,
            convert_to="mlprogram",
            skip_model_load=True,
        )
        mlmodel.short_description = (
            f"StyleTTS2 HiFi-GAN decoder (LibriTTS) @ T_mel={t_mel}. 24 kHz, hop=300."
        )
        mlmodel.save(str(out_path))
        print(f"[04]   saved → {out_path}")


if __name__ == "__main__":
    main()
