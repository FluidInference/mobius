"""Stage 7/7 — Vocoder (HiFi-GAN body, no SineGen).

ANE-resident, fp16 + int8pal kmeans. Single fixed shape at T_mel = MAX_T_A.

The cos-Snake patch (`install_cos_snake_patch`) is applied via
`load_modules_for_ane` so AdaINResBlock1 is composed of mul/add/cos/conv only.

Inputs (T_a = MAX_T_A; the host pads):
  asr        [1, 512, MAX_T_A]               fp32
  F0_curve   [1, MAX_T_A * 2]                fp32  (raw F0 from Prosody, upsampled)
  N          [1, MAX_T_A * 2]                fp32  (raw N  from Prosody, upsampled)
  s          [1, 128]                        fp32  (acoustic half of ref_s — ref_s[:, 128:])
  sine_waves [1, MAX_T_A * 2 * UPSAMPLE_SCALE, harm+1] fp32  (from Stage 6)

Output:
  audio      [1, T_audio]                    fp32
  T_audio = MAX_T_A * 2 * UPSAMPLE_SCALE  (UPSAMPLE_SCALE = 300 for LibriTTS HiFi-GAN)

Notes:
  - StyleTTS2's HiFi-GAN is iSTFT-free, so Kokoro's separate Tail collapses
    here (no second graph needed).
  - F0_conv / N_conv have stride=2 to bring the upsampled F0/N back down to
    T_a so they can be cat'd with asr at T_a; this is identical to upstream
    `Modules.hifigan.Decoder.forward`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from _styletts2_ane_lib import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    MAX_T_A,
    UPSAMPLE_SCALE,
    VocoderTraceable,
    install_sinegen_v2_constfold_fix,
    load_modules_for_ane,
    palettize_int8,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out-dir", type=Path,
        default=THIS_DIR.parent.parent / "coreml" / "build" / "ane",
    )
    parser.add_argument("--no-palettize", action="store_true")
    parser.add_argument("--trace-only", action="store_true")
    parser.add_argument("--max-T-a", type=int, default=MAX_T_A)
    parser.add_argument("--example-T-a", type=int, default=120,
                        help="Example T_a used for tracing (RangeDim will cover 2..max-T-a).")
    parser.add_argument("--fp32", action="store_true",
                        help="Diagnostic: keep compute_precision=FLOAT32 to isolate fp16 bugs.")
    parser.add_argument("--static", action="store_true",
                        help="Force static shapes at max-T-a (legacy, broken at boundary).")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[07-ane] loading modules …")
    modules, cfg = load_modules_for_ane(args.checkpoint)

    # We don't trace through SineGen here (Stage 6 owns it), but the shared
    # SineGen const-fold patch sets `fracs_tensor` for downstream linear+tanh
    # consumers — install for safety.
    install_sinegen_v2_constfold_fix(t_mel=args.max_T_a)

    wrapper = VocoderTraceable(modules["decoder"]).eval()

    # Trace at a small example T_a so the trace path is short. RangeDim below
    # lets the converted graph accept any T_a in [2, max_T_a] at predict-time.
    # NOTE: HiFi-GAN's stack of conv-transpose ups + noise_convs is highly
    # sensitive to input length at the active/zero boundary — running zero-padded
    # at MAX_T_A and slicing the active region produces +20 dB of edge-effect
    # leakage (cos vs unpadded ≈ 0.22). RangeDim avoids this entirely by running
    # at the actual active T_a.
    T_a = args.max_T_a if args.static else args.example_T_a

    # asr is at T_a; F0_curve and N are at T_a*2 (Prosody's F0/N have upsample=True);
    # the F0_conv/N_conv stride=2 brings them down to T_a before cat with asr.
    # The decoder consumes only the acoustic half of ref_s (style_dim, not style_dim*2).
    asr_e = torch.zeros(1, cfg.hidden_dim, T_a, dtype=torch.float32)
    F0_e = torch.zeros(1, T_a * 2, dtype=torch.float32)
    N_e = torch.zeros(1, T_a * 2, dtype=torch.float32)
    s_e = torch.zeros(1, cfg.style_dim, dtype=torch.float32)

    # SineGen output shape is (1, T_a * 2 * UPSAMPLE_SCALE, harm+1).
    harm = modules["decoder"].generator.m_source.l_sin_gen.harmonic_num
    T_audio_chunk = T_a * 2 * UPSAMPLE_SCALE
    sine_e = torch.zeros(1, T_audio_chunk, harm + 1, dtype=torch.float32)

    print(f"[07-ane] sanity forward (static T_a={T_a}, T_audio_chunk={T_audio_chunk}) …")
    with torch.no_grad():
        audio = wrapper(asr_e, F0_e, N_e, s_e, sine_e)
    print(f"[07-ane]   audio: {tuple(audio.shape)} {audio.dtype}")

    print("[07-ane] tracing …")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (asr_e, F0_e, N_e, s_e, sine_e), strict=False)

    if args.trace_only:
        print("[07-ane] --trace-only: skipping CoreML convert.")
        return

    import coremltools as ct

    out_path = args.out_dir / "styletts2_ane_vocoder.mlpackage"
    if args.static:
        asr_shape = (1, cfg.hidden_dim, T_a)
        F0_shape = (1, T_a * 2)
        N_shape = (1, T_a * 2)
        sine_shape = (1, T_audio_chunk, harm + 1)
        mode_str = f"STATIC T_a={T_a}"
    else:
        rd_T_a = ct.RangeDim(lower_bound=2, upper_bound=args.max_T_a, default=T_a)
        rd_T_a2 = ct.RangeDim(lower_bound=4, upper_bound=args.max_T_a * 2, default=T_a * 2)
        rd_T_audio = ct.RangeDim(
            lower_bound=4 * UPSAMPLE_SCALE,
            upper_bound=args.max_T_a * 2 * UPSAMPLE_SCALE,
            default=T_audio_chunk,
        )
        asr_shape = (1, cfg.hidden_dim, rd_T_a)
        F0_shape = (1, rd_T_a2)
        N_shape = (1, rd_T_a2)
        sine_shape = (1, rd_T_audio, harm + 1)
        mode_str = f"RangeDim T_a∈[2,{args.max_T_a}] (default={T_a})"
    print(f"[07-ane] converting → {out_path.name} (cpu_and_ne, fp16, {mode_str}) …", flush=True)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="asr", shape=asr_shape, dtype=np.float32),
            ct.TensorType(name="F0_curve", shape=F0_shape, dtype=np.float32),
            ct.TensorType(name="N", shape=N_shape, dtype=np.float32),
            ct.TensorType(name="s", shape=(1, cfg.style_dim), dtype=np.float32),
            ct.TensorType(name="sine_waves", shape=sine_shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="audio")],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32 if args.fp32 else ct.precision.FLOAT16,
        skip_model_load=True,
    )
    mlmodel.short_description = (
        f"StyleTTS2-ANE Stage 7: HiFi-GAN body (cos-Snake, {mode_str}) — fp16, ANE."
    )
    mlmodel.save(str(out_path))
    print(f"[07-ane]   saved → {out_path}")

    if not args.no_palettize:
        print("[07-ane] palettizing …", flush=True)
        palettize_int8(out_path)
        print(f"[07-ane]   palettized → {out_path}")


if __name__ == "__main__":
    main()
