"""Dump intermediate tensors at every stage boundary for direct
byte-by-byte comparison against Swift's StyleTTS2DebugDump output.

Inputs are pinned via CLI so Python and Swift see the same tokens / ref_s
/ ref_s_pred. With --alpha 0 --beta 0 in Swift, ref_s_pred is unused (the
mix collapses to raw ref_s); pass --no-diffusion here so Python uses
ref_s for both prosody and vocoder.

Usage:
    uv run python 98_dump_for_swift_diff.py \\
        --coreml-dir ../../coreml/build/ane \\
        --tokens-from /tmp/styletts2_dumps_swift/01_tokens_padded.f32.bin \\
        --ref-s ~/.cache/fluidaudio/Models/styletts2/ANE/voices/ref_s_Yinghao.bin \\
        --no-diffusion \\
        --out-dir /tmp/styletts2_dumps_python
"""

from __future__ import annotations

import argparse
import json
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
    install_sinegen_v2_constfold_fix,
    load_modules_for_ane,
)


def dump(name: str, arr: np.ndarray, out_dir: Path) -> None:
    flat = arr.astype(np.float32).reshape(-1)
    bin_path = out_dir / f"{name}.f32.bin"
    json_path = out_dir / f"{name}.json"
    flat.tofile(str(bin_path))
    js = {
        "shape": list(arr.shape),
        "dtype": "f32",
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "sum": float(flat.sum()),
    }
    with open(json_path, "w") as f:
        json.dump(js, f)


def _load_mlmodel(coreml_dir: Path, name: str, compute_units: str = "cpu_and_gpu"):
    import coremltools as ct

    pkg = coreml_dir / f"styletts2_ane_{name}.mlpackage"
    if not pkg.exists():
        raise FileNotFoundError(pkg)
    cu_map = {
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "all": ct.ComputeUnit.ALL,
    }
    return ct.models.MLModel(str(pkg), compute_units=cu_map[compute_units])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--coreml-dir", type=Path,
        default=THIS_DIR.parent.parent / "coreml" / "build" / "ane",
    )
    p.add_argument("--tokens-from", type=Path, required=True,
                   help="Raw fp32 tokens file from Swift dump (cast to int).")
    p.add_argument("--ref-s", type=Path, required=True,
                   help="256-fp32 ref_s.bin path.")
    p.add_argument("--no-diffusion", action="store_true",
                   help="Use ref_s as ref_s_pred (matches Swift --alpha 0 --beta 0).")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/styletts2_dumps_python"))
    p.add_argument("--compute-units", type=str, default="cpu_and_gpu",
                   choices=("cpu_only", "cpu_and_gpu", "cpu_and_ne", "all"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dump] loading modules from {args.checkpoint}")
    modules, cfg = load_modules_for_ane(args.checkpoint)

    # ---- Inputs ----
    tokens = np.fromfile(str(args.tokens_from), dtype=np.float32).astype(np.int64)
    ref_s = np.fromfile(str(args.ref_s), dtype=np.float32)
    assert ref_s.shape == (cfg.style_dim * 2,), f"ref_s shape {ref_s.shape}"

    print(f"[dump] tokens={tokens.tolist()}")
    print(f"[dump] T_tok={tokens.shape[-1]}  ref_s.mean={ref_s.mean():.4f} std={ref_s.std():.4f}")

    dump("01_tokens_padded", tokens.reshape(1, -1), args.out_dir)
    dump("01_ref_s_full", ref_s.reshape(1, -1), args.out_dir)

    style_raw = torch.from_numpy(ref_s).float().unsqueeze(0)  # (1, 256)
    s_acoustic = style_raw[:, : cfg.style_dim]
    s_pros = style_raw[:, cfg.style_dim:]

    # If --no-diffusion, the "predicted" style is the raw ref_s (matches
    # Swift's --alpha 0 --beta 0 mix collapse).
    style_pred = style_raw if args.no_diffusion else None
    if style_pred is None:
        raise NotImplementedError("Run with --no-diffusion for now.")
    s_pros_pred = style_pred[:, cfg.style_dim:]
    s_acoustic_pred = style_pred[:, : cfg.style_dim]

    T_tok = int(tokens.shape[-1])

    print("[dump] loading 6 mlpackages...")
    mls = {}
    for name in ("plbert", "postbert", "alignment", "prosody", "noise", "vocoder"):
        print(f"  - {name}")
        mls[name] = _load_mlmodel(args.coreml_dir, name, args.compute_units)

    with torch.no_grad():
        # ─── Stage 1: PLBert ───
        out = mls["plbert"].predict({
            "tokens": tokens.astype(np.int32).reshape(1, T_tok),
        })
        bert_dur_np = next(iter(out.values())).astype(np.float32)
        print(f"[dump] PLBert out shape={bert_dur_np.shape}")
        dump("02_plbert_bert_dur", bert_dur_np, args.out_dir)
        bert_dur = torch.from_numpy(bert_dur_np)

        # ─── Stage 2: PostBert ───
        out = mls["postbert"].predict({
            "bert_dur": bert_dur.numpy().astype(np.float32),
            "tokens": tokens.astype(np.int32).reshape(1, T_tok),
            "style": style_raw.numpy().astype(np.float32),
        })
        t_en = out["t_en"].astype(np.float32)
        d = out["d"].astype(np.float32)
        pred_dur_log = out["pred_dur_log"].astype(np.float32)
        print(f"[dump] PostBert: t_en={t_en.shape} d={d.shape} pred_dur_log={pred_dur_log.shape}")
        dump("03_postbert_t_en", t_en, args.out_dir)
        dump("03_postbert_d", d, args.out_dir)
        dump("03_postbert_pred_dur_log", pred_dur_log, args.out_dir)
        t_en_t = torch.from_numpy(t_en)
        d_t = torch.from_numpy(d)
        pred_dur_log_t = torch.from_numpy(pred_dur_log)

        # Compute pred_dur identically to Swift's computeDurations:
        # round(sigmoid(logit).sum(-1)).clamp(min=1).
        pred_dur = torch.sigmoid(pred_dur_log_t).sum(-1).round().clamp(min=1).long()  # (1, T_tok)
        T_a = int(pred_dur.sum().item())
        print(f"[dump] pred_dur sum = T_a = {T_a}  (per-token: {pred_dur[0].tolist()})")
        dump("04_pred_dur", pred_dur.float().numpy(), args.out_dir)

        # ─── Stage 3 (Swift labels: 5): Alignment ───
        out = mls["alignment"].predict({
            "pred_dur": pred_dur.float().numpy().astype(np.float32),
            "d": d_t.numpy().astype(np.float32),
            "t_en": t_en_t.numpy().astype(np.float32),
        })
        en = out["en"].astype(np.float32)
        asr = out["asr"].astype(np.float32)
        print(f"[dump] Alignment: en={en.shape} asr={asr.shape}")
        dump("07_alignment_en", en, args.out_dir)
        dump("07_alignment_asr", asr, args.out_dir)
        en_t = torch.from_numpy(en)
        asr_t = torch.from_numpy(asr)

        # Style mix dump for parity (with --no-diffusion these are just halves).
        dump("06_acoustic_mix", s_acoustic_pred.numpy(), args.out_dir)
        dump("06_prosody_mix", s_pros_pred.numpy(), args.out_dir)

        # ─── Stage 5: Prosody ───
        out = mls["prosody"].predict({
            "en": en_t.numpy().astype(np.float32),
            "s": s_pros_pred.numpy().astype(np.float32),
        })
        F0 = out["F0"].astype(np.float32)
        N = out["N"].astype(np.float32)
        print(f"[dump] Prosody: F0={F0.shape} N={N.shape}")
        dump("08_prosody_F0", F0, args.out_dir)
        dump("08_prosody_N", N, args.out_dir)
        F0_t = torch.from_numpy(F0)
        N_t = torch.from_numpy(N)

        # ─── Stage 6: Noise ───
        # Static at MAX_T_A*2 — Stage 5 already returned MAX_T_A*2.
        F0_padded = F0_t
        out = mls["noise"].predict({
            "F0_curve": F0_padded.numpy().astype(np.float32),
        })
        sine = out["sine_waves"].astype(np.float32)
        print(f"[dump] Noise: sine={sine.shape}")
        dump("09_noise_sine_waves", sine, args.out_dir)
        sine_t = torch.from_numpy(sine)

        # ─── Stage 7: Vocoder (active T_a slice — RangeDim) ───
        F0_act = F0_t[:, : T_a * 2]
        N_act = N_t[:, : T_a * 2]
        sine_act = sine_t[:, : T_a * 2 * UPSAMPLE_SCALE, :]
        asr_act = asr_t[:, :, :T_a].contiguous()
        print(f"[dump] Vocoder inputs: asr={asr_act.shape} F0={F0_act.shape} "
              f"N={N_act.shape} sine={sine_act.shape}")
        dump("10_vocoder_input_asr_sliced", asr_act.numpy(), args.out_dir)
        dump("10_vocoder_input_F0_sliced", F0_act.numpy(), args.out_dir)
        dump("10_vocoder_input_N_sliced", N_act.numpy(), args.out_dir)
        dump("10_vocoder_input_sine_sliced", sine_act.numpy(), args.out_dir)

        out = mls["vocoder"].predict({
            "asr": asr_act.numpy().astype(np.float32),
            "F0_curve": F0_act.numpy().astype(np.float32),
            "N": N_act.numpy().astype(np.float32),
            "s": s_acoustic_pred.numpy().astype(np.float32),
            "sine_waves": sine_act.numpy().astype(np.float32),
        })
        audio = out["audio"].astype(np.float32)
        print(f"[dump] Vocoder: audio={audio.shape} peak={np.abs(audio).max():.4f}")
        dump("10_vocoder_audio", audio, args.out_dir)

        # Write a quick WAV for listening.
        import wave
        wav_path = args.out_dir / "python_dump.wav"
        pcm = np.clip(audio.flatten(), -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        with wave.open(str(wav_path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(cfg.sample_rate)
            f.writeframes(pcm16.tobytes())
        print(f"[dump] wav written → {wav_path}")

    print(f"[dump] all dumps in {args.out_dir}")


if __name__ == "__main__":
    main()
