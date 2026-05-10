"""Audibility + ANE-residency + warm-latency bench for tail-fp16 variants.

Mirrors Phase F.2's measurement convention so verdicts compare
apples-to-apples. The reference is the production fp32 nanocodec
(`build/nanocodec_decoder_v3.mlpackage` if present, else the deployed
`~/.cache/fluidaudio/Models/magpie-tts/nanocodec_decoder_v3.mlmodelc`).

For each candidate:
  1) Drive both reference and candidate with the same N seeded random
     T=24 chunked-token batches; concatenate the audio outputs.
  2) Report SNR (dB) = 20 log10( rms(ref) / rms(ref - cand) ),
     plus max|delta| and Pearson correlation.
  3) Compile candidate to .mlmodelc and call coreml-cli --fallback
     for ANE residency.
  4) Bench warm latency at .cpuAndNeuralEngine (median of 3 iter, after
     the cold-compile).

Usage:
    uv run python -m experiments.baseline_fp32.tail_fp16_audibility \\
        --candidate build/fp32/nanocodec_tail_fp16_v1.mlpackage \\
        --label v1 \\
        --num-utterances 5 \\
        --frames-per-utterance 72
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import coremltools as ct

_HERE = Path(__file__).resolve().parent
_COREML_DIR = _HERE.parent.parent
_MAGPIE_CACHE = Path("~/.cache/fluidaudio/Models/magpie-tts").expanduser()


# ──────────────── audibility ────────────────


def _seeded_token_chunks(num_utterances: int, frames_per_utt: int,
                         num_codebooks: int, codebook_size: int,
                         frames_per_chunk: int = 24) -> List[List[np.ndarray]]:
    """Generate a list of utterances, each split into T_in=24 chunks."""
    rng = np.random.default_rng(42)
    out: List[List[np.ndarray]] = []
    for u in range(num_utterances):
        n_chunks = max(1, frames_per_utt // frames_per_chunk)
        chunks = []
        for _ in range(n_chunks):
            chunks.append(
                rng.integers(0, codebook_size,
                             size=(1, num_codebooks, frames_per_chunk),
                             dtype=np.int64).astype(np.int32)
            )
        out.append(chunks)
    return out


def _decode_one(model: Any, chunks: List[np.ndarray]) -> np.ndarray:
    audio_segments = []
    for c in chunks:
        out = model.predict({"tokens": c})
        audio_segments.append(np.asarray(out["audio"]).astype(np.float64).reshape(-1))
    return np.concatenate(audio_segments)


def _load_predict(path: Path) -> Any:
    p = str(path)
    if path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(p, compute_units=ct.ComputeUnit.CPU_ONLY)
    return ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_ONLY)


def _snr_db(ref: np.ndarray, cand: np.ndarray) -> Dict[str, float]:
    err = ref - cand
    sig_rms = float(np.sqrt(np.mean(ref * ref)))
    err_rms = float(np.sqrt(np.mean(err * err)))
    snr = 20.0 * np.log10(sig_rms / err_rms) if err_rms > 0 else float("inf")
    denom = float(np.linalg.norm(ref) * np.linalg.norm(cand))
    cos = float(np.dot(ref, cand) / denom) if denom > 0 else float("nan")
    return {
        "snr_db": snr,
        "rms_signal": sig_rms,
        "rms_error": err_rms,
        "max_abs_delta": float(np.max(np.abs(err))),
        "cosine": cos,
        "n_samples": int(ref.size),
    }


def audibility(reference: Path, candidate: Path,
               num_utterances: int, frames_per_utt: int,
               num_codebooks: int, codebook_size: int) -> Dict[str, Any]:
    print(f"[audibility] reference={reference.name} candidate={candidate.name}",
          file=sys.stderr)
    ref_model = _load_predict(reference)
    cand_model = _load_predict(candidate)

    utterances = _seeded_token_chunks(num_utterances, frames_per_utt,
                                      num_codebooks, codebook_size)

    per_utt = []
    for i, chunks in enumerate(utterances):
        ref_audio = _decode_one(ref_model, chunks)
        cand_audio = _decode_one(cand_model, chunks)
        # Trim to common length in case the converter truncates.
        n = min(len(ref_audio), len(cand_audio))
        m = _snr_db(ref_audio[:n], cand_audio[:n])
        m["utterance_index"] = i
        m["n_chunks"] = len(chunks)
        per_utt.append(m)
        print(f"  utt {i}: SNR={m['snr_db']:.2f} dB  "
              f"max|d|={m['max_abs_delta']:.3e}  cos={m['cosine']:.5f}",
              file=sys.stderr)

    # Aggregate: minimum SNR is the conservative metric (worst-case
    # utterance is what you actually hear).
    snrs = [u["snr_db"] for u in per_utt if np.isfinite(u["snr_db"])]
    return {
        "reference": str(reference),
        "candidate": str(candidate),
        "num_utterances": num_utterances,
        "frames_per_utt": frames_per_utt,
        "snr_db_min": float(min(snrs)) if snrs else float("nan"),
        "snr_db_mean": float(np.mean(snrs)) if snrs else float("nan"),
        "snr_db_max": float(max(snrs)) if snrs else float("nan"),
        "max_abs_delta": float(max(u["max_abs_delta"] for u in per_utt)),
        "min_cosine": float(min(u["cosine"] for u in per_utt)),
        "per_utterance": per_utt,
    }


# ──────────────── ANE residency + warm latency ────────────────


_COREML_CLI_DIR = ("/Users/kikow/brandon/voicelink/FluidAudio/mobius"
                   "/tools/coreml-cli")
_COREML_CLI = ["uv", "run", "--directory", _COREML_CLI_DIR, "coreml-cli"]


def _compile_to_mlmodelc(mlpackage: Path) -> Path:
    mlmodelc = mlpackage.with_suffix(".mlmodelc")
    if mlmodelc.exists():
        return mlmodelc
    proc = subprocess.run(
        ["xcrun", "coremlcompiler", "compile",
         str(mlpackage), str(mlmodelc.parent)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"coremlcompiler failed: {proc.stderr}")
    return mlmodelc


def _coreml_cli(mlmodelc: Path, *extra: str,
                plan_timeout: float = 600.0) -> Dict[str, Any]:
    abspath = str(mlmodelc.resolve())
    proc = subprocess.run(
        _COREML_CLI + [abspath, "--json", "--plan-timeout", str(plan_timeout),
                       *extra],
        capture_output=True, text=True, timeout=plan_timeout * 4,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"coreml-cli failed:\nstdout:\n{proc.stdout}"
                           f"\nstderr:\n{proc.stderr}")
    return json.loads(proc.stdout)


def ane_residency(mlmodelc: Path, plan_timeout: float = 600.0) -> Dict[str, Any]:
    j = _coreml_cli(mlmodelc, "--fallback", plan_timeout=plan_timeout)
    fb = j["models"][0]["fallback"]
    return {
        "total_ops": int(fb["total_ops"]),
        "ane_ops": int(fb["ane_ops"]),
        "gpu_ops": int(fb.get("gpu_ops", 0)),
        "cpu_ops": int(fb["cpu_ops"]),
        "ane_pct": float(fb["ane_percent"]),
        "top_reasons": [
            {"reason": r["reason"], "count": int(r["count"])}
            for r in fb["reasons"][:5]
        ],
    }


def warm_latency(mlmodelc: Path, units: str = "cpu_and_neural_engine",
                 iterations: int = 3,
                 plan_timeout: float = 600.0) -> Dict[str, Any]:
    j = _coreml_cli(mlmodelc, "--units", units,
                    "--iterations", str(iterations),
                    plan_timeout=plan_timeout)
    res = next(r for r in j["models"][0]["results"]
               if r["compute_units"] == units)
    return {
        "compute_units": units,
        "median_ms": float(res["latency"]["median_ms"]),
        "mean_ms": float(res["latency"]["mean_ms"]),
        "iterations": int(res["latency"]["iterations"]),
        "summary_pct": res["summary"],
    }


# ──────────────── main ────────────────


def _resolve_reference(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    local_fp32 = (_COREML_DIR / "build" / "nanocodec_decoder_v3.mlpackage")
    if local_fp32.exists():
        return local_fp32
    cached = _MAGPIE_CACHE / "nanocodec_decoder_v3.mlmodelc"
    if cached.exists():
        return cached
    raise SystemExit(
        "no reference mlpackage/mlmodelc found — pass --reference explicitly."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--reference", type=Path, default=None,
                    help="Defaults to build/nanocodec_decoder_v3.mlpackage if "
                         "present, else the cached .mlmodelc.")
    ap.add_argument("--label", required=True,
                    help="Short tag for the variant (v1 / v2 / …)")
    ap.add_argument("--num-utterances", type=int, default=5)
    ap.add_argument("--frames-per-utterance", type=int, default=72)
    ap.add_argument("--num-codebooks", type=int, default=8)
    ap.add_argument("--codebook-size", type=int, default=2024)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--skip-bench", action="store_true",
                    help="Audibility only — skip coreml-cli ANE/latency.")
    ap.add_argument("--plan-timeout", type=float, default=600.0)
    args = ap.parse_args()

    reference = _resolve_reference(args.reference)
    candidate: Path = args.candidate
    if not candidate.exists():
        raise SystemExit(f"candidate not found: {candidate}")

    # 1) audibility
    audi = audibility(reference, candidate,
                      num_utterances=args.num_utterances,
                      frames_per_utt=args.frames_per_utterance,
                      num_codebooks=args.num_codebooks,
                      codebook_size=args.codebook_size)

    summary: Dict[str, Any] = {
        "label": args.label,
        "candidate": str(candidate),
        "reference": str(reference),
        "audibility": audi,
    }

    # 2 + 3) bench
    if not args.skip_bench:
        print("[bench] compiling candidate to .mlmodelc…", file=sys.stderr)
        mlmodelc = _compile_to_mlmodelc(candidate)
        print("[bench] coreml-cli --fallback …", file=sys.stderr)
        ane = ane_residency(mlmodelc, plan_timeout=args.plan_timeout)
        print("[bench] coreml-cli --units cpu_and_neural_engine …",
              file=sys.stderr)
        ne = warm_latency(mlmodelc, "cpu_and_neural_engine",
                          plan_timeout=args.plan_timeout)
        summary["ane_residency"] = ane
        summary["warm_predict_ms"] = ne

    # Markdown row.
    a = summary["audibility"]
    snr_min = a["snr_db_min"]
    parts = [
        f"| {args.label}",
        f"SNR_min={snr_min:.2f} dB",
        f"max|d|={a['max_abs_delta']:.2e}",
    ]
    if "ane_residency" in summary:
        ar = summary["ane_residency"]
        parts.append(f"ANE {ar['ane_pct']:.1f}% "
                     f"(ANE {ar['ane_ops']} / GPU {ar['gpu_ops']} / "
                     f"CPU {ar['cpu_ops']})")
        wl = summary["warm_predict_ms"]
        parts.append(f"warm @cpu+ne {wl['median_ms']:.2f} ms")
    # PASS / FAIL gate: SNR >= 48 dB is *still* the audible threshold per
    # Phase F.2 (v_convs_fp32 came in at 48 dB and was declared noisy).
    # Use 48 dB as the conservative bar; v_full_fp32 sits at 211 dB.
    parts.append("PASS" if snr_min >= 48.0 else "FAIL")
    print(" | ".join(parts) + " |")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2))
        print(f"[bench] wrote {args.out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
