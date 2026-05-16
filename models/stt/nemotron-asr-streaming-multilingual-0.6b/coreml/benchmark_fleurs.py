#!/usr/bin/env python3
"""
FLEURS multilingual benchmark for Nemotron Multilingual Streaming 0.6B.

Runs the CoreML pipeline across one or more FLEURS test subsets and reports:

  - WER (Latin-script langs) or CER (CJK / Thai)
  - Detection accuracy when `--mode auto` is used (does the leading
    <xx-XX> token match the FLEURS subset label?)
  - RTFx (real-time factor) per language

Reuses `NemotronMultilingualCoreML` from `test_coreml_multilingual.py`
unchanged, so the inference path is byte-identical to the smoke test.

Two ways to provide FLEURS:

  1. `--use-hf`              — stream the dataset via `datasets` from
                               `google/fleurs`. Requires `pip install datasets`.
  2. `--fleurs-root <dir>`   — point at a local FLEURS download with
                               <lang>/test/*.wav + <lang>/test.tsv.

Example:

    uv run python benchmark_fleurs.py \\
        --model-dir ./build_fp16 \\
        --use-hf \\
        --languages cmn_hans_cn,en_us,es_419 \\
        --mode auto \\
        --max-files-per-lang 100
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np

# Reuse the inference class verbatim
sys.path.insert(0, str(Path(__file__).parent))
from test_coreml_multilingual import NemotronMultilingualCoreML  # noqa: E402
from fleurs_lang_map import fleurs_to_nemotron, uses_cer  # noqa: E402


# ---------------------------------------------------------------------------
# Text normalization + scoring
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u0e00-\u0e7f]")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. CJK/Thai chars kept."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    d = np.zeros((n + 1, m + 1), dtype=np.uint32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return int(d[n, m])


def score(reference: str, hypothesis: str, use_cer: bool) -> Tuple[int, int]:
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if use_cer:
        ref_units = [c for c in ref if not c.isspace()]
        hyp_units = [c for c in hyp if not c.isspace()]
    else:
        ref_units = ref.split()
        hyp_units = hyp.split()
    return _edit_distance(ref_units, hyp_units), len(ref_units)


# ---------------------------------------------------------------------------
# FLEURS loaders — two backends with the same iterator contract
# ---------------------------------------------------------------------------

def _iter_fleurs_hf(lang: str, max_files: Optional[int]) -> Iterator[Tuple[str, np.ndarray, int, str]]:
    """Yield (utt_id, audio_float32, sr, reference) from `google/fleurs`.

    Uses `Audio(decode=False)` + soundfile because the default
    decode path requires torchcodec, which is not in the slim macOS
    inference env.
    """
    try:
        from datasets import load_dataset, Audio
    except ImportError as e:
        raise SystemExit(
            "datasets package not installed; run `uv add datasets` or pass --fleurs-root"
        ) from e
    import io
    import soundfile as sf
    ds = load_dataset("google/fleurs", lang, split="test", streaming=False)
    ds = ds.cast_column("audio", Audio(decode=False))
    for i, ex in enumerate(ds):
        if max_files and i >= max_files:
            break
        audio_bytes = ex["audio"]["bytes"]
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        yield str(ex.get("id", i)), audio, int(sr), ex["transcription"]


def _iter_fleurs_local(
    fleurs_root: Path, lang: str, max_files: Optional[int]
) -> Iterator[Tuple[str, np.ndarray, int, str]]:
    """Yield from a local FLEURS layout: <root>/<lang>/test/*.wav + test.tsv."""
    import soundfile as sf
    tsv = fleurs_root / lang / "test.tsv"
    if not tsv.exists():
        raise SystemExit(f"missing {tsv} — check --fleurs-root layout")
    audio_dir = fleurs_root / lang / "audio" / "test"
    if not audio_dir.exists():
        # Fall back to flat layout
        audio_dir = fleurs_root / lang / "test"
    with open(tsv, newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    for i, row in enumerate(rows):
        if max_files and i >= max_files:
            break
        if len(row) < 3:
            continue
        # FLEURS test.tsv columns: id, raw_transcription, normalized_transcription, num_samples, gender
        utt_id, _raw, ref = row[0], row[1], row[2]
        wav = audio_dir / f"{utt_id}.wav"
        if not wav.exists():
            wav = audio_dir / f"{utt_id}.flac"
        if not wav.exists():
            continue
        audio, sr = sf.read(str(wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        yield utt_id, audio, int(sr), ref


def iter_fleurs(
    lang: str,
    use_hf: bool,
    fleurs_root: Optional[Path],
    max_files: Optional[int],
) -> Iterator[Tuple[str, np.ndarray, int, str]]:
    if use_hf:
        yield from _iter_fleurs_hf(lang, max_files)
    else:
        if fleurs_root is None:
            raise SystemExit("Pass either --use-hf or --fleurs-root")
        yield from _iter_fleurs_local(fleurs_root, lang, max_files)


def maybe_resample(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if sr == target_sr:
        return audio
    try:
        from scipy.signal import resample_poly
    except ImportError as e:
        raise SystemExit(
            "scipy required for resampling; install it or pre-resample audio"
        ) from e
    from math import gcd
    g = gcd(sr, target_sr)
    return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-language run
# ---------------------------------------------------------------------------

def run_lang(
    runner: NemotronMultilingualCoreML,
    fleurs_lang: str,
    nemo_lang: str,
    mode: str,
    use_hf: bool,
    fleurs_root: Optional[Path],
    max_files: Optional[int],
) -> dict:
    use_cer = uses_cer(nemo_lang)
    metric = "CER" if use_cer else "WER"

    total_errors = 0
    total_units = 0
    correct_detections = 0
    detection_attempts = 0
    audio_secs = 0.0
    compute_secs = 0.0
    expected_tag = f"<{nemo_lang}>"

    print(f"\n[{fleurs_lang} → {nemo_lang}]  metric={metric}  mode={mode}")

    n = 0
    for utt_id, audio, sr, ref in iter_fleurs(fleurs_lang, use_hf, fleurs_root, max_files):
        audio = maybe_resample(audio, sr, 16000)
        audio_secs += len(audio) / 16000.0

        target_lang = nemo_lang if mode == "forced" else "auto"
        t0 = time.perf_counter()
        detected_tag, hyp, _pid = runner.transcribe_streaming(audio, target_lang=target_lang)
        compute_secs += time.perf_counter() - t0

        if mode == "auto":
            detection_attempts += 1
            if detected_tag == expected_tag:
                correct_detections += 1

        errors, units = score(ref, hyp, use_cer)
        total_errors += errors
        total_units += units
        n += 1

        if n <= 3:
            print(f"   [{n}] {utt_id}  errs={errors}/{units}  det={detected_tag}")
            print(f"       REF: {ref[:80]}")
            print(f"       HYP: {hyp[:80]}")

    if total_units == 0:
        return {"lang": fleurs_lang, "files": 0, "metric": metric, "value": None}

    score_pct = 100.0 * total_errors / total_units
    rtfx = audio_secs / compute_secs if compute_secs > 0 else float("inf")
    det_acc = (
        100.0 * correct_detections / detection_attempts if detection_attempts else None
    )

    print(
        f"   files={n} {metric}={score_pct:.2f}%  RTFx={rtfx:.2f}x"
        + (f"  detect_acc={det_acc:.1f}%" if det_acc is not None else "")
    )

    return {
        "fleurs_lang": fleurs_lang,
        "nemo_lang": nemo_lang,
        "files": n,
        "metric": metric,
        "value": round(score_pct, 2),
        "rtfx": round(rtfx, 2),
        "detect_acc": round(det_acc, 1) if det_acc is not None else None,
        "audio_seconds": round(audio_secs, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Nemotron Multilingual FLEURS benchmark")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory with the 4 .mlpackages + metadata.json + tokenizer.json")
    parser.add_argument("--languages", type=str, required=True,
                        help="Comma-separated FLEURS lang codes, e.g. cmn_hans_cn,en_us,es_419")
    parser.add_argument("--mode", choices=["forced", "auto"], default="auto",
                        help='"forced": pass the per-utterance language as prompt; '
                             '"auto": always pass prompt 101 (default)')
    parser.add_argument("--max-files-per-lang", type=int, default=None,
                        help="Cap files per language for quick runs")
    parser.add_argument("--use-hf", action="store_true",
                        help="Load FLEURS via datasets.load_dataset('google/fleurs', ...)")
    parser.add_argument("--fleurs-root", type=str, default=None,
                        help="Local FLEURS root directory (alternative to --use-hf)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write per-language results to this JSON file")
    args = parser.parse_args()

    print("=" * 70)
    print("NEMOTRON MULTILINGUAL — FLEURS BENCHMARK")
    print("=" * 70)
    print(f"model_dir: {args.model_dir}")
    print(f"mode:      {args.mode}")
    print(f"max/lang:  {args.max_files_per_lang}")

    runner = NemotronMultilingualCoreML(args.model_dir)

    lang_codes = [l.strip() for l in args.languages.split(",") if l.strip()]
    fleurs_root = Path(args.fleurs_root) if args.fleurs_root else None

    results = []
    for fleurs_lang in lang_codes:
        nemo_lang = fleurs_to_nemotron(fleurs_lang)
        if nemo_lang is None:
            print(f"\n[skip] {fleurs_lang}: no Nemotron prompt mapping")
            continue
        r = run_lang(
            runner=runner,
            fleurs_lang=fleurs_lang,
            nemo_lang=nemo_lang,
            mode=args.mode,
            use_hf=args.use_hf,
            fleurs_root=fleurs_root,
            max_files=args.max_files_per_lang,
        )
        results.append(r)

    print("\n" + "=" * 70)
    print(f"SUMMARY (mode={args.mode})")
    print("=" * 70)
    header = f"{'fleurs':<14} {'nemo':<8} {'n':>4} {'metric':<6} {'score':>7} {'RTFx':>6} {'det%':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        det = f"{r['detect_acc']:.1f}" if r.get("detect_acc") is not None else "-"
        val = f"{r['value']:.2f}" if r.get("value") is not None else "-"
        print(f"{r['fleurs_lang']:<14} {r['nemo_lang']:<8} {r['files']:>4} "
              f"{r['metric']:<6} {val:>7} {r['rtfx']:>6.2f} {det:>6}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(
            {"mode": args.mode, "results": results}, indent=2, ensure_ascii=False
        ))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
