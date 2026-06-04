#!/usr/bin/env python3
"""
FLEURS benchmark for the base PyTorch .nemo model.

Mirrors `benchmark_fleurs.py` (CoreML side) one-to-one so the resulting
JSON has the same schema and can be diffed directly. The only differences:

  - Loads `nemotron-asr-streaming-multilingual-0.6b.nemo` via NeMo.
  - Uses `CacheAwareStreamingAudioBuffer` + `conformer_stream_step`
    instead of the CoreML preprocessor/encoder/decoder/joint chain.

WER/CER normalization and FLEURS loaders are imported from
`benchmark_fleurs.py` unchanged so scoring is byte-identical.

Example:

    PYTHONUNBUFFERED=1 conversion_scripts/.venv/bin/python \\
        benchmark_fleurs_pytorch.py \\
        --nemo-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \\
        --use-hf \\
        --languages en_us,cmn_hans_cn,ja_jp,es_419,fr_fr \\
        --mode forced \\
        --max-files-per-lang 5 \\
        --output-json bench_results/fleurs_pytorch_5x5.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
# Reuse normalization + FLEURS loaders verbatim
from benchmark_fleurs import iter_fleurs, maybe_resample, score  # noqa: E402
from fleurs_lang_map import fleurs_to_nemotron, uses_cer  # noqa: E402
from nemo_reference import (  # noqa: E402
    resolve_prompt_id,
    transcribe_streaming,
)


def run_lang(
    model,
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
    audio_secs = 0.0
    compute_secs = 0.0

    print(f"\n[{fleurs_lang} → {nemo_lang}]  metric={metric}  mode={mode}")

    n = 0
    for utt_id, audio, sr, ref in iter_fleurs(fleurs_lang, use_hf, fleurs_root, max_files):
        audio = maybe_resample(audio, sr, 16000)
        audio_secs += len(audio) / 16000.0

        target_lang = nemo_lang if mode == "forced" else "auto"
        t0 = time.perf_counter()
        detected_tag, hyp = transcribe_streaming(model, audio, sr=16000, target_lang=target_lang)
        compute_secs += time.perf_counter() - t0

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

    print(f"   files={n} {metric}={score_pct:.2f}%  RTFx={rtfx:.2f}x")

    return {
        "fleurs_lang": fleurs_lang,
        "nemo_lang": nemo_lang,
        "files": n,
        "metric": metric,
        "value": round(score_pct, 2),
        "rtfx": round(rtfx, 2),
        "audio_seconds": round(audio_secs, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Nemotron PyTorch FLEURS benchmark")
    parser.add_argument("--nemo-path", type=str, required=True,
                        help="Path to nemotron-asr-streaming-multilingual-0.6b.nemo")
    parser.add_argument("--languages", type=str, required=True)
    parser.add_argument("--mode", choices=["forced", "auto"], default="forced")
    parser.add_argument("--max-files-per-lang", type=int, default=None)
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--fleurs-root", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("NEMOTRON MULTILINGUAL — FLEURS PYTORCH BENCHMARK")
    print("=" * 70)
    print(f"nemo:      {args.nemo_path}")
    print(f"mode:      {args.mode}")
    print(f"max/lang:  {args.max_files_per_lang}")

    print("Loading NeMo model (this can take ~30 s)...")
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.restore_from(args.nemo_path)
    model.eval()
    print(f"Loaded. auto prompt_id={resolve_prompt_id(model, 'auto')}")

    lang_codes = [l.strip() for l in args.languages.split(",") if l.strip()]
    fleurs_root = Path(args.fleurs_root) if args.fleurs_root else None

    results = []
    for fleurs_lang in lang_codes:
        nemo_lang = fleurs_to_nemotron(fleurs_lang)
        if nemo_lang is None:
            print(f"\n[skip] {fleurs_lang}: no Nemotron prompt mapping")
            continue
        r = run_lang(
            model=model,
            fleurs_lang=fleurs_lang,
            nemo_lang=nemo_lang,
            mode=args.mode,
            use_hf=args.use_hf,
            fleurs_root=fleurs_root,
            max_files=args.max_files_per_lang,
        )
        results.append(r)

    print("\n" + "=" * 70)
    print(f"SUMMARY (mode={args.mode}, backend=pytorch)")
    print("=" * 70)
    header = f"{'fleurs':<14} {'nemo':<8} {'n':>4} {'metric':<6} {'score':>7} {'RTFx':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        val = f"{r['value']:.2f}" if r.get("value") is not None else "-"
        print(f"{r['fleurs_lang']:<14} {r['nemo_lang']:<8} {r['files']:>4} "
              f"{r['metric']:<6} {val:>7} {r['rtfx']:>6.2f}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(
            {"mode": args.mode, "backend": "pytorch", "results": results},
            indent=2, ensure_ascii=False,
        ))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
