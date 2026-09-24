#!/usr/bin/env python3
"""PyTorch (NeMo, fp32, full-context) transcription of the files in a FluidAudio asr-benchmark JSON, scored the same
way, to separate model-inherent WER from CoreML conversion loss. Runs redux and (optionally) stock v3."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def norm(t: str) -> list:
    t = t.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return t.split()


def wer(ref: list, hyp: list) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[len(hyp)] / max(1, len(ref))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bench_json", type=Path)
    ap.add_argument("--dataset", type=Path,
                    default=Path("~/Library/Application Support/FluidAudio/Datasets/LibriSpeech/test-clean").expanduser())
    ap.add_argument("--hf-dir", type=Path, default=Path("~/Documents/parakeet-redux-work/hf").expanduser())
    ap.add_argument("--models", default="redux,stock")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results = json.load(open(args.bench_json))["results"]
    files = []
    for r in results:
        stem = r["fileName"].replace(".flac", "")
        spk, chap, _ = stem.split("-")
        files.append(args.dataset / spk / chap / r["fileName"])
        assert files[-1].exists(), files[-1]
    coreml_wer = [wer(norm(r["reference"]), norm(r["hypothesis"])) for r in results]
    print(f"CoreML (from json): avg WER {100 * sum(coreml_wer) / len(coreml_wer):.2f}%  n={len(results)}")

    import nemo.collections.asr as nemo_asr
    from redux_weights import load_into_nemo

    out = {}
    for tag in args.models.split(","):
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cpu")
        model.eval()
        if tag == "redux":
            load_into_nemo(model, args.hf_dir, verbose=False)
        hyps = model.transcribe([str(f) for f in files], batch_size=8, verbose=False)
        texts = [h.text if hasattr(h, "text") else h for h in hyps]
        ws = [wer(norm(r["reference"]), norm(t)) for r, t in zip(results, texts)]
        short = [w for w, r in zip(ws, results) if r["audioLength"] <= 15.0]
        agree = [wer(norm(t), norm(r["hypothesis"])) for r, t in zip(results, texts)]
        print(f"NeMo {tag}: avg WER {100 * sum(ws) / len(ws):.2f}% (all)  {100 * sum(short) / len(short):.2f}% (<=15 s, n={len(short)})"
              f"  | CoreML-vs-NeMo hypothesis WER {100 * sum(agree) / len(agree):.2f}%")
        out[tag] = [{"fileName": r["fileName"], "nemo": t, "coreml": r["hypothesis"], "reference": r["reference"],
                     "wer_nemo": w, "wer_coreml": c} for r, t, w, c in zip(results, texts, ws, coreml_wer)]
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
