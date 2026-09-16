"""Matched, free-running baseline/candidate/reference ASR evaluation."""

import argparse
import json
import os
from pathlib import Path
import re
import unicodedata

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import jiwer
from opencc import OpenCC
from scipy.signal import resample_poly
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .artifacts import ROOT, sha256, verify_assets, write_json
from .bundle import load_bundle, synthesize
from .frontend import Frontend
from .model import CompatibleGenerator
from .train import load_data

ASR_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


def normalize(text, language):
    text = OpenCC("t2s").convert(unicodedata.normalize("NFKC", text)).lower()
    text = "".join(c if c.isalnum() or c.isspace() else " " for c in text)
    return " ".join(text.split()) if language == "en" else re.sub(r"\s+", "", text)


def error_counts(reference, hypothesis, language):
    reference, hypothesis = normalize(reference, language), normalize(hypothesis, language)
    result = jiwer.process_words(reference, hypothesis) if language == "en" else jiwer.process_characters(reference, hypothesis)
    return {"errors": result.substitutions + result.deletions + result.insertions,
            "reference_units": result.hits + result.substitutions + result.deletions,
            "substitutions": result.substitutions, "deletions": result.deletions, "insertions": result.insertions,
            "metric": "WER" if language == "en" else "CER"}


def evaluate(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    asset_lock = json.loads((ROOT / "training-assets.lock.json").read_text())
    for entry in asset_lock["files"]:
        if entry["part"] == "asr" and sha256(args.asr / Path(entry["path"]).name) != entry["sha256"]:
            raise ValueError("Evaluation ASR checkpoint/configuration differs from pinned assets")
    if args.output.exists():
        raise ValueError("Refusing to overwrite evaluation evidence")
    args.output.mkdir(parents=True)
    records, contract = load_data(args.data)
    cases = [{**r, "suite": args.split} for r in records if r["split"] == args.split]
    if args.controls:
        cases += [{**r, "suite": "controls"} for r in json.loads((ROOT / "evaluation-prompts.json").read_text())]
    verify_assets(args.assets)
    candidate, candidate_voices, bundle = load_bundle(args.bundle, args.device)
    baseline = CompatibleGenerator(json.loads((args.assets / "config.json").read_text()))
    baseline.load_checkpoint(torch.load(args.assets / "kokoro-v1_1-zh.pth", weights_only=True, map_location="cpu"))
    baseline.to(args.device).eval()
    baseline_voices = torch.load(args.assets / "voices/zf_001.pt", weights_only=True, map_location=args.device)
    frontend = Frontend(candidate.config["vocab"])
    entries = []
    audio_dir = args.output / "audio"
    audio_dir.mkdir()
    for case in cases:
        for name, model, voices in [("baseline", baseline, baseline_voices), ("candidate", candidate, candidate_voices)]:
            audio, metadata = synthesize(model, voices, frontend, case["text"], args.device)
            path = audio_dir / f"{case['id']}-{name}.wav"
            sf.write(path, audio, 24000, subtype="FLOAT")
            entries.append({"id": case["id"], "suite": case["suite"], "language": case["language"], "variant": name,
                            "text": case["text"], "audio": str(path.relative_to(args.output)), "audio_sha256": sha256(path),
                            "signal": {k: metadata[k] for k in ("samples", "peak", "rms", "clipping_fraction")},
                            "input_ids": metadata["input_ids"], "phonemes": metadata["phonemes"]})
        if "targets" in case:
            audio = torch.load(args.data / case["targets"], weights_only=True)["audio"].numpy().reshape(-1)
            path = audio_dir / f"{case['id']}-reference.wav"
            sf.write(path, audio, 24000, subtype="FLOAT")
            entries.append({"id": case["id"], "suite": case["suite"], "language": case["language"], "variant": "real_reference",
                            "text": case["text"], "audio": str(path.relative_to(args.output)), "audio_sha256": sha256(path)})
    del candidate, baseline
    torch.cuda.empty_cache()
    processor = WhisperProcessor.from_pretrained(args.asr, local_files_only=True)
    asr = WhisperForConditionalGeneration.from_pretrained(args.asr, local_files_only=True,
              dtype=torch.float16 if str(args.device).startswith("cuda") else torch.float32).to(args.device).eval()
    for language in ("en", "zh", "mixed"):
        subset = [e for e in entries if e["language"] == language]
        for offset in range(0, len(subset), args.batch_size):
            batch = subset[offset:offset + args.batch_size]
            audios = [resample_poly(sf.read(args.output / e["audio"], dtype="float32")[0], 2, 3) for e in batch]
            if any(len(audio) > 30 * 16000 for audio in audios):
                raise ValueError("ASR short-utterance protocol exceeds 30 seconds; no truncation allowed")
            inputs = processor(audios, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            with torch.no_grad():
                tokens = asr.generate(input_features=inputs.input_features.to(args.device, dtype=asr.dtype),
                                      attention_mask=inputs.attention_mask.to(args.device),
                                      language="english" if language == "en" else "chinese", task="transcribe",
                                      do_sample=False, num_beams=1, max_new_tokens=256)
            hypotheses = processor.batch_decode(tokens, skip_special_tokens=True)
            for entry, hypothesis in zip(batch, hypotheses, strict=True):
                entry["transcript"] = hypothesis
                entry["error"] = error_counts(entry["text"], hypothesis, language)
                print(json.dumps({k: entry[k] for k in ("id", "variant", "transcript", "error")}, ensure_ascii=False), flush=True)
    aggregates = {}
    for entry in entries:
        key = f"{entry['suite']}/{entry['language']}/{entry['variant']}"
        group = aggregates.setdefault(key, {"utterances": 0, "errors": 0, "reference_units": 0, "metric": entry["error"]["metric"]})
        group["utterances"] += 1
        for field in ("errors", "reference_units"):
            group[field] += entry["error"][field]
    for group in aggregates.values():
        group["rate"] = group["errors"] / group["reference_units"]
    result = {"model_sha256": bundle["files"]["model.pth"]["sha256"], "step": bundle["step"],
              "precision": {"generator_tensor_dtype": "float32", "asr_tensor_dtype": str(asr.dtype),
                            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
              "manifest_sha256": contract["manifest_sha256"], "protocol": "whisper-turbo-greedy-forced-language-v1",
              "asr": {"repo": "openai/whisper-large-v3-turbo", "revision": ASR_REVISION,
                      "weights_sha256": sha256(args.asr / "model.safetensors")},
              "normalization": "NFKC, simplified Chinese, lowercase, strip punctuation; WER English, CER Mandarin/mixed",
              "limitations": ["ASR proxy, not listening/tone/naturalness certification", "mixed forces Chinese transcription; English normalization is not numeral expansion"],
              "aggregates": aggregates, "entries": entries}
    write_json(args.output / "results.json", result)
    print(json.dumps(aggregates, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path(".artifacts/baseline"))
    parser.add_argument("--asr", type=Path, default=Path(".artifacts/whisper-turbo"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
