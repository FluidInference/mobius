"""Stage a HuggingFace-ready directory for Magpie TTS Multilingual 357M.

The mobius exporters and converters write into two local directories:

- ``build/`` — compiled ``.mlpackage`` bundles (and, after
  ``compile_mlmodelc.py``, matching ``.mlmodelc`` bundles).
- ``constants/`` — ``.npy`` tensors, ``*.json`` config, the
  ``local_transformer/`` subtree, **and** the per-language tokenizer
  JSONs.

The FluidAudio Swift port and the target HF repo
(``FluidInference/magpie-tts-multilingual-357m-coreml``) expect a slightly
different layout: CoreML models at the root, tokenizer JSONs in a
dedicated ``tokenizer/`` folder, everything else in ``constants/``. This
script assembles that layout into ``hf-upload/`` (configurable), writes a
model card + ``.gitattributes``, validates that nothing required is
missing, and prints the exact ``huggingface-cli upload`` commands for
the user to run.

It does **not** upload anything. Per project policy, HF uploads are
performed manually by the maintainers.

Usage:

    # After running the converter + compiler + constants exporters
    python prepare_hf_upload.py

    # Custom paths / output
    python prepare_hf_upload.py \\
        --build-dir build \\
        --constants-dir constants \\
        --output-dir hf-upload \\
        --repo-id FluidInference/magpie-tts-multilingual-357m-coreml
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Core models expected at the repo root.
REQUIRED_MODELS = [
    "text_encoder.mlmodelc",
    "decoder_step.mlmodelc",
    "nanocodec_decoder.mlmodelc",
]
OPTIONAL_MODELS = [
    "decoder_prefill.mlmodelc",
]

# Keys that MUST survive in constants/. Anything not in this allow-list that
# also isn't a per-language tokenizer file will be flagged as unknown.
CONSTANTS_KEEP_FILES = {
    "constants.json",
    "speaker_info.json",
    "tokenizer_info.json",
    "tokenizer_metadata.json",
    "tokenizer_references.json",
    "text_embedding.npy",
    "speaker_embeddings_raw.npy",
}
CONSTANTS_KEEP_PREFIXES = (
    "speaker_",         # speaker_0.npy .. speaker_N.npy
    "audio_embedding_", # audio_embedding_0.npy .. audio_embedding_7.npy
)
CONSTANTS_KEEP_DIRS = {"local_transformer"}

# Mirror of MagpieTokenizerFiles.files(for:) in the Swift port.
PER_LANGUAGE_TOKENIZER_FILES = {
    "english": [
        "english_phoneme_token2id.json",
        "english_phoneme_phoneme_dict.json",
        "english_phoneme_heteronyms.json",
    ],
    "spanish": [
        "spanish_phoneme_token2id.json",
        "spanish_phoneme_phoneme_dict.json",
    ],
    "italian": [
        "italian_phoneme_token2id.json",
        "italian_phoneme_phoneme_dict.json",
    ],
    "vietnamese": [
        "vietnamese_phoneme_token2id.json",
        "vietnamese_phoneme_phoneme_dict.json",
    ],
    "german": [
        "german_phoneme_token2id.json",
        "german_phoneme_phoneme_dict.json",
        "german_phoneme_heteronyms.json",
    ],
    "french": [
        "french_chartokenizer_token2id.json",
    ],
    "hindi": [
        "hindi_chartokenizer_token2id.json",
    ],
    "mandarin": [
        "mandarin_phoneme_token2id.json",
        "mandarin_phoneme_phoneme_dict.json",
        "mandarin_phoneme_pinyin_dict.json",
        "mandarin_phoneme_tone_dict.json",
        "mandarin_phoneme_ascii_letter_dict.json",
        "mandarin_pypinyin_char_dict.json",
        "mandarin_pypinyin_phrase_dict.json",
        "mandarin_jieba_dict.json",
    ],
}

ALL_TOKENIZER_FILES = {
    fname for files in PER_LANGUAGE_TOKENIZER_FILES.values() for fname in files
}


GITATTRIBUTES = """\
*.mlmodelc filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.mlpackage filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
"""


README_TEMPLATE = """\
---
license: cc-by-4.0
language:
  - en
  - es
  - de
  - fr
  - it
  - vi
  - zh
  - hi
tags:
  - text-to-speech
  - coreml
  - apple-silicon
  - magpie
library_name: coreml
base_model: nvidia/magpie_tts_multilingual_357m
---

# Magpie TTS Multilingual 357M (CoreML)

CoreML export of NVIDIA's [Magpie TTS Multilingual 357M](https://huggingface.co/nvidia/magpie_tts_multilingual_357m), optimized for on-device inference on Apple Silicon. Ships as `.mlmodelc` bundles compiled for macOS 14+ / iOS 17+.

Converted with [FluidInference/mobius](https://github.com/FluidInference/mobius). Consumed by the Swift port in [FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio) (see `Sources/FluidAudio/TTS/Magpie/`).

## Languages

English, Spanish, German, French, Italian, Vietnamese, Mandarin, Hindi. Japanese is not yet included.

## Contents

```
├── text_encoder.mlmodelc/         # Text → (B, 256, 768) encoder output
├── decoder_step.mlmodelc/         # 12-layer AR decoder (stateful KV cache)
├── decoder_prefill.mlmodelc/      # (optional) batched prefill fast path
├── nanocodec_decoder.mlmodelc/    # 8-codebook → PCM vocoder (22050 Hz)
├── constants/
│   ├── constants.json             # d_model, n_layers, EOS ids, ...
│   ├── speaker_info.json          # speaker names + context shape
│   ├── tokenizer_metadata.json    # tokenizer-agnostic EOS + special tokens
│   ├── speaker_0.npy .. speaker_4.npy
│   ├── audio_embedding_0.npy .. audio_embedding_7.npy
│   └── local_transformer/         # 1-layer transformer weights (Swift reads .npy)
└── tokenizer/
    ├── english_phoneme_*.json
    ├── spanish_phoneme_*.json
    ├── german_phoneme_*.json
    ├── french_chartokenizer_*.json
    ├── italian_phoneme_*.json
    ├── vietnamese_phoneme_*.json
    ├── mandarin_*.json
    └── hindi_chartokenizer_*.json
```

## Usage (Swift)

```swift
import FluidAudio

let manager = try await MagpieTtsManager.downloadAndCreate(
    languages: [.english, .spanish]
)
let result = try await manager.synthesize(
    text: "Hello | ˈ n ɛ m o ʊ | from FluidAudio.",
    speaker: .john,
    language: .english
)
let wav = AudioWAV.data(from: result.samples, sampleRate: result.sampleRate)
try wav.write(to: URL(fileURLWithPath: "hello.wav"))
```

The manager lazy-downloads everything in this repo on first use.

## Inline IPA override

Text enclosed in `|...|` is passed straight to the tokenizer as whitespace-separated IPA tokens:

```
"Hello | ˈ n ɛ m o ʊ | world"
```

## License

- CoreML export: CC-BY-4.0 (inherits from the upstream NeMo model).
- Upstream weights: see [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m).
"""


@dataclass
class PrepReport:
    copied_models: list[str] = field(default_factory=list)
    missing_required_models: list[str] = field(default_factory=list)
    missing_optional_models: list[str] = field(default_factory=list)
    copied_constants: list[str] = field(default_factory=list)
    missing_constants: list[str] = field(default_factory=list)
    copied_tokenizer_files: dict[str, list[str]] = field(default_factory=dict)
    missing_tokenizer_files: dict[str, list[str]] = field(default_factory=dict)
    unknown_files: list[str] = field(default_factory=list)

    def has_errors(self) -> bool:
        return bool(self.missing_required_models or self.missing_constants)


def _copy_tree(src: str, dst: str) -> None:
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _copy_models(build_dir: str, output_dir: str, report: PrepReport) -> None:
    for model in REQUIRED_MODELS:
        src = os.path.join(build_dir, model)
        if not os.path.exists(src):
            report.missing_required_models.append(model)
            continue
        dst = os.path.join(output_dir, model)
        _copy_tree(src, dst)
        report.copied_models.append(model)

    for model in OPTIONAL_MODELS:
        src = os.path.join(build_dir, model)
        if not os.path.exists(src):
            report.missing_optional_models.append(model)
            continue
        dst = os.path.join(output_dir, model)
        _copy_tree(src, dst)
        report.copied_models.append(model)


def _copy_constants(constants_dir: str, output_dir: str, report: PrepReport) -> None:
    dst_constants = os.path.join(output_dir, "constants")
    os.makedirs(dst_constants, exist_ok=True)

    required = {"constants.json", "speaker_info.json", "tokenizer_metadata.json"}
    required |= {f"audio_embedding_{i}.npy" for i in range(8)}
    required |= {f"speaker_{i}.npy" for i in range(5)}
    # local_transformer/ is a dir — enumerate expected files separately.
    local_transformer_files = {
        "in_proj_weight.npy",
        "in_proj_bias.npy",
        "pos_emb.npy",
        "norm1_weight.npy",
        "sa_qkv_weight.npy",
        "sa_o_weight.npy",
        "norm2_weight.npy",
        "ffn_conv1_weight.npy",
        "ffn_conv2_weight.npy",
    }
    for i in range(8):
        local_transformer_files.add(f"out_proj_{i}_weight.npy")
        local_transformer_files.add(f"out_proj_{i}_bias.npy")

    for entry in sorted(os.listdir(constants_dir)):
        src = os.path.join(constants_dir, entry)

        # Tokenizer files are moved out to tokenizer/ — skip here.
        if entry in ALL_TOKENIZER_FILES:
            continue

        # Known constants files or dirs.
        is_keep_file = entry in CONSTANTS_KEEP_FILES
        is_keep_prefix = any(entry.startswith(p) for p in CONSTANTS_KEEP_PREFIXES)
        is_keep_dir = entry in CONSTANTS_KEEP_DIRS and os.path.isdir(src)

        if is_keep_file or is_keep_prefix or is_keep_dir:
            dst = os.path.join(dst_constants, entry)
            _copy_tree(src, dst)
            report.copied_constants.append(entry)
        else:
            report.unknown_files.append(os.path.relpath(src, constants_dir))

    copied_set = set(report.copied_constants)
    for req in sorted(required):
        if req not in copied_set:
            report.missing_constants.append(f"constants/{req}")

    lt_src = os.path.join(constants_dir, "local_transformer")
    if os.path.isdir(lt_src):
        present = set(os.listdir(lt_src))
        for req in sorted(local_transformer_files):
            if req not in present:
                report.missing_constants.append(f"constants/local_transformer/{req}")
    else:
        report.missing_constants.append("constants/local_transformer/")


def _copy_tokenizer(constants_dir: str, output_dir: str, report: PrepReport) -> None:
    dst_tokenizer = os.path.join(output_dir, "tokenizer")
    os.makedirs(dst_tokenizer, exist_ok=True)

    for language, files in PER_LANGUAGE_TOKENIZER_FILES.items():
        copied: list[str] = []
        missing: list[str] = []
        for fname in files:
            src = os.path.join(constants_dir, fname)
            if not os.path.exists(src):
                missing.append(fname)
                continue
            dst = os.path.join(dst_tokenizer, fname)
            shutil.copy2(src, dst)
            copied.append(fname)
        if copied:
            report.copied_tokenizer_files[language] = copied
        if missing:
            report.missing_tokenizer_files[language] = missing


def _write_metadata(output_dir: str, report: PrepReport, repo_id: str) -> None:
    with open(os.path.join(output_dir, ".gitattributes"), "w") as f:
        f.write(GITATTRIBUTES)

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(README_TEMPLATE)

    # Machine-readable prep report for auditability.
    summary = {
        "repoId": repo_id,
        "copiedModels": report.copied_models,
        "missingRequiredModels": report.missing_required_models,
        "missingOptionalModels": report.missing_optional_models,
        "copiedConstants": sorted(report.copied_constants),
        "missingConstants": report.missing_constants,
        "copiedTokenizerFiles": report.copied_tokenizer_files,
        "missingTokenizerFiles": report.missing_tokenizer_files,
        "unknownFiles": report.unknown_files,
    }
    with open(os.path.join(output_dir, "_prep_report.json"), "w") as f:
        json.dump(summary, f, indent=2)


def _print_report(report: PrepReport, output_dir: str, repo_id: str) -> int:
    print("")
    print("=" * 72)
    print(f"HF upload staging → {output_dir}")
    print(f"Target repo:       {repo_id}")
    print("=" * 72)

    print("\nCoreML models:")
    for m in report.copied_models:
        print(f"  OK   {m}")
    for m in report.missing_required_models:
        print(f"  MISS {m}  (REQUIRED — re-run convert_*.py + compile_mlmodelc.py)")
    for m in report.missing_optional_models:
        print(f"  skip {m}  (optional)")

    print("\nconstants/:")
    for c in sorted(report.copied_constants):
        print(f"  OK   {c}")
    for c in report.missing_constants:
        print(f"  MISS {c}")

    print("\ntokenizer/:")
    for lang, files in sorted(report.copied_tokenizer_files.items()):
        print(f"  [{lang}] {len(files)} file(s) copied")
    for lang, files in sorted(report.missing_tokenizer_files.items()):
        for fname in files:
            print(f"  MISS tokenizer/{fname}  ({lang})")

    if report.unknown_files:
        print("\nUnknown files under constants/ (not copied — review):")
        for u in report.unknown_files:
            print(f"  ??   {u}")

    print("")
    if report.has_errors():
        print("Staging completed WITH ERRORS — see MISS entries above.")
        print("Re-run the relevant exporter/converter and re-run this script.")
        return 1

    print("Staging OK. Upload with one of:")
    print("")
    print(f"  huggingface-cli upload {repo_id} {output_dir} . \\")
    print("      --repo-type model --commit-message 'upload Magpie TTS CoreML export'")
    print("")
    print("Or, if the repo does not exist yet:")
    print("")
    print(f"  huggingface-cli repo create {repo_id} --type model")
    print(f"  huggingface-cli upload {repo_id} {output_dir} . --repo-type model")
    print("")
    print("Verify from Swift:")
    print("  swift run fluidaudiocli magpie download --languages en")
    print("")
    return 0


def prepare(
    build_dir: str,
    constants_dir: str,
    output_dir: str,
    repo_id: str,
    clean: bool,
) -> int:
    build_dir = os.path.abspath(build_dir)
    constants_dir = os.path.abspath(constants_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(build_dir):
        print(f"error: build dir not found: {build_dir}", file=sys.stderr)
        return 2
    if not os.path.isdir(constants_dir):
        print(f"error: constants dir not found: {constants_dir}", file=sys.stderr)
        return 2

    if clean and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    report = PrepReport()
    _copy_models(build_dir, output_dir, report)
    _copy_constants(constants_dir, output_dir, report)
    _copy_tokenizer(constants_dir, output_dir, report)
    _write_metadata(output_dir, report, repo_id)

    return _print_report(report, output_dir, repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage a HuggingFace-ready directory for Magpie TTS CoreML.",
    )
    parser.add_argument(
        "--build-dir",
        default=os.path.join(SCRIPT_DIR, "build"),
        help="Directory with compiled .mlmodelc bundles (default: ./build)",
    )
    parser.add_argument(
        "--constants-dir",
        default=os.path.join(SCRIPT_DIR, "constants"),
        help="Directory with exported constants + tokenizer files (default: ./constants)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "hf-upload"),
        help="Staging directory to populate (default: ./hf-upload)",
    )
    parser.add_argument(
        "--repo-id",
        default="FluidInference/magpie-tts-multilingual-357m-coreml",
        help="Target HF repo id (only used in the printed upload command)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output dir before staging (fresh build).",
    )
    args = parser.parse_args()

    rc = prepare(
        build_dir=args.build_dir,
        constants_dir=args.constants_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        clean=args.clean,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
