#!/usr/bin/env python3
"""Stage parakeet-unified CoreML packages for HuggingFace upload / Swift host use.

Compiles each .mlpackage to .mlmodelc (what FluidAudio's MLModel.load expects),
exports the SentencePiece vocab as vocab.json ({id: piece}, the format the Swift
`Tokenizer` consumes), and copies metadata.json.

Usage:
    uv run --no-sync python stage_hf.py \
        --coreml-dir ./build/parakeet_unified_coreml --output-dir ./build/hf-staging
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import sentencepiece as spm


def export_vocab(nemo_path: Path, out_path: Path) -> int:
    with tarfile.open(nemo_path, "r") as tar:
        names = [n for n in tar.getnames() if n.endswith("tokenizer.model")]
        assert names, f"no tokenizer.model found in {nemo_path}"
        with tempfile.NamedTemporaryFile(suffix=".model") as tmp:
            tmp.write(tar.extractfile(names[0]).read())
            tmp.flush()
            sp = spm.SentencePieceProcessor()
            sp.load(tmp.name)
    vocab = {str(i): sp.id_to_piece(i) for i in range(sp.get_piece_size())}
    out_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=0))
    return sp.get_piece_size()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreml-dir", type=Path, default=Path("build/parakeet_unified_coreml"))
    parser.add_argument(
        "--int8-dir",
        type=Path,
        default=Path("build/parakeet_unified_coreml_int8"),
        help="quantize_int8.py output; its encoders are staged as *_int8.mlmodelc",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("build/hf-staging"))
    parser.add_argument("--nemo-path", type=Path, default=Path("parakeet-unified-en-0.6b.nemo"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(pkg, pkg.stem) for pkg in sorted(args.coreml_dir.glob("*.mlpackage"))]
    if args.int8_dir.exists():
        jobs += [
            (pkg, pkg.stem + "_int8")
            for pkg in sorted(args.int8_dir.glob("parakeet_unified_encoder*.mlpackage"))
        ]

    for pkg, stem in jobs:
        target = args.output_dir / (stem + ".mlmodelc")
        if target.exists():
            print(f"exists, skipping: {target.name}")
            continue
        print(f"compiling {pkg.name} → {target.name}")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(
                    ["xcrun", "coremlcompiler", "compile", str(pkg), tmp],
                    check=True,
                    capture_output=True,
                )
                compiled = next(Path(tmp).glob("*.mlmodelc"))
                shutil.move(str(compiled), str(target))
        except (subprocess.CalledProcessError, FileNotFoundError):
            # No full Xcode: let the CoreML framework compile on load and copy
            # the compiled bundle out of the cache.
            import coremltools as ct

            model = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_ONLY)
            compiled_path = Path(model.get_compiled_model_path())
            shutil.copytree(compiled_path, target)

    vocab_size = export_vocab(args.nemo_path, args.output_dir / "vocab.json")
    print(f"vocab.json: {vocab_size} pieces")

    metadata = args.coreml_dir / "metadata.json"
    if metadata.exists():
        shutil.copy(metadata, args.output_dir / "metadata.json")
        print("copied metadata.json")

    print(f"Staged to {args.output_dir} — ready for HF upload (user-run).")


if __name__ == "__main__":
    main()
