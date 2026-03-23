#!/usr/bin/env python3
"""
Master conversion script: Convert all 6 Argmax-style CoreML models.

This script orchestrates the conversion of all models needed for the
Argmax-style Qwen3-TTS pipeline:

  1. TextProjector      (W8A16)  — text token → projected embedding
  2. CodeEmbedder       (W16A16) — CB0 token → embedding
  3. MultiCodeEmbedder  (W16A16) — CB1-15 linearized token → embedding
  4. CodeDecoder        (W8A16)  — 28-layer LM transformer with KV cache
  5. MultiCodeDecoder   (W8A16)  — 5-layer code predictor with 15 lm_heads
  6. SpeechDecoder      (W8A16)  — codec decoder (audio generation)

Architecture reverse-engineered from argmaxinc/ttskit-coreml MIL programs.

Usage:
    python convert_argmax_all.py --model-path ./model_0.6b --tokenizer-path ./tokenizer_12hz
    python convert_argmax_all.py --model-path ./model_0.6b --quantize-w8
    python convert_argmax_all.py --model-path ./model_0.6b --only text_projector,code_embedder

Output structure:
    output_dir/
    ├── TextProjector.mlpackage
    ├── CodeEmbedder.mlpackage
    ├── MultiCodeEmbedder.mlpackage
    ├── CodeDecoder.mlpackage
    ├── MultiCodeDecoder.mlpackage
    └── SpeechDecoder.mlpackage
"""

import os
import sys
import time
import argparse
import subprocess


ALL_MODELS = [
    "text_projector",
    "code_embedder",
    "multi_code_embedder",
    "code_decoder",
    "multi_code_decoder",
    "speech_decoder",
]

SCRIPT_MAP = {
    "text_projector": "convert_argmax_text_projector.py",
    "code_embedder": "convert_argmax_embedders.py",
    "multi_code_embedder": "convert_argmax_embedders.py",  # same script does both
    "code_decoder": "convert_argmax_code_decoder.py",
    "multi_code_decoder": "convert_argmax_multi_code_decoder.py",
    "speech_decoder": "convert_argmax_speech_decoder.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert all Argmax-style Qwen3-TTS CoreML models"
    )
    parser.add_argument("--model-path", default="./model_0.6b", help="Path to Qwen3-TTS model")
    parser.add_argument("--tokenizer-path", default="./tokenizer_12hz", help="Path to tokenizer")
    parser.add_argument("--output-dir", default="./argmax_models", help="Output directory")
    parser.add_argument("--quantize-w8", action="store_true", help="Apply W8A16 palettization")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of models to convert")
    parser.add_argument("--skip-verify", action="store_true", help="Skip CoreML verification")
    args = parser.parse_args()

    models = ALL_MODELS
    if args.only:
        models = [m.strip() for m in args.only.split(",")]
        for m in models:
            if m not in ALL_MODELS:
                print(f"Unknown model: {m}")
                print(f"Available: {', '.join(ALL_MODELS)}")
                sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Argmax-style Qwen3-TTS Full Model Conversion")
    print("=" * 60)
    print(f"\nModels to convert: {', '.join(models)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Quantize W8A16: {args.quantize_w8}")
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    total_t0 = time.time()

    results = {}
    for model_name in models:
        print(f"\n{'─' * 60}")
        print(f"Converting: {model_name}")
        print(f"{'─' * 60}")

        script = os.path.join(script_dir, SCRIPT_MAP[model_name])
        cmd = [sys.executable, script, "--model-path", args.model_path, "--output-dir", args.output_dir]

        if args.quantize_w8:
            cmd.append("--quantize-w8")
        if args.skip_verify:
            cmd.append("--skip-verify")
        if model_name == "speech_decoder":
            cmd.extend(["--tokenizer-path", args.tokenizer_path])

        t0 = time.time()
        try:
            # For embedders, both are done in one script, skip if already done
            if model_name == "multi_code_embedder" and "code_embedder" in results:
                print("   (Already converted with CodeEmbedder)")
                results[model_name] = "done"
                continue

            proc = subprocess.run(cmd, capture_output=False, text=True)
            elapsed = time.time() - t0

            if proc.returncode == 0:
                results[model_name] = f"done ({elapsed:.1f}s)"
                print(f"\n   Completed in {elapsed:.1f}s")
            else:
                results[model_name] = f"FAILED (exit {proc.returncode})"
                print(f"\n   FAILED (exit code {proc.returncode})")

        except Exception as e:
            results[model_name] = f"ERROR: {e}"
            print(f"\n   ERROR: {e}")

    total_elapsed = time.time() - total_t0

    print(f"\n\n{'=' * 60}")
    print("CONVERSION SUMMARY")
    print(f"{'=' * 60}")
    for model_name, status in results.items():
        print(f"  {model_name:25s} {status}")
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print(f"Output dir: {args.output_dir}")

    # List output files
    if os.path.exists(args.output_dir):
        print(f"\nOutput files:")
        for f in sorted(os.listdir(args.output_dir)):
            path = os.path.join(args.output_dir, f)
            if os.path.isdir(path):
                size = sum(
                    os.path.getsize(os.path.join(dp, fn))
                    for dp, _, fns in os.walk(path)
                    for fn in fns
                )
                print(f"  {f:40s} {size / 1024 / 1024:.1f} MB")

    print("=" * 60)


if __name__ == "__main__":
    main()
