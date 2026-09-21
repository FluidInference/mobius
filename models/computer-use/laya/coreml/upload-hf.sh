#!/bin/sh
# Upload the compiled buckets, tokenizer and model card to HuggingFace.
# Usage: ./upload-hf.sh [repo]   (default FluidInference/laya-coreml)
set -eu
cd "$(dirname "$0")"
REPO="${1:-FluidInference/laya-coreml}"
hf repos create "$REPO" --type model 2>/dev/null || true
for L in 128 256 512; do
    NAME="laya_multilingual_fp16_L${L}_options32"
    [ -d "build/$NAME.mlmodelc" ] || uv run python -c "import coremltools as ct; ct.utils.compile_model('build/$NAME.mlpackage', destination_path='build/$NAME.mlmodelc')"
    hf upload "$REPO" "build/$NAME.mlmodelc" "$NAME.mlmodelc" --commit-message "Add $NAME"
done
hf upload "$REPO" artifacts/multilingual/tokenizer/tokenizer.json tokenizer.json --commit-message "Add tokenizer.json"
hf upload "$REPO" hf/README.md README.md --commit-message "Add model card"
hf upload "$REPO" reports reports --commit-message "Add verification reports"
echo "Uploaded to https://huggingface.co/$REPO"
