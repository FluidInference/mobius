#!/bin/sh
# Stage and upload the compiled models, tables, tokenizer, card and reports to Hugging Face.
# Usage: ./upload-hf.sh [repo]   (default FluidInference/cua-s1-4b-coreml)
set -eu
cd "$(dirname "$0")"
REPO="${1:-FluidInference/cua-s1-4b-coreml}"
STAGE=build/hf-stage
rm -rf "$STAGE" && mkdir -p "$STAGE/text" "$STAGE/multimodal/vision"
uv run python compile_models.py build/text/L1024 build/text/L1024-w8 build/text/L1024-gptq \
    build/multimodal/L2048 build/multimodal/L2048-w8 build/multimodal/vision
for d in text/L1024 text/L1024-w8 text/L1024-gptq multimodal/L2048 multimodal/L2048-w8; do
    mkdir -p "$STAGE/$d"
    cp "build/$d/config.json" "$STAGE/$d/"
    for m in build/$d/*.mlmodelc; do cp -Rc "$m" "$STAGE/$d/"; done  # APFS clones: no extra disk
done
cp -Rc build/multimodal/vision/CuaS1Vision_P4096.mlmodelc "$STAGE/multimodal/vision/"
cp build/multimodal/vision/pos_embed_table.f16 build/multimodal/vision/vision_config.json "$STAGE/multimodal/vision/"
cp -c build/text/embeddings.f16 "$STAGE/embeddings.f16"
cp build/tokenizer.json hf/README.md hf/NOTICE hf/LICENSE "$STAGE/"
cp -R reports "$STAGE/reports"
hf repos create "$REPO" --type model 2>/dev/null || true
hf upload-large-folder "$REPO" "$STAGE" --repo-type model
echo "Uploaded to https://huggingface.co/$REPO"
