#!/usr/bin/env bash
# Full FLEURS forced-mode WER/CER across 5 languages on the FP32 OV export.
# Runs sequentially (4 cores) and clears each language's HF cache afterward
# to keep disk bounded (~13G peak per language).
set -u
cd ~/nemotron-ov-export
source .venv/bin/activate

LANGS="en_us es_419 fr_fr cmn_hans_cn ja_jp"
for L in $LANGS; do
  echo "########## $L ##########"
  python benchmark_fleurs_ov.py \
    --model-dir ./build_ov \
    --languages "$L" \
    --mode forced \
    --use-hf \
    --output-json "ov_full_${L}.json" 2>&1 \
    | grep -vE "OneLogger|telemetry|Downloading|Generating|Resolving|^\s*$|unauthenticated|HF_TOKEN"
  # Reclaim disk before next language
  rm -rf ~/.cache/huggingface/datasets/google___fleurs* \
         ~/.cache/huggingface/hub/datasets--google--fleurs* 2>/dev/null
  echo "[cache cleared for $L]"
done
echo "ALL DONE"
