#!/usr/bin/env bash
set -u
cd ~/nemotron-ov-export
source .venv/bin/activate
# Set HF_TOKEN in your environment before running (do not hardcode tokens):
#   export HF_TOKEN=hf_xxx
: "${HF_TOKEN:?set HF_TOKEN to your Hugging Face access token}"
clear_cache(){ rm -rf ~/.cache/huggingface/datasets/google___fleurs* ~/.cache/huggingface/hub/datasets--google--fleurs* 2>/dev/null; }
for L in en_us es_419 fr_fr cmn_hans_cn ja_jp; do
  for attempt in 1 2; do
    echo "##### $L (attempt $attempt) #####"
    python benchmark_fleurs_ov.py --model-dir ./build_ov_int8 --languages "$L" --mode forced --use-hf \
      --output-json "int8wo_full_${L}.json" > "int8wo_${L}.log" 2>&1
    [ -f "int8wo_full_${L}.json" ] && { echo "[$L OK]"; clear_cache; break; }
    echo "[$L attempt $attempt failed]"; tail -3 "int8wo_${L}.log"; clear_cache
  done
done
echo "ALL DONE"
