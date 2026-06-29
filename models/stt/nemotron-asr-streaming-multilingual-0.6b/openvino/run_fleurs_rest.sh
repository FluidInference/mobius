#!/usr/bin/env bash
set -u
cd ~/nemotron-ov-export
source .venv/bin/activate
clear_cache() {
  rm -rf ~/.cache/huggingface/datasets/google___fleurs* \
         ~/.cache/huggingface/hub/datasets--google--fleurs* 2>/dev/null
}
run_lang() {
  local L="$1"
  for attempt in 1 2; do
    echo "########## $L (attempt $attempt) ##########"
    python benchmark_fleurs_ov.py \
      --model-dir ./build_ov --languages "$L" --mode forced --use-hf \
      --output-json "ov_full_${L}.json" > "full_${L}.log" 2>&1
    if [ -f "ov_full_${L}.json" ]; then
      echo "[$L OK]"; grep -E "files=|WER=|CER=" "full_${L}.log" | tail -1
      clear_cache; return 0
    fi
    echo "[$L attempt $attempt failed]"; tail -3 "full_${L}.log"
    clear_cache
  done
  echo "[$L GAVE UP]"; return 1
}
for L in fr_fr cmn_hans_cn ja_jp; do run_lang "$L"; done
echo "ALL DONE"
