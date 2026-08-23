#!/bin/zsh
set -x
cd "$(dirname "$0")"
uv run python measure_calibrated_quant.py calibrate || exit 1
for args in \
  "--method fp16 --eval-set both" \
  "--method rtn --nbits 4 --eval-set both" \
  "--method awq --nbits 4 --eval-set both" \
  "--method gptq --nbits 4 --eval-set both" \
  "--method gptq --nbits 4 --head-nbits 8 --head-block 0 --eval-set both" \
  "--method gptq --nbits 5 --eval-set both" \
  "--method awq --nbits 6 --eval-set both" \
  "--method rtn --nbits 8 --eval-set both"; do
  echo "===== evaluate $args"
  uv run python measure_calibrated_quant.py evaluate ${=args} || exit 1
done
echo "ALL RUNS DONE"
