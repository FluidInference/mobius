#!/bin/sh
# Retrain Cua's cua-s1-4b-0.2 SFT recipe (train_4b_v2.py) on a smaller Qwen3.5 base, then score it on
# the GUI-360 text test split. Runs on one CUDA GPU (24 GB is enough for 0.8B and 2B in bf16).
#
#   BASE=Qwen/Qwen3.5-0.8B ./run.sh          # default
#   BASE=Qwen/Qwen3.5-2B   ./run.sh
#   HF_REPO=<org/name> ./run.sh               # also push the adapter + reports (private repo)
set -eu
cd "$(dirname "$0")/.."  # bundle root: README.md, splits/, cloud/
BASE="${BASE:-Qwen/Qwen3.5-0.8B}"
NAME="cua-s1-$(basename "$BASE" | tr '[:upper:].' '[:lower:]-')-text"
CUA_COMMIT=a5f18829df026d7b9ef80c339194b44b1c61f856

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d cua ] || git clone -q https://github.com/trycua/cua cua
git -C cua checkout -q "$CUA_COMMIT"
export PYTHONPATH="$PWD/cua/libs/cua-bench-s1/python/src:$PWD/cua/libs/cua-s1/python/src"

uv venv -q --python 3.12 .venv
# Cua's four-b-train extra pins transformers<5, but Qwen3.5 (qwen3_5) needs transformers 5.x.
uv pip install -q --python .venv "torch==2.7.0" "torchvision==0.22.0" "transformers==5.17.0" "peft>=0.18.1" \
    "accelerate>=0.34" "safetensors>=0.5" "pillow>=10,<12" "numpy<2.3" "huggingface-hub>=0.34"
PY=.venv/bin/python

mkdir -p runs reports
$PY cloud/eval_gui360.py --base-model "$BASE" --tasks splits/test.jsonl --out "reports/$NAME-zeroshot.json"
$PY cua/libs/cua-s1/training/train_4b_v2.py --base-model "$BASE" --modality text \
    --train splits/train.jsonl --val splits/val.jsonl --out "runs/$NAME" --epochs 4 --batch-size 8 --lr 1e-4
$PY cloud/eval_gui360.py --base-model "$BASE" --adapter "runs/$NAME" --tasks splits/test.jsonl \
    --out "reports/$NAME-sft.json"

if [ -n "${HF_REPO:-}" ]; then
    .venv/bin/hf repos create "$HF_REPO" --type model --private 2>/dev/null || true
    .venv/bin/hf upload "$HF_REPO" "runs/$NAME" "$NAME" --commit-message "SFT adapter on $BASE"
    .venv/bin/hf upload "$HF_REPO" reports reports --commit-message "GUI-360 text reports"
fi
echo "done: runs/$NAME, reports/$NAME-{zeroshot,sft}.json"
