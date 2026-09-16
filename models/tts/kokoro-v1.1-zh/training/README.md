# Kokoro bilingual PyTorch training

[Listen to baseline vs trained audio](listening-review/README.md) ·
[Model and measured results](MODEL_CARD.md) · [Result manifest](results.json)

Fine-tune Kokoro v1.1-zh on real English/Mandarin recordings and export a
PyTorch generator with its matching voice table. The selected MF5 model is an
experimental adaptation; production speech quality is not established. Model
weights and real recordings remain local. The PR includes 24 generated review
clips. Core ML is outside the current scope.

## Inference

From this source directory, with a previously exported bundle:

```bash
uv sync --frozen
uv run --frozen python -m kokoro_training.bundle infer \
  --bundle /path/to/bundle \
  --text '请打开 API，然后把结果发给我。' \
  --output .runs/example.wav --device cuda
```

Use `--device cpu` for CPU inference. A bundle contains `model.pth`, `voice.pt`,
`config.json`, `vocab.json`, source, dependency lock, and SHA-256 manifest. Use
its matching voice table. Loading verifies hashes and maps all weights strictly;
inference does not download models. Output is 24 kHz mono WAV plus a JSON sidecar.
Split text above 510 content tokens. Speed is limited to 0.25–4 and output to
4,000 duration frames (100 seconds).

## Reproduce training and selection

Tested on Linux, Python 3.12, Torch 2.11.0+cu130, and an RTX A6000. Install the
system `espeak-ng` library first. `uv.lock`, `baseline.lock.json`, and
`training-assets.lock.json` pin packages, base weights, corpus, JDC, and ASR.
Run from this directory; commands refuse to overwrite existing outputs.

```bash
uv sync --frozen
uv run --frozen kokoro-readiness fetch-baseline --assets .artifacts/baseline
uv run --frozen kokoro-readiness parity \
  --assets .artifacts/baseline --output .runs/reproduce/parity --device cuda
uv run --frozen python -m kokoro_training.acquire data
uv run --frozen python -m kokoro_training.acquire targets
uv run --frozen python -m kokoro_training.acquire asr
uv run --frozen python -m kokoro_training.acquire extract-mf5
uv run --frozen python -m kokoro_training.prepare --output .runs/reproduce/prepared

uv run --frozen python scripts/check-real-training.py \
  --data .runs/reproduce/prepared --output .runs/reproduce/gradients.json
uv run --frozen python scripts/check-resume.py \
  --data .runs/reproduce/prepared --output .runs/reproduce/resume
uv run --frozen python -m kokoro_training.train \
  --data .runs/reproduce/prepared --output .runs/reproduce/micro \
  --micro 8 --steps 100 --lr 0.0001 --validate-every 50
uv run --frozen python -m kokoro_training.train \
  --data .runs/reproduce/prepared --output .runs/reproduce/pilot \
  --steps 500 --lr 0.00002 --train-style --style-lr 0.001 \
  --validate-every 100 --snapshot-every 100

for step in 100 300 500; do
  uv run --frozen python -m kokoro_training.bundle export \
    --checkpoint .runs/reproduce/pilot/snapshot-${step}.pt \
    --output .artifacts/reproduce-step${step}
  uv run --frozen python -m kokoro_training.evaluate \
    --data .runs/reproduce/prepared --bundle .artifacts/reproduce-step${step} \
    --output .runs/reproduce/mf5-v2-step${step}-eval --controls
done
uv run --frozen python scripts/select-checkpoint.py \
  --run .runs/reproduce/pilot --evaluations .runs/reproduce \
  --output .artifacts/reproduce-selected --record .runs/reproduce/selection.json
uv run --frozen python -m kokoro_training.evaluate \
  --data .runs/reproduce/prepared --bundle .artifacts/reproduce-selected \
  --output .runs/reproduce/test --split test
```

The [selection policy](evaluation-policy.json) compares development English WER
and Mandarin CER, breaking ties with acoustic loss. Freeze the selected hash
before test evaluation. `best.pt` alone means lowest acoustic development loss.
`--resume` continues `last.pt` with matching data/settings and a larger step target.

## Checks and artifacts

```bash
uv run --frozen pytest -q
uv run --frozen ruff check src tests scripts
uv run --frozen python scripts/check-bundle.py \
  --bundle .artifacts/reproduce-selected \
  --checkpoint .runs/reproduce/pilot/snapshot-500.pt \
  --output .runs/reproduce/export-check.json
```

For `check-bundle.py`, use the checkpoint step recorded in `selection.json`;
500 was selected in the recorded run. Parity checks compare an independently
loaded upstream `KModel`, all state tensors, intermediate outputs, and RNG state.
The identity affine tensors added by upstream export support are explicitly
initialized and frozen. Model checks require the acquired assets; unit tests
never download them.

Raw training/evaluation reports live under ignored `.runs/`; recordings and
weights under ignored `.artifacts/`. [results.json](results.json) retains the
selected model, hashes, counts, validation outcomes, and quality regressions.
The optional `kokoro-readiness audit-data` and `inventory-emime` commands inspect
new acquisition manifests/archives; they do not run training or approve a voice.
See [NOTICE.md](NOTICE.md) for upstream and dataset attribution.
