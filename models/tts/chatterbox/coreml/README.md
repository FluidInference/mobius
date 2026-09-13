# Chatterbox Multilingual + Nano → CoreML

Conversion toolkit for [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)
multilingual (23 languages) and [chatterbox-nano](https://huggingface.co/ResembleAI/chatterbox-nano)
(110M, English) → CoreML. Published artifacts:
[FluidInference/chatterbox-multilingual-coreml](https://huggingface.co/FluidInference/chatterbox-multilingual-coreml).
Status, parity numbers, and gotchas: [REPORT.md](./REPORT.md) (MTL),
[REPORT-NANO.md](./REPORT-NANO.md) (Nano).

## Try it (no conversion needed)

Requires Apple silicon, macOS 14+, [uv](https://docs.astral.sh/uv/). The
converted models (~2.3 GB) download from HF on first run; the upstream
PyTorch checkpoint is NOT needed.

```bash
cd models/tts/chatterbox/coreml
uv sync
uv run python verify/e2e_coreml.py --lang en
uv run python verify/e2e_coreml.py --lang de
uv run python verify/e2e_coreml.py --lang fr --text "Bonjour tout le monde."
```

Output wavs land in `build/e2e/`. First model load is slow (CoreML compiles
~2 GB of T3 packages once, then caches). Supported `--lang` codes: the 23
languages listed on the HF model card (ar, da, de, el, en, es, fi, fr, he,
hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh).

Notes for feedback:
- Uses the built-in voice only (voice cloning encoders aren't converted yet).
- One sentence per call; generated audio caps at ~13 s (500-token flow bucket).
- Decode uses the I/O-KV model (38 ms/step); the faster MLState variant
  (16.8 ms/step) needs the Swift host and isn't wired into this Python driver.
- Sampling is seeded (`--seed`) but fp16 logits ≠ fp32, so token sequences
  won't match the PyTorch reference run-for-run.
- Do not force `.cpuOnly` compute units — T3 predict crashes (see REPORT).

## Reproduce the conversion

```bash
# Multilingual (Llama-520M + 10-step CFM)
uv run python baseline.py                      # stock PyTorch reference wavs (~3.3 GB checkpoint)
uv run python convert-t3.py --fp16             # T3 prefill + I/O decode + parity
uv run python convert-t3.py --fp16 --stateful  # MLState decode + timing
uv run python convert-s3gen.py --fp16          # Flow + HiFT + parity
uv run python export-tables.py                 # embedding tables + default voice

# Nano (GPT2-small 110M + 2-step meanflow; ~1.9 GB checkpoint)
uv run python baseline-nano.py                      # stock PyTorch reference wavs
uv run python convert-t3-nano.py --fp16             # T3 prefill + I/O decode + parity
uv run python convert-t3-nano.py --fp16 --stateful  # MLState decode + timing
uv run python convert-s3gen-nano.py --fp16          # meanflow Flow + HiFT + parity
uv run python export-tables-nano.py                 # embedding tables + default voice
uv run python verify/e2e_nano_coreml.py             # full CoreML chain
```

If the HF CDN is throttled, fetch the Nano checkpoint out-of-band and point
`CHATTERBOX_NANO_CKPT` at the directory (see `src/nano_ckpt.py`).

## Layout

- `src/t3_coreml.py` — Llama-520M reimplementation (prefill / decode /
  stateful decode), llama3-RoPE from the loaded checkpoint, alignment-head
  attention rows as outputs.
- `src/t3_nano_coreml.py` — GPT2-small reimplementation for Nano/Turbo:
  batch-1 (no CFG), in-graph `wpe`, no alignment heads.
- `src/s3gen_coreml.py` — flow (encoder + 10-step CFG Euler in-graph),
  meanflow flow (2-step, `r` end-time input) and HiFT (matmul STFT/iSTFT,
  host-seeded SineGen randomness).
- `verify/analyzer_port.py` — host-side AlignmentStreamAnalyzer; the Swift
  port spec (MTL only — Nano has no analyzer).
- `verify/e2e_coreml.py` / `verify/e2e_nano_coreml.py` — full chain
  drivers, tables-only hosts.
