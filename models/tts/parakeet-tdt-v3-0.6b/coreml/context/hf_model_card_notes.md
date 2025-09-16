# Hugging Face Model Card Notes: nvidia/parakeet-tdt-0.6b-v3

Key details extracted from the public model card for quick reference.

## Summary
- Task: Multilingual automatic speech recognition (ASR)
- Architecture: FastConformer encoder + RNNT + TDT
- Params: ~0.6B
- License: CC‑BY‑4.0

## Languages (25 EU + Russian/Ukrainian)
bg, hr, cs, da, nl, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, sv, es, en, ru, uk

## Input / Output
- Input: 16 kHz mono audio (`.wav`, `.flac`)
- Output: text with punctuation and capitalization

## Notes relevant to conversion
- The model card reports accurate word/segment timestamps; Parakeet TDT uses duration buckets which we expose from the joint head (last 5 logits along the class axis).
- Long‑audio inference is reported for GPU; our CoreML export intentionally uses a fixed 15‑second window to fit on‑device constraints.

References
- HF repo: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- NeMo FastConformer: https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer
