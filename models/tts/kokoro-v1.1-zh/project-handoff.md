# Kokoro bilingual project status

Current deliverable: a **PyTorch model and matching voice weights** for one female
English/Mandarin voice, including code-switching. Core ML and Apple integration
are deferred. Keep Kokoro v1.1-zh's compact architecture and vocabulary; this is
not a multi-speaker product.

## Start here

- [Training and inference commands](training/README.md)
- [Selected model, methods, results, and limits](training/MODEL_CARD.md)
- [Machine-readable results and hashes](training/results.json)

## What exists

Training ran on an RTX A6000 using real EMIME MF5 English/Mandarin recordings.
The selected deterministic checkpoint is update 500. Strict checkpoint loading,
untouched baseline parity, real-data gradients, micro-overfit, exact restart,
export parity, and offline CPU/GPU inference are implemented and checked.
Model weights, real recordings, and generated evaluation audio remain local.
Run artifacts are under ignored `.runs/` and `.artifacts/`.

The frozen 74-utterance test improved English WER from 6.15% to 4.31%; Mandarin
CER increased from 1.50% to 1.93%. Mixed ASR is unreliable when the recognizer
translates the utterance. This model is not production-qualified.

## Remaining work

- Bilingual listening review: pronunciation, tones, naturalness, language switches,
  and voice consistency. Report failures rather than inferring quality from ASR.
- Better real same-speaker code-switch coverage and session-disjoint evaluation.
- Production voice rights/consent and release acceptance.

Preserve real recordings and test splits. Generated speech is evaluation audio,
never substitute training data. Publishing weights or deploying is a separate
action requiring applicable authorization. Do not restart dataset acquisition
or request Swift/Apple access as a prerequisite to the existing PyTorch work.

The [original Mac demo](coreml/bilingual-demo/README.md) is historical reference;
its saved inputs remain immutable and are also used by baseline parity checks.
