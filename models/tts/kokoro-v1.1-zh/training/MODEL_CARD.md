# Kokoro MF5 English/Mandarin adaptation

## Artifact

A single female voice adapted from Kokoro-82M-v1.1-zh, retaining its architecture,
vocabulary, 24 kHz output, and length-indexed 256-dimensional style interface.
The selected checkpoint is deterministic run v2, update **500**. Weights remain
local, along with generated evaluation audio.

| File | Bytes | SHA-256 |
|---|---:|---|
| `model.pth` | 327,453,363 | `2c27865b7c794124197e49ab9b7c814da088e9ef87f1dd605f95b87e1458a4c0` |
| `voice.pt` | 523,739 | `119eca879d472a5327bc1e01c358958e9028a62d09d3a528d953248e94fbc8b3` |

Use the matching voice table, configuration, and frontend with these weights.
[results.json](results.json) records model/data/report identities and exact counts.

## Data and training

Real EMIME MF5 microphone-0 recordings: 314 utterances, approximately 28 minutes.
Official test IDs and translated parallel passages stay together. Other passages
have deterministic development assignment. There is one session per language,
so these splits are passage-disjoint, not session-disjoint.

| Split | English utterances | Mandarin utterances | Prepared seconds |
|---|---:|---:|---:|
| Train | 96 | 129 | 1,347.95 |
| Development | 6 | 9 | 83.15 |
| Test | 43 | 31 | 274.55 |

The frontend uses pinned Misaki, NFKC and simplified Chinese, explicit API/GitHub
pronunciations, and distinct numeral/erhua handling. Unknown symbols and overly
long inputs fail. Audio is resampled from 22,050 to 24,000 Hz and padded to the
600-sample duration grid. Temporary baseline synthesis supplies token boundaries
transferred by MFCC DTW onto real recordings. This is approximate alignment;
**only real recordings are reconstruction targets**. Pinned JDC supplies F0;
pitch/energy targets use the 300-sample grid and StyleTTS2 mel convention.

Trainable parameters: ALBERT projection, prosody predictor, text encoder, and
one shared voice-style residual. ALBERT, decoder, and identity affine constants
are frozen. AdamW uses LR 2e-5 (style: 1e-3), two accumulated utterances per step,
2.4-second crops, balanced language sampling, and seed 1729. Loss combines mel,
duration, F0, energy, and style regularization. Strict deterministic execution
uses equivalent slice-based reflection padding to avoid nondeterministic CUDA
padding gradients. Checkpoints retain optimizer and all RNG states.

The eight-recording overfit check reduced loss 43.1%. After an exploratory
1,000-update run, the corrected deterministic run started from the base weights
and completed 500 updates. Its development acoustic loss fell 5.98555 → 3.19828.
Candidates 100/300/500 were compared on development ASR; 300 and 500 tied,
with lower acoustic loss selecting 500. Test results did not influence selection.

## Measured quality

Matched baseline/candidate inputs, seed, speed, frontend, and respective voice
tables. Whisper large-v3-turbo uses pinned weights, greedy decoding, and explicit
English/Chinese transcription. Scores apply NFKC, simplified Chinese, lowercase,
and punctuation removal; numeral spellings are not equated. Evaluation uses
FP32 generator tensors, FP16 ASR, matmul TF32 off, and cuDNN TF32 allowed.

| Metric | Baseline | Trained |
|---|---:|---:|
| Development English WER, 53 words | 1.89% | 0.00% |
| Development Mandarin CER, 235 characters | 7.23% | 5.96% |
| Held-out English WER, 325 words / 43 utterances | 6.15% | 4.31% |
| Held-out Mandarin CER, 466 characters / 31 utterances | 1.50% | 1.93% |
| Mixed-control raw CER, 6 utterances | 13.18% | 19.38% |

English ASR improved; Mandarin test CER increased by two character errors.
The mixed score regressed under a recognizer that sometimes translates mixed
speech instead of transcribing it. It is retained as a diagnostic, not treated
as a reliable measurement of code-switch quality. ASR does not establish
naturalness, Mandarin tone correctness, or speaker consistency. No blinded
bilingual listening study has been completed.

## Implementation validation

- Untouched baseline: 688 state tensors and 15 inference cases match upstream.
- Eight real-recording gradient/forward checks pass; alignment negative controls
  distinguish matching text from wrong-text recordings.
- Three updates uninterrupted versus save/restart after update one match
  bit-for-bit, including model, optimizer, RNG, and validation state.
- Exported weights/voice match the selected checkpoint; upstream Kokoro produces
  identical waveforms/durations on 12 controls. Exactly 148 generator tensors
  changed; frozen weights did not change.
- Standalone offline CPU and GPU synthesis pass. Peak training allocation: 2.01 GB.

## Limits and provenance

This is an experimental model. The corpus provides no real code-switch speech,
no session-disjoint validation, and no independently certified production voice
consent. Listening, tone, and naturalness acceptance remain open. No deployment
or Core ML acceptance is implied.

Corpus: [EMIME](https://www.emime.org/participate/emime-bilingual-database.html)
(ODbL/DbCL per its README). Base code/weights and auxiliary models have separate
licenses; see [NOTICE.md](NOTICE.md). Acquisition revisions and checksums are in
[baseline.lock.json](baseline.lock.json) and [training-assets.lock.json](training-assets.lock.json).
The original full run reports remain in
[Git history](https://github.com/FluidInference/mobius/tree/fb13fc48d599b733759a4ecc60d0c10b6499fffc/models/tts/kokoro-v1.1-zh/training/evidence)
and the local ignored run directories; the recorded run's dependency-lock hash
is retained in `results.json`.
