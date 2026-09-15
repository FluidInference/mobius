# English–Mandarin single-voice Kokoro: project handoff

Updated: 2026-09-15. Shared requirements and implementation runbook, not evidence
that a trainer or production model already exists. Start here when continuing
this project in another environment. Internal agent execution plans stay in
ignored `.mobius/`; this project contract is intentionally tracked.

Receiving-environment implementation now begins in [training/](training/README.md).
Its [2026-09-15 report](training/docs/implementation-2026-09-15.md) records strict
checkpoint mapping, exact untouched CUDA inference parity, acquisition-audit
tooling, and data investigation. Supervised training, real-data gradient/overfit
proof, formal quality evaluation, export, and device acceptance remain open.
The historical demo below is preserved unchanged.

Jump to [artifact inventory](#2-what-is-actually-available),
[data requirements](#5-data-establish-what-we-have-the-dataset-means),
[implementation gates](#6-harness-work-packages-and-required-gates),
[evaluation](#7-automated-evaluation-available-proxies-and-required-improvements),
[export/release](#8-export-apple-integration-and-production-gates), or
[open inputs and receiver checklist](#9-evidence-package-and-genuinely-unresolved-inputs).

## 1. Read this before asking the user to repeat the project

Build a compact, good-quality TTS model with **one female voice that speaks
English, Mandarin, and natural English–Mandarin code-switching**. Start with
**Kokoro v1.1-zh**, build a compatible training harness, reproduce the untouched
checkpoint, and only then fine-tune on verified real bilingual recordings.
The deployment direction is **FluidAudio / Core ML on Apple devices**.

The source Mac is for small samples, demos, code preparation, and lightweight
tests—not official quality evaluation or training. The receiving environment
has reportedly been settled already. Inspect its existing setup and prior
approvals; do not restart provider selection or ask the six generic product
questions again. Its actual hardware, dataset paths, and budget are not recorded
in this checkout.

This handoff does **not** independently authorize paid jobs, provisioning,
uploading recordings, publishing weights, or deploying to users. Preserve any
valid authorization already recorded in the receiving environment. Ask only
for a genuinely missing decision that blocks the next consequential action.
Code inspection, implementation, and lightweight tests can continue while
data access or run approval is being resolved.

### Decision ledger

| Subject | Status | Instruction |
| --- | --- | --- |
| Languages | User-decided | English + Mandarin, including within-sentence switching. “Chinese” here means Mandarin, not Cantonese or every Chinese language. |
| Voice count | User-decided | One female output voice; no cloning or runtime multi-speaker product. |
| Priority | User-decided | Quality first, while keeping the model compact; do not expand scope to many voices. |
| Starting model | User-decided | Kokoro v1.1-zh; compatible StyleTTS2-derived fine-tuning, not training a new architecture from scratch. |
| Existing sample voice | Verified demo fact | `zf_001`, speed 1, native 24-kHz mono. This is the baseline reference, not proof that our eventual recorded speaker is `zf_001`. |
| Runtime direction | Established project direction | FluidAudio/Core ML/Apple; exact oldest device, hard size/latency ceilings, and supported OS matrix still need confirmation. |
| Local execution | User-decided | Source Mac is demo/preparation only. Do not run a full suite or optimizer there. |
| Remote environment | Reportedly decided elsewhere | Recover its actual specification and authorization locally in that environment. No automatic provider change. |
| Architecture preservation | Proposed engineering default | Keep the approximately 82M-parameter generator and its interfaces first. Optimize size only after quality/parity work. |
| First training scope | Proposed engineering default | Conservative supervised adaptation before decoder/adversarial changes; validate the schedule rather than assuming it works. |
| Streaming | Not specified | Do not add a new streaming architecture as an implicit first-release requirement. Establish whether chunked playback meets the actual product need. |
| Data | User said datasets were researched | An accessible, audited single-speaker bilingual manifest has not been recovered here. Find the earlier selection/recordings before sourcing replacements. |
| Production acceptance | Not yet fixed here | Calibrate and freeze a protocol and thresholds before selecting a winning run. Historical demo scores are not acceptance targets. |

PocketTTS and Dia2 were explored earlier; they are not the selected baseline.
Kokoro-French was discussed as a conversion/export reference, not as a French
training objective. No exact French repository/revision is established in this
handoff. Do not invent that dependency: the existing Mobius Kokoro-zh exporter
is the concrete starting point. Revisit architecture only if a documented
compatibility or quality gate fails and the user agrees to the change.

## 2. What is actually available

| Artifact | Location/status | What it does **not** establish |
| --- | --- | --- |
| 12-clip demo metadata | [coreml/bilingual-demo](coreml/bilingual-demo/README.md), included in PR #94 | Not a training set, sealed test set, or official quality baseline. |
| Saved phonemes, IDs, diagnostics, hashes | [manifest.json](coreml/bilingual-demo/manifest.json), `evidence/`, `provenance/` | Hashes identify artifacts; they do not transfer source code, models, or WAVs. |
| Offline demo integrity checker | `coreml/bilingual-demo/verify-demo.py`, nine tests | No synthesis, ASR, learned scoring, acoustic assessment, or training. |
| Original 12 WAVs | Source workspace only; Git-ignored; about 4.2 MiB / 44.625 seconds | Public playback links are unavailable until the exact matching WAVs are supplied. |
| Existing Core ML converter | [coreml/scripts/convert-coreml.py](coreml/scripts/convert-coreml.py), other scripts and docs in `coreml/` | Not a trainer or a demonstrated exporter of our future fine-tuned checkpoint. |
| Swift bilingual frontend | Modified FluidAudio source checkout, under `Sources/FluidAudio/TTS/KokoroAne/` | Not shipped by this Mobius PR. An ordinary FluidAudio clone is not proven to reproduce the demo. |
| Local evaluation prototype | FluidAudio `Benchmarks/tts/bilingual/` | Not included here; Swift/Core ML rendering and MLX scoring are not a ready CUDA stack. |
| Local 300-prompt development suite | Source workspace evaluation artifacts | Not sealed; not licensed/consented target-speaker training recordings. |
| Earlier training proposal | Source workspace private `.mobius/kokoro-bilingual-training/PROPOSAL.md` | Not transferred through Git; speculative settings there are not approved requirements. This handoff supersedes it for cross-environment continuation. |
| Compatible training graph, gradient proof, pilot checkpoint | **Not established** | Do not report training readiness or a trained model from the demo. |
| Audited data/splits, official baseline, final acceptance report | **Not established here** | Recover any additional work already done in the receiving environment, with evidence. |

The upstream model card describes an 82M StyleTTS2/iSTFTNet-derived release
supporting English and Chinese, with no released diffusion/style-encoder
training package. Its published training-data history is **not our accessible
dataset or a grant to reuse those recordings**.
[Upstream model card](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh).

### Verify that the receiving checkout contains this handoff

From a clean Mobius checkout, after preserving any unrelated work:

```bash
gh pr checkout 94 --repo FluidInference/mobius
git branch --show-current
git log -1 --oneline
git status --short
test -f models/tts/kokoro-v1.1-zh/project-handoff.md
cd models/tts/kokoro-v1.1-zh/coreml/bilingual-demo
uv run --frozen python verify-demo.py
uv run --frozen python -m unittest discover -s tests -v
```

The PR branch is `docs/kokoro-bilingual-demo`. The initial demo commit was
`50e10ab596e686fefd2422eea5d22ffbcb17e7f2`; that commit alone does **not** contain
this handoff. If the PR has merged, use the branch containing the merged
handoff instead. Do not discard local changes to perform this checkout.
The checker should validate 12 metadata records and explicitly say audio was
not checked. For already-supplied WAVs, add `--audio-dir <existing-directory>`.
This verifies bytes, not quality, and does not invoke models.

### Transfer/acquisition ledger the receiving agent must complete

For each required artifact, record owner, accessible URI/path, revision/hash,
access restrictions, and status: `present`, `verified`, or `missing`.

1. Mobius checkout containing this file and the existing conversion code.
2. FluidAudio source commit **plus the relevant uncommitted patch if needed**.
   Match frontend files against `signature.tts_source_files` in
   [run provenance](coreml/bilingual-demo/provenance/run.json). A commit ID alone
   cannot describe the dirty source tree used by the demo. Transfer only the
   scoped source changes, not the user's whole workspace.
3. Original demo WAVs if byte-level reproduction/listening is required. Missing
   demo WAVs do not prevent designing the harness or text-only tests.
4. The local evaluator/corpus code if reusing it; otherwise implement a new,
   explicitly versioned backend with compatible output records. Do not claim
   this PR already includes that code.
5. Original PyTorch model, configuration, vocabulary, and baseline voice pack
   from approved caches or upstream; separately identify deployed Core ML
   model assets. Converted asset hashes are not the PyTorch checkpoint hash.
6. Actual target-speaker recordings/transcripts, earlier dataset research,
   provenance/consent records, and split manifests. Never substitute demo TTS
   output for these recordings.
7. Existing environment/run authorization, storage destinations, and secret
   access through the environment's normal secret mechanism. Never put access
   tokens, private recording URLs, or personal paths in committed documents.

## 3. Immediate findings: fix inputs before teaching the model

The detailed evidence and uncertainty are in
[ISSUES.md](coreml/bilingual-demo/ISSUES.md). No listed fix was implemented by
the demo or by this documentation handoff.

| ID | Evidence | Required follow-up |
| --- | --- | --- |
| F1: numeral/erhua | `这本书的价格是二十三元五角。` has a saved phoneme sequence consistent with 二 being merged into the preceding syllable; ASR returns 13 rather than 23. | Preserve source-character/word-role information and restrict erhua eligibility. Add regressions for numeral 二, real 儿 suffixes, independent 儿 words, punctuation, and word boundaries. Do not disable all erhua. |
| F2: API | Saved input has word-like `ˈæpi`. | Proposed exact-token A–P–I pronunciation policy; validate vocabulary IDs. Do not spell every uppercase word as letters. |
| F3: GitHub | Saved input contains `ɡˈɪθʌb`. | Proposed exact-token “git hub” pronunciation; inspect existing lexicon hooks. Do not attribute the whole sentence's ASR mismatch to this token. |
| E1: ASR mode | Same mixed WAV has 0% MER with fixed `zh` and 50% with automatic language selection. | Preserve both hypotheses; fix the primary decoding policy before comparisons. Never choose the transcript with the best reference match. |
| U1/U2: short English words | “a” / “The” exist in the input phones but are omitted/merged by ASR. | Acoustic and recognizer explanations remain open. Inspect narrow segments or an independent judge before labeling these training failures. |
| C1: fallback/tones | The recorded run fell back after a g2pW vocabulary request returned 404; some tone/polyphone prompts have zero CER. | Pin frontend assets and log fallback. Zero CER does not certify tones or pronunciation. This is a historical run fact, not a new availability check. |

Implement approved frontend fixes in their owning FluidAudio code with unit
tests, then reconcile the Python training frontend with Swift. Keep the original
demo immutable. A corrected frontend receives a new version and new outputs;
do not overwrite evidence to make old hashes or scores look improved.

The shared frontend contract must cover normalization, mixed-script span
boundaries, acronym/brand policy, numbers/dates/currency, polyphones, tone
sandhi, erhua, punctuation/pauses, out-of-vocabulary symbols, BOS/EOS, and
chunking. Store text → normalized text → language spans → phones → IDs, plus
lexicon/fallback provenance. Unexpected symbol loss is an error or explicitly
reviewed exclusion, never silent success. Add identical text fixtures on both
Python and Swift sides; any intentional divergence must be versioned.

## 4. Architecture and compatibility contract

The existing converter describes seven deployment stages:
**Albert → PostAlbert → Alignment → Prosody → Noise → Vocoder → Tail**.
These are execution/export partitions of one generator, not separate English
and Mandarin acoustic models. The existing Chinese checkpoint can already
produce both languages; that does not certify equally strong pronunciation,
naturalness, speaker consistency, or code-switching.
[Existing conversion documentation](coreml/README.md).

Preserve and verify from the pinned files:

- Model configuration and checkpoint tensor shapes, including **`n_token=178`**.
  The documented vocabulary dictionary has **171 entries**; its entry count
  is not the embedding size. Do not renumber IDs or resize embeddings to 171.
- ALBERT, its projection, duration/prosody predictor, text encoder, and iSTFTNet
  decoder. Record parameter counts and the exact trainable/frozen key lists.
- The 256-dimensional reference style, its two 128-dimensional uses, and the
  baseline length-conditioned voice table (the converter documents a flat
  fp32 `[510, 256]` voice pack). Document style selection in every
  caller, including whether length counts phones, valid IDs, or special tokens;
  test short/long boundary cases and off-by-one behavior.
- Native 24-kHz output, actual alignment/frame strides, padding, masks, tensor
  layouts, duration limits, and chunk boundaries. Measure shapes from code and
  tensors; do not assume every stage uses the same frame grid.
- Existing conversion limits (`T_enc` up to 512, alignment bounds documented
  in `coreml/docs/shape-bounds.md`) are implementation constraints to test, not
  permission to silently truncate long user text.

The public inference path disables gradients and constructs rounded integer
durations and hard alignment. Simply removing `no_grad` does not create a
useful gradient path through those operations. Build a separate supervised
training forward with explicit duration targets and a documented alignment
strategy, while preserving an unchanged inference path for parity.
[Kokoro model implementation](https://github.com/hexgrad/kokoro/blob/main/kokoro/model.py).

“StyleTTS2-compatible” here means adapting the necessary training machinery to
Kokoro's released generator. It does **not** mean feeding its checkpoint into
an unmodified StyleTTS2 trainer or requiring a new diffusion model. Training-only
aligners, target extractors, and any later discriminators must not accidentally
become deployment dependencies. Preserve Kokoro's learned text representation
initially; replacing it with an unrelated PL-BERT is an architecture experiment.

### Exact baseline identity

Upstream repository: `hexgrad/Kokoro-82M-v1.1-zh`.
Model filename: `kokoro-v1_1-zh.pth`.
Published model SHA-256:

```text
b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b
```

Verify the downloaded/cached file, pin the repository revision, and separately
hash `config.json`, vocabulary, `zf_001.pt`, converted voice bytes, code,
dependencies, and any auxiliary models. A mutable `main` URL is a discovery
reference, not a reproducibility pin. Treat untrusted checkpoint formats as
code-execution risks; use trusted sources and appropriate safe loading.
[Published checkpoint identity](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh).

## 5. Data: establish what “we have the dataset” means

First search the receiving environment and prior research for the selected
recordings. This checkout does not establish dataset names, accessible paths,
usable hours, speaker consent, or split membership. Do not ask the user to
select everything again until that search is complete, and do not claim the
absence of records here proves the recordings do not exist elsewhere.

### Audit requirements

- Verify permission for the intended training, derived voice, storage/transfer,
  and release use; preserve consent/license evidence and restrictions. A public
  download URL or upstream model license alone is insufficient evidence.
- Verify that the English and Mandarin recordings really are the **same target
  female speaker**. Combining two unrelated female speakers is not a verified
  single-voice dataset. If one bilingual speaker is unavailable, document the
  adaptation/voice-identity tradeoff and obtain a decision before changing scope.
- Count usable hours and utterances separately for English, Mandarin, and
  actual within-utterance code-switching, plus sessions, accents, recording
  conditions, and rejected material. Do not infer bilingual coverage from a
  dataset's name or from concatenating unrelated monolingual clips.
- Retain original recordings and transcripts immutably. Audit clipping,
  truncation, background speech/music, overlapping speakers, excessive silence,
  duplicate audio, sample rates/channels, transcript mismatch, and segmentation.
  Specify QC cutoffs and record exclusions; do not quietly delete source data.
- Decide the spoken-text convention for numerals, abbreviations, punctuation,
  English contractions, and simplified/traditional Chinese. Preserve both raw
  transcript and model-normalized text; do not “correct” away what was spoken.
- Cover tones/polyphones, short English function words, consonant endings,
  numbers, names, acronyms, and both switch directions. This is coverage to
  measure, not a claim that every proposed corpus already contains it.
- Split by recording session/source passage and near-duplicate text/audio,
  not random neighboring clips. Audit leakage before freezing train/dev/test.
  With one speaker, session-disjoint tests matter; do not promise an impossible
  speaker-disjoint test of that same single-speaker product.
- Keep the 12 demo prompts and their translations/paraphrases in development,
  never a newly claimed sealed acceptance set. Keep generated demo WAVs and
  benchmark reference recordings out of target-speaker training.

No minimum hours, 40/40/20 sampling mixture, or large-data acquisition budget
has been validated. Earlier exploratory hour/mixture suggestions are not
requirements. Choose sampling weights from the audited coverage, log effective
exposure by language, and check that minority-language oversampling does not
turn into memorization.

### Manifest and preprocessing contract to implement

One immutable record per utterance, with a versioned schema:

| Field group | Required information |
| --- | --- |
| Identity | Stable utterance ID, corpus/source ID, real speaker ID, recording session and source-passage IDs. |
| Audio | Restricted-access source URI, original SHA-256, original format/rate/channels, segment offsets, duration; prepared-audio URI/hash and exact transformation chain. |
| Text | Raw transcript, normalized spoken text, language and mixed-span annotations, annotation provenance. |
| Rights | Reference to consent/license/use restrictions and audit status; no private legal documents embedded in public manifests. |
| Split/QC | Train/dev/test assignment, split-manifest hash, duplicate-group ID, QC flags, exclusion/review reason. |
| Model input | Frontend version/hash, phones, token IDs, special-token convention, token length, dropped/unknown symbols, fallback trace. |
| Targets | Aligner version, token-map version, durations/alignment confidence, feature-grid specification, pitch/voicing/other target provenance. |

Do not invent populated records for unavailable recordings. Validate missing
fields explicitly. Prepared audio should match the model's 24-kHz mono contract
with a pinned resampler/channel policy; retain originals and account for any
trimming in alignment timestamps. Training crop boundaries, mel/STFT parameters,
pitch frame grids, masks, and token-duration sums must agree.

Select and validate an aligner against bilingual real recordings. StyleTTS2's
auxiliary aligner is a candidate, **not** a guaranteed drop-in: reconcile symbol
inventory, blank/pad IDs, language phones, sample rate, and time resolution.
Likewise validate pitch/voicing targets rather than interpreting the generator's
`N` feature as an arbitrary waveform-amplitude label. Reject or review low-
confidence alignments. Forced alignment given the answer transcript is not
independent recognition or a pronunciation-accuracy score.
[StyleTTS2 auxiliary modules](https://github.com/yl4579/StyleTTS2#pre-trained-modules).

## 6. Harness work packages and required gates

**These remain the required gates.** The sibling [training/](training/README.md)
toolkit now implements pinned acquisition, strict checkpoint loading,
untouched-inference parity, and an initial data-audit CLI, with its own locked
environment and tests. It does not yet implement a supervised training forward,
optimizer, aligner, or CUDA quality scorer. Consult its implementation report
for measured results and limits; do not force Linux tools to import Core ML
or MLX dependencies.

Use explicit stages (audit, preprocess, parity, gradient check, micro-overfit,
pilot, evaluate, export), each with input/output schemas and failure exit codes.
No default command may silently advance from inspection into an optimizer run.
Pin executable source rather than relying on open-ended package constraints.

### Gate 0 — source/environment/data inventory

Produce an environment record with OS, CPU/GPU/VRAM, driver/CUDA, Python,
Torch/audio libraries, dependency lock, workspace/storage paths, disk quota,
existing run authorization, and compute/storage limits. Do not prescribe an
80-GB GPU or a vendor from historical suggestions. Inspect what is available.

Complete the transfer ledger and data audit. Missing training data blocks
real-data training, not architecture inspection or parity implementation.
Missing budget approval blocks paid execution, not writing tests. Record which
gate each missing item blocks; avoid declaring the entire project blocked.

### Gate 1 — untouched-checkpoint reproduction

1. Instantiate a compatible generator and build an explicit checkpoint-key
   mapping. Report every matched, renamed, missing, unexpected, or shape-
   mismatched key and parameter count. Fail on unexplained mismatches; do not
   use permissive loading to hide newly random parameters.
2. Compare an independently instantiated, frozen upstream inference oracle and
   the candidate implementation with the **same** token IDs, style tensor,
   speed, precision, device/backend, and stochastic inputs. First bypass text
   frontend differences; test frontend equivalence separately.
3. Put both in evaluation mode. Account for dropout, random excitation/phase,
   and random-number consumption. A shared seed alone may not ensure identical
   noise if execution paths consume randomness differently. Capture/inject the
   same noise where necessary without changing normal inference semantics.
4. Capture intermediate outputs: text representation/projection, duration
   logits and integer durations, alignment, text features, F0/noise features,
   decoder intermediates, and final waveform. Compare shapes/lengths, exact
   discrete values, max/mean error, and appropriate signal/spectral differences.
5. Calibrate tight same-backend floating-point tolerances with repeated frozen
   runs. Separately document cross-backend/export tolerances. Existing Core ML
   waveform/mel correlation smoke thresholds are **not** a sufficient proof of
   untouched PyTorch checkpoint reproduction.
6. Cover English, Mandarin, mixed scripts, punctuation/short prompts, numeric
   and acronym fixtures, and permitted length/style-row boundaries. Preserve
   per-case inputs, settings, hashes, tensors or diagnostics, and failures.

**Pass:** a reproducible report with no unexplained weight/input differences and
all declared parity checks passing. Matching one pleasant sentence is not a
pass. Do not optimize weights until this gate passes.

### Gate 2 — gradient and target correctness

Implement a training forward distinct from the oracle. Initially use validated
teacher/forced alignment for acoustic supervision and direct duration loss;
document any differentiable/soft-alignment alternative explicitly. Match all
target frame rates, masks, crop offsets, and duration sums.

For real recordings, audit per loss and parameter group:

- Finite loss and finite gradients; expected nonzero gradients across multiple
  real batches, not a requirement that every element be nonzero every step.
- Which parameters are intended to learn and which are frozen. Check optimizer
  membership and actual weight deltas; frozen modules must remain unchanged.
- No accidental `detach`, inference-only context, inaccessible rounded-duration
  path, padding leakage, or target/prediction length mismatch.
- F0/voicing treatment, unvoiced masks, duration supervision, and mel/STFT
  reconstruction all have explicit definitions. A regularizer alone reducing
  loss is not proof that the speech path learns.
- Training/evaluation mode and checkpoint reload preserve the untouched
  inference behavior until an optimizer update is deliberately made.

**Pass:** a gradient-coverage report and target/alignment checks. Do not label a
forward/backward call successful merely because it did not raise an exception.

### Gate 3 — tiny real-data overfit and restart test

Proposed diagnostic scale: roughly **10–30 real utterances**, spanning English,
Mandarin, and genuine mixed speech if available. This is an engineering test,
not a sufficient training dataset or quality claim. Declare its step/time cap
and optimizer settings before execution in the authorized environment.

Demonstrate that aligned reconstruction and relevant prediction losses improve,
that expected parameters change, and that free-running synthesis still works.
Distinguish teacher-aligned reconstruction from inference with predicted
durations; success on the former does not prove the latter. Inspect held-out
controls for collapse. Save, interrupt, restore, and continue a checkpoint.
Compare uninterrupted/resumed state and outputs under the declared determinism
policy. A test failure goes back to implementation, not a larger training run.

### Gate 4 — bounded bilingual pilot

Proposed first adaptation schedule, subject to Gates 2–3:

- Keep architecture/vocabulary fixed; initially freeze the decoder and ALBERT.
  Adapt the projection, text encoder, and duration/prosody components as justified
  by gradient coverage. If freezing prevents useful learning, change one group
  at a time and record an ablation.
- Start with duration, validated pitch/voicing, and acoustic reconstruction
  losses, plus documented regularization/real-data bilingual retention. Define
  loss weights and masks from observed scales; do not silently import unrelated
  recipe defaults. Real recordings, not synthetic demo audio, supply supervision.
- Keep the baseline style table fixed for the first proof. A small regularized
  style residual or learned single-speaker representation is an optional later
  experiment, not a proven solution. Preserve zero-update compatibility and
  test unseen length rows; do not let sparse per-length parameters memorize clips.
- If the actual recorded speaker differs from `zf_001`, explicitly evaluate
  whether the fixed style can fit that speaker. Decide whether to adapt style
  and/or decoder after the micro-overfit evidence, without adding a cloning API.
- Unfreeze decoder components at a separately justified learning rate only if
  the measured error calls for it. Add adversarial/WavLM losses only after
  supervised stability, licensing, memory, and evaluation checks. Keep every
  added dependency out of the inference package unless deliberately required.
- Prove single-GPU correctness first. Multi-GPU is not a default assumption;
  upstream StyleTTS2 documents distributed-training limitations.
  [Upstream training guidance](https://github.com/yl4579/StyleTTS2#training).

Before a pilot, write the exact trainable-key list, sampling weights, batch size,
gradient accumulation/effective batch, crop/length policy, optimizer/LRs,
scheduler/warmup, precision, loss weights, gradient clipping, seed, validation
cadence, max steps/wall time/cost, checkpoint cadence/retention, and stop rules.
Those numeric settings are **not validated yet**; estimate them from the small
run's memory and throughput, then freeze a bounded pilot config. Mixed precision
must pass stability checks rather than being assumed safe everywhere.

Stop on persistent nonfinite loss/gradients, invalid alignments, collapse,
out-of-budget execution, or agreed language-retention failures. Save diagnostics;
do not restart an unbounded sweep. Evaluate on development data at declared
intervals; reserve sealed tests for the predeclared candidate-selection stage.

### Checkpoint/reproducibility contract

Each run records parent/base hashes, code commits and dirty patches, complete
resolved config, dataset/split/frontend/auxiliary-model hashes, environment,
seed policy, metrics, wall time, and resource use. Resume checkpoints include
generator/style state, optimizer/scheduler/scaler, step/epoch, sampler position,
Python/NumPy/Torch/device RNG state, and any enabled discriminator state.
If exact dataloader replay is not supported, explicitly define the restart
boundary and validate its consequences. Use atomic checkpoint writes and hash
verification; log best-dev and latest separately. A deployment checkpoint is a
separate minimal artifact, not the entire training-state archive.

## 7. Automated evaluation: available proxies and required improvements

The user wants useful automatic accuracy measurement, not a process blocked
on manual listening after every step. Build reproducible automatic reports and
targeted failure triage. Do not relabel proxy scores as ground truth.

### Historical prototype, not production certification

The source workspace has a 300-prompt development corpus: 120 English, 140
Mandarin, and 40 mixed. It combines external evaluation prompts and authored
parallel/challenge prompts. The 12 public demo prompts are all project-authored;
the external corpus and its redistribution obligations are not transferred by
this PR. Record source revisions/licenses if retrieving the larger corpus.

The local corpus builder records 100 English and 100 Chinese prompts from
`MiniMaxAI/TTS-Multilingual-Test-Set`, revision
`cb416f0ac3658da0577e97873065e19fe6488917`, with `CC-BY-SA-4.0` provenance.
Recover `Benchmarks/tts/bilingual/corpus.py` and
`Documentation/TTS/MinimaxCorpus.md` from the FluidAudio source handoff before
reusing or redistributing that material. These are evaluation texts, not the
missing single-speaker bilingual training recordings. Public benchmark results
are comparable only when model/voice, prompt slice, frontend, judge, and scoring
protocol match; do not treat unrelated published WER/MOS values as our baseline.

- ASR: MLX Whisper large-v3-turbo, fixed `en` for English and `zh` for Mandarin
  and mixed; auto-language mixed decoding is an additional diagnostic. No
  expected transcript prompt; temperature 0; no previous-text conditioning.
- Normalization: `nfkc-simplified-lower-punctuation-space-v1`. English WER uses
  words, Mandarin CER uses characters, mixed MER uses Han characters plus
  English-word/numeric runs. Number formatting, acronym spelling, contraction
  variants, and homophones can still affect results.
- Naturalness: one UTMOSv2 configuration/checkpoint, **not human MOS** and not
  established as calibrated for Mandarin/code-switching. Do not turn a scalar
  into a pass/fail quality threshold without calibration.
- Additional local prototype diagnostics: SpeechBrain ECAPA embedding cosine
  consistency, signal checks, and YIN-based F0 summaries. These are not all
  included in the public demo evidence. Without real target-speaker references,
  embedding consistency is not proof of the correct voice; F0 summaries do not
  establish tone accuracy.

The exact saved ASR/naturalness/signal configurations and hashes are in
[demo provenance](coreml/bilingual-demo/provenance/run.json) and adjacent
`*-config.json` files. Historical percentages must not become production
targets or be compared directly to scores from a new backend.

### Formal protocol to build in the designated environment

1. **Content recovery:** report pooled edits/reference units and per-utterance
   distributions separately for EN WER, ZH CER, and mixed MER. Also report
   substitutions/deletions/insertions, completion rate, failed/missing samples,
   and coverage. Never silently drop failures or combine incompatible metrics
   into a single headline “accuracy.” Preserve raw and normalized transcripts.
2. **Semantic-critical content:** report number/currency/date/name/acronym
   failures with reviewed expected spoken forms. A normalization may equate
   twenty-three with 23, but must never equate 13 with 23. Keep strict and
   equivalence-aware diagnostics separate and freeze rules in advance.
3. **Switching:** slice by switch direction, number of switches, position,
   English span length, short function words, and surrounding punctuation.
   Use annotated span alignment for span-level errors; do not guess language
   boundaries solely from the recognizer's automatic language label.
4. **Independent checks:** add a second, independently selected bilingual ASR
   judge or labeled diagnostic subset. Two modes of Whisper are not two
   independent judges. Disagreement flags uncertain cases; do not select the
   favorable transcript. Audit ASR on real reference speech as well as synthesis
   to expose recognizer bias.
5. **Pronunciation/tones:** prepare an annotated diagnostic subset with expected
   lexical pronunciation and context/sandhi rules. A suitable phoneme/tone
   recognizer must be validated before reporting PER/tone accuracy. Until then,
   mark acoustic tone accuracy **unmeasured**, not passed because CER is zero.
6. **Naturalness and voice:** report predicted MOS as a proxy by language and
   condition; test its agreement with a small blinded bilingual calibration set.
   Compare speaker embeddings against held-out real target-speaker audio and
   across languages/sessions, accounting for channel effects. If the target
   voice changes from `zf_001`, that identity change is not automatically drift.
7. **Signal/robustness:** nonfinite samples, clipping, empty/truncated output,
   unexpected silence, repetition, extreme duration/speaking rate, chunk seams,
   punctuation, and maximum supported lengths. Calibrate duration expectations
   separately by language and case type.
8. **Statistics:** paired baseline/candidate comparisons on the same inputs;
   bootstrap at an appropriate utterance/session/topic unit, fixed seed and
   documented confidence intervals. Report sample counts and tails. Repeated
   prompts, stochastic synthesis, judge bias, and corpus design remain sources
   of uncertainty beyond a naive per-utterance interval.
9. **Backend changes:** a CUDA Whisper port is not numerically identical to MLX.
   Pin model/revision, decoding, resampler, VAD/chunking, precision, and text
   normalization. Re-render/rescore both baseline and candidate with the same
   formal protocol; do not compare a new CUDA number to an old Mac percentage.

Automation should select the most informative disagreement/failure clips for
limited human calibration and final listening review. It must not require a
human to approve every checkpoint. Conversely, without calibrated pronunciation
and perceptual checks, do not claim fully automatic proof of human-level quality.

### Separate frontend gains from training gains

Keep three labeled conditions (and a fourth ablation if useful):

| Condition | Weights | Frontend | Purpose |
| --- | --- | --- | --- |
| Historical reference | Original | Historical | Preserve what the demo actually measured; not necessarily the formal baseline. |
| Corrected-input baseline | Original | Fixed/pinned | Measure the benefit of frontend repair under the new formal protocol. |
| Trained candidate | Fine-tuned | Same fixed/pinned frontend | Attribute incremental changes to training, not different input text/phones. |

For a controlled acoustic comparison, hold style/voice policy and speed fixed
where possible. If style adaptation is part of training, report that factor and
its ablation separately. For deployment, also compare the complete intended
pipeline rather than relying only on precomputed-token tests.

## 8. Export, Apple integration, and production gates

### Gate 5 — candidate export and parity

The existing [conversion script](coreml/scripts/convert-coreml.py) constructs
`KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh')`; its current CLI does not expose
a trained-checkpoint argument. **Running it unchanged can export the upstream
model instead of our candidate.** Add explicit local checkpoint/config/style
inputs, strict loading, and embedded candidate hashes. Test that candidate
weights are actually loaded and that missing inputs fail rather than silently
falling back to an upstream download.

- Start with an uncompressed candidate export and verify stage-by-stage and
  end-to-end parity against that exact PyTorch candidate. Then evaluate fp16 /
  palettization/other size changes independently. The existing mixed-precision
  and int8-palettized pipeline is a reference, not a free quality guarantee.
- Preserve the seven-stage interfaces or version any intentional changes.
  Export the one required voice/style artifact; additional source voices need
  not ship in the single-voice product. Do not assume fewer packaged voices
  materially reduce the shared generator's parameter count.
- Audit the deployed FluidAudio stage names, revisions, and fixes. The inspected
  source runtime expects `KokoroNoise_v2.mlmodelc` while the older converter
  documents `KokoroNoise`. This is a compatibility/fix-history check, not a
  filename rename that can be assumed to solve numerical differences.
- Verify vocabulary/embedding consistency, style-row selection, BOS/EOS,
  dynamic-shape limits, duration/alignment bounds, special cases, excitation,
  and noise/tail precision. Validate held-out bilingual audio after export.
- Pin Core ML tools and the export OS/Python environment. Core ML execution and
  device profiling require a designated supported Apple host; a CUDA machine
  alone cannot establish actual ANE/device performance. Do not automatically
  designate the source demo Mac as the official test host.

### Gate 6 — FluidAudio/device acceptance

Implement any integration changes in the owning FluidAudio repository, with
its API/thread-safety conventions and real-model tests. Never use
`@unchecked Sendable`, mock models, or synthetic stand-in audio to claim model
validation. Text-only and metadata tests are fine for their actual scope.

Confirm the actual supported-device/OS matrix. Mobius conversion guidance
uses iOS 17+/macOS 14+; that is an engineering starting point, not a user-approved
oldest-device performance contract. Measure, on designated devices:

- Cold download/compile/load separately from warm synthesis; model/package
  bytes, peak memory, end-to-end latency, and real-time factor (`elapsed / audio
  duration`). Separate frontend time, inference time, and disk I/O.
- Per-stage compute-unit placement and CPU/GPU fallback, not just a claim of
  “ANE compatible.” Use the repository's `coreml-cli` guidance when appropriate.
- Short/medium/long EN/ZH/mixed inputs, chunk seams and pause continuity,
  repeated invocations, cancellation/concurrency as supported by the API,
  initialization failure, invalid inputs, and offline/cache behavior.
- If streaming/chunked playback is required, explicitly define and measure
  time to first playable audio, buffering, continuity, and cancellation. Do
  not infer these from total waveform generation time.
- Full deployed pipeline quality relative to the accepted candidate, including
  any precision/quantization changes. A fast export that changes pronunciation
  or loses words fails the quality requirement.

### Gate 7 — release candidate, not automatic rollout

Before candidate selection, freeze an acceptance record covering each language,
code-switching, semantic-critical errors, voice consistency/naturalness,
robustness, artifact size, supported devices, and latency/memory budgets.
Populate numeric thresholds from baseline/pilot calibration and product needs;
do not invent a 20% WER improvement, +1 CER point allowance, or 5% speed margin
and present it as an approved requirement.

Require: all preceding gates passed, sealed evaluation completed under the
frozen protocol, known limitations disclosed, scoped final listening/calibration
complete, and no unresolved release-blocking regression. Promotion needs the
agreed acceptance authority. “Loss decreased” is not a release criterion.

Release artifacts must include minimal inference weights, config, vocabulary,
single-voice style asset, Core ML packages, hashes, version/compatibility metadata,
model card, provenance and applicable notices, evaluation/device reports, and
integration-test evidence. Training-only assets and private recordings do not
belong in the app bundle. Audit redistribution obligations for the base model,
training data, auxiliary models, and export code individually.

Use versioned download/cache paths with checksum validation and an atomic
activation mechanism; keep the previous known-good artifact for rollback.
Provide a staged rollout/rollback procedure and define allowed failure telemetry
without collecting user text/audio by default. Publishing artifacts and actual
production deployment are separate authorized actions, not automatic side
effects of finishing training.

## 9. Evidence package and genuinely unresolved inputs

Future run artifacts should make each gate inspectable. Suggested logical
outputs below are **contracts to implement**, not files already present:

```text
<run-id>/
  environment.json              # machine, dependencies, limits, authorization reference
  inputs.json                   # code/data/model/frontend hashes and access references
  resolved-config.yaml          # exact settings, no secrets
  data-audit.json               # rights status, counts, QC, overlap, exclusions
  checkpoint-map.json           # every weight mapping and shape result
  parity/                       # same-backend and export reports kept separate
  gradient-audit.json            # intended/observed trainability and target checks
  checkpoints/                  # private training/resume states with retention policy
  renders/                      # immutable WAVs + text/phones/IDs/style/seed records
  evaluation/                   # per-utterance judge outputs, metrics, uncertainty
  export/                       # candidate-bound manifests and parity evidence
  device-report.json            # hardware, cold/warm timing, memory, placement
  acceptance.md                 # frozen criteria, results, limitations, approval
```

Store heavy/private artifacts in the approved artifact store; commit schemas,
code, documentation, and safe summaries only. New sample audio must come from
real recordings or the real model as appropriate, and must be labeled by origin.
Generated evaluation audio is never silently treated as training recordings.

| Unknown to resolve | First action / owner | Blocks |
| --- | --- | --- |
| Receiving machine and existing authorization | Receiving agent inspects its environment and earlier setup decisions. | Actual execution beyond the recorded approval. |
| Dataset location, rights, same-speaker bilingual coverage | Receiving agent retrieves earlier research/data; data owner supplies only missing access/consent evidence. | Audited preprocessing and all real-data optimizer runs. |
| Exact target voice/accent if the recordings do not determine it | Audit the recordings and existing decisions; ask product/data owner only if ambiguous. | Final voice acceptance; may affect style adaptation. |
| Reproducible Swift frontend patch/evaluator transfer | Source-workspace maintainer provides scoped versioned source; receiver checks hashes and portability. | Exact old-demo reproduction and deployment frontend parity. |
| Compatible aligner/targets and differentiable graph | Training implementation work, measured at Gates 1–3. | A credible pilot; not something the user should have to design. |
| CUDA evaluator implementation and judge calibration | Evaluation implementation work, with targeted bilingual validation. | Trustworthy model comparisons and production claims. |
| GPU/time/storage limits and deadline | Recover existing limits first; user/operator fills only missing bounds. | Paid runs/sweeps and retention decisions. |
| Oldest device, hard footprint/latency limits, streaming requirement | Recover product configuration first; product owner resolves missing requirements. | Final device acceptance, not initial checkpoint mapping. |
| Numeric quality gates and release approver | Propose after baseline/pilot calibration, freeze before candidate selection. | Production promotion and rollout. |

### First actions for the receiving agent

1. Confirm this handoff is in the checked-out revision; run the offline demo
   checks. Report exactly which artifacts are present versus referenced.
2. Inspect existing hardware, setup, data, permissions, and prior run budget.
   Fill the transfer/unknown ledger rather than asking the generic model-goal,
   language, speaker-count, or platform questions again.
3. Recover scoped frontend/evaluator source if needed. Implement/test F1 and
   agreed lexical policies before producing canonical training inputs.
4. Build strict checkpoint mapping and untouched-inference parity tooling.
   In parallel in the workflow, audit real data and define splits/target formats;
   do not wait for a final production device choice to start these safe tasks.
5. Implement the actual supervised training path, gradient audit, and restart
   test. Run model checks only in the authorized environment, within its bounds.
6. Present the gate evidence and a capped pilot configuration. Start only if
   existing authority covers that run; otherwise ask for the missing approval.
7. Continue through evaluation, export, device verification, and release
   acceptance. Report failed gates honestly instead of declaring E2E completion
   when only a checkpoint or a WAV exists.

The first status response should say: **what was found, what is implemented,
which gate is next, and the precise missing input (if any)**. Do not imply that
the project is undefined because some production thresholds remain open.

### Copyable receiving-environment task

> Continue the one-female-voice English–Mandarin/code-switching Kokoro v1.1-zh
> project. Read `models/tts/kokoro-v1.1-zh/project-handoff.md` and applicable
> `AGENTS.md` files first. Preserve the product decisions and the Mac demo-only
> boundary. Inspect this environment's already-settled setup, data, and run
> authorizations. Complete the artifact/readiness ledger, then implement the
> checkpoint-parity, data-audit, and training-readiness work packages. Do not
> confuse demo evidence with a trainer or official baseline. Ask only for
> missing access/decisions/authority that actually block the next gate; continue
> safe implementation work where possible. Never launch unapproved paid work,
> publish recordings/weights, or deploy merely because this handoff exists.

## 10. Reference hierarchy

User decisions above and verified artifacts take precedence over exploratory
suggestions. Sources accessed for this handoff are navigation references;
pin executable revisions before using them in a run.

- [Demo evidence and limitations](coreml/bilingual-demo/README.md) and
  [frontend/evaluator issue evidence](coreml/bilingual-demo/ISSUES.md).
- [Existing Mobius conversion](coreml/README.md),
  [architecture](coreml/docs/architecture.md),
  [shape bounds](coreml/docs/shape-bounds.md), and
  [conversion trials](coreml/TRIALS.md).
- [Kokoro v1.1-zh release/model card](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)
  and [inference source](https://github.com/hexgrad/kokoro).
- [StyleTTS2 training reference](https://github.com/yl4579/StyleTTS2): useful
  machinery and limitations, not a Kokoro-specific ready-made recipe.
- [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2),
  [historical MLX ASR model](https://huggingface.co/mlx-community/whisper-large-v3-turbo),
  [local speaker-proxy model](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

Delivery of this file makes the instructions available through Git. It does
not prove the other environment has pulled them or received local-only assets;
the receiving agent must acknowledge its actual revision and artifact inventory.
