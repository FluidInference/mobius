# Kokoro bilingual training readiness

This Linux/CUDA toolkit establishes a reproducible starting point for adapting
Kokoro v1.1-zh to one female English/Mandarin/code-switching voice. It implements
pinned baseline acquisition, strict checkpoint mapping, untouched-inference
parity, and acquisition-manifest auditing. **An optimizer/trainer, validated
alignment targets, and a trained candidate are not implemented yet.**

Read the [project handoff](../project-handoff.md),
[implementation report](docs/implementation-2026-09-15.md), and
[dataset investigation](docs/datasets.md). The source Mac remains for demos;
this toolkit does not depend on Core ML or MLX.

## Reproduce

Run from this directory, in the designated training environment:

```bash
uv sync --frozen
uv run --frozen pytest -q
uv run --frozen ruff check src tests

# Explicit download: approximately 328 MB, only three public pinned files.
uv run --frozen kokoro-readiness fetch-baseline --assets .artifacts/baseline
uv run --frozen kokoro-readiness environment --output .runs/environment.json

# Real model execution, no optimizer. Choose a new output directory every time.
uv run --frozen kokoro-readiness parity \
  --assets .artifacts/baseline --output .runs/parity-new --device cuda
```

Tested matrix: Linux x86_64, Python 3.12.13, Torch 2.11.0+cu130,
Transformers 4.57.6, RTX A6000 (48 GB), NVIDIA driver 610.43.02. `uv.lock`
pins transitive dependencies. The PyTorch index is explicit; Kokoro is pinned
to commit `dfb907a02bba8152ca444717ca5d78747ccb4bec`. The package is intended
to run from this source checkout through `uv`; adjacent schemas/evidence are
part of its input contract.

`baseline.lock.json` pins Hugging Face revision
`01e7505bd6a7a2ac4975463114c3a7650a9f7218` and SHA-256/size for the checkpoint,
configuration (including vocabulary), and `zf_001` voice pack. Acquisition
verifies bytes before activation and refuses to replace mismatched files.
Parity uses only supplied, verified local assets and does not download models.

## What parity proves

The candidate independently assembles pinned upstream layers and implements
its own token-to-waveform orchestration. The oracle is a separate upstream
`KModel` instance loaded independently. This proves the assembly and checkpoint
mapping; shared layer implementations are not an independent architecture
reimplementation or proof of trainability.

- Map every checkpoint key explicitly. Only `module.` removal and legacy
  weight-normalization renames are permitted; unexpected/missing keys,
  collisions, shape/dtype errors, and nonfinite weights fail.
- The release has 548 tensors. Pinned upstream code adds 140 AdaIN affine
  tensors for an export workaround. Their exact identities (weight=1, bias=0)
  are enumerated and frozen, not silently randomized. All 688 resulting state
  tensors must equal the independently loaded oracle before inference.
- Compare ALBERT, projection, duration encoder/LSTM/logits, F0/N projections,
  text features, aligned decoder inputs, excitation/noise source, vocoder
  projection, decoder output, final waveform, integer durations, and post-run
  CPU/CUDA RNG states. The same full RNG state is restored before each forward;
  source-output and final-RNG comparisons verify matching random consumption.
- Repeat the oracle per case to establish determinism. FP32, evaluation mode,
  deterministic algorithms, disabled TF32, and zero absolute/relative tolerance
  are explicit. A different environment may fail exact parity; investigate
  before changing tolerances.
- Cover all 12 historical demo token sequences plus one-phone, punctuation,
  and 510-content-token/512-total-token boundary probes. Style row is content
  token count minus one, excluding BOS/EOS. Invalid lengths/IDs fail rather
  than truncate. The 4,000-alignment-frame cap is checked before decoding.

The historical input sequences deliberately retain known frontend defects.
Passing these checks does not fix pronunciation, reproduce the Mac WAV bytes,
establish a quality baseline, or measure Core ML/device parity. Boundary probes
are interface tests, not natural speech or training data. New waveforms are
generated transiently by the real model; only hashes/diagnostics are retained.

Each run writes `environment.json`, `inputs.json`, `resolved-config.json`,
`checkpoint-map.json`, `loaded-state-parity.json`, `cases/*.json`, and
`summary.json`. The environment includes actual source-file hashes, dependency
lock hash, parent Git revision, and dirty-tree status. Reports fail closed;
exit code 1 means a failed check. Raw run directories and binaries are ignored.
Safe compact evidence is retained under `evidence/`.

## Data acquisition audit

The JSONL schema is [recording.schema.json](schemas/recording.schema.json).
Supply one original real recording per record, with relative audio path,
SHA-256, speaker/session/passage identities, language, raw/normalized text,
split, and external rights/speaker/QC evidence references. Keep private
manifests and recordings outside Git. No invented sample manifest is supplied.

```bash
uv run --frozen kokoro-readiness audit-data \
  --manifest .artifacts/data/recordings.jsonl --audio-root .artifacts/data \
  --output .runs/data-audit.json

# Separate read-only candidate-corpus inspection; does not create training splits.
uv run --frozen kokoro-readiness inventory-emime \
  --archive .artifacts/emime/UEDIN_mandarin_bilingual_data_v1.1.tar.bz2 \
  --output .runs/emime-inventory.json
```

A missing/empty manifest fails. The audit rejects generated/unknown audio
origin, pending rights/QC, ambiguous gender or multiple speaker IDs, missing
English/Mandarin coverage, unassigned/incomplete splits, repeated audio bytes,
session/passage/duplicate/exact-text leakage, development-prompt contamination,
unsafe audio paths, wrong hashes/headers, and empty/nonfinite/all-zero audio.
Audio is read in blocks and peak/clipping are reported. QC thresholds and
semantic transcript correctness still require a reviewed protocol; no invented
automatic threshold substitutes for it. Original rates need not be 24 kHz;
preprocessing/resampling and target grids are a separate, unimplemented stage.

`passed` means only the implemented acquisition checks passed. Every report
still says `training_ready: false`: near-duplicate/paraphrase checks, independent
speaker/rights verification, frontend agreement, alignment, and target checks
remain necessary. Research-only rights never become production voice approval.
The EMIME command inventories archive headers without extraction or choosing
a production speaker; microphone views and test segments are not extra hours.

## Next implementation gates

1. Transfer the source Mac's scoped FluidAudio patch, including untracked
   bilingual source/tests and evaluator if reused; check provenance hashes.
2. Fix/version the frontend and match Python/Swift fixtures. Approve and audit
   real target-speaker recordings, rights, session/passage splits, and coverage.
3. Implement validated alignment/feature targets and a supervised training
   forward. The rounded-duration inference graph is not that training path.
4. Prove real-data gradient coverage, then capped micro-overfit and resume.
5. Freeze a measured pilot budget/config, evaluate, export the actual candidate,
   and verify on designated Apple devices before release acceptance.

No command in this toolkit starts an optimizer, uploads recordings, publishes
weights, or deploys a model. Production-quality thresholds and run budgets
remain decisions to calibrate, not hardcoded claims.
