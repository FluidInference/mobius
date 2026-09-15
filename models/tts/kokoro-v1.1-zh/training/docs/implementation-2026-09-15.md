# Training-environment implementation report — 2026-09-15

## Outcome and scope

The first readiness milestone is implemented: a locked Linux/CUDA toolkit,
strict mapping of the released checkpoint, exact untouched-inference parity,
and an initial acquisition-manifest auditor. **No optimizer step, supervised
training graph, aligned real-data batch, trained checkpoint, official quality
evaluation, or Core ML candidate export is claimed.**

Work started from PR #94 revision
`6fb15048d600046b3b3630c038f3d6e4f50f64bd`, in an isolated worktree on
`feat/kokoro-training-readiness`. The unrelated existing Nemotron checkout was
preserved. The user designated this session for training and asked to start
implementation and document the work. Local code preparation, public baseline
acquisition, bounded parity checks, and candidate-corpus inspection were
performed under that instruction. No provisioning, paid service, uploads,
publication of weights/recordings, or rollout occurred.

## Environment recovered

| Item | Verified value |
| --- | --- |
| Host | Linux x86_64 |
| GPU | NVIDIA RTX A6000, 48 GB class VRAM, initially idle |
| Driver / Torch CUDA build | 610.43.02 / 13.0 |
| Python / Torch | 3.12.13 / 2.11.0+cu130 |
| Container memory | 48 GiB |
| Storage at initial inspection | Approximately 34 GB free on a 100 GB overlay |
| Dependency isolation | Dedicated `uv` environment and committed lockfile |
| New-run budget | Bounded FP32 inference checks; no training budget inferred |

The first generic CUDA forward/backward smoke test established device access
only. It was not a model gradient check. Installing the isolated Torch/CUDA
dependencies consumed additional storage; current bytes free are captured in
each environment record. Keep dataset/checkpoint retention within this volume.

## Work performed

1. Read the shared handoff and model/demo instructions. Ran the offline demo
   checker (12 metadata records, audio explicitly unchecked) and all nine
   original demo unit tests successfully. Preserved the historical evidence.
2. Inspected the workspace for existing target-speaker data, Kokoro weights,
   FluidAudio source, and prior selection notes. No selected recording manifest
   or modified frontend was recovered. The source-workspace report subsequently
   confirmed acquisition was unfinished and frontend changes remained local.
3. Inspected and pinned upstream Kokoro source and Hugging Face release
   revisions. Downloaded only checkpoint/config/one voice initially; verified
   the published checkpoint SHA-256. Added explicit repeatable acquisition.
4. Built a separately assembled generator with an independent inference
   orchestration path, strict checkpoint mapper, input/style contracts, and
   detailed oracle comparison runner. No Core ML/MLX imports enter this toolkit.
5. Investigated the strict-load failure described below. Added a narrowly
   enumerated identity-constant rule, then compared every resulting state tensor
   with a separately instantiated upstream oracle before running inference.
6. Executed two bounded CUDA parity runs. The second run uses the final parity
   implementation and locked environment represented in the committed evidence.
7. Added acquisition JSONL schema and checks for real-recording provenance,
   external evidence references, single-speaker bilingual scope, split leakage,
   duplicate data, path containment, hash/header consistency, and basic signals.
   A missing-manifest invocation intentionally returned failure with
   `training_ready: false`; no placeholder dataset was substituted.
8. Investigated EMIME and ASCEND using primary sources. Acquired the public
   EMIME v1.1 archive with a 1.4-GB download cap for read-only local inventory.
   Added an archive inspector that never extracts files. See the dataset report
   and measured inventory for available recordings and outstanding decisions.
9. Added failure-case tests and portable evidence summaries. Ran lint, format,
   compilation, and unit checks. Wrote reproduction steps, provenance, artifact
   ledger, limitations, and next gates. Heavy/local data remain Git-ignored.

## Strict-loading finding

The pinned upstream `KModel` first tries loading each submodule, then catches
any exception, strips seven characters from every key, and retries with
`strict=False`. That fallback can conceal incompatibility; it is not used by
the candidate loader.

The release has **548 source tensors**, with DDP `module.` prefixes and legacy
weight-normalization keys. The new mapper explicitly removes only that prefix
and maps `.weight_g`/`.weight_v` to parametrization originals 0/1, respectively.

The initial strict audit found **140 absent AdaIN affine tensors**. Pinned
`AdaIN1d` uses `InstanceNorm1d(affine=True)` as a documented ONNX shape workaround,
while the checkpoint omits those affine weights. The implementation enumerates
only actual `AdaIN1d.norm` weight/bias keys, initializes exact identity constants
(one/zero), and freezes them. Every other unexplained missing/unexpected key,
collision, nonfinite value, or shape/dtype mismatch remains fatal. Supplied
malformed tensors are not rescued by this allowlist.

All **688 loaded state tensors** then matched the independent oracle exactly.
The assembled graph contains **81,810,022 parameters**, including the frozen
export-compatibility identities. The full source-to-destination map and identity
entries are in [checkpoint-map.jsonl](../evidence/checkpoint-map.jsonl).

## Validation evidence

| Check | Result and limits |
| --- | --- |
| Original demo checks | 12 metadata records and 9 tests passed; original WAVs not supplied |
| Toolkit unit tests | 38 passed; guard/metadata tests, not speech-quality tests |
| Lint / formatting / compilation | Passed on toolkit source/tests; summary script lint/format checked |
| Strict checkpoint mapping | 548 release tensors + 140 explicit frozen identity tensors; no unexplained differences |
| Loaded oracle/candidate states | 688 tensor comparisons passed exactly |
| CUDA parity run 001 | 15/15 cases, approximately 24.67 seconds |
| CUDA parity run 002 | 15/15 cases, approximately 22.91 seconds; zero maximum absolute error |
| Intermediate capture | 23 comparisons per case, including waveform, integer durations and RNG states |
| Oracle repeat control | Exact repeated outputs for every case |
| Peak CUDA allocation, run 002 | 1,050,369,536 bytes (allocator peak, not full process/device memory) |
| Missing recording manifest | Expected nonzero exit; no training-ready claim |
| EMIME archive inventory | Completed: 9,044 WAV members; primary headers inspected; seven female speakers each have about 26–28 minutes across EN/ZH using one microphone |

Run 002 includes four English, four Mandarin, four mixed development cases and
three token boundary probes. The 510-content-token probe selects style row 509;
one-token probes select row 0. These establish interface coverage, not natural
long-form speech quality. Known historical frontend errors remain in the saved
inputs intentionally. No text frontend is run to obscure token differences.

[parity-002.json](../evidence/parity-002.json) retains the environment, source
hashes, pinned inputs, settings, every case result, checked tensor names, and
SHA-256 values of the detailed raw reports. Raw per-tensor reports remain in
ignored `.runs/parity-002/` and are regenerable with the README command. The
run's parent revision and dirty-tree status are explicit; source-file hashes
identify the implementation before it was committed. The summary utility reads
existing evidence only and never performs model execution.

```bash
uv run --frozen python scripts/summarize-parity.py \
  --run .runs/parity-002 --output evidence/parity-002.json \
  --mapping-output evidence/checkpoint-map.jsonl
```

Exact same-backend parity is a passed milestone. It does not establish gradient
coverage, a bilingual training recipe, perceptual quality, Mac waveform parity,
Core ML compatibility, ANE placement, or production readiness.

## Artifact and transfer ledger

| Artifact / owner | Receiving status | Identity / next action |
| --- | --- | --- |
| Mobius handoff / project | Verified | PR #94 `6fb1504`; tracked locally |
| Model/config/voice / hexgrad | Verified locally, ignored binaries | `baseline.lock.json`: repository revision and all three byte hashes |
| Toolkit dependencies / implementation | Installed and locked | `pyproject.toml`, `uv.lock`, environment/source hashes |
| Historical demo records / source workspace | Verified metadata | Original 12 clips unchanged; nine tests pass |
| Historical WAVs / source workspace | Missing here | Transfer exact bytes only if historical listening/reproduction is needed |
| Modified Swift frontend/tests / source workspace | Missing here | Transfer scoped commit plus dirty patch and new/untracked files; match recorded source hashes |
| Existing evaluator/300-prompt suite / source workspace | Not transferred | Transfer scoped source if reused; otherwise build/version the CUDA evaluator |
| EMIME v1.1 / Edinburgh | Public candidate archive acquired for inspection | Archive inventory does not select speaker or approve production use |
| Target-speaker data/splits/rights / data owner | Unresolved | No approved JSONL training manifest; acquisition/selection remains work |
| Real-data aligner/targets/training forward / implementation | Not implemented | Next technical gate after input/data contracts |
| Formal quality and device acceptance / project | Not established | Calibrate before candidate selection; designate an Apple acceptance host |

The scoped FluidAudio transfer must include tracked diffs **and untracked new
files** under the relevant KokoroAne source/tests and evaluator paths. A Git
commit alone does not reproduce the historical dirty working tree. Do not
transfer credentials or the whole source workspace. Compare received files
with `signature.tts_source_files` in the preserved demo provenance; document
intentional changes separately from historical reproduction.

## Next gate and precise outstanding inputs

Proceed to the shared frontend contract and real-data/target audit. The source
patch transfer and a selected/approved speaker corpus are the missing inputs;
model goal, language scope, voice count, runtime direction, and training host
are settled. EMIME may support small experiments after review, but neither its
availability nor separate bilingual utterances settle the production voice or
code-switching-data requirement.

Then implement the supervised forward and losses against validated duration,
pitch/voicing, and acoustic targets. Prove gradients and a capped real-data
micro-overfit/resume run before proposing a measured pilot. Exact device/OS,
quality/latency limits, listening calibration, and promotion authority are
needed for later production acceptance, not reasons to redo this milestone.
