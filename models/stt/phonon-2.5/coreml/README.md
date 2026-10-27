# Phonon-2.5 → Core ML / FluidAudio

Experimental conversion of [FermionResearch/Phonon-2.5](https://huggingface.co/FermionResearch/Phonon-2.5)
into FluidAudio's existing v3-family component contract. **No new inference runtime is required.**
The resulting directory loads through unmodified FluidAudio **0.17.3**:

```swift
let models = try AsrModels.loadLocal(from: directory, version: .v3,
                                     encoderComputeUnits: .cpuAndNeuralEngine)
```

`.v3` selects the tensor, vocabulary, blank-ID and decoder contract; the loaded
weights remain Phonon-2.5. A shipping catalog integration should use distinct IDs,
URLs and labels, with a macOS 15 / iOS 18 availability gate. Models are not bundled
inside this repository or the Swift package. A separate Phonon-2.5 model registry
entry is not yet present upstream.

## Upstream status (2026-10-27)

No open Phonon-2.5 PR was found in FluidAudio or möbius (including a listing of all
open PRs). The earlier [Phonon-2 conversion PR #111](https://github.com/FluidInference/mobius/pull/111)
was closed without merge. This directory adapts its palette/export recipe and the
existing v3/redux wrappers; see [ATTRIBUTION.md](ATTRIBUTION.md).

## Conversion

Validated on macOS 26.6.2, Apple M5, Python 3.11, PyTorch 2.7.0, NeMo 2.3.1 and
coremltools 9.0. `uv.lock` pins the complete environment. Python 3.11 is a deliberate
deviation from the historical 3.10.12 conversion environment.

```sh
uv sync --locked
# Download phonon-2.5.bps.tar.zst from the source repo at the revision in reports/m5-smoke.json.
# Archive SHA256: 1faa04053a472cabab3bc31d07f30c23f22906f71044ca10eb1f0de36d1f19a9
mkdir -p checkpoint
# The tested archive has four flat members: config.json, model.fermion,
# packed_manifest.json and bps_manifest.json. Inspect members before extraction.
tar -tf phonon-2.5.bps.tar.zst
tar -xf phonon-2.5.bps.tar.zst -C checkpoint
# model.fermion SHA256: 17cb62d08e751a41819eb779db089aa7ac4eb2bab433ba5c2991d8cbd3cfefb9

uv run python export_encoder.py --container checkpoint/model.fermion \
  --work build --audio public-16khz-audio.wav
uv run python fivevalue_encoder.py --container checkpoint/model.fermion \
  --work build --mode sparse-g8 --out-name Encoder
uv run python export_decoder_joint.py --container checkpoint/model.fermion --out build/components
uv run python stage_shared.py checkpoint/config.json build/components
# Optional smaller GPU file, named separately; stage it as Encoder.mlmodelc in its own model directory.
uv run python fivevalue_encoder.py --container checkpoint/model.fermion \
  --work build --mode lut3 --out-name EncoderGPU
uv run python -m unittest discover -s . -p test_weights.py -v
```

The checkpoint already contains the architecture configuration and all weights.
`components.py` constructs the NeMo component tree directly: no redundant 2.5 GB
NVIDIA teacher download. Loading validates names, shapes, duplicate/missing
records and trailing bytes, and processes one tensor at a time. The three
prediction components use the checkpoint's own weights; reusing the teacher's
decoder/joint would be incorrect. Only the shared preprocessor is reused.

All **240** palettized encoder constants are checked bit-exact against the fp16
source. The other 24 five-value matrices are positional projections folded into
fixed-window constants. Output is a single 15-second encoder, with the same
`Preprocessor / Encoder / Decoder / JointDecisionv3` inputs and outputs as v3.

## Validation results

[Machine-readable report](reports/m5-smoke.json). Public data only: three
LibriSpeech smoke clips plus the first 20 members of each FLEURS test archive for
English, French, German and Spanish: **83 clips / 961 seconds**. This is not a full
corpus or leaderboard reproduction. WER uses one case/punctuation normalization
for both paths. Timings are from a SwiftPM debug probe and are diagnostic, not
isolated release benchmarks.

| Execution path | Encoder size | Cosine vs fp32 encoder | Normalized transcripts identical to fp32 | Numerical gate |
|---|---:|---:|---:|---|
| Sparse + CPU | 321 MB | 0.999985 | 10/12 (initial smoke set) | Pass |
| Sparse + Neural Engine | 321 MB | 0.997446 | 72/83 | **Fail**: NRMSE 0.0714 |
| Dense 3-bit + GPU | 253 MB | 0.999997 | 72/83 | Pass: NRMSE 0.00149 |

Full model directory: approximately **358 MB sparse / 290 MB dense**, not the
156 MB source download. Six malformed-container/packing tests pass; all 725
NeMo tensors load strictly. Two-, fifteen- and thirty-second silence inputs
produce empty output. A 43.74-second recording preserves four repeated utterances.
A warm complete sparse/ANE model reload took 0.76 seconds. First preparation was
much slower (117 seconds for the first full ANE probe; 179 seconds with a CPU
encoder), so an app needs an explicit preparation phase.

### Interpretation and remaining gates

- The conversion and reuse of the existing Swift backend are demonstrated.
- The GPU encoder passes the strict single-window numerical check. Its actual
  transcripts still differ in some cases due to fp16/runtime and window/decode
  differences; it is not byte-identical to full-context NeMo.
- The Neural Engine **fails the retained** cosine >0.999 / NRMSE <0.05 gate.
  Its sample transcript WER is close to fp32, but the threshold was not relaxed.
  More corpus/device evidence is needed before treating ANE as validated for a
  default app model. All models were tested only on this M5/macOS 26.6.2 host.
- The 25-language model was exercised in four languages. The English-only 2.5
  checkpoint, the remaining languages, other chips and first-use experience in a
  signed app have not been validated here.
- No model artifacts have been uploaded and no upstream PR has been opened.
  An eventual model-registration PR also needs a hosted, licensed artifact repo.

## Independent reference

`reference.py CHECKPOINT/model.fermion rows.json reference.json` runs fp32 NeMo
TDT decoding. `rows.json` contains `{id, path, language, ref}` objects. The paired
Muesli investigation branch contains a standalone FluidAudio probe and comparison
driver. `validate_encoder.py MODEL.mlmodelc encoder_ref_phonon25.npz report.json
--units ane|gpu|cpu` preserves the strict numerical gate; failures still save their
measured metrics. Corpus-level checks must follow numerical checks, not replace
failed results silently.
