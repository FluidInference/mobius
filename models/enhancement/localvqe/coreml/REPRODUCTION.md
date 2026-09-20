# Continued LocalVQE reproduction investigation — 2026-09-18

**The complete HuggingFace v1.2 table is still not reproduced.** Further investigation found a demonstrable historical GGML state-update defect, but testing the original release engine on every one of the 300 disputed far-end recordings did not reconcile the published echo/ERLE results. The model card defines ERLE as a plain energy ratio, while its numbers resemble a separate gated reconstruction; the original scorer is not public. It is premature to say that the private scoring script is the only possible cause or that the published table is invalid.

## Complete far-end comparison

All rows use published v1.2 weights. The historical run uses the legacy AECMOS model on the first 20 seconds, DNSMOS on the challenge-rated segment, and the previously declared 512/256 gated ERLE reconstruction. The scorer was not tuned for this run. Ordinary far-end has 107 recordings and movement has 193; all are present, with no duplicate recordings or AECMOS/DNSMOS exclusions. Gated ERLE has 106/191 eligible recordings, respectively.

| Metric | Scenario | HF card | Current Core ML | Current GGML | Release-time GGML |
|---|---|---:|---:|---:|---:|
| Echo MOS | Far-end | 3.78 | 4.0723 | 4.1422 | 4.1234 |
| Echo MOS | Far-end with movement | 4.12 | 4.2735 | 4.3187 | 4.2822 |
| Degradation MOS | Far-end | 4.91 | 4.9289 | 4.9240 | 4.9069 |
| Degradation MOS | Far-end with movement | 4.96 | 4.9636 | 4.9654 | 4.9740 |
| Plain ERLE, dB (card's declared formula) | Far-end | 45.7 | 38.4616 | 38.7275 | 33.8070 |
| Plain ERLE, dB (card's declared formula) | Far-end with movement | 40.6 | 30.3501 | 30.4718 | 27.6529 |
| Gated ERLE, dB | Far-end | 45.7 | 47.6362 | 47.9518 | 42.2001 |
| Gated ERLE, dB | Far-end with movement | 40.6 | 41.2546 | 41.1354 | 36.9386 |
| Rated DNSMOS OVRL | Far-end | 1.80 | 1.8943 | — | 1.8034 |
| Rated DNSMOS OVRL | Far-end with movement | 1.75 | 1.7962 | — | 1.7166 |

Current columns are retained from the existing saved reports; Core ML OVRL comes from the existing challenge run whose rated DNSMOS segment is independent of the AECMOS protocol. Historical values all come directly from one new run of the fixed scorer ([per-recording results](validation/release-v12-farend.json)), including DNSMOS. No values from different historical candidates were combined. A dash means that metric was not scored in the cached current-GGML comparison.

The HF column repeats one reported ERLE value across the plain and gated rows
only to expose the definition ambiguity; it does not claim that both metrics
can equal that value. The historical ordinary far-end OVRL happens to round to
the published 1.80. Its echo and either ERLE definition still differ
substantially, so this does not establish reproduction of that row. Historical
versus current GGML lowers gated ERLE by 5.75/4.20 dB but changes mean echo MOS
by only −0.019/−0.036. The runtime defect therefore cannot by itself account
for the far-end echo gap under the tested protocol.

## Isolated runtime defect

Reconstructed top-level C++ source at the actual v1.2 table release, `1d223631b7a4199a9675578cd0aa6bdecaebdcd3` (May 14). On the fixed 12-recording diagnostic set it produces different audio from current upstream `f53063c9eb2a85f96479867d1dd911dc3bf6319b`, with the same published F32 GGUF.

The May 15 single-pair `memcpy`→`memmove` fix produces exactly the same PCM16 WAV samples as the original engine on the selected four diagnostic recordings. It is not the cause of the observed difference on this Mac.

The later two-phase state-copy fix in `b1fbbd219ae2a08ba99a1af10531522f2adab202` snapshots every state output before writing any input. The old sequential copy could overwrite another output before it was read. Applying **only** that change to the May 14 source makes every PCM16 sample exactly equal to the current GGML output on all four selected recordings: ordinary doubletalk, ordinary far-end, moving far-end, and near-end. Before the change, the doubletalk waveform differed at 329,714 samples; after it, zero. This is causal evidence for the runtime difference on this diagnostic set, not proof of which runtime the author used for the HF table.

Build hygiene: the existing GGML vendor checkout contained local additions for a new GRU operator, unused by v1.2. Rebuilt the exact pinned dependency `c044a8eeae2591faa0950c8b5e514cbc4bbfc4ca` without those additions in a separate ignored directory. Linking the historical engine to that pristine dependency gives identical WAV samples on all four diagnostic recordings. The 300-recording run used the original build with the unused GRU additions; the four-recording pristine control supports, but does not overstate, its fidelity.

## Weights and public PyTorch reference

The downloaded checkpoint hashes already match HuggingFace. This follow-up also compared the actual GGUF tensors against the public PyTorch checkpoint after the documented temperature-folding/export transform. For both v1.2 and v1.3, 135 of 137 stored tensors match exactly. The only differences are the two derived S4D coefficient arrays, at maximum absolute difference 5.96e−8 for v1.2 and 1.19e−7 for v1.3. This finds no evidence of different learned weights between these published formats.

Public PyTorch FP32 with the published v1.2 weights agrees with Core ML on the four fixed diagnostic recordings: maximum absolute difference 0.000069 AECMOS, 0.00055 DNSMOS OVRL, and 0.0085 dB gated ERLE where eligible. Decoder gain is adjusted to the GGML/STFT convention for this parity comparison; this is not a claim that the bare public decoder returns the same amplitude without that adjustment.

The v1.2 checkpoint does not store the activation architecture version or delay-window size. The public source had stale bare-constructor defaults briefly on release day. Testing both historical defaults on the fixed 12 recordings: ReLU6 instead of SiLU breaks doubletalk performance; SiLU with the shorter delay window changes far-end echo only slightly on average (−0.016 ordinary / −0.042 moving). Neither diagnostic establishes the missing full-table reproduction. The shorter-delay result is a small-sample observation, not a full-corpus exclusion.

## Precision checks and remaining limits

The first CPU BF16 diagnostic included the frozen input transform inside autocast. The author's public predecessor evaluator instead computes the input transform before its CUDA BF16 context. A separate follow-up therefore kept the transform and synthesis in FP32 and applied CPU BF16 autocast only to the network, using the same four predetermined real recordings.

| Scenario (one recording each) | BF16 versus FP32/Core ML echo delta | Gated ERLE delta, dB | Rated OVRL delta |
|---|---:|---:|---:|
| doubletalk | -0.01438 | -0.0111 | -0.01187 |
| farend-singletalk | +0.02997 | -1.9503 | -0.00959 |
| farend-singletalk-with-movement | -0.00389 | +0.0458 | +0.01065 |
| nearend-singletalk | -0.00000 | — | +0.00287 |

These four recordings do not reproduce or rule out a full-table mixed-precision effect. No CUDA evaluation was performed, and CPU autocast is not equivalent evidence for CUDA kernels. The v1.3 CPU BF16 control was terminated because it ran too slowly; it is not reported as completed. A separate CPU FP16 trial generated nonfinite output on all four v1.2 recordings; those failures are recorded and were not silently excluded from a scored mean.

Scoring the bare public decoder's half-amplitude output was also retained as a four-recording diagnostic. It adds approximately 6.02 dB to eligible ERLE, in the opposite direction to the residual current-Core-ML versus published v1.2 far-end ERLE discrepancy. It is not a consistent explanation for the table.

The tests establish several distinct contributors (scorer/segment definitions, output conventions, and a historical runtime defect). They do not establish the exact private upstream render/scoring provenance. The honest status remains partial reproduction, with the v1.2 far-end rows open. Evidence does not support alleging manipulated results or asserting that no further local investigation is possible.

## Scope and reproducibility

This historical-runtime and precision follow-up ran new enhancement inference only on explicitly selected real recordings: the fixed 12-clip diagnostic manifest, four-clip precision/causal controls, and one evidence-backed expansion to all 300 far-end recordings with four workers and two GGML threads each. It did not rerun the full 800-clip inference benchmark. No synthetic audio, model substitution, outcome-based selection, training, remote jobs, uploads, pushes, or messages to the author.

The prior audit of the main 800-clip benchmark and the separate exploratory 200-example ASR study remains distinct. “All 300 far-end clips” describes this historical-engine follow-up; it does not relabel this run as a fresh complete 800-clip benchmark.

Published artifacts:

- [Benchmark index](validation/benchmark-index.json) and [800-recording manifest](validation/blind-manifest.txt): 13 saved runs, each covering the same 800 recordings, including the challenge reference, upstream comparisons and all five full-corpus PyTorch configuration diagnostics in the README. Per-recording CSVs retain the measured values; summaries, metric availability and CSV SHA256 hashes are in the index. All aggregates and input stem sets were rechecked before publication. The upstream runs skipped DNSMOS, explicitly recorded in the index and as empty CSV fields; their OVRL must not be described as directly scored in those runs.
- [Evidence verifier](verify-benchmark-evidence.py) and pinned [model-card reference](validation/upstream-model-card.json): standard-library verification of every saved CSV hash, exact 800-stem coverage, scenario membership, DNSMOS availability and aggregate, plus a side-by-side printout of plain and gated ERLE. Run `python verify-benchmark-evidence.py`; it performs no inference.
- [Historical far-end results](validation/release-v12-farend.json): the predetermined 300-stem manifest, source revisions, protocol, per-recording measurements, counts and aggregates. `null` gated ERLE marks a recording with no eligible frame.
- [Weight, runtime and precision diagnostics](validation/diagnostics.json): complete tensor comparisons, four-recording state-copy and pristine-dependency controls, fixed diagnostic stem set, and precision outcomes including failed runs.

The full-corpus `dmax=32`, activation and temperature experiments reported in the [README](README.md#quality-icassp-2022-aec-challenge-blind-test-set) are a separate follow-up. Their closer ERLE/degradation values support a historical-configuration hypothesis; they do not confirm upstream's configuration. No configuration reproduces the entire card, and scores from different configurations must not be combined into a claimed reproduction.

The main render and scoring tools remain `render-blind.sh`, `render-torch.py`, `score_blind.py`, and `compare-renders.py`. The historical C++ experiment used the linked release source and later two-phase state-copy change, with the dependency/build qualification above. These saved diagnostics document the experimental results; they do not bundle recordings, weights, machine-specific build scripts or build products.

Public provenance: [v1.2 table release](https://github.com/localai-org/LocalVQE/commit/1d223631b7a4199a9675578cd0aa6bdecaebdcd3), [later state-copy fix](https://github.com/localai-org/LocalVQE/commit/b1fbbd219ae2a08ba99a1af10531522f2adab202), [published model card](https://huggingface.co/LocalAI-io/LocalVQE#performance).
