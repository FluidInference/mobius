# Verdict on Core ML

This toolkit converts the released [`heman10x/rlcd-modernbert-151m`](https://huggingface.co/heman10x/rlcd-modernbert-151m) checkpoint (Apache-2.0) at pinned revision `8af2496eb63c7fa66d7d234e1f62629380030eb4`. It is a **151,378,177 parameter Verdict model**, separate from GLiClass Edge Apps v2. The Core ML graph contains the trained ModernBERT encoder and GLiClass decision head. The host must use the released tokenizer, append Verdict's trained `__insufficient_evidence__` option, and apply the released `calibrator.json` temperature for the complete candidate count. The graph's `probabilities` output is uncalibrated; use its `logits` output with `native_reference.decode` for native behavior.

The source package and its SHA-256 locks are in `assets.lock.json`. `native_reference.py` and `decision_index_engine.py` preserve the audited native formatting and abstention behavior. The upstream code revision audited is [`30f1556`](https://github.com/Heman10x-NGU/Verdict-open-jev/tree/30f15564821626ca5c1ad5b2638c4eb7078787dd). The historical checkpoint and renderer behind the Decision Index score of 13.38 have not been authenticated; running this adapter is a **new reproduction**.

## Reproduce

On Apple Silicon, from this directory:

```bash
uv sync --frozen
uv run python assets.py
uv run python convert-coreml.py --length 128
uv run python verify.py --length 128
uv run python convert-coreml.py --length 512
uv run python verify.py --length 512
uv run pytest -q
```

The source supports up to 512 tokens and 24 substantive candidates plus abstention. The installed L128 and L512 buckets cover that range: 129–511 token requests use L512. If a request exceeds 512 tokens or its candidate markers do not fit, `decision_index_engine.py` raises `Unsupported` without truncating it. The benchmark adapter accepts the complete state, option keys and descriptions; its mapping is frozen in source and has not been tuned to gold labels.

## Measured result

Apple M5 Pro, macOS 27.0, 2026-09-22; FP16, 25 candidate slots, iOS 17/macOS 14 deployment target. L128 is 303,210,832 bytes; L512 is 304,390,482 bytes. Native PyTorch versus export wrapper max raw-logit error is 1.9e-6 for both. L128 passed **4/4** real-model choice, noul and score cases, including two native abstentions, with worst calibrated probability error **0.00341**. L512 passed **5/5**, including a 283-token public Decision Index input and the abstention cases, with worst error **0.00070**. These are protocol/parity smoke tests, not the Decision Index suite. Detailed reports: [`L128`](reports/verification-L128.json), [`L512`](reports/verification-L512.json).

`coreml-cli` reported median model call latency of 3.627 ms on CPU+GPU, 3.710 ms on CPU+ANE, 3.915 ms with automatic units, and 11.721 ms on CPU only (20 iterations each). The profiler's automatic placement estimate was 68.27% ANE / 31.73% CPU. This differs from request latency, which also includes tokenization, rendering, calibration and one call per question. The full report is [`reports/profile-L128.json`](reports/profile-L128.json).

## Current limits

Both L128 and L512 are validated, covering the released 512-token context limit. L512 median Core ML call time was 8.15 ms across five requests; 129–511 tokens use this bucket. The upstream native `DecisionEngine` batches multiple questions in one PyTorch call; this adapter processes them one at a time, preserving individual decisions but changing throughput. Verdict's explicit abstention is surfaced as `NativeAbstention` in the Decision Index adapter. An abstention is not silently turned into a forced choice. There is no historical leaderboard score claim for these Core ML artifacts.

An L128 8-bit k-means per-tensor weight LUT reduced package size from 303,210,832 to 151,878,624 bytes. It passed the four-case native smoke check (worst calibrated probability error 0.00774) but failed the predeclared broader comparison against FP16: **96/100** selection and abstention agreement, below the 99/100 gate, with four abstention flips. The fixed sample selected up to ten L128 requests per family from eleven Decision Index families, without using gold labels. P95 calibrated probability error was 0.0197 and worst was 0.0297. Median measured model call time on those requests was 3.48 ms LUT8 versus 3.34 ms FP16. We do **not** distribute LUT8 as a validated artifact or claim a speed improvement. The full reports are [`suite comparison`](reports/lut8-L128-suite-parity.json) and [`native smoke`](reports/verification-verdict_lut8_kmeans_per_tensor_L128_candidates25.json).

License and attribution: [`Verdict model card`](https://huggingface.co/heman10x/rlcd-modernbert-151m), [`Verdict source`](https://github.com/Heman10x-NGU/Verdict-open-jev), and its `knowledgator/gliclass-modern-base-v2.0` base architecture.
