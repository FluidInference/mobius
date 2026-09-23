# LFM2.5-350M-RLCD ANE precision probes (2026-09-22)

The published L256/B8/V16 FP16 graph on automatic Core ML device selection
passed nine selected upstream task cases. The same graph under forced CPU+ANE
previously produced log-likelihood errors as large as 45,402.8 on the fixed
real-model fixture. No forced-ANE variant is approved for release.

Two mathematically equivalent full-value scoring graphs were converted using
the same pinned weights and exact bucket, with zero PyTorch trace error:

| Candidate | Automatic-device fixture | CPU+ANE fixture | Decision |
| --- | --- | --- | --- |
| `target_logit - logsumexp(logits)`, FP16 | Same selected values; max score error 0.0763 | Different selected values; max error 109.96, including invalid positive log likelihoods | Reject ANE route |
| Same graph, `logsumexp` held FP32 | Not scored on automatic device | Same selected values; max score error 2.678, above 0.5 gate; first model call 220.8 ms | Reject ANE route |

These are one-fixture diagnostic checks, not a broad benchmark. Both candidates
are local only. Keeping `logsumexp` FP32 removes the catastrophic underflow but
does not restore the previously declared full-score parity gate; the single
ANE call is insufficient for a paired speed claim. The
conversion manifests, exact package hashes, outputs, and failure reports are
adjacent. The published FP16 package remains the recommended artifact, using
automatic Core ML device selection.
