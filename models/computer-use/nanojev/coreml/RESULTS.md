# NanoJev Core ML local validation

| Check | Result |
| --- | --- |
| Trained checkpoint | `C-Tianyu/NanoJev@047b927b30882a1138fc504821b82ac145a4b81a` |
| Checkpoint SHA256 | `f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b` |
| Trained parameters | 596,250,498 across 322 tensors, including set and scalar heads |
| Export | FP16, 128-token, four-candidate bucket, iOS 17/macOS 14 minimum |
| Native vs export-wrapper max logit error | `0.00000167` over Choice, Boolean, Score |
| Core ML answer agreement | 3/3 over Choice, Boolean, Score |
| Core ML max probability error | Choice `0.003851`; Boolean `0.000847`; Score `0.005882` |
| Local packages | Encoder about 1.1 GiB; shared decision head about 420 KiB |
| Export time | 22.7 seconds after loading checkpoint |

The local packages have **not** been uploaded: the upstream model card licenses source code but leaves trained-weight redistribution unclear. The current checkpoint is also newer than the tracker-era one; the tracker’s 26.19 score is not a measured Core ML result.
