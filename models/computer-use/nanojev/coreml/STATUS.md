# NanoJev status

- Pinned current checkpoint and audited native source; full checkpoint SHA256 matches upstream and contains 596,250,498 trained parameters across 322 tensors. Historical tracker revision unresolved.
- Pinned full checkpoint downloaded. Native Qwen3 encoder plus decision-head exporter matched the trained model on Choice, Boolean, and Score requests: same argmax for all three, maximum logit error `1.67e-6`.
- FP16 Core ML L128/K4 encoder and full decision-head packages exported in 22.7 seconds. Encoder package is about 1.1 GiB; head package is about 420 KiB.
- Core ML runtime parity passed for Choice, Boolean, and Score: 3/3 chosen answers matched trained PyTorch. Maximum probability errors were `0.003851`, `0.000847`, and `0.005882`, respectively.
- No full Decision Index or 2048 evaluation has been run for this Core ML package.
- Trained-weight redistribution license unresolved. Public HF model artifact is not yet authorized by the source metadata.
