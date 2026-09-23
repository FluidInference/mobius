# Kev 0.6B Core ML results

| Check | Result |
| --- | --- |
| Source | `jaredpalmer/kev-0.6b@dece6dba8d43f0f7ded45e9f5b9df12474d90843` |
| Base | `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| Loaded parameters | 596,574,720, including trained head |
| Export | FP16 and W8, L128, 32 option slots, iOS 17/macOS 14 minimum |
| Native vs export-wrapper max logit error | 0.000000954 |
| FP16 max probability error / agreement | 0.002031 / 4 of 4, including Score |
| W8 max probability error / agreement | 0.008554 / 4 of 4, including Score |
| FP16 median model call | 15.35 ms over four fixtures; 13.10 ms over earlier three, CPU+ANE allowed |
| W8 median model call | 12.27 ms over four fixtures; 23.15 ms over earlier three, CPU+ANE allowed |
| Package sizes | FP16 1,194 MB; W8 599 MB |

The tiny latency samples reverse the FP16/W8 speed ranking and exclude tokenization, so they do not establish which variant is faster. No full Decision Index run or 2048 evaluation has been performed for this Core ML package.

The published-style standalone host was also run with network access disabled, using only the published tokenizer, pinned `training_config.json`, and saved Core ML packages. Unlabelled Choice and Noul requests succeeded with FP16; an unlabelled Score request succeeded with W8. The returned typed answers are recorded in `reports/standalone-runtime.json`. A separate real-checkpoint test confirmed this weight-free renderer produces exactly the same input arrays as native `model.encode` on all four local fixtures.
