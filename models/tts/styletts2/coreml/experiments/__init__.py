"""StyleTTS2 → CoreML experiments — diagnostic and dead-end scripts.

Standalone CLI drivers used to probe specific hypotheses about ANE
placement, op rejection, and graph rewriting. None of these are part of
the production conversion pipeline; production exporters live in
`coreml/exporters/`.

Kept in tree for reproducibility — each entry below points at the
trial entry in `coreml/trials.md` that motivated it.

| Module                              | Trial / outcome                                         |
|-------------------------------------|---------------------------------------------------------|
| `trial10_decoder_upsample_fixed`    | Trial 10 — fp32 fixed-shape ANE probe (dead-end)        |
| `trial10b_decoder_upsample_conv2d`  | Trial 10b — Conv1d→Conv2d rewrite (dead-end)            |
| `trial10d_step1_capture_ane_log`    | Trial 10d Step 1 — captured `Tensor width > 16384` error|
| `trial10e1_t_mel_cap`               | Trial 10e1 — T_mel cap sweep (dead-end; structural)     |
| `trial10e3_bisection`               | Trial 10e3 — op-bisection sweep (Snake = trigger)       |
| `trial10e4_snake_cosine`            | Trial 10e4 — Snake → cos-identity rewrite (no flip)     |

The mlpackages produced by these scripts land in `coreml/packages/`
and are gitignored — they're throwaway diagnostic artifacts, not part
of any HF upload.
"""
