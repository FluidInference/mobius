# q8 EOS Bias Diagnosis

After applying the three host-side fixes
([HOST_SIDE_FIXES.md](./HOST_SIDE_FIXES.md)), the INT8 weight-quantized
stateful decoder shipped on HuggingFace
(`FluidInference/cohere-transcribe-03-2026-coreml`, `q8/` subdir) still
over-generates on short utterances: the hypothesis is correct through the
true sentence boundary, then continues with hallucinated extra text.

## Diagnosis

Three candidate causes, from a priori most → least destructive:

  a. EOS logit is quantized so low it never wins (systematic under-estimate).
  b. EOS is near-top but lexical tokens get a random quantization boost.
  c. Model is semantically uncertain at the true boundary and quantization
     tips a close decision.

We instrumented the q8 stateful decoder and dumped, per step, the top-5
tokens plus the EOS (id=3) logit and rank. On EN and FR FLEURS samples
where over-generation occurs:

| Observation | Cause |
|-------------|-------|
| EOS logit is **within the top 5** at the true boundary | Not (a) |
| EOS is **rank 1–2** (not top), with **2–3 logit gap** to the winner | Matches (c) |
| Token at rank 0 is the first lexical token of the spurious continuation | Consistent with (c) |

The EOS logit is *not* systematically under-estimated; it simply sits a
small logit gap below the top token at the true boundary, and INT8 weight
quantization noise is enough to keep it there when fp16 weights would
have made it win.

## Fix: EOS Logit Bias

Adding a small constant bias to the EOS logit during decoding closes the
gap. We swept +0.0 / +2.0 / +4.0 on 3 FLEURS samples each of EN / FR / ES
/ Mandarin, using the fixed q8 pipeline and no other changes:

```python
# In the decoder loop, after getting logits and before argmax:
if eos_bias != 0.0 and step >= len(prompt) - 1:
    logits[EOS] += eos_bias
```

Findings:

| Bias | Non-CJK WER trend | Mandarin CER | Risk |
|------|-------------------|--------------|------|
| +0.0 | Over-generation, high WER from trailing hallucination | ~same as +2 / +4 | baseline |
| +2.0 | Recovers many close cases | ~same | safe |
| +4.0 | Recovers most cases, close to f16 quality | ~same | checked: no premature EOS observed on these slices |

Mandarin CER was stable across bias values because CJK samples in the
slice did not hit the over-generation pattern.

**Recommendation**: `logit_bias_eos = +4.0` when shipping q8. This is a
pure inference-side constant; no re-quantization or retraining required.

## Reproduction

The diagnostic (dumps per-step top-5 logits, EOS rank, EOS gap, running
hypothesis) and the bias sweep are not kept as scripts in this PR because
they are one-shot experiments. The canonical q8 FLEURS benchmark — which
is runnable and reproducible — is:

```bash
uv run python tests/bench-q8-fleurs.py --language en_us --n 3
```

The host-side code that applies `logit_bias_eos` lives in:

- `q8/quickstart.py` (Python reference)
- `CohereFixedPipeline` in FluidAudio (Swift shipping path)

## Relationship to [HOST_SIDE_FIXES.md](./HOST_SIDE_FIXES.md)

These are **independent** problems and **independent** fixes:

| | Cause | Layer | Required for |
|-|-------|-------|--------------|
| Host-side fixes | inference code bugs | preprocessing / decoder call / detokenization | f16 **and** q8 |
| EOS bias | INT8 weight quantization noise | decoding | q8 only |

On f16, only the host-side fixes are needed. On q8, both are needed for
short-utterance quality parity with f16.
