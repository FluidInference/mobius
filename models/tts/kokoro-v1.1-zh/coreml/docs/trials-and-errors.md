# Trials and errors

What didn't work, and why. Future agents: read this before fighting the same
ANE scheduler / coremltools issue.

## coremltools 9.0 sdist fallback (BlobWriter not loaded)

**Symptom**: `scripts/convert-coreml.py` fails midway through stage 1 with
`RuntimeError: BlobWriter not loaded`.

**Cause**: `uv sync` resolved coremltools to the pure-python sdist
(`Tag: py3-none-any`) instead of the platform wheel
(`coremltools-9.0-cp311-none-macosx_11_0_arm64.whl`). The sdist is missing
`libcoremlpython.so` and `libmilstoragepython.so`, which `convert.convert(...)`
needs to write the MLProgram blob format.

**Fix**: After `uv sync`, force the platform wheel:
```bash
uv pip install --reinstall coremltools==9.0
```

Verify:
```bash
ls .venv/lib/python3.11/site-packages/coremltools/libcoremlpython.so
# (file must exist and be ELF/Mach-O, not absent)
```

Don't pin a different version — coremltools 9.0 is the only release that has
both the iOS17 mlprogram features used by convert.py AND working int8
palettization. Older `coremltools<8` lacks compute_units / palettize APIs;
newer 9.x point releases reintroduce sdist regression intermittently.

## ANE rejection of fp32 sin/exp/iSTFT (Noise, Tail must stay fp32)

Initial attempt converted Noise + Tail with `compute_precision=FLOAT16`. ANE
silently fell back to CPU mid-graph because:

* `sin(x)` for large `x` (the F0 curve hits ~ k·π for many k) saturates fp16.
* `exp(x)` overflows fp16 above ~11.

`coremltools` does not warn — it just produces a model that silently runs
slower than expected. The fix was `compute_precision=FLOAT32` for Noise and
Tail, and accepting they run on CPU/GPU rather than ANE.

## ANE scheduler thrashing on `compute_units=ALL`

For Albert/PostAlbert/Alignment/Vocoder, `compute_units=ALL` was 1.4–2.1×
slower than `CPU_AND_NE` because the scheduler kept migrating small ops to
GPU and paying the dispatch tax. For Prosody/Noise/Tail, the opposite was
true — `ALL` was faster because parts of those graphs simply can't run on
ANE.

The per-stage assignment in `scripts/convert-coreml.py` and the Swift `KokoroLaiModelStore`
is the empirical optimum found by laishere; do not "simplify" by setting all
stages to one value.

## Vocoder dual-output trick (single-output vocoder fails parity)

Tried collapsing Vocoder + Tail into one fp16 model. The output had visible
buzzing and `mel_corr` dropped from 0.99 to 0.83. Root cause: the iSTFT's
`ifft` requires fp32 numerical range that ANE's fp16 doesn't preserve through
the `exp/sin/iSTFT` chain.

Splitting the vocoder so the body runs fp16 on ANE up to `conv_post` and the
final iSTFT runs fp32 on CPU/GPU recovered parity (corr=0.81, mel=0.99).
Keep the dual output (`anchor` + `x_pre`) — `anchor` is unused at inference
but its presence prevents a coremltools trace bug where single-output traces
of the body lose static-shape inference.

## RangeDim cap (don't try to support arbitrary length)

Set `--max-frames 2000` (T_a). Higher values blow up the alignment matrix
construction in `KokoroAlignment` past a 4 GB CoreML tensor limit. Lower
values (< 1000) reject any sentence past ~6 words.

The 2000 frame cap covers up to ~24-second utterances which is well past
typical TTS chunking practice.

## Don't trace the prosody encoder with weight_norm intact

`torch.jit.trace` on weight_norm'd LSTMs produces a graph that silently uses
the unnormalized weights, breaking parity. Run `_remove_weight_norm(model.x)`
before tracing — see `scripts/convert-coreml.py:485`.

## Don't trace with dropout layers active

Same issue — dropout layers in train mode get baked into the traced graph as
random masks. Replace with `nn.Identity()` after `model.eval()`. See the
loop at `scripts/convert-coreml.py:595`.

## CpuOnly tracing (mobius rule)

mobius's `Documentation/ModelConversion.md` requires tracing with
`device='cpu'` to avoid `torch.jit.trace` baking in MPS-specific ops. Already
honored — no `.to('mps')` calls in `scripts/convert-coreml.py`. The
`PYTORCH_ENABLE_MPS_FALLBACK=1` env var is a belt-and-suspenders: if any
upstream Kokoro op tries to use MPS at trace time, it falls back to CPU
instead of failing.
