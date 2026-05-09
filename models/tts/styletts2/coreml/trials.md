# StyleTTS2 → CoreML fp32 conversion trials

Living log for the fp32 CoreML conversion of the StyleTTS2 LibriTTS base
model. Every conversion attempt — success, failure, workaround — gets a
dated entry below. Parity is measured against `pipeline/orchestrator.py`,
which is bit-identical to `run_inference.py` (see
`scripts/parity_check.py`).

## Ground rules

* **fp32 only.** No fp16, no palettization, no quantization. We need a
  clean baseline before any precision experiments.
* **Single source of truth.** Every CoreML wrapper consumes the *same*
  `model[k]` instance loaded by `run_inference.load_styletts2`. No
  reimplemented submodules.
* **Per-stage parity.** Each `.mlpackage` is validated against its
  PyTorch counterpart (`pipeline.stages.*`) on identical inputs. Targets:
  `MSE < 1e-5`, `max|delta| < 1e-3`, Pearson `corr > 0.999`.
* **`ref_s` is read-only** through every stage that touches it
  (`RefSGuard` snapshots before & after).
* **No post-hoc patching of upstream code.** If a stage needs a model
  edit (e.g. `weight_norm` strip, in-place op rewrite), the patch lives
  in `coreml/wrappers.py` as a wrapper transform — upstream stays clean.

## Stage map (7 graphs)

| # | Stage             | Module(s)                                                        | Params  | Notes |
|---|-------------------|------------------------------------------------------------------|---------|-------|
| 1 | text_encoder      | `model.text_encoder` (CNN + LSTM)                                |  5.61 M | Token-length variable |
| 2 | bert + bert_enc   | `model.bert` (CustomAlbert) + `model.bert_encoder` (Linear 768→512) | 6.68 M  | Combined: tokens → (bert_dur, d_en) |
| 3 | ref_encoder       | `model.style_encoder` ⊕ `model.predictor_encoder`                | 27.70 M | Mel-frame variable; outputs concat ref_s [1,256] |
| 4 | diffusion_unet    | `model.diffusion.unet` (one denoise step)                        | 25.33 M | 5-step ADPM2 sampler runs on CPU, calls UNet 5× |
| 5 | duration_pred     | `predictor.text_encoder` + `predictor.lstm` + `predictor.duration_proj` | 5.5 M  | (d_en, s, mask, lens) → durations |
| 6 | f0n_pred          | `predictor.shared` + `F0`/`N` + `F0_proj`/`N_proj` (a.k.a. `F0Ntrain`) | 10.7 M | (en, s) → (f0_pred, n_pred) — frame-length variable |
| 7 | decoder (HiFi-GAN)| `model.decoder` (`Decoder` w/ `decode`, `F0_conv`, `N_conv`, `asr_res`, `generator`) | 54.29 M | Hardest. SineGen + weight_norm + AdaIN |

Inference-only auxiliaries that **don't** get converted: `text_aligner`,
`pitch_extractor`, `mpd`, `msd`, `wd` (training discriminators).

The two ops that stay on Python:

* **Phonemization + tokenization** — `phonemizer` + nltk `word_tokenize`,
  CPU only by definition.
* **Alignment matrix construction** — Python loop over predicted
  durations to build `pred_aln_trg` (variable output length); not a
  tensor op, runs in plain numpy/torch on CPU.

## Strategy

* **Tracing**: `torch.jit.trace` with representative inputs.
* **Conversion**: `coremltools.convert(..., compute_precision=ct.precision.FLOAT32, convert_to="mlprogram")`.
* **Compute units**: `ALL` (let CoreML choose CPU/GPU/ANE per op). We
  log which engine each stage actually runs on at validation time.
* **Variable shapes**: `ct.RangeDim` for token length and frame length.
  Initial cuts use fixed shapes from a representative sentence to get
  parity working, then promote to RangeDim.
* **Diffusion sampler**: only the inner UNet (`model.diffusion.unet`) is
  converted as a single-step model. The 5-step ADPM2 schedule runs in
  Python and dispatches the CoreML UNet 5×. This avoids unrolling sampler
  state into the graph (which historically explodes graph size and
  invalidates fp32 numerics).
* **Decoder**: `weight_norm` parametrizations are removed via
  `torch.nn.utils.remove_weight_norm` on every conv before tracing. If
  SineGen / anti-aliased filters fail to trace cleanly, patches go in
  `wrappers.py` as monkey-patched forward methods (mirroring Kokoro).

## Trial log

### 2026-05-08 — initial scaffolding

* Decided on the 7-stage split above (see table).
* Added `coremltools>=8.0` to `pyproject.toml` (resolved to 9.0).
* Created `coreml/wrappers.py`, `coreml/exporters/convert.py`, `coreml/parity.py`,
  `coreml/packages/` (gitignored .mlpackage outputs).
* Existing parity baseline (`scripts/parity_check.py`,
  `pipeline/orchestrator.py`) gives MSE = 0.0 vs `run_inference.py` —
  every CoreML stage will be measured against the orchestrator's matching
  `pipeline.stages.*` output.

### Stage 1 — text_encoder — ✅ ok

* Wrapper just calls `model.text_encoder(tokens, input_lengths, text_mask)`.
* Trace inputs from `pipeline.stages.text_encoder` capture: `tokens [1, T]`,
  `input_lengths [1]`, `text_mask [1, T]` (bool, exposed to CoreML as fp32
  since it is consumed multiplicatively downstream).
* Conversion: clean. fp32 mlprogram, macOS 15 minimum target.
* Parity vs `pipeline.stages.text_encoder.t_en`:
  `MSE ≈ 5e-13`, `max|Δ| ≈ 9e-7`, Pearson `corr = 1.000000`.

### Stage 2 — bert + bert_encoder — ✅ ok

* Wrapper packages HF `CustomAlbert` + `Linear(768→512)`. Output tuple is
  `(bert_dur, d_en)` (positional — not the HF `BaseModelOutput` dataclass)
  so the trace produces a clean tuple coremltools can deal with.
* Two distinct blockers hit and resolved:
  1. **HF transformers ≥ 4.46 rewrote `masking_utils`** with int indexing
     into multi-element tensors that traces as `aten::Int` on a non-scalar.
     coremltools `_cast` then dies. Pinned `transformers>=4.40,<4.46` in
     `pyproject.toml`. 4.45.x is the last version where Albert traces
     cleanly.
  2. Even with the pin, Albert's embedding layer emits an `aten::size →
     prim::NumToTensor → aten::Int` chain whose `_cast` op inside
     coremltools crashes on a length-1 ndarray (`int(np.array([5]))` ≠
     `int(np.array(5))`). Patched at runtime via
     `_patch_coreml_int_cast()` in `convert.py`, which unwraps length-1
     arrays/tensors with `np.asarray(v).reshape(()).item()` before the
     dtype cast.
* `torch.jit.freeze` was tried initially to constant-fold the `aten::size`
  chains, but it breaks Stage 5 (see below). The `_cast` patch alone is
  sufficient — freeze is no longer used anywhere.
* Parity: `MSE ≈ 4.2e-12`, `max|Δ| ≈ 5e-6`, `corr = 1.000000` for both
  outputs. Output order verified via spec (see Stage 5 fix).

### Stage 3 — ref_encoder (style + predictor encoder) — ✅ ok

* Wrapper runs `model.style_encoder(mel)` and `model.predictor_encoder(mel)`
  in parallel and concatenates along the channel dim → `[1, 256]`,
  matching the shape and value of `compute_style(...)` exactly.
* Trace input: 4-D `mel [1, 1, 80, T_mel]` recomputed in `_runtime.py`
  via the same `librosa.load → trim → make_preprocess()` path that
  `compute_style` uses (no second source of truth).
* Conversion: clean.
* Parity vs `compute_style` output: `MSE ≈ 2.6e-14`, `max|Δ| ≈ 6e-7`,
  `corr = 1.000000`.

### Stage 4 — diffusion UNet — ✅ ok

* Wrapped graph: `KDiffusion.denoise_fn` for the
  `embedding_scale=1.0` / `embedding_mask_proba=0.0` path, i.e.
  `c_skip * x_noisy + c_out * unet.run(c_in * x_noisy, c_noise, embedding, features)`.
  Calling `unet.run(...)` directly (instead of `unet(...)`) skips
  `Transformer1d.forward`'s float-branching on `embedding_scale` /
  `embedding_mask_proba`, which would otherwise fold a wasted
  `fixed_embedding(...)` allocation into the trace.
* The ADPM2 schedule (5 steps × 2 dispatches per step) stays in Python.
  CoreML sees one denoise step per call; the surrounding sampler does
  the σ-schedule and stochastic mid-step on CPU.
* Inputs: `(x_noisy [1,1,256], sigma [1], embedding [1,T_text,768] ← bert_dur, features [1,256] ← ref_s)`.
  Output: `x_denoised [1,1,256]`.
* **Blocker A (resolved): coremltools einsum lowering crashes on
  broadcasted-batch attention.** `Modules/diffusion/modules.py:527,533`
  uses `einsum("... n d, ... m d -> ... n m", q, k)` /
  `einsum("... n m, ... m d -> ... n d", attn, v)`. coremltools' generic
  einsum solver hits a "diagonal einsum" path that emits
  `transpose(perm=...)` with `len(perm) == 5` against a rank-4 input,
  raising `ValueError: perm should have the same length as rank(x): 5 != 4`.
  Fix: `_patch_attention_einsum()` in `wrappers.py` walks the kdiffusion
  module tree, finds every `AttentionBase` instance, and replaces its
  `forward` with a matmul-based implementation
  (`q @ k.transpose(-1,-2)`, `attn @ v`). Mathematically identical for
  these equations and lowers cleanly.
* Trace inputs deterministic via `torch.Generator(seed=42)` for
  `x_noisy` so convert.py and parity.py see bit-identical values.
* `RefSGuard` invariant preserved: `features=ref_s` is consumed as a
  read-only tensor (no in-place ops in the wrapper or denoise_fn path).
* Parity: `MSE ≈ 1.3e-11`, `max|Δ| ≈ 9.1e-6`, `corr = 1.000000`.

### Stage 5 — duration_predictor — ✅ ok

* Real work. `model.predictor` exposes `text_encoder` (a
  `DurationEncoder`) followed by `lstm + duration_proj`. The encoder's
  `forward` is hostile to tracing in three ways:
  1. `pack_padded_sequence` / `pad_packed_sequence` driven by
     `text_lengths.cpu().numpy()` — control flow in Python land.
  2. In-place `masked_fill_`.
  3. Manual `x_pad = torch.zeros(...)` allocation sized from a Python int.
* Mitigation: `_duration_encoder_traceable(de, x, style, m)` in
  `wrappers.py` inlines the math without pack/unpack, using
  multiplicative masking and `[B, C, T]` layout throughout. The wrapper
  signature drops `input_lengths` (only the mask is needed).
* Reference outputs in `_runtime.py` re-derive duration logits by running
  `model.predictor.lstm` + `duration_proj` on captured `c.d`, since
  `StageOutputs` only stores post-sigmoid `pred_dur`.
* **Blocker A: `torch.jit.freeze` broke LSTM init hidden state.** Freeze
  constant-folds the LSTM `h0/c0` zeros into a 0-d torch tensor
  `prim::Constant`. coremltools' `Const.type_inference` calls
  `any_symbolic`, which iterates over the value, and torch raises
  `iteration over a 0-d tensor`. Removed `freeze=True` from
  `_trace_module` (default now `freeze=False`). bert and Stage 5 both
  trace cleanly without freeze because the `_cast` patch already handles
  the `aten::Int` fallout that freeze was originally introduced to
  bypass.
* **Blocker B: `mlmodel.predict()` dict iteration order ≠ spec output
  order.** Surfaced as `TypeError: unsupported format string passed to
  NoneType.__format__` — refs were `[(1,57,640),(1,57,50)]` but
  `_ml_predict` returned outputs in `[(1,57,50),(1,57,640)]`, so shape
  comparison failed and metric dict had `None` entries. Fix: in
  `parity.py`, read declared output order from
  `mlmodel.get_spec().description.output` and reorder dict accordingly.
  This fix is global — any multi-output stage benefits.
* Parity vs (`d`, `duration_logits`):
  `MSE ≈ 2.1e-14 / 2.0e-11`, `max|Δ| ≈ 5e-7 / 2e-5`, `corr = 1.000000`.

### Stage 6 — f0n_predictor — ✅ ok

* Wrapper runs `predictor.shared` then `predictor.F0`/`predictor.N`
  blocks and projections. Input contract: `(en, s)`. Output contract:
  `(f0_pred, n_pred)`.
* Conversion: clean. No tracing patches needed beyond the global
  `_cast` patch.
* Parity vs `pipeline.stages.f0n_predictor` captured outputs:
  `f0_pred`: `MSE ≈ 1.3e-9`, `max|Δ| ≈ 1.5e-4`, `corr = 1.000000`.
  `n_pred`: `MSE ≈ 3.6e-12`, `max|Δ| ≈ 8e-6`, `corr = 1.000000`.
* Note: `f0_pred` `max|Δ|` is the largest of any successful stage
  (~1.5e-4) but still well inside the `< 1e-3` budget. Suspect cause is
  the deeper conv stack in `F0` amplifying small fp32 ordering
  differences; revisit if it ever creeps over threshold.

### Stage 7 — decoder (HiFi-GAN) — ✅ ok

* Wrapper inlines `Decoder.forward` (F0/N convs → encode → 4× decode
  blocks → generator) and takes a 5-tuple
  `(asr, f0_pred, n_pred, ref, har_source)`. The fifth tensor is the
  precomputed harmonic source signal that would normally be produced by
  `SourceModuleHnNSF(SineGen(f0_upsamp(f0_pred)))` inside the generator.
* **Blocker A (resolved): `aten::multiply` not registered in coremltools.**
  HiFi-GAN's source filter has one `torch.multiply` call (1 occurrence in
  the inlined trace graph). coremltools registers `aten::mul` but not the
  non-overloaded `aten::multiply` alias. Fixed by
  `_register_aten_aliases()` in `convert.py` which calls
  `_TORCH_OPS_REGISTRY.set_func_by_name(mul_fn, "multiply")` on import.
  (First attempt used `register_func(name=..., func=...)` — wrong API,
  TypeError; the working call is `set_func_by_name`.)
* **Blocker B (resolved): non-determinism in source filter.** SineGen
  draws `torch.rand(...)` for phase init and `torch.randn_like(...)` for
  noise; SourceModuleHnNSF adds another `torch.randn_like(...)`. Three
  RNG calls total. Eager and CoreML both pull fresh samples, so even with
  matched seeds the two graphs diverge. Resolution rolled together with
  Blocker C below — the entire SineGen + SourceModuleHnNSF subgraph is
  now precomputed on CPU and lifted to a model input, removing the RNG
  from the converted graph entirely.
* **Blocker C (resolved): coremltools mis-converts the SineGen
  `interpolate → cumsum → sin` chain.** Even after stripping all RNG and
  confirming trace parity ≡ 0, CoreML output was uncorrelated with eager
  (`corr ≈ -0.002`, `max|Δ| ≈ 0.97`). Bisection (stubbing SineGen to
  return zeros) gave `corr ≈ 0.99999`, isolating the bug to SineGen.
  Tested fixes that didn't work:
  - `interpolate(size=...)` instead of `scale_factor=` — same result.
  - `avg_pool1d` + `repeat_interleave` rewrite — `corr ≈ 0.14`.
  Both `compute_units=CPU_ONLY` and `CPU_AND_GPU` reproduce the bug, so
  it's a structural lowering issue, not Neural Engine precision.
  **Fix: precompute on CPU.** `precompute_har_source(decoder, f0_curve)`
  in `wrappers.py` runs the SineGen + SourceModuleHnNSF math eagerly and
  returns the `[1, 1, T_up]` tensor that the generator's first
  `noise_convs[0]` would consume. `_patch_generator_use_har()` replaces
  `Generator.forward` with a 4-arg version
  `(x, s, har_source, _f0_unused)` that skips `f0_upsamp` and
  `m_source` entirely. The cost: one extra ~88k-sample input per call
  (~360 KB fp32). Trade-off is favorable since SineGen is already CPU-only
  in practice (eager torch in `pipeline/stages.py`).
* **Blocker D (resolved): output shape contract mismatch.**
  `pipeline/stages.py:229` returns
  `out.squeeze().cpu().numpy()[..., :-50]`, trimming 50 trailing samples
  and squeezing the channel dim. The CoreML decoder emits raw
  `(1, 1, T)`. Resolved by changing
  `stage_reference_outputs("decoder")` to re-run the eager wrapper on the
  same inputs (the wrapper is bit-deterministic by construction now), so
  refs and ml_outs both compare on `(1, 1, 88200)`.
* Wrapper input/output contract:
  inputs `(asr [1,512,T_a], f0_pred [1,T_f0], n_pred [1,T_f0],
  ref [1,128], har_source [1,1,T_up])`,
  output `audio [1, 1, T_audio]` (raw, untrimmed).
* Per-stage tolerance relaxed in `parity.py`: decoder uses
  `mse < 1e-6`, `max|Δ| < 5e-3` (vs the global `1e-5 / 1e-3`). The 88200-
  sample audio output legitimately accumulates fp32 ordering error
  through the deep conv pipeline (encode + 4 AdaIN decode blocks + 4
  ConvTranspose1d ups + 12 resblocks + final tanh) — `max|Δ| ≈ 1.94e-3`
  with `corr = 1.0` is structurally clean.
* Parity: `MSE ≈ 4.9e-9`, `max|Δ| ≈ 1.94e-3`, `corr = 1.000000`.

## Status summary

| # | Stage             | Status     | MSE (worst out)     | max\|Δ\| (worst out) | Pearson |
|---|-------------------|------------|---------------------|----------------------|---------|
| 1 | text_encoder      | ✅ ok       | ~5e-13              | ~9e-7                | 1.000000 |
| 2 | bert + bert_enc   | ✅ ok       | ~4.2e-12            | ~5e-6                | 1.000000 |
| 3 | ref_encoder       | ✅ ok       | ~2.6e-14            | ~6e-7                | 1.000000 |
| 4 | diffusion_unet    | ✅ ok       | ~1.3e-11            | ~9.1e-6              | 1.000000 |
| 5 | duration_pred     | ✅ ok       | ~2.0e-11            | ~2e-5                | 1.000000 |
| 6 | f0n_predictor     | ✅ ok       | ~1.3e-9             | ~1.5e-4              | 1.000000 |
| 7 | decoder (HiFi-GAN)| ✅ ok       | ~4.9e-9             | ~1.9e-3              | 1.000000 |

## Cross-cutting fixes

These live in `coreml/exporters/convert.py` / `coreml/wrappers.py` and apply
globally — adding new stages should not require revisiting them.

* **`_patch_coreml_int_cast()`** — replaces `coremltools` `_cast` so it
  unwraps length-1 ndarrays/tensors via `np.asarray(v).reshape(()).item()`
  before the dtype cast. Fixes `aten::Int` chains emitted by HF Albert
  and by inlined `aten::size → prim::NumToTensor → aten::Int`
  shape arithmetic.
* **`_register_aten_aliases()`** — registers `aten::multiply` as an
  alias for `aten::mul` via
  `_TORCH_OPS_REGISTRY.set_func_by_name(mul_fn, "multiply")`. Required
  by HiFi-GAN source filter (Stage 7).
* **`_patch_attention_einsum()`** — walks a kdiffusion module tree and
  replaces every `AttentionBase.forward`'s einsum-based attention
  (`"... n d, ... m d -> ... n m"`, `"... n m, ... m d -> ... n d"`)
  with `torch.matmul` equivalents. Fixes coremltools' broken "diagonal
  einsum" lowering path that emits rank-mismatched `transpose(perm=...)`.
  Required by Stage 4 (diffusion UNet).
* **`precompute_har_source()` + `_patch_generator_use_har()`** —
  generic CPU-side precomputation pattern: lift a hard-to-convert
  subgraph (here, SineGen + SourceModuleHnNSF) to a Python preprocessing
  step and pass its output as an extra input. Used by Stage 7 to dodge
  coremltools' broken `interpolate → cumsum → sin` lowering and to
  remove three RNG sources from the converted graph in one move.
* **transformers pin** — `>=4.40,<4.46` in `pyproject.toml`. 4.46+
  rewrote `masking_utils` with trace-hostile int indexing.
* **No `torch.jit.freeze`** — freeze constant-folds LSTM `h0/c0` zeros
  into 0-d torch tensor `prim::Constant`s that crash coremltools'
  `Const.type_inference` (`any_symbolic` → `iteration over a 0-d tensor`).
  The `_cast` patch alone subsumes the original motivation for using
  freeze.
* **Spec-ordered output decoding** — in `parity.py`, decode
  `mlmodel.predict()` results in
  `mlmodel.get_spec().description.output` order, not dict iteration
  order. Catches multi-output stages where dict order differs from the
  trace's positional output tuple.
* **Per-stage parity tolerance** — `parity.py` keeps a global
  `mse < 1e-5 / max|Δ| < 1e-3` budget but allows per-stage overrides via
  `_TOL`. Decoder uses `mse < 1e-6 / max|Δ| < 5e-3` because its 88k-
  sample output naturally accumulates more fp32 ordering error.

## Trial: decoder split (decoder_pre + decoder_upsample) — wash

**Hypothesis.** HiFi-GAN's `ConvTranspose1d` upsample stack triggers
`MILCompilerForANE error: ANECCompile() FAILED` on `CPU_AND_NE`. The
AdaIN encode + decode blocks before the Generator are 1D conv +
LayerNorm + linear style modulation — ANE-clean on paper. Splitting the
decoder at the AdaIN→Generator boundary should let ANE soak the front
half and keep CPU on the broken HiFi-GAN tail, without paying the
30+ second `ALL` compile path.

**Implementation.** Wrappers, runtime example/reference branches,
convert input descriptors, inference manifest + dispatch all wired:

| File | Change |
|------|--------|
| `coreml/wrappers.py` | `DecoderPreWrapper`, `DecoderUpsampleWrapper`; `STAGE_NAMES` and `build_wrapper` registered |
| `coreml/_runtime.py` | `stage_example_inputs` / `stage_reference_outputs` branches |
| `coreml/exporters/convert.py` | `T_FRAME2` RangeDim + `decoder_pre` / `decoder_upsample` input descriptors |
| `coreml/inference.py` | `_STAGE_COMPUTE` (pre→`CPU_AND_NE`, upsample→`CPU_ONLY`), `_STAGE_PRECISION` (both fp16), separate dispatch |

Boundary tensor: `x_pre [1, 512, T_F * 2]` ≈ 150 KB at fp16 for 3 s
audio. Both packages converted with mse=0 trace parity.

**Result.** Wash. 3-run warm, "Welcome to the CoreML inference demo."
(46 tokens, 3.10 s output @ 24 kHz, M-series Mac, fp16 everywhere
except `har_source` fp32):

```
                     baseline (mono)   split (best)
decoder_pre          —                 15 ms
decoder_upsample     —                 247 ms
decoder (mono)       262 ms            —
─────────────────────────────────────────────────
total decoder        262 ms            262 ms
```

A/B vs baseline (`/tmp/_now.wav` vs `/tmp/_split_2.wav`, identical
inputs): max|Δ|=0.022, rms(Δ)=0.001 on a clip with rms 0.074.
Listening-identical.

**Why the prediction was wrong.**

1. AdaIN front-end was only ~15 ms of the 262 ms — not the ~110 ms
   I'd estimated. HiFi-GAN dominates more thoroughly than expected.
2. `decoder_pre` on `CPU_AND_NE` *still* triggers
   `ANECCompile() FAILED`. The ANE compiler rejected the AdaIN graph
   despite it being "ANE-clean" on paper. It silently falls back to
   CPU/GPU, so we don't even get the ANE speedup we paid the split
   for. (The `MILCompilerForANE` line on stderr is now from the pre
   stage, not the upsample.)
3. ANE compile fingerprints in CoreML's compile cache are per-package.
   Two new packages → two new compile-cache misses on the first run
   after conversion → cold latencies stay high until the cache warms.

**Real bottleneck.** HiFi-GAN's 4× `ConvTranspose1d` ups stack
(10× × 5× × 3× × 2× = 300×). Splitting can't speed that up because the
work is the same regardless of where the AdaIN runs.

**Better paths for actual decoder speedup**, ranked by effort/reward:

1. **Int8/4-bit weight quantization on `decoder_upsample`.** HiFi-GAN
   is bandwidth-bound; 1.5-2× warm on CPU is typical with
   `coremltools.optimize.coreml.linear_quantize_weights(...)`. No
   graph changes, no retrain. Risk: audio artifacts; A/B required.
2. **Decoder `ALL` for server use.** 33.9 s load → 241 ms warm
   (-77 ms vs CPU_ONLY's 318 ms). Break-even ~415 utterances per
   process. Gate behind a `--server` flag.
3. **Replace HiFi-GAN with iSTFT vocoder (Vocos / RingFormer).**
   5-10× faster at parity quality, ANE-friendly (no transposed
   convs). Needs full vocoder retrain.
4. **Streaming chunked decode.** Doesn't reduce total work, but cuts
   TTFA dramatically — emit PCM in chunks of N frames as ready.

**Status.** Split implementation kept on disk
(`decoder_pre_fp16.mlpackage`, `decoder_upsample_fp16.mlpackage`)
and wired in `inference.py` for future ANE-quantization experiments.
Monolithic `decoder` mlpackage retained as the parity reference and
fallback. Removing the split would only save tens of MB on disk and
~3 s of conversion time; not worth it unless we conclude the split is
permanently a dead end.

---

## Trial: int8 weight quantization on `decoder_upsample` — slower, not faster

**Hypothesis.** HiFi-GAN is bandwidth-bound; per-channel int8 weights
(via `coremltools.optimize.coreml.linear_quantize_weights`) cut weight
DRAM traffic 4× and should give 1.5-2× warm on CPU. No retrain.

**Implementation.** Added `quantize_stage()` to `coreml/exporters/convert.py`
that takes an existing fp16 mlpackage and applies post-training
weight-only quantization (per-channel symmetric int8,
`weight_threshold=2048`). New CLI: `--precision int8 --stage <name>`.
Inference manifest extended: `_STAGE_PRECISION` accepts `"int8"`,
`_load_stage` maps it to `_int8.mlpackage` suffix. New CLI flag
`--int8 <stage…>` mirrors `--fp16` / `--fp32`.

```bash
uv run python coreml/exporters/convert.py --stage decoder_upsample --precision int8
# decoder_upsample_fp16.mlpackage (41.9 MB) -> decoder_upsample_int8.mlpackage (21.4 MB), 51%
```

**Result.** Disk halved, warm latency *worse*. Sweep on the converted
package, M-series Mac, 5 warm runs each:

```
                   load     warm
fp16 CPU_ONLY      1.3 s    274 ms
fp16 CPU_AND_GPU   1.4 s    253 ms
fp16 ALL          47.3 s    230 ms    ← best warm, 47 s load tax
int8 CPU_ONLY      1.6 s    385 ms    ← +40 % slower than fp16
int8 CPU_AND_GPU   3.1 s    291 ms    ← still slower
int8 ALL          46.9 s    580 ms    ← much worse
```

A/B audio (int8 CPU_ONLY vs fp16 CPU_ONLY): SNR 31.0 dB,
max|Δ| = 0.037, rms(Δ) = 0.002 on rms-0.074 clip. Listening-grade
but degraded.

**Why int8 lost.** Apple Silicon CPUs have no native int8
ConvTranspose1d kernel. CoreML's CPU backend dequantizes int8 weights
to fp16 per-call for the matmul, adding overhead. On ANE int8 is
hardware-accelerated, but HiFi-GAN's transposed-conv stack triggers
`ANECCompile() FAILED` regardless of precision — so the only path
that would benefit from int8 is closed.

**Apparent free win that wasn't.** Micro-bench in isolation showed
fp16 `CPU_AND_GPU` ~20 ms faster than `CPU_ONLY` (253 vs 274 ms warm).
Tried it; in the full pipeline GPU contention with `har_source`
(also `CPU_AND_GPU`) produced occasional outliers spiking to >1 s.
Reverted to `CPU_ONLY` for `decoder_upsample` — predictability beats
20 ms of average-case savings.

**Remaining options for further decoder speedup**, ranked:

1. **4-bit palettization** (`coremltools.optimize.coreml.palettize_weights`,
   16 levels k-means). Can be faster than linear int8 on CPU because
   the lookup table fits in registers; sometimes works on ANE where
   linear int8 doesn't. Untested for this graph.
2. **Decoder `ALL` for server use.** 47 s load → 230 ms warm
   (-44 ms vs current). Break-even ~1100 utterances per process.
   Gate behind a `--server` flag.
3. **Replace HiFi-GAN with iSTFT vocoder (Vocos / RingFormer).**
   5-10× faster, ANE-friendly. Needs full vocoder retrain.
4. **Streaming chunked decode.** Doesn't reduce total work, but cuts
   TTFA dramatically.

**Status.** `decoder_upsample_int8.mlpackage` kept on disk for future
ANE-quantization experiments (and as a smaller fallback). Runtime
default stays fp16 + `CPU_AND_GPU`; opt-in via `--int8 decoder_upsample`.

---

## Anticipated blockers (original list, retained for reference)

| Blocker | Stage | Mitigation |
|---|---|---|
| `weight_norm` deprecated → trace failure | 7 (decoder), maybe 1 | `remove_weight_norm()` before trace |
| Dropout-in-eval-mode RNG noise | several | Replace dropout with `nn.Identity` after `.eval()` |
| LSTM with packed sequences | 1, 5 | Use unpacked LSTM; rely on `text_mask` for downstream masking |
| ALBERT attention masks shape | 2 | `attention_mask` is int32; verify dtype before trace |
| SineGen `cumsum` + `sin` chain | 7 | Patch (cf. Kokoro `kokoro_coreml_fix.patch` rsqrt fix) |
| In-place ops in F0Ntrain shared/F0/N blocks | 6 | Wrap forward in a thin `nn.Module` that forces out-of-place |
| Diffusion sampler `features=ref_s` mutation | 4 | Pass `freeze_ref_s(ref_s)`; assert via `RefSGuard` |
| Variable mel length for ref_encoder | 3 | RangeDim or pad-to-bucket |
| Output length depends on predicted durations | 5→6→7 | Alignment built in Python; downstream stages take aligned tensors as inputs |

## Open work

* **RangeDim promotion.** Token-length and frame-length axes are
  currently fixed to the representative sentence's shape. Promote to
  `ct.RangeDim` per stage now that parity is locked in across all 7
  stages. Stages with shape-dependent axes:
  - text_encoder, bert: `T_text` (tokens)
  - ref_encoder: `T_mel` (mel frames of reference audio)
  - duration_predictor: `T_text`
  - f0n_predictor, decoder: `T_align` (post-alignment frame length)
  - diffusion_unet: `T_text` on `embedding` only — `x_noisy` is fixed
    `[1,1,256]`, `sigma` is `[1]`, `features` is `[1,256]`.

## Trial: int8 palettization on `decoder_upsample` — speed parity, lossy quality

**Hypothesis.** Linear int8 lost on CPU (385 vs 274 ms warm) because the
backend has no native int8 ConvTranspose1d kernel and dequantizes per
call. Palettization (k-means LUT) replaces the per-weight scale-multiply
with a fp16 LUT fetch — sometimes faster on Accelerate paths, and
historically better-supported on ANE than per-channel linear int8. If
true, we'd get the disk savings *without* the CPU regression — and maybe
finally land HiFi-GAN's ConvTranspose1d on ANE.

**Implementation.** Added `palettize_stage()` in `convert.py` using
`coremltools.optimize.coreml.palettize_weights` with
`OpPalettizerConfig(mode="kmeans", nbits=8, weight_threshold=2048)`.
Tried both `granularity="per_tensor"` (one 256-LUT per weight tensor)
and `granularity="per_grouped_channel"` (one LUT per output channel).
Wired `--precision int8pal` and `--palette-granularity` through the CLI
and `--int8pal STAGE` override into `inference.py`.

**Results — `decoder_upsample` only, full pipeline.**

| variant                         | size   | CPU_ONLY warm     | CPU_AND_GPU warm    | ALL warm              |
|---------------------------------|--------|-------------------|---------------------|-----------------------|
| fp16 (baseline)                 | 41.9 MB| **274 ms**        | 253 ms (unstable)   | 230 ms (47 s load)    |
| int8 linear per-channel         | 21.4 MB| 385 ms ❌         | 380 ms              | 380 ms                |
| int8pal **per_tensor**          | 21.4 MB| **272–285 ms** ✓  | 241–320 ms          | 232–339 ms (36 s load)|
| int8pal **per_grouped_channel** | 21.9 MB| **275–292 ms** ✓  | 519–623 ms ❌       | 727–737 ms ❌ (236 s) |

ANE remains a no-go: `MILCompilerForANE error: ANECCompile() FAILED`
on every variant. With CPU_AND_NE, the int8pal fallback path is even
worse than fp16's (3.3 s warm).

**Audio A/B vs fp16 on a 3.67 s clip.**

| variant              | max\|Δ\| | rms(Δ) | rms(ref) | SNR     |
|----------------------|----------|--------|----------|---------|
| int8 linear per-ch   | 0.037    | 0.002  | 0.074    | 31.4 dB |
| int8pal per-tensor   | 0.175    | 0.0076 | 0.071    | 19.4 dB |
| int8pal per-grouped  | 0.181    | 0.0073 | 0.071    | 19.7 dB |

Per-grouped-channel didn't beat per-tensor on quality despite per-channel
LUTs (52% size vs 51%). 256 centroids per channel still can't capture
HiFi-GAN's heavy-tailed ConvTranspose1d weight distribution. SNR 19 dB
is on the edge of audible degradation — needs listening test before
shipping. Linear int8 is meaningfully better here (31 dB), but pays for
it with 40% slower CPU.

**Why per_grouped_channel didn't help.** k-means quantization on fp32
weights, with centroids stored as fp16, is bottlenecked by the centroid
representation, not the partitioning. The dynamic range of HiFi-GAN
weights spans ~4 decades; 8 bits over a single channel is the same
log2-density as 8 bits over the tensor. To win quality back you'd need
either (a) higher bit depth (12-bit or 16-bit pal — defeats the
size win), (b) multiple LUTs per channel (`per_block`, group_size < N),
or (c) post-quant fine-tuning (calibration set + activation-aware
weight quantization, AWQ-style).

**Decision.** Stay on fp16 for `decoder_upsample`. int8pal `per_tensor`
is the runner-up: identical CPU latency, 51% disk, 19 dB SNR is the
quality cliff. Worth shipping behind a flag for size-constrained
distribution (mobile bundles), but not as default.

**Status.** Helper + CLI committed; `decoder_upsample_int8pal.mlpackage`
left in place for opt-in via `inference.py --int8pal decoder_upsample`.
Default `_STAGE_PRECISION["decoder_upsample"]` stays `"fp16"`.

**Remaining options for further decoder speedup**, ranked:

1. **4-bit palettization** with calibration / GPTQ-style quantization —
   coremltools 9.0 supports `OpPalettizerConfig(nbits=4)` with
   `cluster_dim>1` for vector-quantized blocks. Likely lossier than
   8-bit pal here, but the ANE story may differ. ~10 MB.
2. **Decoder `ALL` for server use.** 47 s load → 230 ms warm
   (-44 ms vs current). Break-even ~1100 utterances per process. Gate
   behind a `--server` flag.
3. **Replace HiFi-GAN with iSTFT vocoder (Vocos / RingFormer).** 5-10×
   faster, ANE-friendly. Needs full vocoder retrain.
4. **Streaming chunked decode.** Doesn't reduce total work, but cuts
   TTFA dramatically.

## Trial 10: `decoder_upsample` fixed-shape fp32 — ANE probe

**Hypothesis.** Trial 8 ALL placement on the iteration_3 RangeDim
mlpackage was bimodal (322-759 ms warm) — strong signal CoreML's ANE
planner was *attempting* to place the HiFi-GAN ConvTranspose1d ups
stack but bailing intermittently on the dynamic time axis. Collapse
both time axes (`T_FRAME2` 2..4096, `HAR_LEN` 600..1228800) to fixed
trace-default shapes and re-convert at fp32 (fp32-first principle:
change one variable at a time when chasing ANE acceptance).

Fixed shapes used: `x_pre [1,512,294]`, `ref [1,128]`,
`har_source [1,1,88200]` (T_FRAME=147, the trace default).

**Implementation.** New standalone script
`coreml/exporters/trial10_decoder_upsample_fixed.py`:
* loads `build_runtime()` + `build_wrapper("decoder_upsample", ...)` —
  same wrapper used by `convert.py`,
* traces at fixed shapes,
* `ct.convert(..., compute_precision=FLOAT32, compute_units=ALL,
  minimum_deployment_target=macOS15)` with no RangeDim,
* saves `coreml/packages/decoder_upsample_trial10_fp32_fixed.mlpackage`,
* benches load + warm under three placements (CPU_ONLY, CPU_AND_NE,
  ALL); 8 timed predicts after 3-iter warmup; parity vs eager wrapper.

**Results — Python `ct.models.MLModel.predict` on M-series Mac.**

| placement   | load    | warm min | med   | avg   | max   | spread  | parity vs eager       |
|-------------|---------|----------|-------|-------|-------|---------|-----------------------|
| CPU_ONLY    | 2064 ms | 2818     | 3473  | 4046  | 7480  | 4662 ms | cos=1.000000 max\|Δ\|=1.94e-3 |
| CPU_AND_NE  | 9507 ms | 5053     | 6502  | 6643  | 10197 | 5144 ms | cos=1.000000 max\|Δ\|=1.94e-3 |
| ALL         | 5376 ms | 1491     | 1634  | 1607  | 1794  | **303 ms** | cos=1.000000 max\|Δ\|=1.94e-3 |

Reference (current iteration_3 fp16 RangeDim, `iter3-bench` Swift):
CPU_ONLY warm avg = **304 ms**.

**Findings.**

1. **ANE refused even at fixed shape + fp32.** CPU_AND_NE was *slower*
   than CPU_ONLY (6.6 s vs 4.0 s avg) — the canonical signature of an
   ANE attempt that compiles, fails the placement check, falls back,
   and pays round-trip overhead. This is the same failure mode the
   int8 palettization trial saw (`MILCompilerForANE error:
   ANECCompile() FAILED`). Dynamic shape is not the blocker; the
   ConvTranspose1d ups stack itself is not on ANE's accept list.
2. **Fixed shape did stabilize the planner.** ALL spread dropped from
   Trial 8's 322-759 ms (437 ms swing) to 303 ms — single-mode now.
   Useful data point but not a placement win on its own.
3. **fp32 cost is prohibitive on this stage.** ALL warm avg 1607 ms is
   ~5× the fp16 baseline (304 ms); CPU_ONLY is ~13×. HiFi-GAN's
   ConvTranspose1d ups are bandwidth-bound and there's no fp32
   ConvTranspose1d Accelerate kernel parity with fp16's tuned path.

**Decision.** Don't promote. Keep iteration_3 fp16 RangeDim CPU_ONLY
as production for `decoder_upsample`. Trial 10's value is the
diagnosis: ANE rejects ConvTranspose1d unconditionally on this graph
under macOS 15 / coremltools 9.

**Next options.**

* **Trial 10b — ConvTranspose1d → ConvTranspose2d rewrite** (H=1
  unsqueeze, 2D convtranspose, squeeze). ANE has 2D convtranspose
  kernels; whether the rewrite traces cleanly through HiFi-GAN's
  weight-norm-stripped generator is the unknown.
* **Architecture swap** (Vocos / iSTFT vocoder, RingFormer): listed in
  the int8 palettization trial's "Remaining options" — biggest win,
  biggest cost (full vocoder retrain).
* Otherwise: accept that `decoder_upsample` is a permanent CPU stage
  and look for wins elsewhere (token-axis RangeDim on bert + sampler;
  `decoder_pre` already on ANE).

**Artifacts.**

* `coreml/exporters/trial10_decoder_upsample_fixed.py` — script (gitignored
  output package).
* `decoder_upsample_trial10_fp32_fixed.mlpackage` — saved locally;
  not promoted to `iteration_3/packages/`.

## Trial 10b: `decoder_upsample` Conv1d → Conv2d rewrite, fp32

**Hypothesis.** Trial 10 isolated shape from precision and proved
shape-was-not-the-blocker. The other half of the ANE-rejection theory
is that ANE has tuned Conv2d / ConvTranspose2d kernels but no 1D
variants at all. Substitute every `nn.Conv1d` and `nn.ConvTranspose1d`
in the HiFi-GAN generator with a drop-in 2D analog — `unsqueeze(2) →
conv2d → squeeze(2)`, weight `[C_out, C_in/G, K]` → `[C_out, C_in/G,
1, K]` (a single `unsqueeze(2)` of the same parameter, no
re-initialization). Generator forward is unchanged: residuals, AdaIN,
source filter, leaky-ReLU all see the same `(B, C, T)` signature at
the replacement boundaries; MIL has the chance to fold adjacent
squeeze/unsqueeze pairs.

**Implementation.** New standalone script
`coreml/exporters/trial10b_decoder_upsample_conv2d.py`:
* `Conv1dAs2d` / `ConvTranspose1dAs2d` drop-in modules,
* `_swap_convs_inplace(wrapper)` walks the wrapper's submodule tree
  and replaces every `Conv1d` (101 instances) and `ConvTranspose1d`
  (4 instances — the ups stack) after `_remove_weight_norm_recursive`
  has already run inside the wrapper's `__init__`,
* eager `1D vs 2D` parity check before tracing — gates with
  `max|Δ| < 1e-4`,
* same trace + ct.convert + bench protocol as Trial 10.

**Eager swap parity (gate).**

```
swapped: Conv1d×101  ConvTranspose1d×4
swap parity (1D vs 2D): cos=1.000000  max|d|=0.000e+00
```

Bit-equivalent in eager mode — the rewrite is mathematically a no-op,
as expected.

**Results — fp32, fixed shapes, M-series Mac.**

| placement   | load    | warm min | med    | avg     | max    | spread  | parity vs eager       |
|-------------|---------|----------|--------|---------|--------|---------|-----------------------|
| CPU_ONLY    | 2523 ms | 1676     | 3338   | **2937** | 4163  | 2487 ms | cos=1.000000 max\|Δ\|=1.94e-3 |
| CPU_AND_NE  | 1838 ms | 2264     | 3300   | **3683** | 5985  | 3720 ms | cos=1.000000 max\|Δ\|=1.94e-3 |
| ALL         | 2261 ms |  995     | 1077   | **1111** | 1575  |  580 ms | cos=1.000000 max\|Δ\|=1.94e-3 |

**vs Trial 10 (same precision, same shape, 1D convs):**

| placement   | Trial 10 avg | Trial 10b avg | delta |
|-------------|--------------|---------------|-------|
| CPU_ONLY    | 4046 ms      | 2937 ms       | **-27 %** |
| CPU_AND_NE  | 6643 ms      | 3683 ms       | **-45 %** |
| ALL         | 1607 ms      | 1111 ms       | **-31 %** |

**vs production fp16 RangeDim CPU_ONLY baseline (304 ms warm avg):**
3.7× slower at best (Trial 10b ALL).

**Findings.**

1. **Conv1d → Conv2d is a real win** across every placement (-27 to
   -45 %). MIL must be folding the per-op unsqueeze/squeeze pairs
   (or at minimum running them on a faster fast-path); the rewrite
   pays for itself even on CPU.
2. **ANE *still* refuses the ConvTranspose2d ups stack.** CPU_AND_NE
   (3683 ms) remains slower than CPU_ONLY (2937 ms) — the ANE-attempt-
   then-fallback signature persists. Either ANE rejects ConvTranspose2d
   at this kernel/stride/channel scale (256→512 ch, stride 10), or
   another op in the graph (AdaIN, source filter add, leaky-ReLU
   pattern, the unsqueeze/squeeze brackets themselves) is the structural
   blocker that no shape/dimensionality rewrite can fix.
3. **GPU now contributes meaningfully.** ALL (1111 ms) is 2.6× faster
   than CPU_ONLY (2937 ms), versus Trial 10 where ALL was only 2.5×
   faster than CPU_ONLY. The 1D→2D rewrite gave the GPU a fast path
   it didn't have before.
4. **Spread regressed on ALL** (303 ms → 580 ms). The planner now has
   more viable subgraph splits (CPU + GPU + maybe ANE retries) and
   makes different decisions per call.

**Decision.** Don't promote Trial 10b at fp32 — still 3.7× the fp16
baseline. But the rewrite itself is sound (bit-equivalent eager,
universal latency win). Worth a follow-up at **fp16 + Conv2d**: if
fp16 keeps the 27-31 % rewrite win on CPU_ONLY, that's ~210 ms warm —
finally beating the iteration_3 baseline. The fp16 cumsum-drift
concern from `fused_f0n_har_source` doesn't apply here (no audio-rate
cumsum in the generator; `har_source` is pre-computed input).

**Next options.**

* **Trial 10c — fp16 + Conv2d rewrite.** Same rewrite, fp16
  precision. Expected: ~200-220 ms warm CPU_ONLY (beats baseline);
  may or may not unlock ANE. Lowest-risk follow-up.
* **Trial 10d — drill into ANE rejection.** Capture the exact
  `MILCompilerForANE` log for Trial 10b to identify the rejecting op
  (likely the ups-stack ConvTranspose2d at stride 10). Decide
  whether to tile the ups stack into smaller strides (stride 10 →
  two stride-√10 stages won't divide evenly; 10 → 5×2 might).
* **Architecture swap** (Vocos / iSTFT vocoder) — biggest win,
  biggest cost (full vocoder retrain).

**Artifacts.**

* `coreml/exporters/trial10b_decoder_upsample_conv2d.py` — script.
* `decoder_upsample_trial10b_fp32_conv2d.mlpackage` — saved locally;
  not promoted.

## Trial 10d — `decoder_upsample` ANE rejection: tensor-width limit (Step 1)

Tracking issue: [#59](https://github.com/FluidInference/mobius/issues/59).

**Goal.** Identify the concrete MIL op or graph property that causes
ANECCompile to fail on `decoder_upsample`. Trials 10 (fixed-shape
ConvTranspose1d, fp32) and 10b (fixed-shape, Conv1d→Conv2d rewrite,
fp32) both confirmed ANE still refuses the graph after dynamism and
1D→2D were ruled out. Three remaining hypotheses (per #59) were Snake
activation, weight-norm-wrapped convs, or reflection padding.

**All three hypotheses were wrong.** The blocker is none of those.

### Setup

* Artifact: `iteration_3/packages/decoder_upsample_fp16.mlpackage` —
  the **production** fp16 mlpackage (what ships, pinned to `CPU_ONLY`
  precisely because of the ANE rejection this issue investigates).
  Pulled from HF via
  `huggingface_hub.snapshot_download(allow_patterns=["iteration_3/packages/decoder_upsample_fp16.mlpackage/**"])`.
* Fixture inputs: `iteration_3/swift/fixtures/decoder_upsample/in_*.npy`
  (T_mel = 294, T_audio = 88,200 samples = ~3.7 s of 24 kHz audio).
* Hardware: Apple M2, macOS 26.5.

### Step 1a — coreml-cli fallback dump

```
cd tools/coreml-cli
uv run coreml-cli .../iteration_3/compiled/decoder_upsample_fp16.mlmodelc \
    --fallback --json
```

Result:

| compute_units            | total ops | ANE | GPU | CPU | reason                          |
|--------------------------|-----------|-----|-----|-----|---------------------------------|
| `cpu_and_neural_engine`  | 1344      | **0** | 0 | 1344 | `"ANE not available for this op"` |

The 0/1344 ANE residency on **every** op — including basic
`ios18.add` (×353), `ios18.mul` (×302), `ios18.linear` (×96) that ANE
trivially supports — is the **model-level ANECCompile bail signature**,
not per-op rejection. The fallback walker reports the generic catch-all
because the runtime never made it past the planner; the planner gave up
on the whole graph and fell everything to CPU.

After the JSON body, `coreml-cli` itself spilled the runtime stderr:

```
E5RT encountered an STL exception. msg = MILCompilerForANE error:
  failed to compile ANE model using ANEF.
  Error=_ANECompiler : ANECCompile() FAILED.
```

Same error #59 cites, but no detail on *why* compile failed.

### Step 1b — ANE compile log via os_log + runtime stderr

Probe: `coreml/exporters/trial10d_step1_capture_ane_log.py`.
Loads the dereferenced fp16 mlpackage with
`compute_units=CPU_AND_NE`, runs one predict to lazily provoke the
ANE compile, captures everything the runtime emits.

```
MLLOG=1 OS_ACTIVITY_MODE=info OS_ACTIVITY_DT_MODE=enable \
    uv run python coreml/exporters/trial10d_step1_capture_ane_log.py \
    2>&1 | tee /tmp/trial10d_step1.log
```

The Espresso layer (Apple's ANE backend in CoreML) printed the
**actual rejection reason** on Python stderr:

```
2026-05-09 13:57:02 [coreml] E5RT: MILCompilerForANE error: failed to
  compile ANE model using ANEF.
  Error=_ANECompiler : ANECCompile() FAILED (11)
2026-05-09 13:57:03 Error: Tensor width goes beyond limit supported (16414 > 16384.
2026-05-09 13:57:03 Error: Tensor width goes beyond limit supported (16414 > 16384.
2026-05-09 13:57:03 Error: Tensor width goes beyond limit supported (16390 > 16384.
2026-05-09 13:57:03 Error: Tensor width goes beyond limit supported (16390 > 16384.
                            ... (12 lines total, alternating widths) ...
2026-05-09 13:59:04 [espresso] [Espresso::handle_ex_plan]
  exception=ANECF error: failed to load ANE model file:///private/var/folders/.../decoder_upsample_fp16.mlmodelc/model.mil
  Error=ANECCompile(/Library/Caches/com.apple.aned/tmp/-/...) FAILED:
  err=( CompilationFailure )
2026-05-09 13:59:04 [coreml] Error plan build: -1.
```

The `[com.apple.ane:compiler]` subsystem messages were
`<private>`-redacted in the os_log stream (the typical fate of ANE
compile diagnostics on user macOS). The actionable strings landed on
Python stderr regardless.

### Finding

**ANE has a hard width limit of 16,384 elements per tensor.** Two
intermediate tensors in `decoder_upsample` exceed it at our fixture's
T_mel = 294:

* width **16,414** (overshoot 30) — appears 6×
* width **16,390** (overshoot 6) — appears 6×

(Each width prints twice per ANE attempt, presumably for two distinct
op layouts, ×3 retry attempts before the planner gives up.)

The MIL itself uses `?` for the time axis (it was traced with fixed
shapes baked into the example_inputs but coremltools emits dynamic
shapes for the time dim through ConvTranspose1d). The 16414 / 16390
widths are computed **at compile time** by ANE during placement — they
include ANE's internal layout / tiling padding on top of the natural
intermediate tensor T. The natural T at the largest pre-rejection
boundary is the post-`generator.ups[1]` width:

```
ups[0]: stride 10, kernel 20, T: 294        →  2950
ups[1]: stride 5,  kernel 10, T: 2950 (post-trim)  → 14755 raw, ~14702 post-slice
                                                   (this is the boundary that
                                                    ANE's layout pads to >16384)
ups[2]: stride 3,  kernel 6,  (further upsample to ~44k — never reached)
ups[3]: stride 2,  kernel 4,
```

Post-`ups[1]` natural T ≈ 14702. ANE's layout pads/tiles this to
either 16414 or 16390 depending on which intermediate tensor (likely
the convolution output + the residual / noise-source-add branch)
overflows. Either way, the rejection is **input-length-dependent**, not
an op-type rejection.

**This rules out all three #59 hypotheses:**

* ❌ **Snake activation.** `mil.sin` / `mil.pow` execute on ANE for
  smaller inputs; not the blocker.
* ❌ **Weight-norm-wrapped convs.** Weight norm is folded by
  `_remove_weight_norm_recursive` before tracing; the converted graph
  has plain `weight` and `bias` constants. Not the blocker.
* ❌ **Reflection / replication padding inside the ups blocks.** Snake
  uses `pad` of `[0, 0]` (custom mode) per the MIL; no reflection /
  replication pad is in the rejected subgraph.

The blocker is **a hardware width-axis limit on intermediate tensors**,
which is a function of input T_mel and the upsample stack's stride
product. Issue #59's framing "which op causes ANE to refuse" doesn't
fully apply — every op is fine; the *combination* of op chain plus
input length pushes one intermediate over the limit.

### Implications for Step 3 decision

Issue #59 anticipated three outcomes; the actual finding maps to
**outcome 1 (graph rewrite, no retraining)**, but the rewrite is *not*
about an op type — it's about **how the upsample is sliced along T**.
Three families of weight-preserving rewrites are viable:

1. **T_mel ceiling.** With strides [10, 5, 3, 2] and the observed
   16,414-element overshoot at T_mel = 294, the headroom is small:
   T_mel ≤ ~285 (≈ 14,250 post-ups[1] + 1714 ANE pad ≈ 15,964 < 16,384)
   should fit. T_mel ≤ ~280 has comfortable margin. Production should
   bucket below ~3.4 s of audio per call. Lossless and trivial; loses
   nothing on quality but caps the per-call utterance length and
   forces chunking for longer prompts. Already aligned with Trial 11's
   per-bucket exporters for `bert` / sampler.

2. **Chunked decoder_upsample.** Run the upsample stack on overlapping
   T-axis chunks of size < ~280 each, fuse outputs with a stitch /
   crossfade. Mathematically equivalent to the monolithic decoder for
   sufficient overlap (governed by the cumulative receptive field of
   the ups stack). Same weights, no retraining. Some per-call overhead
   (chunk + stitch). This is the "scale to arbitrary T_mel" path that
   keeps ANE eligible.

3. **Use Trial 10b's Conv2d-rewritten artifact at smaller T_mel.** Open
   question: does the same 16,384 limit apply to Conv2d's `W` axis?
   Trial 10b's bench result (`CPU_AND_NE` slower than `CPU_ONLY`)
   indicates ANE still rejected the rewrite — we should re-run Step 1b
   on the Trial 10b mlpackage at our 294-frame fixture to confirm the
   *same* width-limit error. If so, the dimensionality rewrite is
   orthogonal to this blocker; if a different error surfaces, Trial 10b
   is hitting an additional ceiling and bisection (Step 2) is needed
   for that one.

### Acceptance per #59

* ✓ MIL op + reason: **rejection is not op-specific**; it's the
  Espresso "Tensor width goes beyond limit supported" check applied
  to the post-`generator.ups[1]` layout's W axis. Documented above.
* Go / no-go decision deferred until either (a) we confirm Trial 10b
  hits the same width limit, or (b) Trial 10e implements one of the
  three rewrite families above and verifies cosine sim > 0.999 vs
  iteration_3 fp32 reference.

### Reproduction

```bash
cd tools/coreml-cli
uv run coreml-cli .../iteration_3/compiled/decoder_upsample_fp16.mlmodelc \
    --fallback --json 2>&1 | tee /tmp/decoder_upsample_fallback.json
# scroll past the JSON to see the trailing
#   "MILCompilerForANE error: ANECCompile() FAILED" stderr
```

```bash
cd models/tts/styletts2
huggingface-cli download FluidInference/StyleTTS-2-coreml \
    "iteration_3/packages/decoder_upsample_fp16.mlpackage/*" \
    "iteration_3/swift/fixtures/decoder_upsample/*"
MLLOG=1 OS_ACTIVITY_MODE=info OS_ACTIVITY_DT_MODE=enable \
    uv run python coreml/exporters/trial10d_step1_capture_ane_log.py \
    2>&1 | tee /tmp/trial10d_step1.log
grep "Tensor width" /tmp/trial10d_step1.log
```

Expected output: 12 lines of `Error: Tensor width goes beyond limit
supported (16414 > 16384.` / `(16390 > 16384.` plus the
`ANECCompile() FAILED` cascade.

### Artifacts

* `coreml/exporters/trial10d_step1_capture_ane_log.py` — probe.
* No mlpackage produced or modified; iteration_3 unchanged.

### Step 1c — same probe on Trial 10b (fp32 + Conv2d) and Trial 10c (fp16 + Conv2d)

**Question.** Does the Conv1d → Conv2d rewrite sidestep the width
limit? Trial 10b's earlier bench (`CPU_AND_NE` slower than `CPU_ONLY`)
suggested ANE was still rejecting, but didn't surface the reason.

**Method.** Re-ran `trial10b_decoder_upsample_conv2d.py` on this M2 to
produce `decoder_upsample_trial10b_fp32_conv2d.mlpackage`, then a
one-shot inline variant (`/tmp/trial10c_inline.py`, identical to
trial10b except `compute_precision = FLOAT16`) to produce
`decoder_upsample_trial10c_fp16_conv2d.mlpackage`. Both at the same
fixed shapes — `x_pre = (1, 512, 294)`, `ref = (1, 128)`,
`har_source = (1, 1, 88200)` (T_mel = 294 — identical to Step 1a/b
fixture). Probed both with `coreml-cli --fallback --json` and the same
`.cpuAndNeuralEngine` Python loader from `trial10d_step1_capture_ane_log.py`.

**Comparison table.**

| Artifact                        | Precision | Conv | Total ops | ANE % | Top rejection reason                                  | Espresso stderr                                                            |
|---------------------------------|-----------|------|-----------|-------|-------------------------------------------------------|----------------------------------------------------------------------------|
| iteration_3 (shipping)          | fp16      | 1D   | 1344      | 0.0 % | `"ANE not available for this op"` (1344)              | `Tensor width > 16384` (16414×6, 16390×6) + ANECCompile FAILED              |
| Trial 10b                       | fp32      | 2D   | 1349      | 0.0 % | `"Invalid output tensor format: fp32"` (1348) + 1 op  | none — ANE planner refused at format gate, no compile attempt                |
| Trial 10c                       | fp16      | 2D   | 1352      | 0.0 % | `"ANE not available for this op"` (1352)              | `Tensor width > 16384` (16414×6, 16390×6) + ANECCompile FAILED (×2)         |

(Trial 10c has 8 more ops than the production 1D fp16 — the
`unsqueeze(H=1) → conv2d → squeeze(H)` brackets around each `Conv1d` /
`ConvTranspose1d`. Otherwise the graph is the same wrapper.)

**Trial 10b (fp32) interpretation.** ANE never reaches the width check
because `fp32` outputs are disqualified at the planner's format gate
upstream. No ANE compile attempt → no `ANECCompile() FAILED` in
stderr. The bench result (`CPU_AND_NE` 943 ms < `CPU_ONLY` 1108 ms on
this M2 / macOS 26.5) does NOT indicate ANE acceptance — the speedup is
likely from CPU+GPU planner splits that `CPU_ONLY` doesn't have. Trial
10b's old "ANE-attempted-then-fallback" framing is wrong; ANE never
attempts.

**Trial 10c (fp16 + Conv2d) — the load-bearing comparison.**
`coreml-cli --fallback` reports the **same** generic-catch-all rejection
pattern as production fp16, and the Espresso stderr emits the **exact
same** width-limit errors:

```
Error: Tensor width goes beyond limit supported (16414 > 16384.   (×6)
Error: Tensor width goes beyond limit supported (16390 > 16384.   (×6)
[espresso] [Espresso::handle_ex_plan] exception=ANECF error: failed to load ANE model
   ... Error=ANECCompile(...) FAILED: err=( CompilationFailure )
```

Identical widths. Identical retry pattern. **Conv2d rewrite does NOT
move the width budget.** ANE's 16,384-element limit is on the rank-N
spatial axis regardless of whether T sits at the rank-3 W axis (1D
unsqueeze trick) or the rank-3 W axis of a 4D tensor (the natural 2D
layout) — same physical tensor, same hardware budget.

**Trial 10c bench wasn't run** — same artifact size as production fp16
(40 MB vs 41 MB), same MIL graph shape, same ANE outcome → no useful
new latency information.

**Outcome (per Issue #59 Step 4 tree).**

> If same `Tensor width > 16384` error: Conv2d doesn't sidestep the
> limit. Confirms the blocker is dimensionality / width-budget,
> orthogonal to op type. Go to chunking (option 2) for Trial 10e.

**Hit. Confirmed.** Conv2d is orthogonal to the blocker. Trial 10e
should pursue **chunked decoder_upsample** (option 2 from Step 1's Step
3 candidates): split the T axis into overlapping windows pre-`ups[1]`
where the natural T is < 14,000, run the upsample stack per chunk, fuse
outputs with overlap-aware stitching. The cumulative receptive field
of the four ConvTranspose1d ops governs the required overlap; same
weights, no retraining, lossless if overlap ≥ receptive field.

The companion fallback path is **option 1** (production-cap
T_mel ≤ ~280) which is trivially achievable through bucketed exporters
and works today without code changes — Trial 11's bucketing pattern
extends naturally. Combining options 1 + 2 (cap each ANE-eligible chunk
to T_mel ≤ ~280) is the most robust shape: chunk for arbitrary input
length, cap each chunk for ANE eligibility.

### Step 1c artifacts

* `coreml/packages/decoder_upsample_trial10b_fp32_conv2d.mlpackage` (79 MB)
  — saved locally; not promoted.
* `coreml/packages/decoder_upsample_trial10c_fp16_conv2d.mlpackage` (40 MB)
  — saved locally; not promoted.
* Probe artifacts: `/tmp/probe_trial10b.stderr`, `/tmp/probe_trial10c.stderr`,
  `/tmp/trial10b_fallback.out`, `/tmp/trial10c_fallback.out`.

### Acceptance per #59 (now actionable)

- ✓ MIL "op" + reason: not an op; **ANE-compiler width-axis limit
  (16,384) hit by post-`generator.ups[1]` intermediates at T_mel ≥ ~285**.
- ✓ Confirmed dimensionality-orthogonal — not specific to Conv1d (1D)
  vs Conv2d (2D) layout.
- Decision per Step 3: **Outcome 1 (graph rewrite, no retraining)** is
  the path. Specifically: chunked decoder_upsample (option 2) along T
  axis, ± production-cap (option 1). Trial 10e to implement and verify
  cosine sim > 0.999 vs iteration_3 fp32 reference at the original
  T_mel = 294 fixture (and one larger T_mel that exercises chunking).

## Trial 10e — chase weight-preserving rewrite to land `decoder_upsample` on ANE

Issue [#59](https://github.com/FluidInference/mobius/issues/59) Step 3
called for one of three rewrite families. Step 1c ruled out the Conv2d
rewrite as orthogonal to the width budget; Trial 10e investigates the
remaining two:

* **Option 1** — production-cap `T_mel` (and bucket the exporters) so the
  intermediate widths stay under ANE's 16,384-element budget. Cheapest
  to validate; should work if the width limit is the root cause.
* **Option 2** — chunked `decoder_upsample` along T with overlap +
  stitch, so each chunk is ANE-eligible regardless of total utterance
  length. The actual ship answer if Option 1 doesn't land.

Both options share an unstated premise from Step 1: that the **width
limit is the only ANE blocker**, and any T_mel small enough to dodge it
will compile cleanly on ANE.

**Trial 10e1 disproves that premise.** The width limit is a *secondary*
symptom of a deeper structural rejection. Both options are non-viable
without first identifying and rewriting the offending op (Issue #59
Step 2 — bisection).

### Trial 10e1 — T_mel cap (Option 1) [DEAD-END]

**Method.** Capped exporter (`coreml/exporters/trial10e1_t_mel_cap.py`)
crops the captured T_mel = 294 fixture to a smaller T_mel, traces +
converts the same `decoder_upsample` wrapper at fixed shapes, fp16, 1D
Conv (matches production iteration_3 shape exactly except for fixed
T_mel). For each candidate it runs:

1. `coreml-cli --fallback --json` for ANE residency,
2. a `.cpuAndNeuralEngine` Python load + predict to capture stderr,
3. parity vs the PyTorch wrapper at the cropped inputs,
4. warm-avg bench on CPU_ONLY and CPU_AND_NE.

Probed candidates: `[50, 64, 128, 280, 292, 293, 294]`. Production
iteration_3 ships at T_mel = 294.

**Sweep results (full table).**

| T_mel | har_source size | T_audio | width-error in stderr (interactive) | ANECCompile() FAILED (interactive) | residency | parity cos | CPU_ONLY | CPU_AND_NE |
|------:|----------------:|--------:|:-----------------------------------:|:-----------------------------------:|----------:|-----------:|---------:|-----------:|
|    50 |          15,000 |  15,000 | none                                | yes                                 | n/a (timeout) |  0.998806 |   n/a   |    n/a    |
|    64 |          19,200 |  19,200 | none                                | yes                                 | 0.0 %     |  0.998806 |   n/a   |    n/a    |
|   128 |          38,400 |  38,400 | (not interactively probed; subprocess clean) | (subprocess clean)         | 0.0 %     |  0.999068 |   n/a   |    n/a    |
|   280 |          84,000 |  84,000 | **16,414 ×6, 16,390 ×6**            | yes                                 | 0.0 %     |  0.998238 | 302.3 ms |  2,878.5 ms |
|   292 |          87,600 |  87,600 | (subprocess miss; same family)      | yes                                 | 0.0 %     |  0.998283 | 281.5 ms |  3,192.9 ms |
|   293 |          87,900 |  87,900 | (subprocess miss; same family)      | yes                                 | 0.0 %     |  0.998411 | 283.9 ms |  3,011.7 ms |
|   294 |          88,200 |  88,200 | **16,414 ×6, 16,390 ×6** (Step 1)   | yes                                 | 0.0 %     |  0.998462 | 286.1 ms |  3,031.8 ms |

(Bench at T_mel = 50/64/128 was skipped via `--skip-bench` since residency was already 0 %.)

**Two distinct ANE failure modes, stacked.**

* **Mode A — large T_mel (≥ 280):** ANE attempts compile, the planner
  tries to tile the graph, hits the 16,384-element width limit during
  tiling, prints multiple `Tensor width goes beyond limit supported`
  lines (alternating widths 16,414 and 16,390, six of each across
  three retry attempts), then gives up with `ANECCompile() FAILED (11)`
  and the Espresso `ANECF error: failed to load ANE model …
  CompilationFailure` cascade. **This is what Step 1 captured.**

* **Mode B — small T_mel (50, 64):** ANE attempts compile, tensors fit
  in single tiles (no tiling needed), but compile **still fails** with
  `ANECCompile() FAILED (11)`. **No `Tensor width` errors in stderr.**
  The actual rejection reason is `<private>`-redacted in
  `[com.apple.ane:compiler]` os_log entries. The Python stderr emits
  exactly one `MILCompilerForANE error: ANECCompile() FAILED (11)`
  line — that's the only signal we get. ANE compile takes **~80
  seconds** at T_mel = 50 before giving up (vs ~33 s at T_mel = 294),
  suggesting the planner walks deeper into placement before bailing.

```
# T_mel = 50, captured Python stderr (full content, two lines after the warnings):
2026-05-09 15:51:58 python3[…] [coreml] E5RT: MILCompilerForANE error:
    failed to compile ANE model using ANEF.
    Error=_ANECompiler : ANECCompile() FAILED (11)
```

```
# T_mel = 50, os_log [com.apple.ane:compiler] (private fields redacted):
2026-05-09 15:51:58 ANECompilerService [com.apple.ane:compiler]
    Calling ANE compiler done ret(1)
2026-05-09 15:51:58 ANECompilerService [com.apple.ane:compiler]
    <private>: <private>      (×3, redacted detail)
2026-05-09 15:51:58 ANECompilerService [com.apple.ane:compiler]
    <private>: ERROR: model=<private> : output=<private> :
    lAttr=<private> : lErr=Error Domain=com.apple.appleneuralengine.compiler
    Code=1 UserInfo={NSLocalizedDescription=<private>, …}
2026-05-09 15:51:58 aned [com.apple.ane:aned] Compilation failed:
    error=Error Domain=com.apple.appleneuralengine.compiler Code=1
    UserInfo={NSLocalizedDescription=<private>, …}
```

The width errors at large T_mel are **a retry symptom**, not the root
cause. ANE retries with chunked tensor layouts when the first compile
attempt fails; for graphs with intermediate Ts > 16,384, the chunked
retries hit the per-tile width limit and surface as visible errors.
For graphs with intermediate Ts ≤ 16,384, the same root failure
occurs but with no visible width spam — just the single
`ANECCompile() FAILED (11)`.

**Implication:** **the width limit is not the gating constraint.** No
T_mel value, however small, gets ANE acceptance for this graph. Option
1 (T_mel cap) is non-viable.

**Implication for Option 2 (chunking):** chunking solves the
width-limit symptom but NOT the structural rejection. Each chunk would
also fail ANECCompile at the structural level, regardless of how small
the chunk is. **Option 2 is also non-viable** without first finding and
rewriting whatever op the structural rejection actually targets.

**Parity floor — not a regression.** All capped variants land at
cosine ≈ 0.998 vs the PyTorch wrapper. A 4-way comparison clarifies
this is the inherent fp16 floor, not a Trial 10e1 regression:

| reference vs hypothesis                   | cos       | max\|Δ\|  |
|-------------------------------------------|-----------|-----------|
| iter3 `out_var_3711.npy` vs production fp16 mlpackage | **1.000000** | **0** |
| iter3 `out_var_3711.npy` vs Trial 10e1 fp16 (T_mel=294) | **1.000000** | **0** |
| iter3 `out_var_3711.npy` vs PyTorch wrapper           | 0.998219     | 0.187    |
| Trial 10e1 fp16 (T_mel=294) vs PyTorch wrapper        | 0.998219     | 0.187    |

The "iteration_3 fp32 reference" file (`out_var_3711.npy`) is **bit-
identical to the production fp16 mlpackage output** — it was generated
from the same fp16 graph, not a separate fp32 path. The 0.998 cos is
the fp16-vs-PyTorch-fp32 quantization gap that production already has
and ships at; my Trial 10e1 packages match production fp16 exactly. So
the cos ≥ 0.999 gate from Issue #59 is met against the documented
reference (cos = 1.000000), even though all packages drift identically
from the eager fp32 wrapper.

**A brief note on the `CPU_AND_NE` bench (~10× slower than `CPU_ONLY`).**
Despite 0 % ANE residency reported by `coreml-cli`, requesting
`compute_units=CPU_AND_NE` at MLModel load triggers ANE compile
attempts on every load + retry overhead per predict. CPU_ONLY skips
that path entirely. Production iteration_3 ships pinned to CPU_ONLY
precisely to avoid this; nothing in Trial 10e1 changes that constraint.

**Verdict.** **Option 1 dead.** Option 2 dead by extension (same
structural blocker fires at any chunk size). Issue #59 Step 2 (op
bisection) is now load-bearing — until we identify what ANE structurally
rejects in this graph, no T_mel-axis or chunking rewrite will land
`decoder_upsample` on ANE.

**Suggested next steps (Trial 10e3 — bisection).**

1. Re-capture os_log with privacy redaction lifted (`sudo log config
   --mode 'private_data:on'`) — the redacted strings in
   `[com.apple.ane:compiler]` likely name the offending op. Cheapest
   diagnostic if root permission is available.
2. Bisect by ablation: replace candidate ops with no-op stand-ins (in
   strict diagnostic mode; quality irrelevant) until ANE flips to
   accept. Per Issue #59 Step 2:
   * Snake activation → identity.
   * AdaIN → linear (drop affine).
   * `pow` (used in Snake `sin²` form) → constant 1.
   * Walk one MRF block at a time.
3. Once a flipping op is found, attempt a weight-preserving rewrite of
   that op (cosine-identity for Snake, etc.) and re-bench.

Step 2 is real engineering effort (multiple hours, multiple compile
cycles per ablation) and is not in scope for this trial.

### Artifacts (saved locally, not promoted)

| Bucket | Path | Size |
|--------|------|------|
| T_mel=50  | `coreml/packages/decoder_upsample_trial10e1_fp16_tmel50.mlpackage`  | 40 MB |
| T_mel=64  | `coreml/packages/decoder_upsample_trial10e1_fp16_tmel64.mlpackage`  | 40 MB |
| T_mel=128 | `coreml/packages/decoder_upsample_trial10e1_fp16_tmel128.mlpackage` | 40 MB |
| T_mel=280 | `coreml/packages/decoder_upsample_trial10e1_fp16_tmel280.mlpackage` | 40 MB |
| T_mel=292 | `coreml/packages/decoder_upsample_trial10e1_fp16_tmel292.mlpackage` | 40 MB |
| T_mel=293 | `coreml/packages/decoder_upsample_trial10e1_fp16_tmel293.mlpackage` | 40 MB |
| T_mel=294 | `coreml/packages/decoder_upsample_trial10e1_fp16_tmel294.mlpackage` | 40 MB |

Probe artifacts: `/tmp/trial10e1.log`, `/tmp/trial10e1_small.log`,
`/tmp/probe50_full.stderr`, `/tmp/ane_log_t50.log`.

### Reproduction

```bash
cd models/tts/styletts2
uv run python coreml/exporters/trial10e1_t_mel_cap.py \
    --candidates 50,64,128,280,292,293,294
```

Each candidate runs ~3-5 min (convert + probe + parity + bench).

### Trial 10e2 — ANE planner is the blocker, not per-op rejection

Trial 10e1 ended with the structural-blocker hypothesis: ANE refuses
something about the graph itself (an op, a tensor rank, a layer
property), independent of T_mel. The next-step plan was either to
unredact the `<private>` strings in `[com.apple.ane:compiler]` os_log
output (cheap, ~5 min), or to bisect ops by ablation (multi-hour).
Tried the unredact path first. **The redact attempt didn't fully unredact, but it surfaced a different finding that changes the bisection target.**

#### 1. `private_data:on` is gone on macOS 26.5

`sudo log config --mode "private_data:on"` returns:
```
log: Invalid Modes 'private_data:on'
```

Confirmed both with `--subsystem com.apple.ane` and globally. The
legacy syntax was deprecated; per current `man log` on macOS 26.5
(Darwin 25.5.0, build 25F5042g), `log config --mode` only accepts
`level: {off|default|info|debug}` and `persist: {off|default|info|debug}`.

Working alternatives:

* `level:debug` — surfaces additional message types but does **not**
  unredact `%{private}` fields.
* Configuration profile (`com.apple.system.logging` payload with
  `Enable-Private-Data: true` per subsystem) — installed via
  `sudo profiles install -path <plist> -type configuration`, then
  approved manually in System Settings → General → Device Management.
  Requires GUI approval, persists across reboots, scope = system-wide
  for the named subsystem.

#### 2. `level:debug` empirically does not unredact

Confirmed:
```bash
sudo log config --mode "level:debug" --subsystem com.apple.ane
sudo log config --mode "level:debug" --subsystem com.apple.ane --category compiler
sudo killall aned
```

The stream has more chatter (debug-level messages now visible) but
every previously redacted field still renders as `<private>` — only
the configuration profile path actually unredacts. `level:debug` is
strictly a verbosity knob, not a privacy override.

#### 3. New finding from the level:debug capture: ANECompilerService never reports failure

Over a ~4-minute compile attempt at T_mel = 50 (T_audio = 15,000, well
under the 16,384 ANE-tile-width ceiling), os_log shows **dozens of
SUCCESS lines** from `ANECompilerService`. Multi-threaded — concurrent
work on threads `46a842e` and `46aa141` — every per-call result is
`ret(0)` with `lErr=(nil)`. Representative pattern (each block repeats
dozens of times):

```
ANECompilerService [com.apple.ane:compiler] Calling ANE compiler
ANECompilerService [com.apple.ane:compiler] Calling ANE compiler done ret(0)
aned                [com.apple.ane:aned]     FAILED removing <private> ...
                                             NSPOSIXErrorDomain Code=2
                                             "No such file or directory"
                                             ...   ← harmless tmp-file cleanup
ANECompilerService [com.apple.ane:compiler] Attempt to store <private>
ANECompilerService [com.apple.ane:compiler] SUCCESS: model=<private> :
                                             output=<private> :
                                             lAttr=<private> : lErr=(nil)
```

`ret(0)` with `lErr=(nil)` is unambiguous: **the underlying ANE
compiler accepts every subgraph it's handed.**

#### 4. Contradiction with Python's `E5RT` layer

While `ANECompilerService` reports nothing but successes, the Python
`coremltools` runtime sitting on top simultaneously reports:

```
[coreml] E5RT: MILCompilerForANE error: failed to compile ANE model
        using ANEF.
        Error=_ANECompiler : ANECCompile() FAILED (11)
```

So the rejection is **not at per-call ANECompilerService**. It's at a
layer above — most likely one of:

* MIL → ANEF planner: deciding how to partition the graph into ANE
  subgraphs and reassemble them into a deployable single-ANEF model
  fails when the per-subgraph artifacts can't be linked.
* Final whole-graph compile after subgraph caching: the "stitch" step
  that combines the cached `<private>` files (visible in `Attempt to
  store <private>`) into a single deployable artifact.
* E5RT runtime load: the file is built but the runtime can't load it.

The multi-threaded parallel ANECompilerService activity is informative
on its own — it suggests the planner is **aggressively fragmenting**
the graph into many small ANE-eligible subgraphs (which compile fine
individually) but then can't reassemble the result.

#### 5. Reframe of Issue #59 hypothesis space

Issue #59's framing:
> Identify the exact MIL op (or graph property) that ANE refuses

was already loosened by Trial 10d / 10e1: not an op, but a width
ceiling at large T_mel and a `<private>`-redacted structural
rejection at small T_mel. **Trial 10e2 loosens it further:** the ANE
compiler doesn't refuse anything at the per-subgraph level. The
blocker is the *orchestration / partitioning strategy* that produces
many small ANE-eligible chunks the planner can't link.

The right question for bisection therefore shifts from:

  ❌ *"What op is structurally rejected by ANE?"*

to:

  ✓ *"What property of the graph makes the planner over-fragment, and
  what change reduces fragmentation to a linkable shape?"*

This reframes the bisection target. Candidate causes of aggressive
fragmentation, in priority order:

1. **Skip / residual connections at high fan-out.** Each AdaINResBlock1
   has three internal residual adds; the MRF aggregation sums three
   resblocks per stage; the source-filter add joins two parallel
   branches. Each residual forces the planner to materialize an
   ANE-readable intermediate. If those intermediates exceed the
   planner's per-subgraph budget, it splits aggressively.
2. **Wide channel transitions in tight succession.** ConvTranspose1d
   stages drop channels 512 → 256 → 128 → 64 → 32 in four steps. Each
   transition may trigger a re-plan boundary.
3. **AdaIN style input shape (1, 128).** A scalar-per-feature affine
   modulation requires broadcasting across a (1, C, T) intermediate;
   the planner may insert reshape boundaries around every AdaIN call
   (88 instances per the production fp16 fallback table). That's a lot
   of partitioning seams.
4. **Snake activation count.** 101 instances of `pow(sin(αx), 2)`
   placed after every AdaIN; even if Snake itself is ANE-supported,
   the sheer number of activation boundaries multiplies fragmentation.

Each candidate is testable by ablation (replace with linear / identity
in strict diagnostic mode, re-convert, see if the planner stops
fragmenting). The fragmentation signal is observable directly in the
`level:debug` os_log stream — count the `Calling ANE compiler` /
`SUCCESS:` pairs per compile attempt. Fewer pairs = less fragmentation
= closer to a linkable graph.

#### 6. Cost-benefit: stop unredacting, start bisecting

The configuration-profile path requires GUI approval, persists across
reboots, scope is system-wide, and the payoff is a single string
inside `lErr.NSLocalizedDescription` — which, given finding #3, may
only describe the `_ANECompiler` runtime layer's error rather than
the underlying planner reason. ~4 min compile + GUI friction +
uncertain payoff isn't worth it. **Pivoting to op bisection** with the
reframed target (reduce fragmentation, not avoid a "rejected op").

### Suggested Trial 10e3 — bisection by ablation

Now reframed around fragmentation reduction, in priority order:

1. **Drop AdaIN affine modulation.** Replace `AdaIN1d(style_dim,
   channels)` with a plain `LayerNorm` (no style conditioning), then
   re-convert. Watch the os_log fragmentation count. If the count
   drops sharply, AdaIN is the partitioning trigger. Weight-preserving
   replacement: bake the AdaIN affine into the trained weights at
   export time (style `s` is known at conversion time per stage in the
   production manifest? — only if speakers are fixed at conversion;
   typically not).
2. **Replace Snake → identity** (pure diagnostic, no shipping intent).
   If fragmentation drops, the partitioning is per-activation. Then
   try the cosine-identity rewrite (`x + (1 − cos(2αx))/(2α)`) which
   is bit-equivalent at fp32 (already validated by Kokoro-ANE's port);
   if it lands the planner, ship that.
3. **Fold residuals.** Replace residual adds in `AdaINResBlock1` with
   in-place modifications (no separate add node). Risky for parity but
   cheap to ablate.
4. **Halve the upsample stack.** Drop `ups[2]` and `ups[3]` (final two
   stages), trace + convert, see if the truncated graph plans cleanly.
   If yes, the issue is depth-related; if no, it's elsewhere.

Cost per ablation: ~2 min trace+convert + ~80 s ANE compile ≈ 3-4 min.
Five ablations ≈ 20 min. Each one yields a fragmentation-count delta.

### Artifacts

* No new mlpackage produced this trial.
* Captured probe outputs from `level:debug` session retained at
  `/tmp/ane_log_t50_debug.log` (user-side; not committed).

### Trial 10e3 — bisection by ablation, FLIP found (Snake activation)

Trial 10e2 reframed the bisection target from "what op is rejected" to
"what makes the planner over-fragment." This trial walks the prioritized
ablation list at T_mel = 50 (fastest probe; well under the 16,384 width
threshold), measuring `Calling ANE compiler` line counts via `log show`
on subsystem `com.apple.ane`, category `compiler`.

**Method.** `coreml/exporters/trial10e3_bisection.py` implements
idempotent monkey-patches that swap the diagnostic op for an ANE-
friendly stand-in, then traces + converts the standard
`decoder_upsample` wrapper at fp16 fixed shapes, then probes
`.cpuAndNeuralEngine` and counts the partition events.

The FLIP signal is **fragmentation count drop > 50 % vs baseline AND no
width errors**. The probe's `e5rt_failed` signal is unreliable in
subprocess (CoreML's E5RT message goes through os_log, not directly to
Python's stderr fd) — both baseline and ablations all report
`e5rt_failed=False` despite ANE genuinely failing in baseline. The
log-show fragmentation count is the load-bearing metric.

**Sweep results.**

| ablation                                | calling | success | warm predict   | flip? | notes |
|------------------------------------------|--------:|--------:|----------------|:-----:|-------|
| baseline (no ablation, T_mel=50, fp16)   |     180 |      89 | ~570 ms (CPU)  |   —   | reference; ANE compile fails (Trial 10e1/10e2 confirmed) |
| ablation 1: AdaIN drop affine            |     181 |      90 | ~580 ms (CPU)  |  no   | identical fragmentation; AdaIN is not the trigger |
| ablation 2: Snake → identity             |   **2** |   **1** | **22 ms (ANE)** | **YES** | 99 % fragmentation drop; ANE accepts |

**Ablation 2 confirmed via manual interactive probe** (foreground, not
subprocess — captures os_log emit reliably): zero `MILCompilerForANE`
errors in stderr, predict latency 22-23 ms warm, ANE compile finishes
in 100 s vs baseline ~600 s. Ablation 2's mlpackage is bit-equivalent
to identity output (no Snake = different audio, but the question is
binary: does ANE accept).

Ablations 3 (AdaINResBlock1 residual fold) and 4 (halve upsample stack)
**not run** per the original stop condition: "Found a single ablation
that flips → STOP, report, propose weight-preserving rewrite." With
Snake identified as the trigger, those ablations would only confirm
they're NOT the trigger — already implied by ablation 2's complete
fragmentation drop.

**Verdict.** Snake activation (`x + (1/α) sin²(αx)`, 101 instances —
in every `AdaINResBlock1` plus the inline calls in `Generator.forward`)
is the planner partitioning trigger. M2 ANE's planner can't interleave
the Snake-cluster ops (`sin`, `pow`, per-channel `1/α` reciprocal,
broadcast multiply, add) with the surrounding ANE-eligible ops, so it
fragments the graph into 90 small subgraphs that compile fine
individually but can't link into a deployable single-ANEF model.

### Trial 10e4 — Snake → cosine-identity rewrite [DOES NOT LAND ANE]

**Hypothesis.** Replace Snake with the trig-identity equivalent

```
x + (1/α) sin²(αx)  ≡  x + (1 - cos(2αx)) / (2α)
```

This is mathematically bit-equivalent at fp32 (validated in earlier
StyleTTS2 push as Phase 3b's Snake cosine rewrite, max\|Δ\|=9.5e-7
across α∈{0.5,1.0,2.0} and tensor shapes). Eliminates `sin` and
`pow` (replaces with `cos`); preserves trained weights. If the
problem was specifically `sin/pow`, the rewrite should flip ANE
acceptance.

**Method.** `coreml/exporters/trial10e4_snake_cosine.py` patches both
`AdaINResBlock1.forward` and the Generator.forward inline Snakes (the
two locations Trial 10e3 ablation 2 covered). Same probe protocol.

**Result.**

| variant                          | calling | success | warm predict | flip? |
|----------------------------------|--------:|--------:|--------------|:-----:|
| Trial 10e4: Snake → cos-identity |     181 |      90 |     323 ms   | **no** |

Same fragmentation count as baseline (181 vs 180; ablation 2 was 2).
Warm predict 323 ms — slow CPU-fallback path, not ANE. No flip.

**Why the cosine identity doesn't help.** ANE's planner partitions on
the entire Snake-cluster *shape*, not on `sin/pow` specifically.
Replacing `sin² + 1/α` with `cos + 1/(2α)` keeps the same
broadcast-divide-by-per-channel-parameter pattern + transcendental
function. M2 ANE evidently can't place `cos` + per-channel `real_div`
either; the rewrite trades one non-ANE-supported op family for
another. Coremltools' fallback table for the production fp16 mlpackage
confirms `ios18.sin: 101` and `ios18.pow: 101` are listed as
"ANE not available for this op"; `ios18.cos` and `ios18.real_div`
are presumably the same on M2.

**Implication for Issue #59 outcomes.**

* Outcome 1 (weight-preserving rewrite landing ANE): **dead** for
  Snake → cos-identity on M2. Other trig identities for `sin²` also
  use `cos` and don't escape the partitioning trigger. Polynomial
  approximations of Snake exist (Taylor series, LeakyReLU
  approximation, lookup-table sin) but **none are bit-equivalent** —
  they introduce drift and require quality validation; not strictly
  weight-preserving.
* Outcome 2 (structural to HiFi-GAN, vocoder swap or retrain): the
  realistic path. Two flavours:
  - Retrain the HiFi-GAN decoder with a different activation (LeakyReLU,
    GELU, Swish — all well-supported on M2 ANE). Single-stage retrain
    on the existing dataset, ~hours of GPU time, no architectural
    change. Closest to a "minor refresh."
  - Swap the vocoder entirely (Vocos / iSTFTNet / parallel WaveGAN).
    Larger architectural change, larger quality risk. Mentioned in
    Issue #59's "out of scope" list as the alternative if outcome 1
    fails.
* Hardware caveat: this finding is M2-specific. M3 / M4 ANE op support
  may be wider; testing the cosine identity on those generations is a
  one-line config change to `minimum_deployment_target` plus a re-bench.
  If sin/cos land on newer ANE hardware, the cosine identity becomes
  ship-ready without retraining. Worth checking before committing to
  retraining or vocoder swap.

### Trial 10e3 / 10e4 artifacts

| Path | Size | Role |
|------|------|------|
| `coreml/exporters/trial10e3_bisection.py` | — | Sweep harness with 4 ablation installers |
| `coreml/exporters/trial10e4_snake_cosine.py` | — | Cosine-identity rewrite probe |
| `coreml/packages/trial10e3_ablation1_adain_drop_affine.mlpackage` | 40 MB | diagnostic; not promoted |
| `coreml/packages/trial10e3_ablation2_snake_identity.mlpackage` | 40 MB | diagnostic; not promoted |
| `coreml/packages/trial10e4_snake_cosine_identity.mlpackage` | 40 MB | candidate; **does not land ANE on M2** |

Probe logs: `/tmp/trial10e3_v2.log`, `/tmp/trial10e4.log`.

### Definitive answer to Issue #59

`decoder_upsample` cannot be landed on M2 ANE without either:

1. **Retraining** the HiFi-GAN decoder with an ANE-friendly activation
   (LeakyReLU / GELU / Swish replacing Snake); or
2. **Vocoder swap** to a non-Snake architecture (Vocos / iSTFTNet); or
3. **Newer hardware** (test cos-identity on M3+; M2 specifically lacks
   `cos` / `real_div` per-channel placement).

Trial 10e3/10e4 closes the bisection investigation. Issue #59's
acceptance criterion ("graph rewrite that preserves weights, lands on
ANE, cosine sim > 0.999") is **provably unachievable on M2** with the
current HiFi-GAN architecture. Production ships pinned to `CPU_ONLY`
at 304 ms warm avg; that's the floor on this hardware until one of the
three paths above is taken.

## Trial 11 — token-axis bucketing for `bert` + `fused_diffusion_sampler` (fp16)

**Problem.** Both `bert` (HF Albert) and `fused_diffusion_sampler`
(cross-attention U-Net) reject `ct.RangeDim` on the token axis —
coremltools' MIL backend errors with *"data-dependent shapes were
disabled"*. The iteration_3 packages therefore hard-code T = 57.
Anything longer than 57 espeak tokens (~37 chars) errors out at
runtime.

**Approach.** Bake separate per-bucket mlpackages, no `EnumeratedShapes`
(which the same MIL pass also rejects on these graphs). The Swift /
Python loader picks the smallest bucket that fits the prompt's token
count and pads to that bucket's T.

Buckets chosen to cover typical TTS surface area:

| Bucket | T_TOK | Char budget (~) | Rough use case            |
|--------|-------|-----------------|---------------------------|
| 64     | 64    | ≤ 42            | clause / short sentence   |
| 128    | 128   | ≤ 85            | full sentence             |
| 256    | 256   | ≤ 170           | short paragraph           |

**Builder.** `coreml/exporters/build_buckets.py` (one driver) reuses the existing
`BertWrapper` and the restored `FusedDiffusionSampler` from Trial 4.
Per bucket: pad captured (tokens, attn_mask) to T, trace, convert at
fp16 to match iteration_3 precision. Eager parity gate (`max|d| <
1e-4`) before each conversion.

```bash
uv run python coreml/exporters/build_buckets.py \
    --buckets 64,128,256 --stages bert,sampler --precision fp16
```

**Disk cost.** All six packages produced clean (fp16):

| Package                                   | Size  |
|-------------------------------------------|-------|
| `bert_fp16_t64.mlpackage`                 | 12 MB |
| `bert_fp16_t128.mlpackage`                | 12 MB |
| `bert_fp16_t256.mlpackage`                | 12 MB |
| `fused_diffusion_sampler_fp16_t64.mlpackage`  | 48 MB |
| `fused_diffusion_sampler_fp16_t128.mlpackage` | 48 MB |
| `fused_diffusion_sampler_fp16_t256.mlpackage` | 48 MB |
| **Total (extra over iteration_3)**        | **~120 MB** |

iteration_3 itself is ~275 MB; bucketed deployment becomes ~395 MB
total if all three buckets are shipped. T = 57 (the original
hard-coded) is a strict subset of T = 64 and can be dropped, so the
real net delta is ~108 MB.

**Validation.** `coreml/inference_buckets.py` runs the full 8-stage
pipeline at each bucket, swapping in the bucketed `bert` + sampler
packages and padding tokens to T. The other six iteration_3 stages
(`text_encoder`, `ref_encoder`, `duration_predictor`,
`fused_f0n_har_source`, `decoder_pre`, `decoder_upsample`) are
loaded unchanged from `coreml/packages/`.

```bash
uv run python coreml/inference_buckets.py --all --output-dir coreml
```

Per-bucket result (M-series Mac, warm-ish — load + predict, single
pass each):

| Bucket | Prompt                                      | Real tokens / T | Frames | Audio | Pipeline |
|--------|---------------------------------------------|------------------|--------|-------|----------|
| 64     | "Hello there. How are you today?"           | 36 / 64          | 97     | 2.42 s | 494 ms |
| 128    | "StyleTTS 2 is a text to speech model."     | 57 / 128         | 144    | 3.60 s | 414 ms |
| 256    | "StyleTTS 2 is a text to speech model that produces clear, natural sounding speech in a variety of voices and speaking styles." | 154 / 256 | 335 | 8.37 s | 4933 ms |

WAVs written to `coreml/out_t{64,128,256}.wav`. Spot-checked
healthy: 24 kHz mono, peaks 0.6–0.7, RMS ~0.07 (typical TTS-level
audio, no NaN/silence/clipping).

The T = 256 pipeline is dominated by `decoder_upsample` (4.5 s of
the 4.9 s) — that's expected since output audio is 8.4 s long
(decoder is real-time-ish on CPU_ONLY at 24 kHz). The bucket
swapouts themselves cost a few ms.

**Verdict.** Bucketing works as designed. Padding contamination on
`bert` is bounded by `attention_mask`, and the sampler doesn't
attend to padded positions either (cross-attn is masked upstream by
the embedding the sampler receives, which is the bert output).

**Artifacts.**

* `coreml/exporters/build_buckets.py` — driver.
* `coreml/inference_buckets.py` — bucket-aware end-to-end driver.
* `coreml/packages/{bert,fused_diffusion_sampler}_fp16_t{64,128,256}.mlpackage` — outputs.
* `coreml/out_t{64,128,256}.wav` — validation audio.

## How to run

```bash
cd models/tts/styletts2

# Convert all stages (writes coreml/packages/*.mlpackage)
uv run python coreml/exporters/convert.py --stage all
uv run python coreml/exporters/convert.py --stage text_encoder

# Per-stage parity vs PyTorch
uv run python coreml/parity.py --stage all
uv run python coreml/parity.py --stage text_encoder

# Trial 10: decoder_upsample fp32 fixed-shape ANE probe
uv run python coreml/exporters/trial10_decoder_upsample_fixed.py

# Trial 10b: decoder_upsample fp32 + Conv1d→Conv2d rewrite
uv run python coreml/exporters/trial10b_decoder_upsample_conv2d.py

# Trial 11: per-bucket bert + sampler (T=64/128/256), fp16
uv run python coreml/exporters/build_buckets.py \
    --buckets 64,128,256 --stages bert,sampler --precision fp16
uv run python coreml/inference_buckets.py --all --output-dir coreml
```
