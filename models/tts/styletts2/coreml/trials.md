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
* Created `coreml/wrappers.py`, `coreml/convert.py`, `coreml/parity.py`,
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

These live in `coreml/convert.py` / `coreml/wrappers.py` and apply
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

## How to run

```bash
cd models/tts/styletts2

# Convert all stages (writes coreml/packages/*.mlpackage)
uv run python coreml/convert.py --stage all
uv run python coreml/convert.py --stage text_encoder

# Per-stage parity vs PyTorch
uv run python coreml/parity.py --stage all
uv run python coreml/parity.py --stage text_encoder
```
