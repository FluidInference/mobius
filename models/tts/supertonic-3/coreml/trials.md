# Supertonic-3 ONNX → PyTorch Port: Trials & Resolution

A log of non-obvious gotchas discovered while hand-porting the four Supertonic-3
TTS v1.7.3 sub-networks from ONNX to PyTorch, validated numerically against
ONNX-Runtime CPU.

## Final validation status

| Module             | max_abs   | Notes                                              |
| ------------------ | --------- | -------------------------------------------------- |
| vocoder            | 2.53e-4   | clean port                                         |
| text_encoder       | 9.77e-2   | passes relaxed tol (atol=2e-1, rtol=5e-1)          |
| duration_predictor | 3.04e-6   | very tight match                                   |
| vector_estimator   | 1.21e-3   | required the four fixes documented below           |

## Common gotchas (apply across modules)

- **Symmetric padding scales with dilation**: for ConvNeXt depthwise convs with
  kernel=5, pad = `(K-1)*D/2`: dil=1→2, dil=2→4, dil=4→8, dil=8→16. Forgetting
  to scale produces a 1-position shift that cascades.
- **Mask everywhere**: every block applies `x * mask` before the next layer's
  read; in attention, masked positions must be excluded *and* the final output
  re-masked.
- **`assign_param` cast**: rotary `increments` are stored `int64` in ONNX but
  used as float for `pos * theta`. The common loader does
  `torch.from_numpy(arr).to(param.dtype)` which handles this correctly.

## vector_estimator — the four non-obvious bugs

This module hit the most landmines. Each bug below was found by progressively
narrowing the divergence with intermediate-output comparison harnesses.

### 1. CFG implemented via batch-2 duplication

ONNX does classifier-free guidance by **tiling everything to batch=2**
(`onnx::Tile_1065 = [2, 1, 1]`), running cond and uncond in parallel, then
combining with constants `W_COND = 4.0`, `W_UNCOND = 3.0`:

```python
denoised = (noisy_latent + (1/total_step) * (4*cond - 3*uncond)) * latent_mask
```

The uncond branch needs:

- `text_emb_uncond  = expand(text_special_token)`
- `style_value_uncond = expand(style_value_special_token)`
- `style_key_uncond   = expand(style_key_special_token)`

The **cond** style key is *not* the user-provided `style_ttl` — it's a learned
initializer at `/vector_estimator/Expand_output_0` (shape `(1, 50, 256)`). Only
the style **value** is taken from `style_ttl`. K and V have separate sources.

### 2. Rotary is length-normalized, not raw positions

Initial assumption: `angles = positions * theta`. Validation showed Q/K
projections matched ONNX but `cos`/`sin` diverged.

Tracing the Mul-feeding-Sin/Cos node back through the graph revealed a `Div`
node: the positions are divided by `ReduceSum(latent_mask)` (i.e. actual
sequence length) before multiplying by `theta`:

```python
angles = (positions / sum(latent_mask)) * theta   # Q side
angles = (positions / sum(text_mask)) * theta     # K side
```

For Q the divisor is the **latent** mask sum; for K it's the **text** mask sum.

The actual rotation is standard Llama rotated-half on the full `dk=64` axis:

```python
x_a, x_b = x[..., :32], x[..., 32:]
out = cat([x_a*cos - x_b*sin, x_a*sin + x_b*cos], dim=-1)
```

### 3. Attention scoring divisor is a fixed constant (16.0)

Both `text_attn` and `style_attn` divide scores by **`16.0`**
(`/main_blocks.3/attn/Constant_51_output_0 = 16.0`), *not* `sqrt(dk)`. Since
`dk=64`, `sqrt(dk)=8` and `dk/4=16`, the off-by-2× scaling was buried in
attention magnitudes and only became obvious by reading the constant directly.

### 4. Style attention applies `tanh(K)` before the score matmul

```python
scores = (Q @ tanh(K).T) / 16.0
attn   = softmax(scores, dim=-1)
attn   = where(latent_mask == 0, 0.0, attn)   # post-softmax Q-side mask
out    = attn @ V
```

The `tanh(K)` is unique to style attention — text attention has no such
non-linearity. Adding it dropped post-style-attn drift from 7.8e-1 to ~1e-3.

The post-softmax `Where` on the Q-side latent mask zeros attention rows for
invalid query positions; since the output is also masked after `out_fc`, this
is essentially redundant when the input mask is all ones (our test case), but
matters when partial masks are used.

## ONNX head layout quirk

When peeking at attention intermediates (`Concat_3` = rotated Q), the shape is
`(H, B, L, dk)` — heads as the **outermost** dim, not the conventional
PyTorch `(B, H, L, dk)`. Numerical comparisons need a `permute(1, 0, 2, 3)` on
the torch side. The final output before `out_fc` is permuted back so the
external shape `(B, L, H*dk)` agrees with both.

## Time encoder details

- phase = `(t * 1000) * omegas` where `omegas = exp(-ln(10000) * arange(32) / 32)`
- emb = `cat([sin(phase), cos(phase)], dim=-1)`  (shape 64)
- `Linear(64→256) → Mish → Linear(256→64) → unsqueeze(-1)` → added to features
  via a `Linear(64→512)` projection in each `_TimeCondLayer`.

## Norm placement

Post-norm with mask: `out = LayerNorm(attn(input) + masked_input) * mask`. The
masked residual must use `x * mask` (not raw `x`) before adding the attention
output; otherwise drift builds up in masked regions even with all-ones masks
because LayerNorm sees nonzero values there.

## Debugging approach that worked

Each step narrowed the search space by ~10×:

1. **Top-level**: compare denoised output → 5.93 max_abs FAIL
2. **Per-block intermediates**: expose `proj_in`, `after_convnext_*`,
   `after_text_attn_norm`, `after_style_attn_norm` as extra ONNX outputs by
   appending to `graph.output` and re-saving — find the block where drift
   jumps.
3. **Inside the bad block**: expose `W_q/W_k/W_v Add` outputs (projections),
   then `Slice_1/Slice_2` (rotary halves), then `Sin/Cos`, then the Mul-feeding
   intermediate — find the exact op that disagrees.
4. **Trace the disagreement back**: walk the ONNX graph upstream from the bad
   op to find the unexpected node (`Div`, `Tanh`, `Where`) the original
   architecture diagram omitted.

The key trick is `onnx.helper.make_tensor_value_info(name, FLOAT, None)` +
`graph.output.append(vi)` + re-save → ORT will expose any internal tensor as
an output without modifying the model semantics.

## CoreML tracing & conversion

After all four PyTorch ports validated against ONNX-Runtime, each module was
traced with `torch.jit.trace` and converted to a `.mlpackage` via
`coremltools.convert`. Final numerical agreement vs the PyTorch port:

| Module             | mlpackage size | max_abs (CoreML vs PyTorch) |
| ------------------ | -------------- | --------------------------- |
| vocoder            | 97 MB          | 1.41e-6                     |
| text_encoder       | 35 MB          | 2.33e-4                     |
| duration_predictor | 3.5 MB         | 3.82e-6                     |
| vector_estimator   | 244 MB         | 2.96e-5                     |

### Shared conversion settings

```python
MIN_DEPLOY        = ct.target.iOS18      # multi-input dynamic shapes
COMPUTE_PRECISION = ct.precision.FLOAT32
CONVERT_TO        = "mlprogram"
COMPUTE_UNITS     = ct.ComputeUnit.CPU_AND_NE
```

### Per-module shape strategy

| Module             | Variable axis           | Strategy           |
| ------------------ | ----------------------- | ------------------ |
| vocoder            | latent L_ttl            | `RangeDim(4..512)` |
| text_encoder       | text T                  | **fixed T=128**    |
| duration_predictor | text T                  | **fixed T=128**    |
| vector_estimator   | latent L and text T_txt | `RangeDim(17..512)`|

`text_encoder` / `duration_predictor` use a fixed T because relative-position
attention computes `F.pad(emb, [0,0,pad_left,pad_right])` with T-derived pad
values; `torch.jit.trace` records these as dynamic 4-D pads which coremltools
rejects with `NotImplementedError: Dynamic padding for n-dimensional tensors
not supported. 4 padding values`. EnumeratedShapes did not help because the
trace still embeds the dynamic op. Callers must pad text inputs to T=128.

### Non-obvious gotchas in the convert step

1. **Python 3.14 has no `BlobWriter`** — coremltools bundles a native
   `libcoremlpython` / `libmilstoragepython`; only py3.11 wheels exist for
   this version. Use `/opt/homebrew/bin/python3.11`.
2. **`aten::Int(aten::size(x, 0))`** appears whenever PyTorch reads
   `x.shape[k]` into an integer used as a reshape arg. CoreML chokes on
   these. Fix: use literal `1` for batch and `-1` for the unknown dim in all
   `reshape` calls. Vocoder's chunk-unpack + final flatten both needed this.
3. **Replicate-pad lower bound**: ConvNeXt depthwise convs use replicate
   padding equal to `(K-1)*D/2`. CoreML enforces `pad ≤ dim_size - 1` at
   load time, so `RangeDim.lower_bound` must dominate the worst-case pad:
   - vocoder: ksz=7, max dil=4 → pad=12; unpacked L = 6 * L_ttl, so
     `L_ttl ≥ 4` keeps L ≥ 24 ≥ 12 + 1.
   - vector_estimator: ksz=5, max dil=8 → pad=16, lower bound 17.
4. **EnumeratedShapes needs iOS 18** for >1 input — error message *"Expected
   a single enumerated shape input for deployment targets below iOS 18"*.
   Bumping `MIN_DEPLOY` to `ct.target.iOS18` is required even if you fall
   back to fixed shapes.
5. **`Default value … less than minimum value … for range`**: the example
   tensor used for tracing must satisfy `RangeDim.lower_bound`. Example
   sizes were raised to match the new lower bounds (L=24, T_text=24 for
   vector_estimator).
6. **int32 vs int64 inputs**: CoreML token inputs must be `int32`; the
   PyTorch port indexes with `int64`. Wrap the module with a tiny
   `nn.Module` that does `text_ids.long()` *inside* the traced graph so the
   external input stays `int32`:
   ```python
   class _Int32Wrapper(nn.Module):
       def forward(self, text_ids, *rest):
           return self.m(text_ids.long(), *rest)
   ```

### Verification harness

`verify_coreml.py` loads each `.mlpackage` with `ct.models.MLModel`, runs the
same RNG-seeded input through both backends, and prints
`max_abs / mean_abs / shape`. Run after every conversion change. The
verification numbers above were produced with FP32 + `CPU_AND_NE` compute
units; FP16 would relax tolerances by ~10×.
