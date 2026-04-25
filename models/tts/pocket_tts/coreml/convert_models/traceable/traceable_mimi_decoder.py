"""Traceable Mimi streaming decoder for CoreML conversion.

Flattens the stateful Mimi decoder into explicit input/output tensors
so it can be traced with torch.jit.trace and converted to CoreML.

Input:  latent [1, 32] + 24 state tensors
Output: audio [1, 1, 1920] + 24 updated state tensors

The model internally denormalizes and quantizes the 32-dim latent:
  denorm = latent * emb_std + emb_mean        [1, 32]
  quantized = Conv1d(denorm, 32→512)           [1, 512, 1]
  audio = mimi_decode(quantized, state)        [1, 1, 1920]

NOTE: The original Mimi model uses in-place tensor mutations (state[:] = ...)
in StreamingConv1d, StreamingConvTranspose1d, and StreamingMultiheadAttention.
torch.jit.trace cannot handle in-place ops, so this module monkey-patches them
with functional equivalents before tracing.

V2 (pocket_tts >= 2.0.0): the attention cache schema dropped `end_offset`
(only `offset` + `cache` remain), cache layout is `[2, B, T, H, D]` (T moved
before H), and `MimiStreamingMultiheadAttention` was merged into the shared
`StreamingMultiheadAttention` in `pocket_tts.modules.transformer`.
"""
import torch
import torch.nn as nn


# Ordered list of (state_name, shape) for the 24 Mimi streaming state tensors.
# Must match the manifest.json order used by the Swift loader.
MIMI_STATE_SPEC = [
    ("upsample_partial", [1, 512, 16]),
    # Cache capacity 256 with modulo-wrap inside `_functional_attention_forward`
    # (Trial 12). The decoder transformer offset bumps by 16 per latent frame
    # (upsample stride), so the cache wraps every 16 frames. Trial 13's
    # capacity-bump-to-4096 alternative regressed german/italian/portuguese,
    # so we keep the wrap path. See TRIALS.md.
    ("attn0_cache", [2, 1, 256, 8, 64]),
    ("attn0_offset", [1]),
    ("attn1_cache", [2, 1, 256, 8, 64]),
    ("attn1_offset", [1]),
    ("conv0_prev", [1, 512, 6]),
    ("conv0_first", [1]),
    ("convtr0_partial", [1, 256, 6]),
    ("res0_conv0_prev", [1, 256, 2]),
    ("res0_conv0_first", [1]),
    ("res0_conv1_prev", [1, 128, 0]),
    ("res0_conv1_first", [1]),
    ("convtr1_partial", [1, 128, 5]),
    ("res1_conv0_prev", [1, 128, 2]),
    ("res1_conv0_first", [1]),
    ("res1_conv1_prev", [1, 64, 0]),
    ("res1_conv1_first", [1]),
    ("convtr2_partial", [1, 64, 4]),
    ("res2_conv0_prev", [1, 64, 2]),
    ("res2_conv0_first", [1]),
    ("res2_conv1_prev", [1, 32, 0]),
    ("res2_conv1_first", [1]),
    ("conv_final_prev", [1, 64, 2]),
    ("conv_final_first", [1]),
]

# Mapping from MIMI_STATE_SPEC names to (module_path, key) in the nested state dict.
# Paths are the absolute `_module_absolute_name` assigned by TTSModel.load_model at
# `mimi/<path>` — e.g. `upsample.convtr`, `decoder.model.0`,
# `decoder_transformer.transformer.layers.0.self_attn`.
_SPEC_TO_NESTED = [
    ("upsample_partial", "upsample.convtr", "partial"),
    ("attn0_cache", "decoder_transformer.transformer.layers.0.self_attn", "cache"),
    ("attn0_offset", "decoder_transformer.transformer.layers.0.self_attn", "offset"),
    ("attn1_cache", "decoder_transformer.transformer.layers.1.self_attn", "cache"),
    ("attn1_offset", "decoder_transformer.transformer.layers.1.self_attn", "offset"),
    ("conv0_prev", "decoder.model.0", "previous"),
    ("conv0_first", "decoder.model.0", "first"),
    ("convtr0_partial", "decoder.model.2", "partial"),
    ("res0_conv0_prev", "decoder.model.3.block.1", "previous"),
    ("res0_conv0_first", "decoder.model.3.block.1", "first"),
    ("res0_conv1_prev", "decoder.model.3.block.3", "previous"),
    ("res0_conv1_first", "decoder.model.3.block.3", "first"),
    ("convtr1_partial", "decoder.model.5", "partial"),
    ("res1_conv0_prev", "decoder.model.6.block.1", "previous"),
    ("res1_conv0_first", "decoder.model.6.block.1", "first"),
    ("res1_conv1_prev", "decoder.model.6.block.3", "previous"),
    ("res1_conv1_first", "decoder.model.6.block.3", "first"),
    ("convtr2_partial", "decoder.model.8", "partial"),
    ("res2_conv0_prev", "decoder.model.9.block.1", "previous"),
    ("res2_conv0_first", "decoder.model.9.block.1", "first"),
    ("res2_conv1_prev", "decoder.model.9.block.3", "previous"),
    ("res2_conv1_first", "decoder.model.9.block.3", "first"),
    ("conv_final_prev", "decoder.model.11", "previous"),
    ("conv_final_first", "decoder.model.11", "first"),
]


# ---------------------------------------------------------------------------
# Functional (no in-place) forward replacements for tracing
# ---------------------------------------------------------------------------

def _functional_streaming_conv1d_forward(self, x, model_state):
    """StreamingConv1d.forward without in-place tensor mutations."""
    B, C, T = x.shape
    S = self._stride
    assert T > 0 and T % S == 0
    if model_state is None:
        state = self.init_state(B, 0)
    else:
        state = self.get_state(model_state)
    # Derive TP from static module attributes (Python int) rather than the
    # state tensor's shape. PyTorch 2.9 otherwise routes tensor.shape[-1]
    # through prim::NumToTensor, producing an aten::Int node in the trace
    # that coremltools fails to fold into a compile-time constant.
    TP = self._effective_kernel_size - self._stride
    previous = state["previous"]
    first = state["first"]

    if TP and self.pad_mode == "replicate":
        assert T >= TP
        init = x[..., :1]
        previous = torch.where(first.view(-1, 1, 1), init, previous)

    if TP:
        x = torch.cat([previous, x], dim=-1)
    y = self.conv(x)
    if TP:
        state["previous"] = x[..., -TP:]          # dict assign (not [:]=)
        if self.pad_mode == "replicate":
            state["first"] = torch.zeros_like(first)
    return y


def _functional_streaming_conv_transpose1d_forward(self, x, mimi_state):
    """StreamingConvTranspose1d.forward without in-place tensor mutations."""
    layer_state = self.get_state(mimi_state)
    partial = layer_state["partial"]
    y = self.convtr(x)
    # Derive PT from static module attributes (Python int) rather than the
    # state tensor's shape. PyTorch 2.9 otherwise routes tensor.shape[-1]
    # through prim::NumToTensor, producing an aten::Int node for the
    # `y[..., :-PT]` slice that coremltools fails to fold.
    PT = self.convtr.kernel_size[0] - self.convtr.stride[0]
    if PT > 0:
        # Overlap-add without in-place (no += or [:]=)
        y = torch.cat([y[..., :PT] + partial, y[..., PT:]], dim=-1)
        # Save new partial
        new_partial = y[..., -PT:]
        bias = self.convtr.bias
        if bias is not None:
            new_partial = new_partial - bias[:, None]
        layer_state["partial"] = new_partial       # dict assign (not [:]=)
        y = y[..., :-PT]
    return y


def _functional_apply_rope(q, k, offset, D, max_period=10000.0):
    """RoPE with a Python-int D to keep shape-derived values out of the trace.

    Upstream `apply_rope` does `B, T, H, D = q.shape` and then `2 / D`, which
    PyTorch 2.9's tracer captures as `aten::reciprocal(D_as_long_tensor)`.
    coremltools then sees an int32 input to its `inverse` op and rejects it.
    Passing D explicitly keeps the expression purely scalar.
    """
    import math

    B, T, H, _ = q.shape
    D_half = D // 2
    ds = torch.arange(D_half, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2.0 / D))

    ts = torch.arange(T, device=q.device, dtype=torch.float32)
    ts = ts + offset  # offset is a 0-d long tensor; broadcasting promotes to fp32
    ts = ts.view(-1, 1, 1)

    q = q.view(B, T, H, D_half, 2)
    k = k.view(B, T, H, D_half, 2)

    qr = q[..., 0].float()
    qi = q[..., 1].float()
    kr = k[..., 0].float()
    ki = k[..., 1].float()

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
    return qo.view(B, T, H, D), ko.view(B, T, H, D)


def _functional_attention_forward(self, query, model_state):
    """v2 StreamingMultiheadAttention.forward — no in-place KV writes, float scale.

    Matches upstream `pocket_tts.modules.transformer.StreamingMultiheadAttention.forward`
    semantics but with two tracing-friendly changes:

    1. The KV cache update is done with `torch.scatter` on a cloned cache instead
       of `cache[0, :, offset:offset+T] = k` (which torch.jit.trace can't capture).
    2. Manual attention (q @ k^T * scale) instead of F.scaled_dot_product_attention,
       because SDPA derives 1/sqrt(d) as an int32 reciprocal that CoreML rejects.

    State schema (v2): `{"cache": [2, B, T_cap, H, D], "offset": [B]}` — no end_offset.
    The caller is responsible for advancing `offset` after forward (via
    `increment_step`, mirrored by our wrapper's post-forward offset bump).
    """
    import math

    B, T, _ = query.shape
    H = self.num_heads
    D = self.dim_per_head

    state = None if model_state is None else self.get_state(model_state)
    if state is None:
        offset = torch.zeros(B, device=query.device, dtype=torch.long)
    else:
        offset = state["offset"]

    # In-proj and split into Q/K/V: [B, T, 3, H, D]
    projected = self.in_proj(query)
    packed = projected.view(B, T, 3, H, D)
    q, k, v = torch.unbind(packed, dim=2)  # each [B, T, H, D]

    # RoPE uses a scalar offset (matches v2's _LinearKVCacheBackend.rope_offset).
    rope_offset = offset.view(-1)[0]
    max_period = float(getattr(self.rope, "max_period", 10000.0))
    q, k = _functional_apply_rope(q, k, rope_offset, D, max_period)

    # Transform Q to attention layout [B, H, T, D]
    q_attn = q.transpose(1, 2)

    # Functional KV cache update with modulo-wrap (Trial 12, see TRIALS.md).
    # Cache capacity is fixed at 256 (MIMI_STATE_SPEC) but the decoder
    # transformer offset bumps by 16 per latent frame (upsample stride), so
    # after 16 frames the offset reaches 256 and writes have to wrap modulo
    # capacity. The attention mask is rebuilt over absolute logical positions
    # so each slot still attends with the correct distance/context.
    if state is None:
        k_attn = k.transpose(1, 2)
        v_attn = v.transpose(1, 2)
        pos_k = torch.arange(k.shape[1], device=q.device, dtype=torch.long)
        pos_k = pos_k.view(1, -1).expand(B, -1)
    else:
        cache = state["cache"]  # [2, B, T_cap, H, D]
        capacity = cache.shape[2]

        # Modulo-wrap writes: slot = (offset + t) % capacity.
        write_base = offset.long().view(B, 1)
        write_range = torch.arange(T, device=q.device, dtype=torch.long).view(1, T)
        abs_idx = write_base + write_range  # [B, T] absolute logical positions
        wrapped = abs_idx % capacity  # CoreML-friendly mod via `%`
        write_indexes = wrapped.view(B, T, 1, 1).expand(-1, -1, H, D)
        new_k_cache = cache[0].scatter(1, write_indexes, k)
        new_v_cache = cache[1].scatter(1, write_indexes, v)
        state["cache"] = torch.stack([new_k_cache, new_v_cache])

        # Replace NaN fill (init_state uses NaN) with 0 so masked-out lanes don't
        # produce NaN * 0 in softmax. The mask below ensures they don't contribute.
        new_k_cache = torch.where(torch.isnan(new_k_cache), torch.zeros_like(new_k_cache), new_k_cache)
        new_v_cache = torch.where(torch.isnan(new_v_cache), torch.zeros_like(new_v_cache), new_v_cache)

        k_attn = new_k_cache.permute(0, 2, 1, 3)  # [B, H, T_cap, D]
        v_attn = new_v_cache.permute(0, 2, 1, 3)

        # Reconstruct each slot's logical position. With wrap, slot i holds
        # the K/V for logical position p where `p % capacity == i` and p is
        # the largest such value <= last_pos. Equivalent to:
        #   slot_logical_pos[i] = last_pos - ((last_pos - i) % capacity)
        last_pos = (offset.view(B, 1) + (T - 1)).long()  # [B, 1]
        slot_idx = torch.arange(capacity, device=q.device, dtype=torch.long).view(1, -1)
        diff = last_pos - slot_idx  # [B, capacity]
        pos_k = last_pos - (diff % capacity)  # [B, capacity]

    # Build attention mask over absolute logical positions. A slot is valid
    # iff its logical pos has been written (pos_k <= last_pos) and within
    # the sliding-window context (delta >= 0 and delta < context).
    pos_q = offset.view(B, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(1, T)
    delta = pos_q[:, :, None] - pos_k[:, None, :]
    valid = pos_k[:, None, :] >= 0
    valid = valid & (pos_k[:, None, :] <= (offset.view(B, 1, 1) + (T - 1)))
    attn_mask = valid & (delta >= 0)
    if self.context is not None:
        attn_mask = attn_mask & (delta < self.context)
    attn_mask = attn_mask[:, None]  # [B, 1, T, T_cap]

    # Manual attention with float scale.
    scale = 1.0 / math.sqrt(float(D))
    attn = torch.matmul(q_attn, k_attn.transpose(-2, -1)) * scale
    attn = attn.masked_fill(~attn_mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    x = torch.matmul(attn, v_attn)  # [B, H, T, D]

    x = x.transpose(1, 2).reshape(B, T, H * D)
    x = self.out_proj(x)
    return x


def _patch_for_tracing(module):
    """Monkey-patch all in-place ops in the module tree for torch.jit.trace."""
    import types
    from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
    from pocket_tts.modules.transformer import StreamingMultiheadAttention

    for name, child in module.named_modules():
        if isinstance(child, StreamingConv1d):
            child.forward = types.MethodType(
                _functional_streaming_conv1d_forward, child
            )
        elif isinstance(child, StreamingConvTranspose1d):
            child.forward = types.MethodType(
                _functional_streaming_conv_transpose1d_forward, child
            )
        elif isinstance(child, StreamingMultiheadAttention):
            child.forward = types.MethodType(
                _functional_attention_forward, child
            )


# ---------------------------------------------------------------------------
# Main traceable wrapper
# ---------------------------------------------------------------------------

class TraceableMimiDecoder(nn.Module):
    """Wrapper that exposes Mimi's streaming state as flat tensor I/O.

    Accepts a raw 32-dim latent and internally applies denormalization
    (latent * emb_std + emb_mean) and quantization (Conv1d 32→512)
    before feeding into the Mimi streaming decoder.

    State tensors are ordered according to MIMI_STATE_SPEC (matching
    the manifest.json order expected by the Swift loader).
    """

    def __init__(self, mimi_model, emb_mean, emb_std, quantize_proj):
        super().__init__()
        self.mimi = mimi_model
        self.register_buffer("emb_mean", emb_mean)
        self.register_buffer("emb_std", emb_std)
        self.quantize_proj = quantize_proj

        # Build the nested state dict from init_states on the full mimi model so
        # keys carry the absolute `_module_absolute_name` paths (`upsample.convtr`,
        # `decoder.model.0`, `decoder_transformer.transformer.layers.0.self_attn`),
        # which is what `StatefulModule.get_state` looks up at runtime.
        from pocket_tts.modules.stateful_module import init_states
        # sequence_length must match MIMI_STATE_SPEC's attn cache capacity
        # (Trial 12: 256 + modulo-wrap path; Trial 13's 4096 simplification
        # regressed multiple language packs).
        full_state = init_states(self.mimi, batch_size=1, sequence_length=256)
        # Keep only the decode-path entries referenced by _SPEC_TO_NESTED so we don't
        # carry encoder/downsample state that the decode pipeline never touches.
        wanted_modules = {module_name for _, module_name, _ in _SPEC_TO_NESTED}
        self._nested_state = {k: v for k, v in full_state.items() if k in wanted_modules}

        # Validate the mapping covers all referenced state entries
        nested_keys = set()
        for module_name, module_state in self._nested_state.items():
            for key in module_state:
                nested_keys.add((module_name, key))
        spec_keys = set((m, k) for _, m, k in _SPEC_TO_NESTED)
        assert nested_keys == spec_keys, (
            f"Mapping mismatch:\n"
            f"  In nested but not spec: {nested_keys - spec_keys}\n"
            f"  In spec but not nested: {spec_keys - nested_keys}"
        )

        # Attention module paths in model_state for offset increment in forward().
        # These match the module_name entries in _SPEC_TO_NESTED for offset keys.
        self._attn_module_names = [
            module_name
            for spec_name, module_name, key in _SPEC_TO_NESTED
            if key == "offset"
        ]

        # Upsample stride determines the offset increment per frame.
        # Each latent frame [1, 512, 1] is upsampled to [1, 512, stride] tokens,
        # so the decoder transformer attention processes `stride` tokens per call.
        self._upsample_stride = int(self.mimi.upsample.convtr.convtr.stride[0])

        # Patch in-place ops for tracing
        _patch_for_tracing(self.mimi)

    @classmethod
    def from_tts_model(cls, tts_model) -> "TraceableMimiDecoder":
        return cls(
            tts_model.mimi,
            emb_mean=tts_model.flow_lm.emb_mean,
            emb_std=tts_model.flow_lm.emb_std,
            quantize_proj=tts_model.mimi.quantizer.output_proj,
        )

    def _pack_state(self, flat_tensors: tuple) -> dict:
        """Convert flat tensor tuple (MIMI_STATE_SPEC order) into nested dict."""
        # Ensure all module_name keys exist in the dict
        state = {}
        for module_name in self._nested_state:
            state[module_name] = {}

        for i, (spec_name, module_name, key) in enumerate(_SPEC_TO_NESTED):
            state[module_name][key] = flat_tensors[i]
        return state

    def _unpack_state(self, state: dict) -> tuple:
        """Extract flat tensor tuple (MIMI_STATE_SPEC order) from nested dict.

        Several state tensors are pass-throughs (the `*_first` scalars after
        first frame, and the zero-length `res{0,1,2}_conv1_prev` tensors
        that are never written when their layer's kernel is empty). At the
        MIL level these outputs share an SSA value with the corresponding
        input parameter. We deliberately do NOT try to break that aliasing
        here — `clone()`, `+0`, etc. all get folded by `noop_elimination`
        anyway. Instead the converter's rename pass is taught to skip
        pass-through outputs (see `rename_outputs_semantic` in
        `convert_mimi_decoder.py`), and the Swift schema loader has a
        fallback that accepts the bare input name as the output for those
        tensors.
        """
        tensors = []
        for spec_name, module_name, key in _SPEC_TO_NESTED:
            tensors.append(state[module_name][key])
        return tuple(tensors)

    def forward(self, latent, *state_tensors):
        """
        Args:
            latent: [1, 32] raw latent frame
            *state_tensors: 24 flat state tensors (MIMI_STATE_SPEC order)

        Returns:
            audio: [1, 1, 1920] decoded audio frame
            *updated_states: 24 updated state tensors (MIMI_STATE_SPEC order)
        """
        # Denormalize: latent * std + mean
        denorm = latent * self.emb_std + self.emb_mean  # [1, 32]
        # Reshape to Conv1d input and quantize: [1, 32, 1] → [1, 512, 1]
        quantized = self.quantize_proj(denorm.unsqueeze(-1))
        model_state = self._pack_state(state_tensors)
        audio = self.mimi.decode_from_latent(quantized, model_state)
        # Functional increment_steps: advance attention offsets by upsample_stride.
        # Each latent frame is upsampled to `stride` encoder tokens, so the
        # decoder transformer attention processes `stride` tokens per frame.
        # The original code calls increment_steps(mimi, state, increment=16).
        for attn_name in self._attn_module_names:
            layer_state = model_state[attn_name]
            layer_state["offset"] = layer_state["offset"] + self._upsample_stride
        updated = self._unpack_state(model_state)
        return (audio,) + updated


def test_traceable_mimi():
    import sys
    import os
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    sys.path.insert(0, _project_dir)

    from pocket_tts import TTSModel

    print("Loading model...")
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    print("Creating traceable Mimi decoder...")
    traceable = TraceableMimiDecoder.from_tts_model(model)
    traceable.eval()

    # Build initial state from MIMI_STATE_SPEC
    print("Building initial state from MIMI_STATE_SPEC...")
    state_tensors = []
    for name, shape in MIMI_STATE_SPEC:
        state_tensors.append(torch.zeros(*shape))

    print(f"State tensors: {len(state_tensors)}")
    for i, (name, shape) in enumerate(MIMI_STATE_SPEC):
        print(f"  [{i}] {name}: {shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    latent = torch.randn(1, 32)
    with torch.no_grad():
        outputs = traceable(latent, *state_tensors)

    audio = outputs[0]
    print(f"Audio shape: {audio.shape}")
    print(f"Audio range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")
    print(f"Updated state tensors: {len(outputs) - 1}")
    print("Done!")


if __name__ == "__main__":
    test_traceable_mimi()
