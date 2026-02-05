"""Traceable Mimi streaming decoder for CoreML conversion.

Flattens the stateful Mimi decoder into explicit input/output tensors
so it can be traced with torch.jit.trace and converted to CoreML.

Input:  latent [1, 32] + 26 state tensors
Output: audio [1, 1, 1920] + 26 updated state tensors

The model internally denormalizes and quantizes the 32-dim latent:
  denorm = latent * emb_std + emb_mean        [1, 32]
  quantized = Conv1d(denorm, 32→512)           [1, 512, 1]
  audio = mimi_decode(quantized, state)        [1, 1, 1920]

NOTE: The original Mimi model uses in-place tensor mutations (state[:] = ...)
in StreamingConv1d, StreamingConvTranspose1d, and MimiStreamingMultiheadAttention.
torch.jit.trace cannot handle in-place ops, so this module monkey-patches them
with functional equivalents before tracing.
"""
import torch
import torch.nn as nn


# Ordered list of (state_name, shape) for the 26 Mimi streaming state tensors.
# Must match the manifest.json order used by the Swift loader.
MIMI_STATE_SPEC = [
    ("upsample_partial", [1, 512, 16]),
    ("attn0_cache", [2, 1, 8, 256, 64]),
    ("attn0_offset", [1]),
    ("attn0_end_offset", [1]),
    ("attn1_cache", [2, 1, 8, 256, 64]),
    ("attn1_offset", [1]),
    ("attn1_end_offset", [1]),
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
# module_path comes from init_states(mimi.decoder), init_states(mimi.decoder_transformer),
# and init_states(mimi.upsample).
_SPEC_TO_NESTED = [
    ("upsample_partial", "convtr", "partial"),
    ("attn0_cache", "transformer.layers.0.self_attn", "cache"),
    ("attn0_offset", "transformer.layers.0.self_attn", "offset"),
    ("attn0_end_offset", "transformer.layers.0.self_attn", "end_offset"),
    ("attn1_cache", "transformer.layers.1.self_attn", "cache"),
    ("attn1_offset", "transformer.layers.1.self_attn", "offset"),
    ("attn1_end_offset", "transformer.layers.1.self_attn", "end_offset"),
    ("conv0_prev", "model.0", "previous"),
    ("conv0_first", "model.0", "first"),
    ("convtr0_partial", "model.2", "partial"),
    ("res0_conv0_prev", "model.3.block.1", "previous"),
    ("res0_conv0_first", "model.3.block.1", "first"),
    ("res0_conv1_prev", "model.3.block.3", "previous"),
    ("res0_conv1_first", "model.3.block.3", "first"),
    ("convtr1_partial", "model.5", "partial"),
    ("res1_conv0_prev", "model.6.block.1", "previous"),
    ("res1_conv0_first", "model.6.block.1", "first"),
    ("res1_conv1_prev", "model.6.block.3", "previous"),
    ("res1_conv1_first", "model.6.block.3", "first"),
    ("convtr2_partial", "model.8", "partial"),
    ("res2_conv0_prev", "model.9.block.1", "previous"),
    ("res2_conv0_first", "model.9.block.1", "first"),
    ("res2_conv1_prev", "model.9.block.3", "previous"),
    ("res2_conv1_first", "model.9.block.3", "first"),
    ("conv_final_prev", "model.11", "previous"),
    ("conv_final_first", "model.11", "first"),
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
    TP = state["previous"].shape[-1]
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
    PT = partial.shape[-1]
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


def _functional_complete(cache, end_offset, k, v):
    """Attention KV cache update without in-place scatter or assignment."""
    capacity = cache.shape[3]
    B, H, T, D = k.shape
    assert T > 0
    indexes = torch.arange(T, device=end_offset.device, dtype=torch.long)
    indexes = indexes + end_offset.long().view(-1, 1)
    indexes = indexes % capacity

    this_indexes = indexes.view(B, 1, T, 1).expand(-1, H, T, D).long()
    # Non-inplace scatter (returns new tensor)
    new_k = cache[0].scatter(2, this_indexes, k)
    new_v = cache[1].scatter(2, this_indexes, v)

    keys = new_k
    values = new_v

    indexes = torch.arange(capacity, device=end_offset.device, dtype=torch.long)
    last_offset = end_offset.view(-1, 1) + T - 1
    end_index = last_offset % capacity
    delta = indexes - end_index
    positions = torch.where(
        delta <= 0, last_offset + delta, last_offset + delta - capacity
    )
    new_end_offset = end_offset + T
    invalid = indexes >= new_end_offset.view(-1, 1)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)

    new_cache = torch.stack([new_k, new_v])
    return keys, values, positions, new_cache, new_end_offset


def _functional_complete_kv(self, k, v, model_state):
    """MimiStreamingMultiheadAttention._complete_kv without in-place ops."""
    from pocket_tts.modules.mimi_transformer import KVCacheResult
    if model_state is None:
        return KVCacheResult.from_kv(k, v)
    layer_state = self.get_state(model_state)
    keys, values, positions, new_cache, new_end_offset = _functional_complete(
        layer_state["cache"], layer_state["end_offset"], k, v
    )
    layer_state["cache"] = new_cache
    layer_state["end_offset"] = new_end_offset
    return KVCacheResult(keys, values, positions)


def _functional_attention_forward(self, query, model_state):
    """MimiStreamingMultiheadAttention.forward with explicit float scale.

    F.scaled_dot_product_attention computes 1/sqrt(d_k) from the tensor shape,
    which produces an int32 reciprocal that CoreML cannot handle. This patch
    passes scale as a pre-computed float constant.
    """
    import math
    from einops import rearrange

    B, T = query.shape[:2]

    if model_state is None:
        offset = torch.zeros(B, device=query.device, dtype=torch.long)
    else:
        offset = self.get_state(model_state)["offset"]

    projected = self.in_proj(query)
    q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    q, k = self.rope(q, k, offset)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)

    k, v, pos_k = self._complete_kv(k, v, model_state)
    pos_k = pos_k[:, None]
    pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(-1, 1)
    delta = pos_q - pos_k
    attn_bias = (pos_k >= 0) & (delta >= 0)
    attn_bias = attn_bias & (delta < self.context)
    attn_bias = attn_bias[:, None]

    # Manual attention to avoid int32 reciprocal from F.scaled_dot_product_attention.
    # The built-in computes 1/sqrt(D) where D=head_dim as an int tensor in the trace.
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(float(head_dim))
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = attn.masked_fill(~attn_bias, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    x = torch.matmul(attn, v)

    x = rearrange(x, "b h t d -> b t (h d)")
    x = self.out_proj(x)
    return x


def _patch_for_tracing(module):
    """Monkey-patch all in-place ops in the module tree for torch.jit.trace."""
    import types
    from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
    from pocket_tts.modules.mimi_transformer import MimiStreamingMultiheadAttention

    for name, child in module.named_modules():
        if isinstance(child, StreamingConv1d):
            child.forward = types.MethodType(
                _functional_streaming_conv1d_forward, child
            )
        elif isinstance(child, StreamingConvTranspose1d):
            child.forward = types.MethodType(
                _functional_streaming_conv_transpose1d_forward, child
            )
        elif isinstance(child, MimiStreamingMultiheadAttention):
            child._complete_kv = types.MethodType(
                _functional_complete_kv, child
            )
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

        # Build the nested state dict from init_states.
        from pocket_tts.modules.stateful_module import init_states
        self._nested_state = init_states(self.mimi.decoder, batch_size=1, sequence_length=256)
        self._nested_state.update(
            init_states(self.mimi.decoder_transformer, batch_size=1, sequence_length=256)
        )
        if hasattr(self.mimi, 'upsample'):
            self._nested_state.update(
                init_states(self.mimi.upsample, batch_size=1, sequence_length=256)
            )

        # Validate the mapping covers all nested state entries
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
        """Extract flat tensor tuple (MIMI_STATE_SPEC order) from nested dict."""
        tensors = []
        for spec_name, module_name, key in _SPEC_TO_NESTED:
            tensors.append(state[module_name][key])
        return tuple(tensors)

    def forward(self, latent, *state_tensors):
        """
        Args:
            latent: [1, 32] raw latent frame
            *state_tensors: 26 flat state tensors (MIMI_STATE_SPEC order)

        Returns:
            audio: [1, 1, 1920] decoded audio frame
            *updated_states: 26 updated state tensors (MIMI_STATE_SPEC order)
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
