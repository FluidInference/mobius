"""Export-time patches that replace FlexAttention with trace-friendly ops.

The NeMo Speech ``TransformerEncoder`` (RoPE, attn_mode="full") computes
``flex_attention(q, k, v, block_mask=padding_mask)`` with no score_mod, which is
mathematically plain softmax attention with a key-padding mask. FlexAttention and
``create_block_mask`` are not traceable by torch.jit / coremltools, so for export we:

1. Replace ``MultiHeadAttention.forward`` with an explicit matmul-softmax that treats
   the ``block_mask`` argument as an additive float bias of shape ``(B, 1, 1, T)``.
2. Replace ``TransformerEncoder.forward_internal``'s ``create_block_mask`` call with
   that additive bias built from ``length`` (valid frames are packed at the start,
   so RoPE absolute positions are unaffected by padding).

``apply_patches`` mutates the module instances in place (eval/export only).
"""

import types

import torch

# Finite fp16-safe "minus infinity" for the softmax bias. Padded queries attend
# uniformly and produce garbage rows, which downstream masking zeroes out.
NEG_BIAS = -30000.0


def _mha_forward_export(self, x, block_mask=None, pos_emb=None):
    B, T, _ = x.shape
    H, D = self.n_heads, self.head_dim

    qkv = self.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    if self.qk_norm:
        q = self.q_norm(q).to(v.dtype)
        k = self.k_norm(k).to(v.dtype)

    if self._uses_rope:
        q, k = self.rope(q, k)

    if self._uses_rel_pos:
        raise NotImplementedError("export patch supports 'rope'/'abs_pos'/'no_pos' only")

    scores = torch.matmul(q, k.transpose(-2, -1)) * (D**-0.5)
    if block_mask is not None:
        scores = scores + block_mask  # additive (B, 1, 1, T) key-padding bias
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).reshape(B, T, self.d_model)
    return self.out_proj(out)


def _encoder_forward_internal_export(self, audio_signal, length, bypass_pre_encode=False):
    if not bypass_pre_encode:
        # Only the FeatureStacking pre-encode path is exercised by the diarization export.
        x, length = self.pre_encode(audio_signal, length)
        length = length.to(torch.int64)
    else:
        x = audio_signal
        length = length.to(torch.int64)

    if self.self_attention_model == "rope":
        if self.xscale:
            x = x * self.xscale
        x = self.dropout_pre_encoder(x)
    elif self.pos_enc is not None:
        x, _ = self.pos_enc(x=x)
    x = self.embed_norm(x)

    B, T, _ = x.shape
    if self.attn_mode != "full":
        raise NotImplementedError("export patch supports attn_mode='full' only")
    pad = torch.arange(T, device=x.device).unsqueeze(0) >= length.unsqueeze(1)  # (B, T)
    attn_bias = pad.to(x.dtype).view(B, 1, 1, T) * NEG_BIAS

    for layer in self.layers:
        x = layer(x, block_mask=attn_bias, pos_emb=None)

    x = self.final_norm(x)
    if self.out_proj is not None:
        x = self.out_proj(x)
    x = x.transpose(1, 2)  # (B, T, D) -> (B, D, T)
    return x, length.to(torch.int64)


def _encoder_forward_export(self, audio_signal, length, bypass_pre_encode=False):
    # Skip update_max_seq_length (touches torch.distributed); pos buffers are
    # pre-sized to pos_emb_max_len which exceeds every export shape.
    return self.forward_internal(audio_signal, length, bypass_pre_encode=bypass_pre_encode)


def apply_patches(model):
    """Patch a restored SortformerEncLabelModel in place for export. Returns the model."""
    enc = model.encoder
    enc.forward_internal = types.MethodType(_encoder_forward_internal_export, enc)
    enc.forward = types.MethodType(_encoder_forward_export, enc)
    for layer in enc.layers:
        layer.attn.forward = types.MethodType(_mha_forward_export, layer.attn)
    return model
