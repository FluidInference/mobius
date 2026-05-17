"""text_encoder.onnx → PyTorch port.

Pipeline (per tts.json/ttl.text_encoder + ONNX graph):

    text_ids [B, T_text]                style_ttl [B, 50, 256]
            │                                    │
    char_embedder (8322→256)                     │
            │                                    │
    transpose to [B, 256, T_text]                │
            │                                    │
    × text_mask                                  │
            │                                    │
    6× ConvNeXt-1D (k=5, C=256, hdim=1024)       │
       (mask applied before each block)          │
            │                                    │
    4× attn_encoder layer (rel-pos MHA + FFN)    │
            │                                    │
    speech_prompted_text_encoder ── attention1 ──┤  (cross-attn, text Q vs style K,V)
            │                                    │
                       attention2 ───────────────┤
            │
    LayerNorm + × text_mask
            │
        text_emb [B, 256, T_text]
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    ConvNeXt1DBlock,
    LayerNorm1d,
    RelPosSelfAttention,
    TransformerFFN,
    assign_param,
    find_orphan_by_shape,
    load_convnext_block,
    load_relpos_attn,
    onnx_initializers,
)


# -- Constants from tts.json (ttl.text_encoder) ------------------------------
CHAR_VOCAB = 8322
EMB_DIM = 256
CONVNEXT_NUM_LAYERS = 6
CONVNEXT_KSZ = 5
CONVNEXT_HDIM = 1024
CONVNEXT_DILATIONS = [1, 1, 2, 2, 4, 4]
ATTN_NUM_LAYERS = 4
ATTN_HEADS = 4
ATTN_HDIM = 1024  # filter_channels in tts.json
ATTN_WINDOW = 4  # window_size, gives 2W+1 = 9
STYLE_TOKENS = 50
STYLE_DIM = 256
SPEECH_PROMPTED_HEADS = 2


class _AttnEncoderLayer(nn.Module):
    """One transformer-like layer: norm1 -> rel-pos MHA -> add -> norm2 -> FFN -> add.

    Structure follows the upstream `encoder.py`: pre-norm vs post-norm depends
    on whether `norm_layers_1[i]` is applied before or after the attention. We
    use post-norm (residual + norm), which is consistent with the VITS-style
    implementation that ships with this model.
    """

    def __init__(self, channels: int, n_heads: int, hdim: int, window_size: int):
        super().__init__()
        self.attn = RelPosSelfAttention(channels, n_heads, window_size)
        self.norm_1 = LayerNorm1d(channels)
        self.ffn = TransformerFFN(channels, hdim)
        self.norm_2 = LayerNorm1d(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]; mask: [B, 1, T] (1=valid, 0=pad).
        # Matches ONNX wiring exactly:
        #   x_m = x * mask
        #   add1 = x_m + attn(x_m)
        #   n1   = LN(add1)
        #   add2 = n1 + ffn(n1, mask)
        #   n2   = LN(add2)
        x_m = x * mask
        y = self.attn(x_m, attn_mask=mask)
        n1 = self.norm_1(x_m + y)
        y = self.ffn(n1, mask=mask)
        n2 = self.norm_2(n1 + y)
        return n2


class _SpeechPromptedAttention(nn.Module):
    """Single cross-attention block in speech_prompted_text_encoder.

    Per the ONNX graph:
        K source = `style_key` (learned parameter, broadcast to batch)
        V source = `style_ttl` (runtime input)
        scores  = softmax(Q @ tanh(K.T) / sqrt(dk))
        out     = (out_fc(softmax_out @ V)) * mask
    Heads are formed by splitting along the channel axis (contiguous halves),
    which is equivalent to `reshape(B, T, H, dk).transpose(1, 2)`.

    Mask semantics: rows where text_mask == 0 are zeroed AFTER softmax via
    `Where(mask == 0, 0, softmax_out)`. For all-ones masks this is a no-op.
    """

    def __init__(self, text_dim: int, style_dim: int, n_units: int, n_heads: int):
        super().__init__()
        assert n_units % n_heads == 0
        self.n_units = n_units
        self.n_heads = n_heads
        self.dk = n_units // n_heads
        self.W_query = _LinearWrap(text_dim, n_units)
        self.W_key = _LinearWrap(style_dim, n_units)
        self.W_value = _LinearWrap(style_dim, n_units)
        self.out_fc = _LinearWrap(n_units, text_dim)

    def forward(
        self,
        text: torch.Tensor,       # [B, T, C_text]
        key_src: torch.Tensor,    # [B, S, C_style]
        value_src: torch.Tensor,  # [B, S, C_style]
        mask: torch.Tensor,       # [B, 1, T]
    ) -> torch.Tensor:
        B, T, _ = text.shape
        S = key_src.shape[1]
        H, dk = self.n_heads, self.dk

        q = self.W_query(text).reshape(B, T, H, dk).transpose(1, 2)       # [B, H, T, dk]
        k = self.W_key(key_src).reshape(B, S, H, dk).transpose(1, 2)      # [B, H, S, dk]
        v = self.W_value(value_src).reshape(B, S, H, dk).transpose(1, 2)  # [B, H, S, dk]

        # NOTE: the divisor in ONNX is hardcoded 16.0 (= sqrt(n_units), not sqrt(dk)).
        scores = torch.matmul(q, torch.tanh(k).transpose(-2, -1)) / math.sqrt(self.n_units)
        attn = F.softmax(scores, dim=-1)
        # Post-softmax mask: zero rows where text_mask == 0.
        # ANE-friendly multiplicative mask: float mult ≡ where(mask==0, 0, attn)
        # but avoids bool tile/select that the ANE compiler rejects.
        mask_q = mask.transpose(1, 2).unsqueeze(1)  # [B, 1, T, 1]
        attn = attn * mask_q
        out = torch.matmul(attn, v)                  # [B, H, T, dk]
        out = out.transpose(1, 2).reshape(B, T, H * dk)
        out = self.out_fc(out)
        out = out * mask.transpose(1, 2)             # [B, T, 1]
        return out


class _LinearWrap(nn.Module):
    """Linear wrapped so ONNX names `<scope>.linear.bias` load directly.

    Weight is loaded from an `onnx::MatMul_*` orphan (shape [in_features, out_features]).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _SpeechPromptedTextEncoder(nn.Module):
    """K source is `style_key` (learned), V source is `style_ttl` (input).
    Both attention residuals are added against the ORIGINAL `text` features,
    not the running sum (per the ONNX graph wiring).
    """

    def __init__(self, text_dim: int = EMB_DIM, style_dim: int = STYLE_DIM, n_units: int = EMB_DIM, n_heads: int = SPEECH_PROMPTED_HEADS, style_tokens: int = STYLE_TOKENS):
        super().__init__()
        self.attention1 = _SpeechPromptedAttention(text_dim, style_dim, n_units, n_heads)
        self.attention2 = _SpeechPromptedAttention(text_dim, style_dim, n_units, n_heads)
        self.norm = LayerNorm1d(text_dim)
        # style_key: learned [1, S, C_style], broadcast across batch at runtime.
        self.style_key = nn.Parameter(torch.zeros(1, style_tokens, style_dim))

    def forward(self, text: torch.Tensor, style_ttl: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # text: [B, C, T] → transpose for attention.
        B = text.shape[0]
        t = text.transpose(1, 2)  # [B, T, C]
        key_src = self.style_key.expand(B, -1, -1)
        attn1_out = self.attention1(t, key_src, style_ttl, mask)
        y1 = t + attn1_out
        attn2_out = self.attention2(y1, key_src, style_ttl, mask)
        # Residual against ORIGINAL t (not y1) — matches the ONNX wiring.
        y_final = t + attn2_out
        y_final = y_final.transpose(1, 2)  # back to [B, C, T]
        return self.norm(y_final)


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.char_embedder = nn.Embedding(CHAR_VOCAB, EMB_DIM)
        self.convnext = nn.ModuleList(
            [
                ConvNeXt1DBlock(EMB_DIM, CONVNEXT_HDIM, CONVNEXT_KSZ, dilation=d, wrap_dwconv=False, causal=False)
                for d in CONVNEXT_DILATIONS
            ]
        )
        self.attn_layers = nn.ModuleList(
            [_AttnEncoderLayer(EMB_DIM, ATTN_HEADS, ATTN_HDIM, ATTN_WINDOW) for _ in range(ATTN_NUM_LAYERS)]
        )
        self.speech_prompted = _SpeechPromptedTextEncoder()

    def forward(self, text_ids: torch.Tensor, style_ttl: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        # text_ids: [B, T] (long); style_ttl: [B, 50, 256]; text_mask: [B, 1, T]
        x = self.char_embedder(text_ids)               # [B, T, C]
        x = x.transpose(1, 2) * text_mask              # [B, C, T]

        for blk in self.convnext:
            x = blk(x * text_mask, mask=text_mask)
        conv_out = x                                    # save for skip

        # attn_encoder
        for layer in self.attn_layers:
            x = layer(x, text_mask)
        x = x * text_mask

        # skip connection: convnext output bypasses attn stack.
        x = (x + conv_out) * text_mask                  # proj_out (mask-only)

        x = self.speech_prompted(x, style_ttl, text_mask)
        return x * text_mask


def build_text_encoder_from_onnx(onnx_path: Path) -> TextEncoder:
    inits = onnx_initializers(onnx_path)
    model = TextEncoder()
    _load_text_encoder(model, inits)
    model.eval()
    return model


def _load_text_encoder(model: TextEncoder, inits: Dict[str, np.ndarray]) -> None:
    # Char embedder.
    assign_param(model.char_embedder.weight, inits["tts.ttl.text_encoder.text_embedder.char_embedder.weight"], name="char_embedder.weight")

    # ConvNeXt blocks (unwrapped dwconv).
    for i, blk in enumerate(model.convnext):
        load_convnext_block(blk, inits, prefix=f"tts.ttl.text_encoder.convnext.convnext.{i}", wrap_dwconv=False)

    # Attn encoder layers.
    for i, layer in enumerate(model.attn_layers):
        attn_prefix = f"tts.ttl.text_encoder.attn_encoder.attn_layers.{i}"
        load_relpos_attn(layer.attn, inits, attn_prefix)
        # norm_layers_1[i] / norm_layers_2[i]
        assign_param(layer.norm_1.norm.weight, inits[f"tts.ttl.text_encoder.attn_encoder.norm_layers_1.{i}.norm.weight"], name=f"norm_layers_1.{i}.weight")
        assign_param(layer.norm_1.norm.bias, inits[f"tts.ttl.text_encoder.attn_encoder.norm_layers_1.{i}.norm.bias"], name=f"norm_layers_1.{i}.bias")
        assign_param(layer.norm_2.norm.weight, inits[f"tts.ttl.text_encoder.attn_encoder.norm_layers_2.{i}.norm.weight"], name=f"norm_layers_2.{i}.weight")
        assign_param(layer.norm_2.norm.bias, inits[f"tts.ttl.text_encoder.attn_encoder.norm_layers_2.{i}.norm.bias"], name=f"norm_layers_2.{i}.bias")
        # FFN
        assign_param(layer.ffn.conv_1.weight, inits[f"tts.ttl.text_encoder.attn_encoder.ffn_layers.{i}.conv_1.weight"], name=f"ffn.{i}.conv_1.weight")
        assign_param(layer.ffn.conv_1.bias, inits[f"tts.ttl.text_encoder.attn_encoder.ffn_layers.{i}.conv_1.bias"], name=f"ffn.{i}.conv_1.bias")
        assign_param(layer.ffn.conv_2.weight, inits[f"tts.ttl.text_encoder.attn_encoder.ffn_layers.{i}.conv_2.weight"], name=f"ffn.{i}.conv_2.weight")
        assign_param(layer.ffn.conv_2.bias, inits[f"tts.ttl.text_encoder.attn_encoder.ffn_layers.{i}.conv_2.bias"], name=f"ffn.{i}.conv_2.bias")

    # Speech-prompted cross-attention.
    # 8 orphan MatMuls of shape [256, 256] in order:
    #   3680 Q1, 3681 K1, 3682 V1, 3683 Out1, 3684 Q2, 3685 K2, 3686 V2, 3687 Out2
    consumed: set = set()
    orphan_keys = sorted([k for k in inits if k.startswith("onnx::MatMul_")])
    assert len(orphan_keys) == 8, f"Expected 8 orphan MatMuls, got {len(orphan_keys)}"
    # PyTorch Linear weight is [out, in], ONNX MatMul weight is [in, out] → transpose.
    def _load_linear(linwrap: _LinearWrap, orphan_key: str, bias_key: str):
        w = inits[orphan_key]
        assign_param(linwrap.linear.weight, w.T, name=f"{orphan_key}->linear.weight")
        assign_param(linwrap.linear.bias, inits[bias_key], name=bias_key)

    sp_prefix = "tts.ttl.speech_prompted_text_encoder"
    _load_linear(model.speech_prompted.attention1.W_query, orphan_keys[0], f"{sp_prefix}.attention1.W_query.linear.bias")
    _load_linear(model.speech_prompted.attention1.W_key, orphan_keys[1], f"{sp_prefix}.attention1.W_key.linear.bias")
    _load_linear(model.speech_prompted.attention1.W_value, orphan_keys[2], f"{sp_prefix}.attention1.W_value.linear.bias")
    _load_linear(model.speech_prompted.attention1.out_fc, orphan_keys[3], f"{sp_prefix}.attention1.out_fc.linear.bias")
    _load_linear(model.speech_prompted.attention2.W_query, orphan_keys[4], f"{sp_prefix}.attention2.W_query.linear.bias")
    _load_linear(model.speech_prompted.attention2.W_key, orphan_keys[5], f"{sp_prefix}.attention2.W_key.linear.bias")
    _load_linear(model.speech_prompted.attention2.W_value, orphan_keys[6], f"{sp_prefix}.attention2.W_value.linear.bias")
    _load_linear(model.speech_prompted.attention2.out_fc, orphan_keys[7], f"{sp_prefix}.attention2.out_fc.linear.bias")

    # speech_prompted final norm.
    assign_param(model.speech_prompted.norm.norm.weight, inits[f"{sp_prefix}.norm.norm.weight"], name="sp.norm.weight")
    assign_param(model.speech_prompted.norm.norm.bias, inits[f"{sp_prefix}.norm.norm.bias"], name="sp.norm.bias")

    # style_key (learned key source for speech_prompted cross-attention).
    assign_param(
        model.speech_prompted.style_key,
        inits["tts.ttl.style_encoder.style_token_layer.style_key"],
        name="speech_prompted.style_key",
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    onnx_path = Path(sys.argv[1] if len(sys.argv) > 1 else "build/supertonic-3-coreml/_onnx/text_encoder.onnx")
    model = build_text_encoder_from_onnx(onnx_path)
    print(f"Built TextEncoder with {sum(p.numel() for p in model.parameters())} params")
    text_ids = torch.zeros(1, 8, dtype=torch.long)
    style = torch.randn(1, 50, 256)
    mask = torch.ones(1, 1, 8)
    with torch.no_grad():
        y = model(text_ids, style, mask)
    print(f"Output shape: {tuple(y.shape)}")
