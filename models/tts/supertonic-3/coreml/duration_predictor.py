"""duration_predictor.onnx → PyTorch port.

Pipeline (per tts.json/dp + ONNX graph):

    text_ids [B, T_text]                style_dp [B, 8, 16]
            │                                    │
    char_embedder (8322→64)                      │
            │                                    │
    transpose → [B, 64, T]                       │
            │                                    │
    × text_mask                                  │
            │                                    │
    prepend sentence_token [1, 64, 1] (broadcast)│
            │                                    │
    prepend a `1` to text_mask → [B, 1, T+1]     │
            │                                    │
    6× ConvNeXt-1D (k=5, C=64, hdim=256, dil=1, symmetric pad)
            │
    2× attn_encoder layer (rel-pos MHA, n_heads=2, FFN ReLU)
            │
    skip add (attn_encoder.out + convnext[-1].out)
            │
    slice first position → [B, 64, 1]
            │
    proj_out Conv1d(64,64,k=1, no bias)
            │
    × sliced_mask[:, :, 0:1]   (always 1)
            │
    flatten → [B, 64] ── concat axis=1 ──── flatten(style_dp) → [B, 128]
                                  │
                              [B, 192]
                                  │
                       Gemm(192→128) + PReLU(1) + Gemm(128→1)
                                  │
                              Exp + Squeeze → duration [B]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .common import (
    ConvNeXt1DBlock,
    LayerNorm1d,
    RelPosSelfAttention,
    TransformerFFN,
    assign_param,
    load_convnext_block,
    load_relpos_attn,
    onnx_initializers,
)


# -- Constants from tts.json (dp.*) ------------------------------------------
CHAR_VOCAB = 8322
EMB_DIM = 64
CONVNEXT_NUM_LAYERS = 6
CONVNEXT_KSZ = 5
CONVNEXT_HDIM = 256
CONVNEXT_DILATIONS = [1, 1, 1, 1, 1, 1]
ATTN_NUM_LAYERS = 2
ATTN_HEADS = 2
ATTN_HDIM = 256
ATTN_WINDOW = 4
N_STYLE = 8
STYLE_DIM = 16
PREDICTOR_HDIM = 128
PREDICTOR_IN = EMB_DIM + N_STYLE * STYLE_DIM  # 64 + 128 = 192


class _AttnEncoderLayer(nn.Module):
    """Mirrors text_encoder._AttnEncoderLayer (post-norm, masked residual)."""

    def __init__(self, channels: int, n_heads: int, hdim: int, window_size: int):
        super().__init__()
        self.attn = RelPosSelfAttention(channels, n_heads, window_size)
        self.norm_1 = LayerNorm1d(channels)
        self.ffn = TransformerFFN(channels, hdim, activation="relu")
        self.norm_2 = LayerNorm1d(channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_m = x * mask
        y = self.attn(x_m, attn_mask=mask)
        n1 = self.norm_1(x_m + y)
        y = self.ffn(n1, mask=mask)
        n2 = self.norm_2(n1 + y)
        return n2


class _Predictor(nn.Module):
    """Final MLP: Linear(192,128) + PReLU + Linear(128,1) + Exp + Squeeze."""

    def __init__(self):
        super().__init__()
        # ONNX uses Gemm with weight stored as [out, in] (transB=1) — matches nn.Linear.
        self.layers = nn.ModuleList([
            nn.Linear(PREDICTOR_IN, PREDICTOR_HDIM, bias=True),
            nn.Linear(PREDICTOR_HDIM, 1, bias=True),
        ])
        self.activation = nn.PReLU(num_parameters=1)

    def forward(self, sentence_emb: torch.Tensor, style_dp: torch.Tensor) -> torch.Tensor:
        # sentence_emb: [B, 64, 1]  → [B, 64]
        # style_dp:     [B, 8, 16]  → [B, 128]
        B = sentence_emb.shape[0]
        s = sentence_emb.reshape(B, -1)
        v = style_dp.reshape(B, -1)
        x = torch.cat([s, v], dim=1)  # [B, 192]
        x = self.layers[0](x)
        x = self.activation(x)
        x = self.layers[1](x)  # [B, 1]
        x = torch.exp(x)
        return x.squeeze(-1)  # [B]


class DurationPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.char_embedder = nn.Embedding(CHAR_VOCAB, EMB_DIM)
        # Learned sentence token, prepended as time step 0 to the encoder input.
        self.sentence_token = nn.Parameter(torch.zeros(1, EMB_DIM, 1))

        self.convnext = nn.ModuleList(
            [
                # dwconv weights ship at `.dwconv.weight` (not `.dwconv.net.weight`)
                ConvNeXt1DBlock(EMB_DIM, CONVNEXT_HDIM, CONVNEXT_KSZ, dilation=d, wrap_dwconv=False, causal=False)
                for d in CONVNEXT_DILATIONS
            ]
        )

        self.attn_layers = nn.ModuleList(
            [_AttnEncoderLayer(EMB_DIM, ATTN_HEADS, ATTN_HDIM, ATTN_WINDOW) for _ in range(ATTN_NUM_LAYERS)]
        )

        # proj_out is a wrapped Conv1d with NO bias (only `.net.weight`).
        self.proj_out_conv = nn.Conv1d(EMB_DIM, EMB_DIM, kernel_size=1, bias=False)

        self.predictor = _Predictor()

    def forward(
        self,
        text_ids: torch.Tensor,    # [B, T_text]
        style_dp: torch.Tensor,    # [B, 8, 16]
        text_mask: torch.Tensor,   # [B, 1, T_text]
    ) -> torch.Tensor:
        B = text_ids.shape[0]

        # 1. char_embedder + mask.
        x = self.char_embedder(text_ids)         # [B, T, 64]
        x = x.transpose(1, 2) * text_mask        # [B, 64, T]

        # 2. Prepend sentence_token at time step 0.
        st = self.sentence_token.expand(B, -1, -1)  # [B, 64, 1]
        x = torch.cat([st, x], dim=2)               # [B, 64, T+1]

        # 3. Prepend a `1` to text_mask (sentence token is always valid).
        ones = torch.ones(B, 1, 1, dtype=text_mask.dtype, device=text_mask.device)
        mask = torch.cat([ones, text_mask], dim=2)  # [B, 1, T+1]

        # 4. ConvNeXt stack.
        for blk in self.convnext:
            x = blk(x * mask, mask=mask)
        conv_out = x  # save for skip connection

        # 5. Attn encoder stack.
        for layer in self.attn_layers:
            x = layer(x, mask)
        x = x * mask

        # 6. Skip add (attn + convnext) — matches ONNX `/sentence_encoder/Add`.
        x = x + conv_out  # [B, 64, T+1]

        # 7. Slice first position (the sentence token aggregation).
        sentence = x[:, :, 0:1]                  # [B, 64, 1]
        sentence_mask = mask[:, :, 0:1]          # [B, 1, 1]  (always 1)

        # 8. proj_out + mask.
        sentence = self.proj_out_conv(sentence) * sentence_mask

        # 9. Predictor MLP.
        return self.predictor(sentence, style_dp)


def build_duration_predictor_from_onnx(onnx_path: Path) -> DurationPredictor:
    inits = onnx_initializers(onnx_path)
    model = DurationPredictor()
    _load_duration_predictor(model, inits)
    model.eval()
    return model


def _load_duration_predictor(model: DurationPredictor, inits: Dict[str, np.ndarray]) -> None:
    p = "tts.dp"

    # Char embedder + sentence token.
    assign_param(
        model.char_embedder.weight,
        inits[f"{p}.sentence_encoder.text_embedder.char_embedder.weight"],
        name="char_embedder.weight",
    )
    assign_param(
        model.sentence_token,
        inits[f"{p}.sentence_encoder.sentence_token"],
        name="sentence_token",
    )

    # ConvNeXt blocks (unwrapped dwconv).
    for i, blk in enumerate(model.convnext):
        load_convnext_block(
            blk,
            inits,
            prefix=f"{p}.sentence_encoder.convnext.convnext.{i}",
            wrap_dwconv=False,
        )

    # Attn encoder layers.
    for i, layer in enumerate(model.attn_layers):
        attn_prefix = f"{p}.sentence_encoder.attn_encoder.attn_layers.{i}"
        load_relpos_attn(layer.attn, inits, attn_prefix)
        n1 = f"{p}.sentence_encoder.attn_encoder.norm_layers_1.{i}.norm"
        n2 = f"{p}.sentence_encoder.attn_encoder.norm_layers_2.{i}.norm"
        assign_param(layer.norm_1.norm.weight, inits[f"{n1}.weight"], name=f"{n1}.weight")
        assign_param(layer.norm_1.norm.bias, inits[f"{n1}.bias"], name=f"{n1}.bias")
        assign_param(layer.norm_2.norm.weight, inits[f"{n2}.weight"], name=f"{n2}.weight")
        assign_param(layer.norm_2.norm.bias, inits[f"{n2}.bias"], name=f"{n2}.bias")
        ffn = f"{p}.sentence_encoder.attn_encoder.ffn_layers.{i}"
        assign_param(layer.ffn.conv_1.weight, inits[f"{ffn}.conv_1.weight"], name=f"{ffn}.conv_1.weight")
        assign_param(layer.ffn.conv_1.bias, inits[f"{ffn}.conv_1.bias"], name=f"{ffn}.conv_1.bias")
        assign_param(layer.ffn.conv_2.weight, inits[f"{ffn}.conv_2.weight"], name=f"{ffn}.conv_2.weight")
        assign_param(layer.ffn.conv_2.bias, inits[f"{ffn}.conv_2.bias"], name=f"{ffn}.conv_2.bias")

    # proj_out (wrapped, no bias).
    assign_param(
        model.proj_out_conv.weight,
        inits[f"{p}.sentence_encoder.proj_out.net.weight"],
        name="proj_out.net.weight",
    )

    # Predictor MLP.
    assign_param(model.predictor.layers[0].weight, inits[f"{p}.predictor.layers.0.weight"], name="predictor.layers.0.weight")
    assign_param(model.predictor.layers[0].bias, inits[f"{p}.predictor.layers.0.bias"], name="predictor.layers.0.bias")
    assign_param(model.predictor.layers[1].weight, inits[f"{p}.predictor.layers.1.weight"], name="predictor.layers.1.weight")
    assign_param(model.predictor.layers[1].bias, inits[f"{p}.predictor.layers.1.bias"], name="predictor.layers.1.bias")
    assign_param(model.predictor.activation.weight, inits[f"{p}.predictor.activation.weight"], name="predictor.activation.weight")


if __name__ == "__main__":  # pragma: no cover
    import sys

    onnx_path = Path(sys.argv[1] if len(sys.argv) > 1 else "build/supertonic-3-coreml/_onnx/duration_predictor.onnx")
    model = build_duration_predictor_from_onnx(onnx_path)
    print(f"Built DurationPredictor with {sum(p.numel() for p in model.parameters())} params")
    text_ids = torch.zeros(1, 8, dtype=torch.long)
    style = torch.randn(1, 8, 16)
    mask = torch.ones(1, 1, 8)
    with torch.no_grad():
        y = model(text_ids, style, mask)
    print(f"Output shape: {tuple(y.shape)}  (expected (1,))")
    print(f"Sample value: {y.item():.4f}")
