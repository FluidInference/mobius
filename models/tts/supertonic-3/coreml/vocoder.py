"""Vocoder (AE decoder) port for Supertonic-3.

ONNX graph: `vocoder.onnx`
Inputs:
    latent : [B, 144, L_ttl]   (144 == latent_dim * chunk_compress_factor)
Outputs:
    wav_tts: [B, 512 * L_ae]   where L_ae == 6 * L_ttl

Pipeline (validated against the ONNX node order):
    1. Divide by `tts.ttl.normalizer.scale` to undo ttl scaling.
    2. Unpack chunk-compress: [B, 24*6, L_ttl] → [B, 24, 6*L_ttl].
    3. Denormalize: `x * latent_std + latent_mean`.
    4. embed Conv1d(24 → 512, k=7, pad=3).
    5. 10× ConvNeXt-1D blocks (ksz=7, hdim=2048, dilations [1,2,4,1,2,4,1,1,1,1]).
    6. final_norm BatchNorm1d(512).
    7. head.layer1 Conv1d(512 → 2048, k=3, pad=1).
    8. PReLU (shared single param).
    9. head.layer2 Conv1d(2048 → 512, k=1).
   10. Reshape [B, 512, L_ae] → [B, 512 * L_ae] via transpose+flatten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .common import (
    BatchNorm1dWrap,
    CausalConv1d,
    ConvNeXt1DBlock,
    LayerNorm1d,
    assign_param,
    find_orphan_by_shape,
    load_convnext_block,
    onnx_initializers,
)


# Constants from tts.json (`ae.*`) and `ttl.normalizer.scale`.
LATENT_DIM = 24
CHUNK_COMPRESS_FACTOR = 6
DECODER_CHANNELS = 512
DECODER_INTERMEDIATE = 2048
DECODER_KSZ = 7
DECODER_DILATIONS = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
HEAD_HDIM = 2048
HEAD_KSZ = 3
BASE_CHUNK_SIZE = 512


class _Head(nn.Module):
    """AE decoder head.

    ONNX names:
        tts.ae.decoder.head.layer1.net.weight, .bias  (Conv1d 512→2048, k=3)
        onnx::PRelu_*                                  (single PReLU param)
        tts.ae.decoder.head.layer2.weight, .bias       (Conv1d 2048→512, k=1)
    """

    def __init__(self):
        super().__init__()
        # layer1 is wrapped (`.net`).
        self.layer1 = CausalConv1d(
            in_ch=DECODER_CHANNELS,
            out_ch=HEAD_HDIM,
            kernel_size=HEAD_KSZ,
            groups=1,
            dilation=1,
        )
        # PReLU(num_parameters=1) — ONNX shape is [1, 1] so we squeeze on load.
        self.act = nn.PReLU(num_parameters=1)
        self.layer2 = nn.Conv1d(HEAD_HDIM, DECODER_CHANNELS, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x


class Vocoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Scalars exposed as buffers so they participate in tracing.
        self.register_buffer("ttl_norm_scale", torch.tensor(0.25), persistent=True)
        self.register_buffer("latent_mean", torch.zeros(1, LATENT_DIM, 1), persistent=True)
        self.register_buffer("latent_std", torch.ones(1, LATENT_DIM, 1), persistent=True)

        # embed: Conv1d(24 → 512, k=7, pad=3).
        self.embed = CausalConv1d(
            in_ch=LATENT_DIM,
            out_ch=DECODER_CHANNELS,
            kernel_size=DECODER_KSZ,
            groups=1,
            dilation=1,
        )

        self.convnext = nn.ModuleList(
            [
                ConvNeXt1DBlock(
                    channels=DECODER_CHANNELS,
                    intermediate_dim=DECODER_INTERMEDIATE,
                    kernel_size=DECODER_KSZ,
                    dilation=d,
                )
                for d in DECODER_DILATIONS
            ]
        )

        self.final_norm = BatchNorm1dWrap(DECODER_CHANNELS)
        self.head = _Head()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent: [B, 144, L_ttl] → wav: [B, 512 * 6 * L_ttl]."""
        # 1. Undo ttl normalization.
        x = latent / self.ttl_norm_scale

        # 2. Unpack chunk-compress [B, 24*6, L] → [B, 24, 6L].
        # Use -1 for dynamic dims so CoreML doesn't see aten::Int(size) ops.
        x = x.reshape(1, LATENT_DIM, CHUNK_COMPRESS_FACTOR, -1)
        x = x.transpose(2, 3).contiguous()  # [B, 24, L, 6]
        x = x.reshape(1, LATENT_DIM, -1)

        # 3. Denormalize.
        x = x * self.latent_std + self.latent_mean

        # 4. embed.
        x = self.embed(x)

        # 5. ConvNeXt stack.
        for blk in self.convnext:
            x = blk(x)

        # 6. final_norm (BatchNorm1d).
        x = self.final_norm(x)

        # 7-9. head.
        x = self.head(x)

        # 10. Reshape to waveform: [B, 512, L_ae] → [B, L_ae, 512] → [B, L_ae*512].
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(1, -1)
        return x


def build_vocoder_from_onnx(onnx_path: Path, tts_json_path: Path | None = None) -> Vocoder:
    """Construct a Vocoder and load all weights from `vocoder.onnx`."""
    inits = onnx_initializers(onnx_path)
    model = Vocoder()
    _load_vocoder(model, inits, tts_json_path=tts_json_path)
    model.eval()
    return model


def _load_vocoder(
    model: Vocoder,
    inits: Dict[str, np.ndarray],
    *,
    tts_json_path: Path | None,
) -> None:
    # Optional: cross-check against tts.json constants.
    if tts_json_path is not None:
        with open(tts_json_path) as f:
            cfg = json.load(f)
        assert cfg["ae"]["ldim"] == LATENT_DIM
        assert cfg["ae"]["base_chunk_size"] == BASE_CHUNK_SIZE
        assert cfg["ae"]["decoder"]["num_layers"] == len(DECODER_DILATIONS)
        assert cfg["ae"]["decoder"]["dilation_lst"] == DECODER_DILATIONS
        assert cfg["ae"]["decoder"]["intermediate_dim"] == DECODER_INTERMEDIATE
        assert cfg["ae"]["decoder"]["ksz"] == DECODER_KSZ

    # Normalizers.
    assign_param(model.ttl_norm_scale, inits["tts.ttl.normalizer.scale"], name="ttl_norm_scale")
    assign_param(model.latent_mean, inits["tts.ae.latent_mean"], name="latent_mean")
    assign_param(model.latent_std, inits["tts.ae.latent_std"], name="latent_std")

    # embed: from orphan onnx::Conv_*. Two orphans with shapes [512,24,7] and [512].
    consumed: set = set()
    w_name = find_orphan_by_shape(inits, (DECODER_CHANNELS, LATENT_DIM, DECODER_KSZ), consumed=consumed)
    b_name = find_orphan_by_shape(inits, (DECODER_CHANNELS,), consumed=consumed)
    assign_param(model.embed.net.weight, inits[w_name], name="embed.weight")
    assign_param(model.embed.net.bias, inits[b_name], name="embed.bias")

    # ConvNeXt blocks.
    for i, blk in enumerate(model.convnext):
        load_convnext_block(blk, inits, prefix=f"tts.ae.decoder.convnext.{i}")

    # final_norm (BatchNorm1d).
    bn = model.final_norm.norm
    assign_param(bn.weight, inits["tts.ae.decoder.final_norm.norm.weight"], name="final_norm.weight")
    assign_param(bn.bias, inits["tts.ae.decoder.final_norm.norm.bias"], name="final_norm.bias")
    assign_param(bn.running_mean, inits["tts.ae.decoder.final_norm.norm.running_mean"], name="final_norm.running_mean")
    assign_param(bn.running_var, inits["tts.ae.decoder.final_norm.norm.running_var"], name="final_norm.running_var")

    # head.layer1 (wrapped Conv1d).
    assign_param(model.head.layer1.net.weight, inits["tts.ae.decoder.head.layer1.net.weight"], name="head.layer1.weight")
    assign_param(model.head.layer1.net.bias, inits["tts.ae.decoder.head.layer1.net.bias"], name="head.layer1.bias")

    # head.act (PReLU). ONNX shape is [1, 1] but nn.PReLU(num_parameters=1) is [1].
    prelu_arr = find_orphan_by_shape(inits, (1, 1), consumed=consumed)
    prelu_val = np.asarray(inits[prelu_arr]).reshape(-1)
    assign_param(model.head.act.weight, prelu_val, name="head.act.weight")

    # head.layer2 (bare Conv1d).
    assign_param(model.head.layer2.weight, inits["tts.ae.decoder.head.layer2.weight"], name="head.layer2.weight")


if __name__ == "__main__":  # pragma: no cover
    import sys

    onnx_path = Path(sys.argv[1] if len(sys.argv) > 1 else "build/supertonic-3-coreml/_onnx/vocoder.onnx")
    json_path = onnx_path.parent / "tts.json"
    model = build_vocoder_from_onnx(onnx_path, tts_json_path=json_path)
    print(f"Built Vocoder with {sum(p.numel() for p in model.parameters())} params")
    x = torch.randn(1, 144, 4)
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {tuple(y.shape)}  (expected (1, {512 * 6 * 4}))")
