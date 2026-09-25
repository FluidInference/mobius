"""Split-graph conversion: pure floating-point transformer+head, everything else host-side.

The monolithic graph's ANECCompile cliff triggers on the chunk branch (FeatureStacking
input length: mel 1376 compiles, 1440 fails), and its int32 index arithmetic both
falls to CPU and poisons W8A8 activation quantization. This variant moves feature
stacking (a free reshape), the 1024->512 projection, state packing, and mask/bias
construction to the host, leaving a graph of pure LN/matmul/softmax/conv ops:

    inputs:  packed (1, T, 512) fp32       — [spkcache | fifo | chunk] embs, zero-padded
             attn_bias (1, 1, 1, T) fp32   — 0 for valid keys, -30000 for padding
             output_mask (1, T, 1) fp32    — 1 for valid frames, 0 for padding
    outputs: speaker_preds (1, T, 8), speaker_preds_10ms (1, T*8, 8)

Host needs `pre_encode_proj_t.bin` ([1024, 512] fp32 row-major) for the projection.

Usage: uv run python convert_split.py --variants fast32 c192
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn

import config
from convert import apply_variant, load_model
from export_patches import apply_patches


class Nemotron3SplitWrapper(nn.Module):
    """packed embs + fp masks -> speaker preds (80 ms + 10 ms). Pure fp tensor ops."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, packed, attn_bias, output_mask):
        enc = self.model.encoder
        sm = self.model.sortformer_modules

        x = enc.dropout_pre_encoder(packed)  # eval no-op; xscaling is false
        x = enc.embed_norm(x)
        for layer in enc.layers:
            x = layer(x, block_mask=attn_bias, pos_emb=None)
        x = enc.final_norm(x)
        emb_seq = sm.encoder_proj(x)

        T = emb_seq.shape[1]
        hidden = sm.upsample_hidden(emb_seq)
        up = self.model.upsample_factor
        mask_hires = (
            output_mask.expand(-1, T, up).reshape(-1, T * up, 1)
        )
        preds_hires = sm.forward_speaker_sigmoids(hidden) * mask_hires
        preds = sm.downsample_preds(preds_hires, up)
        return preds, preds_hires


def convert_variant(model, name, v, out_dir, batch=1):
    apply_variant(model, v)
    wrapper = Nemotron3SplitWrapper(model).eval()
    T = config.packed_frames(v)

    torch.manual_seed(0)
    ex = (
        torch.randn(batch, T, config.EMB_DIM) * 0.5,
        torch.zeros(batch, 1, 1, T),
        torch.ones(batch, T, 1),
    )
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, ex)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="packed", shape=(batch, T, config.EMB_DIM), dtype=np.float32),
            ct.TensorType(name="attn_bias", shape=(batch, 1, 1, T), dtype=np.float32),
            ct.TensorType(name="output_mask", shape=(batch, T, 1), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="speaker_preds", dtype=np.float32),
            ct.TensorType(name="speaker_preds_10ms", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        convert_to="mlprogram",
    )
    mlmodel.short_description = (
        f"Nemotron 3 Diarization preview split-graph ({name}, T={T}). "
        "Internal evaluation only — NVIDIA eval license, do not redistribute."
    )
    out_path = out_dir / f"Nemotron3Diarizer_{name}_split.mlpackage"
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=["fast32"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output-dir", default="build")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    model = apply_patches(load_model())
    for name in args.variants:
        suffix = f"_b{args.batch}" if args.batch > 1 else ""
        convert_variant(model, name + suffix if False else name, config.VARIANTS[name], out_dir, batch=args.batch)


if __name__ == "__main__":
    main()
