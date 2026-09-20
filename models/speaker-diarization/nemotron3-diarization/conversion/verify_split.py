"""Parity gate for the split graph: host-side pipeline + CoreML vs forward_for_export.

Replicates exactly what the Swift host will do — feature stacking as a reshape, the
1024->512 projection via the exported weight file, state packing, additive attention
bias and output mask — then compares against the unpatched NeMo reference.

Usage: uv run python verify_split.py --variants s32 c192 c256
"""

import argparse

import coremltools as ct
import numpy as np
import torch

import config
from convert import apply_variant, load_model
from verify import cases

NEG_BIAS = -30000.0


def host_pipeline(mlmodel, proj_t, v, ex):
    """chunk mel + states -> preds via host stacking/packing + split CoreML graph."""
    chunk, chunk_len, spkcache, sc_len, fifo, fifo_len = ex
    T = config.packed_frames(v)
    sub = config.SUBSAMPLING

    # Feature stacking = reshape on the zero-padded fixed-size mel buffer.
    mel = chunk[0].numpy()  # [mel_frames, 128]
    stacked = mel.reshape(-1, 128 * sub)  # [chunk_enc_capacity, 1024]
    chunk_embs = stacked @ proj_t  # [chunk_enc_capacity, 512]
    enc_len = (int(chunk_len[0]) + sub - 1) // sub

    sc_n, fifo_n = int(sc_len[0]), int(fifo_len[0])
    packed = np.zeros((1, T, config.EMB_DIM), np.float32)
    pos = 0
    for part, n in ((spkcache[0].numpy(), sc_n), (fifo[0].numpy(), fifo_n), (chunk_embs, enc_len)):
        packed[0, pos : pos + n] = part[:n]
        pos += n

    attn_bias = np.zeros((1, 1, 1, T), np.float32)
    attn_bias[..., pos:] = NEG_BIAS
    output_mask = np.zeros((1, T, 1), np.float32)
    output_mask[0, :pos] = 1.0

    out = mlmodel.predict(
        {"packed": packed, "attn_bias": attn_bias, "output_mask": output_mask}
    )
    return out["speaker_preds"], out["speaker_preds_10ms"], chunk_embs[:enc_len]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=["s32"])
    args = parser.parse_args()

    ref_model = load_model()
    proj_t = np.fromfile("build/pre_encode_proj_t.bin", np.float32).reshape(1024, 512)

    for name in args.variants:
        v = config.VARIANTS[name]
        apply_variant(ref_model, v)
        mlmodel = ct.models.MLModel(
            f"build/Nemotron3Diarizer_{name}_split.mlpackage",
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        print(f"=== {name}_split ===")
        for case_name, ex in cases(v):
            with torch.no_grad():
                ref = ref_model.forward_for_export(
                    ex[0], ex[1].long(), ex[2], ex[3].long(), ex[4], ex[5].long()
                )
            preds, hires, host_embs = host_pipeline(mlmodel, proj_t, v, ex)
            d_preds = np.abs(preds - ref[0].numpy()).max()
            d_embs = np.abs(host_embs - ref[1][0, : host_embs.shape[0]].numpy()).max()
            print(f"  {case_name:22s} preds {d_preds:.3e}  host-embs-vs-torch {d_embs:.3e}")


if __name__ == "__main__":
    main()
