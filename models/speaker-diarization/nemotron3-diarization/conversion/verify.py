"""Parity check: CoreML mlpackage vs unpatched NeMo torch reference (FlexAttention).

For each variant, runs several randomized state configurations (empty, half-full and
full spkcache/FIFO, partial final chunk) through:
  ref  — the restored model's own ``forward_for_export`` (FlexAttention path),
  wrap — the patched export wrapper in torch (adds the 10 ms output), and
  cml  — the converted mlpackage (fp16 CoreML runtime).

Reports max abs diff of speaker predictions. Torch-vs-torch must be ~0; CoreML is
fp16 so ~1e-2 on sigmoid probabilities is the expected ceiling.

Usage: uv run python verify.py --variants low offline
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

import config
from convert import apply_variant, load_model
from export_patches import apply_patches
from wrappers import Nemotron3ExportWrapper


def cases(v: dict, seed: int = 0):
    mel = config.chunk_mel_frames(v)
    torch.manual_seed(seed)
    scenarios = [
        ("cold-start", 0, 0, mel),
        ("warm-half", v["spkcache_len"] // 2, v["fifo_len"] // 2, mel),
        ("steady-full", v["spkcache_len"], v["fifo_len"], mel),
        ("final-partial-chunk", v["spkcache_len"], v["fifo_len"] // 3, max(mel - 24, 8)),
    ]
    for name, sc_len, fifo_len, chunk_len in scenarios:
        yield name, (
            torch.randn(1, mel, config.FEAT_DIM),
            torch.tensor([chunk_len], dtype=torch.int32),
            torch.randn(1, v["spkcache_len"], config.EMB_DIM) * 0.5,
            torch.tensor([sc_len], dtype=torch.int32),
            torch.randn(1, v["fifo_len"], config.EMB_DIM) * 0.5,
            torch.tensor([fifo_len], dtype=torch.int32),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(config.VARIANTS))
    parser.add_argument("--build-dir", default="build")
    args = parser.parse_args()

    ref_model = load_model()
    patched_model = apply_patches(load_model())

    for name in args.variants:
        v = config.VARIANTS[name]
        pkg = Path(args.build_dir) / f"Nemotron3Diarizer_{name}.mlpackage"
        print(f"\n=== {name} ({pkg.name}) ===")
        mlmodel = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)

        apply_variant(ref_model, v)
        apply_variant(patched_model, v)
        wrapper = Nemotron3ExportWrapper(patched_model, config.packed_frames(v)).eval()

        for case_name, ex in cases(v):
            ex64 = list(ex)
            with torch.no_grad():
                ref = ref_model.forward_for_export(
                    ex64[0], ex64[1].long(), ex64[2], ex64[3].long(), ex64[4], ex64[5].long()
                )
                wrap = wrapper(*ex)

            torch_diff = (ref[0] - wrap[0]).abs().max().item()

            feed = {
                "chunk": ex[0].numpy(),
                "chunk_lengths": ex[1].numpy().astype(np.int32),
                "spkcache": ex[2].numpy(),
                "spkcache_lengths": ex[3].numpy().astype(np.int32),
                "fifo": ex[4].numpy(),
                "fifo_lengths": ex[5].numpy().astype(np.int32),
            }
            out = mlmodel.predict(feed)
            cml_diff = np.abs(out["speaker_preds"] - ref[0].numpy()).max()
            cml_hires_diff = np.abs(out["speaker_preds_10ms"] - wrap[3].numpy()).max()
            embs_diff = np.abs(out["chunk_pre_encode_embs"] - ref[1].numpy()).max()
            lens_ok = int(out["chunk_pre_encode_lengths"].ravel()[0]) == int(ref[2].ravel()[0])
            print(
                f"  {case_name:22s} torch {torch_diff:.2e} | cml preds {cml_diff:.3e} "
                f"hires {cml_hires_diff:.3e} embs {embs_diff:.3e} lens_ok {lens_ok}"
            )


if __name__ == "__main__":
    main()
