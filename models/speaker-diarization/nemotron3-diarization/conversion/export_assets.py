"""Export the root-level host assets that sit alongside the CoreML bundles.

    uv run python export_assets.py --output-dir build

`learnable_sil_emb.bin`  — the silence embedding the host seeds empty speaker-cache
                           slots with, [EMB_DIM] fp32.
`pre_encode_proj_t.bin`  — transpose of the FeatureStacking projection, [1024, 512]
                           fp32 row-major, used by the split graphs where the host
                           does the pre-encode sgemm itself.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

import config


def export(model, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sil = model.sortformer_modules.learnable_sil_emb.detach().float().reshape(-1).cpu().numpy()
    assert sil.shape == (config.EMB_DIM,), sil.shape
    sil.astype(np.float32).tofile(out_dir / "learnable_sil_emb.bin")
    print(f"learnable_sil_emb.bin  {sil.shape} fp32")

    # nn.Linear stores [out_features, in_features]; the host multiplies row-vectors, so
    # it needs the transpose laid out row-major.
    w = model.encoder.pre_encode.proj.weight.detach().float().cpu().numpy()
    assert w.shape == (config.EMB_DIM, config.FEAT_DIM * config.SUBSAMPLING), w.shape
    proj_t = np.ascontiguousarray(w.T)
    assert proj_t.shape == (config.FEAT_DIM * config.SUBSAMPLING, config.EMB_DIM), proj_t.shape
    proj_t.astype(np.float32).tofile(out_dir / "pre_encode_proj_t.bin")
    print(f"pre_encode_proj_t.bin  {proj_t.shape} fp32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build")
    parser.add_argument("--checkpoint", default=None, help="override config.NEMO_CHECKPOINT")
    args = parser.parse_args()

    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(
        restore_path=args.checkpoint or config.NEMO_CHECKPOINT, map_location="cpu", strict=False
    )
    model.eval()
    with torch.no_grad():
        export(model, Path(args.output_dir))


if __name__ == "__main__":
    main()
