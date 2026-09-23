"""Convert Nemotron 3 Diarization to CoreML, one mlpackage per latency variant.

Usage:
    uv run python convert.py --variants low            # single variant
    uv run python convert.py --manifest campaign.json  # generated campaign variants
    uv run python convert.py                           # all variants
"""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

import config
from export_patches import apply_patches
from wrappers import Nemotron3ExportWrapper


def load_model():
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(
        restore_path=config.NEMO_CHECKPOINT, map_location="cpu", strict=False
    )
    model.eval()
    return model


def apply_variant(model, v: dict):
    sm = model.sortformer_modules
    sm.spkcache_len = v["spkcache_len"]
    sm.fifo_len = v["fifo_len"]
    sm.chunk_len = v["chunk_len"]
    sm.chunk_right_context = v["right_context"]
    sm.chunk_left_context = v["left_context"]
    sm.spkcache_update_period = v["update_period"]
    model._check_streaming_parameters()


def example_inputs(v: dict):
    mel_frames = config.chunk_mel_frames(v)
    torch.manual_seed(0)
    return (
        torch.randn(1, mel_frames, config.FEAT_DIM),
        torch.tensor([mel_frames], dtype=torch.int32),
        torch.randn(1, v["spkcache_len"], config.EMB_DIM),
        torch.tensor([v["spkcache_len"] // 2], dtype=torch.int32),
        torch.randn(1, v["fifo_len"], config.EMB_DIM),
        torch.tensor([v["fifo_len"] // 2], dtype=torch.int32),
    )


def convert_variant(model, name: str, v: dict, out_dir: Path):
    apply_variant(model, v)
    wrapper = Nemotron3ExportWrapper(model, config.packed_frames(v))
    wrapper.eval()

    ex = example_inputs(v)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, ex)

    mel_frames = config.chunk_mel_frames(v)
    inputs = [
        ct.TensorType(name="chunk", shape=(1, mel_frames, config.FEAT_DIM), dtype=np.float32),
        ct.TensorType(name="chunk_lengths", shape=(1,), dtype=np.int32),
        ct.TensorType(name="spkcache", shape=(1, v["spkcache_len"], config.EMB_DIM), dtype=np.float32),
        ct.TensorType(name="spkcache_lengths", shape=(1,), dtype=np.int32),
        ct.TensorType(name="fifo", shape=(1, v["fifo_len"], config.EMB_DIM), dtype=np.float32),
        ct.TensorType(name="fifo_lengths", shape=(1,), dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="speaker_preds", dtype=np.float32),
        ct.TensorType(name="chunk_pre_encode_embs", dtype=np.float32),
        ct.TensorType(name="chunk_pre_encode_lengths", dtype=np.int32),
        ct.TensorType(name="speaker_preds_10ms", dtype=np.float32),
    ]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        convert_to="mlprogram",
    )
    mlmodel.short_description = (
        f"NVIDIA Nemotron 3 Diarization ({name}, latency {(v['chunk_len'] + v['right_context']) * 0.08:.2f}s, "
        f"8 speakers). Converted from the OpenMDW-1.1 general-access checkpoint."
    )
    out_path = out_dir / f"Nemotron3Diarizer_{name}.mlpackage"
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", help="variant names (defaults to every available variant)")
    parser.add_argument("--manifest", help="JSON campaign manifest with additional/overridden variants")
    parser.add_argument("--output-dir", default="build")
    args = parser.parse_args()

    variants = dict(config.VARIANTS)
    if args.manifest:
        variants.update(config.load_variant_manifest(args.manifest))
    selected = args.variants if args.variants is not None else list(variants)
    unknown = sorted(set(selected) - set(variants))
    if unknown:
        parser.error(f"unknown variants: {', '.join(unknown)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = apply_patches(load_model())
    for name in selected:
        convert_variant(model, name, variants[name], out_dir)


if __name__ == "__main__":
    main()
