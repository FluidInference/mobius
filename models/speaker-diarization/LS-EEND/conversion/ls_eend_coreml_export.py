from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from coreml_inference.ls_eend_coreml import initial_state_tensors, load_coreml_step_module
from torch_inference.ls_eend_runtime import DEFAULT_CHECKPOINT, DEFAULT_CONFIG, load_config, save_json

_MIXED_FP16_INCLUDE_MARKERS = (
    "model.enc.",
    "model.cnn.",
    "enc_ret_",
    "enc_conv_cache",
)

_MIXED_FP16_EXCLUDE_MARKERS = (
    "model.dec.",
    "dec_ret",
    "candidate_dec",
    "attractor",
    "full_logits",
    "decode",
    "convert",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the online LS-EEND step model to CoreML.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "coreml" / "ls_eend_step.mlpackage",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional metadata JSON path. Defaults next to the exported package.",
    )
    parser.add_argument("--device", default="cpu", help="Torch device used during tracing.")
    parser.add_argument(
        "--deployment-target",
        choices=("macos13", "macos14", "macos15"),
        default="macos15",
    )
    parser.add_argument(
        "--compute-precision",
        choices=("float16", "float32", "mixed_float16"),
        default="float32",
        help=(
            "CoreML compute precision. 'mixed_float16' keeps the decoder path in float32 "
            "because the recurrent decoder is numerically unstable in full fp16."
        ),
    )
    return parser


def _deployment_target(name: str):
    mapping = {
        "macos13": ct.target.macOS13,
        "macos14": ct.target.macOS14,
        "macos15": ct.target.macOS15,
    }
    return mapping[name]


def _collect_op_identifiers(op) -> str:
    parts = [getattr(op, "name", ""), getattr(op, "op_type", "")]
    for source, values in getattr(op, "scopes", {}).items():
        parts.append(str(source))
        parts.extend(str(value) for value in values)
    for value in getattr(op, "inputs", {}).values():
        if isinstance(value, (list, tuple)):
            values = value
        else:
            values = (value,)
        for item in values:
            parts.append(getattr(item, "name", ""))
            producer = getattr(item, "op", None)
            if producer is not None:
                parts.append(getattr(producer, "name", ""))
    for output in getattr(op, "outputs", ()):
        parts.append(getattr(output, "name", ""))
    return " ".join(parts)


def _mixed_fp16_selector(op) -> bool:
    identifiers = _collect_op_identifiers(op)
    if any(marker in identifiers for marker in _MIXED_FP16_EXCLUDE_MARKERS):
        return False
    return any(marker in identifiers for marker in _MIXED_FP16_INCLUDE_MARKERS)


def _precision(name: str):
    if name == "float16":
        return ct.precision.FLOAT16
    if name == "float32":
        return ct.precision.FLOAT32
    if name == "mixed_float16":
        return ct.transform.FP16ComputePrecision(op_selector=_mixed_fp16_selector)
    raise ValueError(f"Unsupported compute precision: {name}")


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)

    module, layout, engine = load_coreml_step_module(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    example_state = initial_state_tensors(layout, dtype=np.float32)
    example_inputs = (
        torch.zeros((1, 1, layout.input_dim), dtype=torch.float32, device=args.device),
        torch.from_numpy(example_state["enc_ret_kv"]).to(args.device),
        torch.from_numpy(example_state["enc_ret_scale"]).to(args.device),
        torch.from_numpy(example_state["enc_conv_cache"]).to(args.device),
        torch.from_numpy(example_state["dec_ret_kv"]).to(args.device),
        torch.from_numpy(example_state["dec_ret_scale"]).to(args.device),
        torch.from_numpy(example_state["top_buffer"]).to(args.device),
        torch.tensor([1.0], dtype=torch.float32, device=args.device),
        torch.tensor([0.0], dtype=torch.float32, device=args.device),
    )
    traced = torch.jit.trace(module, example_inputs)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=_deployment_target(args.deployment_target),
        compute_precision=_precision(args.compute_precision),
        inputs=[
            ct.TensorType(name="frame", shape=(1, 1, layout.input_dim), dtype=np.float32),
            ct.TensorType(name="enc_ret_kv", shape=example_state["enc_ret_kv"].shape, dtype=np.float32),
            ct.TensorType(name="enc_ret_scale", shape=example_state["enc_ret_scale"].shape, dtype=np.float32),
            ct.TensorType(name="enc_conv_cache", shape=example_state["enc_conv_cache"].shape, dtype=np.float32),
            ct.TensorType(name="dec_ret_kv", shape=example_state["dec_ret_kv"].shape, dtype=np.float32),
            ct.TensorType(name="dec_ret_scale", shape=example_state["dec_ret_scale"].shape, dtype=np.float32),
            ct.TensorType(name="top_buffer", shape=example_state["top_buffer"].shape, dtype=np.float32),
            ct.TensorType(name="ingest", shape=(1,), dtype=np.float32),
            ct.TensorType(name="decode", shape=(1,), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="full_logits", dtype=np.float32),
            ct.TensorType(name="enc_ret_kv_out", dtype=np.float32),
            ct.TensorType(name="enc_ret_scale_out", dtype=np.float32),
            ct.TensorType(name="enc_conv_cache_out", dtype=np.float32),
            ct.TensorType(name="dec_ret_kv_out", dtype=np.float32),
            ct.TensorType(name="dec_ret_scale_out", dtype=np.float32),
            ct.TensorType(name="top_buffer_out", dtype=np.float32),
        ],
    )
    mlmodel.save(str(args.output))

    metadata_path = args.metadata_json or args.output.with_suffix(".json")
    feat_config = config["data"]["feat"]
    save_json(
        {
            "checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "input_dim": layout.input_dim,
            "full_output_dim": layout.full_output_dim,
            "real_output_dim": layout.real_output_dim,
            "encoder_layers": layout.encoder_layers,
            "decoder_layers": layout.decoder_layers,
            "encoder_dim": layout.encoder_dim,
            "num_heads": layout.num_heads,
            "key_dim": layout.key_dim,
            "head_dim": layout.head_dim,
            "encoder_conv_cache_len": layout.encoder_conv_cache_len,
            "top_buffer_len": layout.top_buffer_len,
            "conv_delay": layout.conv_delay,
            "max_nspks": layout.max_nspks,
            "max_speakers": int(config["data"]["max_speakers"]),
            "frame_hz": engine.model_frame_hz,
            "target_sample_rate": engine.target_sample_rate,
            "sample_rate": int(feat_config["sample_rate"]),
            "win_length": int(feat_config["win_length"]),
            "hop_length": int(feat_config["hop_length"]),
            "n_fft": int(feat_config["n_fft"]),
            "n_mels": int(feat_config["n_mels"]),
            "context_recp": int(config["data"]["context_recp"]),
            "subsampling": int(config["data"]["subsampling"]),
            "feat_type": str(config["data"]["feat_type"]),
            "compute_precision": args.compute_precision,
            "state_shapes": {key: list(value.shape) for key, value in example_state.items()},
            "mixed_fp16_include_markers": list(_MIXED_FP16_INCLUDE_MARKERS),
            "mixed_fp16_exclude_markers": list(_MIXED_FP16_EXCLUDE_MARKERS),
        },
        metadata_path,
    )
    print(f"Saved CoreML package: {args.output}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
