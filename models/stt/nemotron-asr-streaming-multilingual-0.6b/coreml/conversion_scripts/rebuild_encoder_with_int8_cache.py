"""Rebuild ONLY the encoder.mlpackage with Int8 cache I/O + dynamic scales.

C2 cache state INT8 compression — compile-only attempt to verify
whether the modified encoder graph compiles + loads. Swift integration
is a separate (large) effort.

Schema changes vs encprojsplit:
  Inputs:
    cache_channel_int8 (Int8) replacing cache_channel (Float32)
    cache_channel_scale (Float32 [1]) NEW
    cache_time_int8 (Int8) replacing cache_time (Float32)
    cache_time_scale (Float32 [1]) NEW
  Outputs:
    cache_channel_out_int8 (Int8) replacing cache_channel_out (Float32)
    cache_channel_out_scale (Float32 [1]) NEW
    cache_time_out_int8 (Int8) replacing cache_time_out (Float32)
    cache_time_out_scale (Float32 [1]) NEW

Note: tokenizer-driven keep-set (uses existing shipped tokenizer.json)
so vocab matches an existing decoder/joint bundle.

Usage:
    .venv/bin/python conversion_scripts/rebuild_encoder_with_int8_cache.py \\
        --nemo-path ... \\
        --reference-tokenizer-json /path/to/existing/build/tokenizer.json \\
        --output-dir build_lp_engprune_42_13_4480ms_v4_int8cache \\
        --att-context 42,13 --chunk-mel-frames 448
"""
from __future__ import annotations

import json as _json
import sys
from pathlib import Path
from typing import List

import coremltools as ct
import numpy as np
import torch
import typer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

_ENG_COREML = (
    THIS_DIR.parent.parent.parent
    / "nemotron-speech-streaming-0.6b"
    / "coreml"
    / "conversion_scripts"
)
sys.path.insert(0, str(_ENG_COREML))

from individual_components import ExportSettings, _coreml_convert  # type: ignore
from multilingual_components_int8cache import (  # type: ignore
    EncoderStreamingWithInt8Cache,
)
from patch_uniquebias import patch_and_log  # type: ignore
from patch_vocab_prune import prune_vocab_english  # type: ignore
from build_joint_noencproj_batched_from_tokenizer import (  # type: ignore
    recover_keep_ids_from_tokenizer_json,
)


def _tensor_shape(t):
    return tuple(int(s) for s in t.shape)


def _parse_cu(s: str) -> ct.ComputeUnit:
    return {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }[s.upper()]


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def build(
    nemo_path: Path = typer.Option(..., "--nemo-path"),
    reference_tokenizer_json: Path = typer.Option(..., "--reference-tokenizer-json"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    att_context: str = typer.Option("42,13", "--att-context"),
    chunk_mel_frames: int = typer.Option(448, "--chunk-mel-frames"),
    pre_encode_cache: int = typer.Option(9, "--pre-encode-cache"),
    encoder_cu: str = typer.Option("CPU_AND_GPU", "--encoder-cu"),
):
    import nemo.collections.asr as nemo_asr

    typer.echo(f"Loading {nemo_path}...")
    m = nemo_asr.models.ASRModel.restore_from(str(nemo_path), map_location="cpu")
    m.eval()
    patch_and_log(m.encoder)

    old_blank = int(m.decoder.blank_idx)
    keep_ids = recover_keep_ids_from_tokenizer_json(
        m.tokenizer, reference_tokenizer_json, old_blank_idx=old_blank
    )
    typer.echo(f"  [vocab-prune] recovered {len(keep_ids)} ids from tokenizer.json")
    prune_vocab_english(m, keep_ids)

    left, right = [int(x) for x in att_context.split(",")]
    m.encoder.set_default_att_context_size([left, right])

    NUM_PROMPTS = 128
    encoder_streaming = EncoderStreamingWithInt8Cache(
        m.encoder.eval(),
        m.prompt_kernel.eval(),
        joint_enc=m.joint.enc.eval(),
        num_prompts=NUM_PROMPTS,
    )

    mel_features = 128
    total_mel_frames = chunk_mel_frames + pre_encode_cache

    cc, ct_, cl = m.encoder.get_initial_cache_state(batch_size=1, device="cpu")
    cache_channel_b = cc.transpose(0, 1).contiguous()
    cache_time_b = ct_.transpose(0, 1).contiguous()
    cache_len = cl.to(torch.int32)

    mel = torch.randn(1, mel_features, total_mel_frames)
    mel_len = torch.tensor([total_mel_frames], dtype=torch.int32)
    prompt_id = torch.tensor([0], dtype=torch.int32)

    # Trace with Float caches and Float scales (PyTorch trace must use Float)
    cache_channel_f = cache_channel_b.float()
    cache_time_f = cache_time_b.float()
    cache_channel_scale_init = torch.tensor([1.0], dtype=torch.float32)
    cache_time_scale_init = torch.tensor([1.0], dtype=torch.float32)

    typer.echo("Tracing encoder with Int8 cache I/O...")
    traced = torch.jit.trace(
        encoder_streaming,
        (mel, mel_len, cache_channel_f, cache_channel_scale_init,
         cache_time_f, cache_time_scale_init, cache_len, prompt_id),
        strict=False,
    )

    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        max_audio_seconds=80.0,
        max_symbol_steps=1,
        chunk_size_frames=chunk_mel_frames // 8,
        cache_size=cache_channel_b.shape[2],
    )

    typer.echo("Converting encoder mlprogram (Int8 cache + scale I/O)...")
    mlmodel = _coreml_convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=_tensor_shape(mel), dtype=np.float32),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
            # Int8 cache inputs (declared as Int8; coremltools auto-casts to Float at boundary)
            ct.TensorType(
                name="cache_channel_int8",
                shape=_tensor_shape(cache_channel_b),
                dtype=np.int8,
            ),
            ct.TensorType(name="cache_channel_scale", shape=(1,), dtype=np.float32),
            ct.TensorType(
                name="cache_time_int8",
                shape=_tensor_shape(cache_time_b),
                dtype=np.int8,
            ),
            ct.TensorType(name="cache_time_scale", shape=(1,), dtype=np.float32),
            ct.TensorType(name="cache_len", shape=(1,), dtype=np.int32),
            ct.TensorType(name="prompt_id", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="encoded", dtype=np.float32),
            ct.TensorType(name="encoded_length", dtype=np.int32),
            ct.TensorType(name="cache_channel_out_int8", dtype=np.int8),
            ct.TensorType(name="cache_channel_out_scale", dtype=np.float32),
            ct.TensorType(name="cache_time_out_int8", dtype=np.int8),
            ct.TensorType(name="cache_time_out_scale", dtype=np.float32),
            ct.TensorType(name="cache_len_out", dtype=np.int32),
            ct.TensorType(name="encoder_proj", dtype=np.float32),
        ],
        settings=settings,
        compute_units_override=_parse_cu(encoder_cu),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pkg = output_dir / "encoder.mlpackage"
    mlmodel.save(str(out_pkg))
    typer.echo(f"Saved {out_pkg}")


if __name__ == "__main__":
    app()
