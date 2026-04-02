#!/usr/bin/env python3
"""Export standalone CTC decoder head from parakeet-ctc-0.6b-zh-cn (hybrid model) to CoreML.

The CTC head is a single linear projection (1024 -> vocab_size+1) that maps encoder
features to CTC log-probabilities for Mandarin Chinese (Simplified) transcription.
This model is a hybrid RNNT+CTC model; we extract only the CTC decoder component.

Usage:
    uv run python export-ctc-zh-cn.py --nemo-path ../parakeet-ctc-riva-0-6b-unified-zh-cn_vtrainable_v3.0/*.nemo --output-dir ./build
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import typer

import nemo.collections.asr as nemo_asr

from individual_components import CTCDecoderWrapper

AUTHOR = "Fluid Inference"

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


@app.command()
def export(
    nemo_path: Path = typer.Option(
        ...,
        "--nemo-path",
        exists=True,
        resolve_path=True,
        help="Path to zh-CN pure CTC .nemo checkpoint",
    ),
    output_dir: Path = typer.Option(
        Path("build"),
        help="Output directory for CtcHeadZhCn.mlpackage",
    ),
    max_audio_seconds: float = typer.Option(
        15.0, "--max-audio-seconds", help="Fixed waveform window (seconds)"
    ),
    compute_units: str = typer.Option(
        "CPU_AND_NE",
        "--compute-units",
        help="Compute units: ALL, CPU_ONLY, CPU_AND_NE",
    ),
) -> None:
    """Export CTC decoder head from zh-CN pure CTC 0.6B model as standalone CoreML model."""

    cu_mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    cu = cu_mapping.get(compute_units.upper())
    if cu is None:
        raise typer.BadParameter(f"Unknown compute units: {compute_units}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load hybrid model (has both RNNT and CTC decoders)
    typer.echo(f"Loading zh-CN hybrid model from {nemo_path}...")
    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        str(nemo_path), map_location="cpu"
    )
    asr_model.eval()

    # Probe dimensions
    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    max_samples = int(round(max_audio_seconds * sample_rate))
    vocab_size = int(asr_model.tokenizer.vocab_size)

    # Get encoder hidden size by running a reference forward pass
    typer.echo("Running reference forward pass to determine shapes...")
    dummy_audio = torch.randn(1, max_samples, dtype=torch.float32)
    dummy_length = torch.tensor([max_samples], dtype=torch.int32)

    with torch.inference_mode():
        mel, mel_length = asr_model.preprocessor(
            input_signal=dummy_audio, length=dummy_length.long()
        )
        encoded, encoded_length = asr_model.encoder(
            audio_signal=mel, length=mel_length.long()
        )

    encoder_dim = int(encoded.shape[1])  # [B, D, T]
    time_steps = int(encoded.shape[2])
    typer.echo(f"  encoder output: [{encoded.shape[0]}, {encoder_dim}, {time_steps}]")
    typer.echo(f"  vocab_size: {vocab_size}, ctc_classes: {vocab_size + 1}")

    # Wrap CTC decoder head (hybrid model uses .ctc_decoder)
    ctc_decoder = CTCDecoderWrapper(asr_model.ctc_decoder.eval())

    # Reference CTC output
    with torch.inference_mode():
        ctc_ref = ctc_decoder(encoded)
    typer.echo(f"  ctc_ref output: {_tensor_shape(ctc_ref)}")

    # Trace
    typer.echo("Tracing CTC decoder head...")
    dummy_encoder_output = torch.randn(1, encoder_dim, time_steps, dtype=torch.float32)
    traced = torch.jit.trace(ctc_decoder, (dummy_encoder_output,), strict=False)
    traced.eval()

    # Verify trace output
    with torch.inference_mode():
        trace_out = traced(dummy_encoder_output)
    typer.echo(f"  traced output: {_tensor_shape(trace_out)}")
    max_diff = (ctc_ref - trace_out).abs().max().item()
    typer.echo(f"  max diff (ref vs traced): {max_diff:.6e}")

    # Convert to CoreML
    typer.echo("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="encoder_output",
                shape=(1, encoder_dim, time_steps),
                dtype=np.float32,
            ),
        ],
        outputs=[
            ct.TensorType(name="ctc_logits", dtype=np.float32),
        ],
        compute_units=cu,
        minimum_deployment_target=ct.target.iOS17,
    )

    # Save
    mlpackage_path = output_dir / "CtcHeadZhCn.mlpackage"
    mlmodel.short_description = (
        f"CTC decoder head from parakeet-ctc-0.6b-zh-cn "
        f"(encoder_dim={encoder_dim}, vocab={vocab_size}+1 blank)"
    )
    mlmodel.author = AUTHOR
    mlmodel.save(str(mlpackage_path))
    typer.echo(f"Saved: {mlpackage_path}")

    # Save vocabulary
    sp = asr_model.tokenizer.tokenizer
    vocab_list = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab_list, ensure_ascii=False, indent=2))
    typer.echo(f"Saved vocabulary ({len(vocab_list)} tokens) to {vocab_path}")

    # Save metadata
    metadata = {
        "model": "parakeet-ctc-0.6b-zh-cn-ctc-head",
        "language": "zh-CN (Mandarin Chinese Simplified)",
        "source": str(nemo_path),
        "encoder_dim": encoder_dim,
        "time_steps": time_steps,
        "vocab_size": vocab_size,
        "ctc_classes": vocab_size + 1,
        "blank_id": vocab_size,
        "max_audio_seconds": max_audio_seconds,
        "sample_rate": sample_rate,
        "input": {"encoder_output": [1, encoder_dim, time_steps]},
        "output": {"ctc_logits": list(_tensor_shape(ctc_ref))},
    }
    meta_path = output_dir / "ctc_head_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    typer.echo(f"Saved metadata: {meta_path}")

    typer.echo(f"\nDone! To compile:")
    typer.echo(f"  xcrun coremlcompiler compile {mlpackage_path} {output_dir}/")


if __name__ == "__main__":
    app()
