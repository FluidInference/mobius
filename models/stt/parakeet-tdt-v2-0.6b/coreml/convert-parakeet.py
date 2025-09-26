#!/usr/bin/env python3
"""CLI for exporting Parakeet TDT v2 components to CoreML."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import typer

import nemo.collections.asr as nemo_asr

from individual_components import (
    DecoderWrapper,
    EncoderWrapper,
    ExportSettings,
    JointWrapper,
    JointDecisionWrapper,
    JointDecisionSingleStep,
    PreprocessorWrapper,
    MelEncoderWrapper,
    _coreml_convert,
)

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
AUTHOR = "Fluid Inference"


def _compute_length(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def _prepare_audio(
    validation_audio: Optional[Path],
    sample_rate: int,
    max_samples: int,
    seed: Optional[int],
) -> torch.Tensor:
    if validation_audio is None:
        if seed is not None:
            torch.manual_seed(seed)
        audio = torch.randn(1, max_samples, dtype=torch.float32)
        return audio

    data, sr = sf.read(str(validation_audio), dtype="float32")
    if sr != sample_rate:
        raise typer.BadParameter(
            f"Validation audio sample rate {sr} does not match model rate {sample_rate}"
        )

    if data.ndim > 1:
        data = data[:, 0]

    if data.size == 0:
        raise typer.BadParameter("Validation audio is empty")

    if data.size < max_samples:
        pad_width = max_samples - data.size
        data = np.pad(data, (0, pad_width))
    elif data.size > max_samples:
        data = data[:max_samples]

    audio = torch.from_numpy(data).unsqueeze(0).to(dtype=torch.float32)
    return audio


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    # Ensure iOS 17+ target for MLProgram ops and ANE readiness
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def _tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _parse_compute_units(name: str) -> ct.ComputeUnit:
    """Parse a human-friendly compute units string into ct.ComputeUnit.

    Accepted (case-insensitive): ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE.
    """
    normalized = str(name).strip().upper()
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_NEURALENGINE": ct.ComputeUnit.CPU_AND_NE,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute units '{name}'. Choose from: " + ", ".join(mapping.keys())
        )
    return mapping[normalized]


def _parse_compute_precision(name: Optional[str]) -> Optional[ct.precision]:
    """Parse compute precision string into ct.precision or None.

    Accepted (case-insensitive): FLOAT32, FLOAT16. If None/empty, returns None (tool default).
    """
    if name is None:
        return None
    normalized = str(name).strip().upper()
    if normalized == "":
        return None
    mapping = {
        "FLOAT32": ct.precision.FLOAT32,
        "FLOAT16": ct.precision.FLOAT16,
    }
    if normalized not in mapping:
        raise typer.BadParameter(
            f"Unknown compute precision '{name}'. Choose from: " + ", ".join(mapping.keys())
        )
    return mapping[normalized]


# Validation logic removed; use compare-compnents.py for comparisons.


# Fixed export choices: CPU_ONLY + FP32, min target iOS17


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    nemo_path: Optional[Path] = typer.Option(
        None,
        "--nemo-path",
        exists=True,
        resolve_path=True,
        help="Path to parakeet-tdt-0.6b-v2 .nemo checkpoint (skip to auto-download)",
    ),
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="Model identifier to download when --nemo-path is omitted",
    ),
    output_dir: Path = typer.Option(Path("parakeet_coreml"), help="Directory where mlpackages and metadata will be written"),
    preprocessor_cu: str = typer.Option(
        "CPU_ONLY",
        "--preprocessor-cu",
        help="Compute units for preprocessor (default CPU_ONLY)",
    ),
    mel_encoder_cu: str = typer.Option(
        "CPU_ONLY",
        "--mel-encoder-cu",
        help="Compute units for fused mel+encoder (default CPU_ONLY)",
    ),
    compute_precision: Optional[str] = typer.Option(
        None,
        "--compute-precision",
        help="Export precision: FLOAT32 (default) or FLOAT16 to shrink non-quantized weights.",
    ),
) -> None:
    """Export all Parakeet sub-modules to CoreML with a fixed 15-second window."""
    # Runtime CoreML contract keeps U=1 so the prediction net matches the streaming decoder.
    export_settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,  # Default: CPU-only for all components
        deployment_target=ct.target.iOS17,  # iOS 17+ features and kernels
        compute_precision=_parse_compute_precision(compute_precision),
        max_audio_seconds=15.0,
        max_symbol_steps=1,
    )

    typer.echo("Export configuration:")
    typer.echo(asdict(export_settings))

    output_dir.mkdir(parents=True, exist_ok=True)
    pre_cu = _parse_compute_units(preprocessor_cu)
    melenc_cu = _parse_compute_units(mel_encoder_cu)

    if nemo_path is not None:
        typer.echo(f"Loading NeMo model from {nemo_path}…")
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(
            str(nemo_path), map_location="cpu"
        )
        checkpoint_meta = {
            "type": "file",
            "path": str(nemo_path),
        }
    else:
        typer.echo(f"Downloading NeMo model via {model_id}…")
        asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
            model_id, map_location="cpu"
        )
        checkpoint_meta = {
            "type": "pretrained",
            "model_id": model_id,
        }
    asr_model.eval()

    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    max_samples = _compute_length(export_settings.max_audio_seconds, sample_rate)
    # Prefer a bundled 15s 16kHz audio if available
    default_audio = (Path(__file__).parent / "audio" / "yc_first_minute_16k_15s.wav").resolve()
    if not default_audio.exists():
        raise typer.BadParameter(f"Expected 15s trace audio at {default_audio}; add the file to proceed.")
    typer.echo(f"Using trace audio: {default_audio}")
    audio_tensor = _prepare_audio(default_audio, sample_rate, max_samples, seed=None)
    audio_length = torch.tensor([max_samples], dtype=torch.int32)

    preprocessor = PreprocessorWrapper(asr_model.preprocessor.eval())
    encoder = EncoderWrapper(asr_model.encoder.eval())
    decoder = DecoderWrapper(asr_model.decoder.eval())
    joint = JointWrapper(asr_model.joint.eval())

    decoder_export_flag = getattr(asr_model.decoder, "_rnnt_export", False)
    asr_model.decoder._rnnt_export = True

    try:
        with torch.inference_mode():
            mel_ref, mel_length_ref = preprocessor(audio_tensor, audio_length)
            mel_length_ref = mel_length_ref.to(dtype=torch.int32)
            encoder_ref, encoder_length_ref = encoder(mel_ref, mel_length_ref)
            encoder_length_ref = encoder_length_ref.to(dtype=torch.int32)

        # Clone Tensors to drop the inference tensor flag before tracing
        mel_ref = mel_ref.clone()
        mel_length_ref = mel_length_ref.clone()
        encoder_ref = encoder_ref.clone()
        encoder_length_ref = encoder_length_ref.clone()

        vocab_size = int(asr_model.tokenizer.vocab_size)
        num_extra = int(asr_model.joint.num_extra_outputs)
        decoder_hidden = int(asr_model.decoder.pred_hidden)
        decoder_layers = int(asr_model.decoder.pred_rnn_layers)

        targets = torch.full(
            (1, export_settings.max_symbol_steps),
            fill_value=asr_model.decoder.blank_idx,
            dtype=torch.int32,
        )
        target_lengths = torch.tensor(
            [export_settings.max_symbol_steps], dtype=torch.int32
        )
        zero_state = torch.zeros(
            decoder_layers,
            1,
            decoder_hidden,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            decoder_ref, h_ref, c_ref = decoder(targets, target_lengths, zero_state, zero_state)
            joint_ref = joint(encoder_ref, decoder_ref)

        decoder_ref = decoder_ref.clone()
        h_ref = h_ref.clone()
        c_ref = c_ref.clone()
        joint_ref = joint_ref.clone()

        typer.echo("Tracing and converting preprocessor…")
        # Ensure tracing happens on CPU explicitly
        preprocessor = preprocessor.cpu()
        audio_tensor = audio_tensor.cpu()
        audio_length = audio_length.cpu()
        traced_preprocessor = torch.jit.trace(
            preprocessor, (audio_tensor, audio_length), strict=False
        )
        traced_preprocessor.eval()
        preprocessor_inputs = [
            # Allow variable-length audio up to the fixed 15s window using RangeDim
            ct.TensorType(
                name="audio_signal",
                shape=(1, ct.RangeDim(1, max_samples)),
                dtype=np.float32,
            ),
            ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
        ]
        preprocessor_outputs = [
            ct.TensorType(name="mel", dtype=np.float32),
            ct.TensorType(name="mel_length", dtype=np.int32),
        ]
        # Preprocessor compute units (parametrized; default CPU_ONLY)
        preprocessor_model = _coreml_convert(
            traced_preprocessor,
            preprocessor_inputs,
            preprocessor_outputs,
            export_settings,
            compute_units_override=pre_cu,
        )
        preprocessor_path = output_dir / "parakeet_preprocessor.mlpackage"
        _save_mlpackage(
            preprocessor_model,
            preprocessor_path,
            "Parakeet preprocessor (15 s window)",
        )

        typer.echo("Tracing and converting encoder…")
        traced_encoder = torch.jit.trace(
            encoder, (mel_ref, mel_length_ref), strict=False
        )
        traced_encoder.eval()
        encoder_inputs = [
            ct.TensorType(name="mel", shape=_tensor_shape(mel_ref), dtype=np.float32),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ]
        encoder_outputs = [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
        ]
        # Encoder: CPU only
        encoder_model = _coreml_convert(
            traced_encoder,
            encoder_inputs,
            encoder_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        encoder_path = output_dir / "parakeet_encoder.mlpackage"
        _save_mlpackage(
            encoder_model,
            encoder_path,
            "Parakeet encoder (15 s window)",
        )

        # Optional fused export: Preprocessor + Encoder
        typer.echo("Tracing and converting fused mel+encoder…")
        mel_encoder = MelEncoderWrapper(preprocessor, encoder)
        traced_mel_encoder = torch.jit.trace(
            mel_encoder, (audio_tensor, audio_length), strict=False
        )
        traced_mel_encoder.eval()
        mel_encoder_inputs = [
            # Keep fixed 15s window for fused Mel+Encoder
            ct.TensorType(name="audio_signal", shape=(1, max_samples), dtype=np.float32),
            ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
        ]
        mel_encoder_outputs = [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
        ]
        # Fused mel+encoder compute units (parametrized; default CPU_ONLY)
        mel_encoder_model = _coreml_convert(
            traced_mel_encoder,
            mel_encoder_inputs,
            mel_encoder_outputs,
            export_settings,
            compute_units_override=melenc_cu,
        )
        mel_encoder_path = output_dir / "parakeet_mel_encoder.mlpackage"
        _save_mlpackage(
            mel_encoder_model,
            mel_encoder_path,
            "Parakeet fused Mel+Encoder (15 s window)",
        )

        typer.echo("Tracing and converting decoder…")
        traced_decoder = torch.jit.trace(
            decoder,
            (targets, target_lengths, zero_state, zero_state),
            strict=False,
        )
        traced_decoder.eval()
        decoder_inputs = [
            ct.TensorType(name="targets", shape=_tensor_shape(targets), dtype=np.int32),
            ct.TensorType(name="target_length", shape=(1,), dtype=np.int32),
            ct.TensorType(name="h_in", shape=_tensor_shape(zero_state), dtype=np.float32),
            ct.TensorType(name="c_in", shape=_tensor_shape(zero_state), dtype=np.float32),
        ]
        decoder_outputs = [
            ct.TensorType(name="decoder", dtype=np.float32),
            ct.TensorType(name="h_out", dtype=np.float32),
            ct.TensorType(name="c_out", dtype=np.float32),
        ]
        # Decoder: CPU only
        decoder_model = _coreml_convert(
            traced_decoder,
            decoder_inputs,
            decoder_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        decoder_path = output_dir / "parakeet_decoder.mlpackage"
        _save_mlpackage(
            decoder_model,
            decoder_path,
            "Parakeet decoder (RNNT prediction network)",
        )

        typer.echo("Tracing and converting joint…")
        traced_joint = torch.jit.trace(
            joint,
            (encoder_ref, decoder_ref),
            strict=False,
        )
        traced_joint.eval()
        joint_inputs = [
            ct.TensorType(name="encoder", shape=_tensor_shape(encoder_ref), dtype=np.float32),
            ct.TensorType(name="decoder", shape=_tensor_shape(decoder_ref), dtype=np.float32),
        ]
        joint_outputs = [
            ct.TensorType(name="logits", dtype=np.float32),
        ]
        # Joint: CPU only
        joint_model = _coreml_convert(
            traced_joint,
            joint_inputs,
            joint_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        joint_path = output_dir / "parakeet_joint.mlpackage"
        _save_mlpackage(
            joint_model,
            joint_path,
            "Parakeet joint network (RNNT)",
        )

        # Joint + decision head (split logits, softmax, argmax)
        typer.echo("Tracing and converting joint decision head…")
        vocab_size = int(asr_model.tokenizer.vocab_size)
        num_extra = int(asr_model.joint.num_extra_outputs)
        joint_decision = JointDecisionWrapper(joint, vocab_size=vocab_size, num_extra=num_extra)
        traced_joint_decision = torch.jit.trace(
            joint_decision,
            (encoder_ref, decoder_ref),
            strict=False,
        )
        traced_joint_decision.eval()
        joint_decision_inputs = [
            ct.TensorType(name="encoder", shape=_tensor_shape(encoder_ref), dtype=np.float32),
            ct.TensorType(name="decoder", shape=_tensor_shape(decoder_ref), dtype=np.float32),
        ]
        joint_decision_outputs = [
            ct.TensorType(name="token_id", dtype=np.int32),
            ct.TensorType(name="token_prob", dtype=np.float32),
            ct.TensorType(name="duration", dtype=np.int32),
        ]
        # JointDecision: CPU only
        joint_decision_model = _coreml_convert(
            traced_joint_decision,
            joint_decision_inputs,
            joint_decision_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        joint_decision_path = output_dir / "parakeet_joint_decision.mlpackage"
        _save_mlpackage(
            joint_decision_model,
            joint_decision_path,
            "Parakeet joint + decision head (split, softmax, argmax)",
        )

        # Single-step JointDecision for [1,1024,1] x [1,640,1] -> [1,1,1]
        typer.echo("Tracing and converting single-step joint decision…")
        jd_single = JointDecisionSingleStep(joint, vocab_size=vocab_size, num_extra=num_extra)
        # Create single-step slices from refs
        enc_step = encoder_ref[:, :, :1].contiguous()
        dec_step = decoder_ref[:, :, :1].contiguous()
        traced_jd_single = torch.jit.trace(
            jd_single,
            (enc_step, dec_step),
            strict=False,
        )
        traced_jd_single.eval()
        jd_single_inputs = [
            ct.TensorType(name="encoder_step", shape=(1, enc_step.shape[1], 1), dtype=np.float32),
            ct.TensorType(name="decoder_step", shape=(1, dec_step.shape[1], 1), dtype=np.float32),
        ]
        jd_single_outputs = [
            ct.TensorType(name="token_id", dtype=np.int32),
            ct.TensorType(name="token_prob", dtype=np.float32),
            ct.TensorType(name="duration", dtype=np.int32),
        ]
        # Single-step JointDecision: CPU only
        jd_single_model = _coreml_convert(
            traced_jd_single,
            jd_single_inputs,
            jd_single_outputs,
            export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        jd_single_path = output_dir / "parakeet_joint_decision_single_step.mlpackage"
        _save_mlpackage(
            jd_single_model,
            jd_single_path,
            "Parakeet single-step joint decision (current frame)",
        )

        metadata: Dict[str, object] = {
            "model_id": model_id,
            "sample_rate": sample_rate,
            "max_audio_seconds": export_settings.max_audio_seconds,
            "max_audio_samples": max_samples,
            "max_symbol_steps": export_settings.max_symbol_steps,
            "vocab_size": vocab_size,
            "joint_extra_outputs": num_extra,
            "checkpoint": checkpoint_meta,
            "coreml": {
                "compute_units": export_settings.compute_units.name,
                "compute_precision": (
                    export_settings.compute_precision.name
                    if export_settings.compute_precision is not None
                    else "FLOAT32"
                ),
            },
            "components": {
                "preprocessor": {
                    "inputs": {
                        "audio_signal": list(_tensor_shape(audio_tensor)),
                        "audio_length": [1],
                    },
                    "outputs": {
                        "mel": list(_tensor_shape(mel_ref)),
                        "mel_length": [1],
                    },
                    "path": preprocessor_path.name,
                },
                "encoder": {
                    "inputs": {
                        "mel": list(_tensor_shape(mel_ref)),
                        "mel_length": [1],
                    },
                    "outputs": {
                        "encoder": list(_tensor_shape(encoder_ref)),
                        "encoder_length": [1],
                    },
                    "path": encoder_path.name,
                },
                "mel_encoder": {
                    "inputs": {
                        "audio_signal": [1, max_samples],
                        "audio_length": [1],
                    },
                    "outputs": {
                        "encoder": list(_tensor_shape(encoder_ref)),
                        "encoder_length": [1],
                    },
                    "path": mel_encoder_path.name,
                },
                "decoder": {
                    "inputs": {
                        "targets": list(_tensor_shape(targets)),
                        "target_length": [1],
                        "h_in": list(_tensor_shape(zero_state)),
                        "c_in": list(_tensor_shape(zero_state)),
                    },
                    "outputs": {
                        "decoder": list(_tensor_shape(decoder_ref)),
                        "h_out": list(_tensor_shape(h_ref)),
                        "c_out": list(_tensor_shape(c_ref)),
                    },
                    "path": decoder_path.name,
                },
                "joint": {
                    "inputs": {
                        "encoder": list(_tensor_shape(encoder_ref)),
                        "decoder": list(_tensor_shape(decoder_ref)),
                    },
                    "outputs": {
                        "logits": list(_tensor_shape(joint_ref)),
                    },
                    "path": joint_path.name,
                },
                "joint_decision": {
                    "inputs": {
                        "encoder": list(_tensor_shape(encoder_ref)),
                        "decoder": list(_tensor_shape(decoder_ref)),
                    },
                    "outputs": {
                        "token_id": [
                            _tensor_shape(encoder_ref)[0],
                            _tensor_shape(encoder_ref)[1],
                            _tensor_shape(decoder_ref)[1],
                        ],
                        "token_prob": [
                            _tensor_shape(encoder_ref)[0],
                            _tensor_shape(encoder_ref)[1],
                            _tensor_shape(decoder_ref)[1],
                        ],
                        "duration": [
                            _tensor_shape(encoder_ref)[0],
                            _tensor_shape(encoder_ref)[1],
                            _tensor_shape(decoder_ref)[1],
                        ],
                    },
                    "path": joint_decision_path.name,
                },
                "joint_decision_single_step": {
                    "inputs": {
                        "encoder_step": [1, _tensor_shape(encoder_ref)[2-1], 1],
                        "decoder_step": [1, _tensor_shape(decoder_ref)[2-1], 1],
                    },
                    "outputs": {
                        "token_id": [1, 1, 1],
                        "token_prob": [1, 1, 1],
                        "duration": [1, 1, 1],
                    },
                    "path": jd_single_path.name,
                },
            },
        }

        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        typer.echo(f"Export complete. Metadata written to {metadata_path}")

    finally:
        asr_model.decoder._rnnt_export = decoder_export_flag


if __name__ == "__main__":
    app()
