#!/usr/bin/env python3
"""Export hybrid TDT-CTC model (EncDecHybridRNNTCTCBPEModel) to CoreML.

This exports the TDT components (encoder, RNNT decoder, joint with duration)
instead of the CTC head, which is blank-dominant and unsuitable for greedy
transcription in hybrid models.

Usage:
    uv run python convert-tdt-coreml.py \
        --nemo-path ./parakeet-tdt-ctc-110m-atco-v2.nemo \
        --output-dir parakeet_tdt_coreml \
        --audio-path audio/trace_15s.wav
"""
from __future__ import annotations

import json
import shutil
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
    JointDecisionSingleStep,
    JointDecisionWrapper,
    JointWrapper,
    MelEncoderWrapper,
    PreprocessorWrapper,
    _coreml_convert,
)

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt_ctc-110m"
AUTHOR = "Fluid Inference"


def _compute_length(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def _tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _save_mlpackage(
    model: ct.models.MLModel, path: Path, description: str
) -> None:
    model.minimum_deployment_target = ct.target.iOS17
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def _prepare_audio(
    audio_path: Path, sample_rate: int, max_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not audio_path.exists():
        raise typer.BadParameter(f"Audio not found: {audio_path}")
    data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if sr != sample_rate:
        raise typer.BadParameter(
            f"Audio sample rate {sr} != model rate {sample_rate}"
        )
    if data.ndim > 1:
        data = data[:, 0]
    if data.size == 0:
        raise typer.BadParameter("Audio is empty")
    original_len = int(data.size)
    if data.size < max_samples:
        data = np.pad(data, (0, max_samples - data.size))
    elif data.size > max_samples:
        data = data[:max_samples]
    audio_tensor = torch.from_numpy(data).unsqueeze(0).to(dtype=torch.float32)
    length = torch.tensor([min(original_len, max_samples)], dtype=torch.int32)
    return audio_tensor, length


def _parse_compute_units(name: str) -> ct.ComputeUnit:
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
            f"Unknown compute units '{name}'. Choose from: "
            + ", ".join(mapping.keys())
        )
    return mapping[normalized]


def _parse_compute_precision(name: Optional[str]) -> Optional[ct.precision]:
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
            f"Unknown compute precision '{name}'. Choose from: "
            + ", ".join(mapping.keys())
        )
    return mapping[normalized]


def _save_vocabulary(asr_model, output_dir: Path) -> None:
    """Save SentencePiece vocabulary as vocab.json (array format)."""
    sp = asr_model.tokenizer.tokenizer
    vocab_list = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab_list, ensure_ascii=False, indent=2))
    typer.echo(
        f"Saved vocabulary ({len(vocab_list)} tokens) to {vocab_path}"
    )


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    nemo_path: Optional[Path] = typer.Option(
        None,
        "--nemo-path",
        exists=True,
        resolve_path=True,
        help="Path to hybrid TDT-CTC .nemo checkpoint",
    ),
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="Model identifier when --nemo-path is omitted",
    ),
    output_dir: Path = typer.Option(
        Path("parakeet_tdt_coreml"),
        help="Output directory for mlpackage files",
    ),
    audio_path: Optional[Path] = typer.Option(
        None,
        "--audio-path",
        resolve_path=True,
        help="Path to 15s 16kHz WAV for tracing",
    ),
    max_audio_seconds: float = typer.Option(
        15.0, "--max-audio-seconds", help="Fixed waveform window (seconds)"
    ),
    mel_encoder_cu: str = typer.Option(
        "CPU_AND_NE",
        "--mel-encoder-cu",
        help="Compute units for fused mel+encoder",
    ),
    compute_precision: Optional[str] = typer.Option(
        None,
        "--compute-precision",
        help="Export precision: FLOAT32 (default) or FLOAT16",
    ),
    reuse_encoder: Optional[Path] = typer.Option(
        None,
        "--reuse-encoder",
        resolve_path=True,
        help="Path to existing mel+encoder mlpackage to reuse (skips encoder export)",
    ),
) -> None:
    """Export TDT components from a hybrid TDT-CTC model to CoreML."""
    export_settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS17,
        compute_precision=_parse_compute_precision(compute_precision),
        max_audio_seconds=max_audio_seconds,
        max_symbol_steps=1,
    )

    typer.echo(f"Export configuration: {asdict(export_settings)}")
    output_dir.mkdir(parents=True, exist_ok=True)

    melenc_cu = _parse_compute_units(mel_encoder_cu)

    # -- Load hybrid model --------------------------------------------------
    typer.echo(f"Loading hybrid TDT-CTC model ({'file' if nemo_path else 'pretrained'})...")
    if nemo_path is not None:
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
            str(nemo_path), map_location="cpu"
        )
        checkpoint_meta = {"type": "file", "path": str(nemo_path)}
    else:
        asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
            model_id, map_location="cpu"
        )
        checkpoint_meta = {"type": "pretrained", "model_id": model_id}
    asr_model.eval()

    sample_rate = int(asr_model.cfg.preprocessor.sample_rate)
    max_samples = _compute_length(max_audio_seconds, sample_rate)

    # -- Probe model dimensions ---------------------------------------------
    vocab_size = int(asr_model.tokenizer.vocab_size)
    num_extra = int(asr_model.joint.num_extra_outputs)
    decoder_hidden = int(asr_model.decoder.pred_hidden)
    decoder_layers = int(asr_model.decoder.pred_rnn_layers)
    blank_idx = int(asr_model.decoder.blank_idx)

    typer.echo(f"  vocab_size={vocab_size}, num_extra={num_extra} (TDT duration bins)")
    typer.echo(f"  decoder: hidden={decoder_hidden}, layers={decoder_layers}, blank_idx={blank_idx}")

    if num_extra == 0:
        raise typer.BadParameter(
            "joint.num_extra_outputs=0 -- this model has no TDT duration head. "
            "It appears to be a plain RNNT, not TDT. Use the standard RNNT export instead."
        )

    # -- Trace audio --------------------------------------------------------
    if audio_path is None:
        candidates = [
            Path(__file__).resolve().parent / "audio" / "trace_15s.wav",
            Path(__file__).resolve().parent / "trace_15s.wav",
        ]
        for c in candidates:
            if c.exists():
                audio_path = c
                break
    if audio_path is None:
        raise typer.BadParameter("Provide --audio-path (15s 16kHz WAV)")
    typer.echo(f"Trace audio: {audio_path}")
    audio_tensor, audio_length = _prepare_audio(audio_path, sample_rate, max_samples)

    # -- Wrap components ----------------------------------------------------
    preprocessor = PreprocessorWrapper(asr_model.preprocessor.eval())
    encoder = EncoderWrapper(asr_model.encoder.eval())
    decoder = DecoderWrapper(asr_model.decoder.eval())
    joint = JointWrapper(asr_model.joint.eval())

    decoder_export_flag = getattr(asr_model.decoder, "_rnnt_export", False)
    asr_model.decoder._rnnt_export = True

    try:
        # -- Reference forward pass -----------------------------------------
        with torch.inference_mode():
            mel_ref, mel_length_ref = preprocessor(audio_tensor, audio_length)
            mel_length_ref = mel_length_ref.to(dtype=torch.int32)
            encoder_ref, encoder_length_ref = encoder(mel_ref, mel_length_ref)
            encoder_length_ref = encoder_length_ref.to(dtype=torch.int32)

        mel_ref = mel_ref.clone()
        mel_length_ref = mel_length_ref.clone()
        encoder_ref = encoder_ref.clone()
        encoder_length_ref = encoder_length_ref.clone()

        targets = torch.full(
            (1, export_settings.max_symbol_steps),
            fill_value=blank_idx,
            dtype=torch.int32,
        )
        target_lengths = torch.tensor(
            [export_settings.max_symbol_steps], dtype=torch.int32
        )
        zero_state = torch.zeros(
            decoder_layers, 1, decoder_hidden, dtype=torch.float32
        )

        with torch.inference_mode():
            decoder_ref, h_ref, c_ref = decoder(
                targets, target_lengths, zero_state, zero_state
            )
            joint_ref = joint(encoder_ref, decoder_ref)

        decoder_ref = decoder_ref.clone()
        h_ref = h_ref.clone()
        c_ref = c_ref.clone()
        joint_ref = joint_ref.clone()

        typer.echo(f"  encoder_ref shape: {_tensor_shape(encoder_ref)}")
        typer.echo(f"  decoder_ref shape: {_tensor_shape(decoder_ref)}")
        typer.echo(f"  joint_ref  shape: {_tensor_shape(joint_ref)}")

        # -- 1. Fused Mel+Encoder -------------------------------------------
        mel_encoder_path = output_dir / "Preprocessor.mlpackage"
        if reuse_encoder is not None:
            typer.echo(f"Reusing existing mel+encoder from: {reuse_encoder}")
            if mel_encoder_path.exists():
                shutil.rmtree(mel_encoder_path)
            shutil.copytree(reuse_encoder, mel_encoder_path)
            typer.echo(f"  Copied to {mel_encoder_path}")
        else:
            typer.echo("Tracing fused mel+encoder...")
            mel_encoder = MelEncoderWrapper(preprocessor, encoder).cpu().eval()
            traced_mel_encoder = torch.jit.trace(
                mel_encoder,
                (audio_tensor.cpu(), audio_length.cpu()),
                strict=False,
            )
            traced_mel_encoder.eval()

            mel_encoder_model = _coreml_convert(
                traced_mel_encoder,
                inputs=[
                    ct.TensorType(
                        name="audio_signal",
                        shape=(1, max_samples),
                        dtype=np.float32,
                    ),
                    ct.TensorType(
                        name="audio_length", shape=(1,), dtype=np.int32
                    ),
                ],
                outputs=[
                    ct.TensorType(name="encoder", dtype=np.float32),
                    ct.TensorType(name="encoder_length", dtype=np.int32),
                ],
                settings=export_settings,
                compute_units_override=melenc_cu,
            )
            _save_mlpackage(
                mel_encoder_model,
                mel_encoder_path,
                f"Parakeet TDT-CTC Mel+Encoder ({max_audio_seconds:.0f}s window)",
            )

        # -- 2. Decoder (LSTM prediction network) --------------------------
        typer.echo("Tracing decoder (RNNT prediction network)...")
        traced_decoder = torch.jit.trace(
            decoder,
            (targets, target_lengths, zero_state, zero_state),
            strict=False,
        )
        traced_decoder.eval()

        decoder_model = _coreml_convert(
            traced_decoder,
            inputs=[
                ct.TensorType(
                    name="targets",
                    shape=_tensor_shape(targets),
                    dtype=np.int32,
                ),
                ct.TensorType(
                    name="target_length", shape=(1,), dtype=np.int32
                ),
                ct.TensorType(
                    name="h_in",
                    shape=_tensor_shape(zero_state),
                    dtype=np.float32,
                ),
                ct.TensorType(
                    name="c_in",
                    shape=_tensor_shape(zero_state),
                    dtype=np.float32,
                ),
            ],
            outputs=[
                ct.TensorType(name="decoder", dtype=np.float32),
                ct.TensorType(name="h_out", dtype=np.float32),
                ct.TensorType(name="c_out", dtype=np.float32),
            ],
            settings=export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        decoder_path = output_dir / "Decoder.mlpackage"
        _save_mlpackage(
            decoder_model,
            decoder_path,
            "Parakeet TDT-CTC decoder (RNNT prediction network)",
        )

        # -- 3. JointDecision (full T x U grid) ----------------------------
        typer.echo("Tracing joint decision head...")
        joint_decision = JointDecisionWrapper(
            joint, vocab_size=vocab_size, num_extra=num_extra
        )
        traced_joint_decision = torch.jit.trace(
            joint_decision,
            (encoder_ref, decoder_ref),
            strict=False,
        )
        traced_joint_decision.eval()

        joint_decision_model = _coreml_convert(
            traced_joint_decision,
            inputs=[
                ct.TensorType(
                    name="encoder",
                    shape=_tensor_shape(encoder_ref),
                    dtype=np.float32,
                ),
                ct.TensorType(
                    name="decoder",
                    shape=_tensor_shape(decoder_ref),
                    dtype=np.float32,
                ),
            ],
            outputs=[
                ct.TensorType(name="token_id", dtype=np.int32),
                ct.TensorType(name="token_prob", dtype=np.float32),
                ct.TensorType(name="duration", dtype=np.int32),
            ],
            settings=export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        joint_decision_path = output_dir / "JointDecision.mlpackage"
        _save_mlpackage(
            joint_decision_model,
            joint_decision_path,
            "Parakeet TDT-CTC joint + decision head",
        )

        # -- 4. Single-step JointDecision (for streaming) ------------------
        typer.echo("Tracing single-step joint decision...")
        jd_single = JointDecisionSingleStep(
            joint, vocab_size=vocab_size, num_extra=num_extra
        )
        enc_step = encoder_ref[:, :, :1].contiguous()
        dec_step = decoder_ref[:, :, :1].contiguous()
        traced_jd_single = torch.jit.trace(
            jd_single, (enc_step, dec_step), strict=False
        )
        traced_jd_single.eval()

        jd_single_model = _coreml_convert(
            traced_jd_single,
            inputs=[
                ct.TensorType(
                    name="encoder_step",
                    shape=(1, enc_step.shape[1], 1),
                    dtype=np.float32,
                ),
                ct.TensorType(
                    name="decoder_step",
                    shape=(1, dec_step.shape[1], 1),
                    dtype=np.float32,
                ),
            ],
            outputs=[
                ct.TensorType(name="token_id", dtype=np.int32),
                ct.TensorType(name="token_prob", dtype=np.float32),
                ct.TensorType(name="duration", dtype=np.int32),
                ct.TensorType(name="top_k_ids", dtype=np.int32),
                ct.TensorType(name="top_k_logits", dtype=np.float32),
            ],
            settings=export_settings,
            compute_units_override=ct.ComputeUnit.CPU_ONLY,
        )
        jd_single_path = output_dir / "JointDecisionSingleStep.mlpackage"
        _save_mlpackage(
            jd_single_model,
            jd_single_path,
            "Parakeet TDT-CTC single-step joint decision",
        )

        # -- Save vocabulary -----------------------------------------------
        _save_vocabulary(asr_model, output_dir)

        # -- Metadata -------------------------------------------------------
        enc_dim = int(encoder_ref.shape[1])
        dec_dim = int(decoder_ref.shape[1])

        metadata: Dict[str, object] = {
            "model_id": model_id,
            "model_type": "hybrid_tdt_ctc",
            "sample_rate": sample_rate,
            "max_audio_seconds": max_audio_seconds,
            "max_audio_samples": max_samples,
            "max_symbol_steps": export_settings.max_symbol_steps,
            "vocab_size": vocab_size,
            "blank_id": blank_idx,
            "joint_extra_outputs": num_extra,
            "duration_bins": [0, 1, 2, 3, 4] if num_extra == 5 else list(range(num_extra)),
            "encoder_dim": enc_dim,
            "decoder_dim": dec_dim,
            "decoder_hidden": decoder_hidden,
            "decoder_layers": decoder_layers,
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
                "joint_decision": {
                    "inputs": {
                        "encoder": list(_tensor_shape(encoder_ref)),
                        "decoder": list(_tensor_shape(decoder_ref)),
                    },
                    "outputs": {
                        "token_id": "int32",
                        "token_prob": "float32",
                        "duration": "int32",
                    },
                    "path": joint_decision_path.name,
                },
                "joint_decision_single_step": {
                    "inputs": {
                        "encoder_step": [1, enc_dim, 1],
                        "decoder_step": [1, dec_dim, 1],
                    },
                    "outputs": {
                        "token_id": [1, 1, 1],
                        "token_prob": [1, 1, 1],
                        "duration": [1, 1, 1],
                        "top_k_ids": "int32",
                        "top_k_logits": "float32",
                    },
                    "path": jd_single_path.name,
                },
            },
        }

        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        typer.echo(f"\nExport complete. Metadata: {metadata_path}")
        typer.echo(f"Output directory: {output_dir}")

    finally:
        asr_model.decoder._rnnt_export = decoder_export_flag


if __name__ == "__main__":
    app()
