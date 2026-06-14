#!/usr/bin/env python3
"""Export nvidia/parakeet-unified-en-0.6b components to CoreML.

Offline components use a fixed 15-second window (matching the parakeet-tdt-v3
pipeline contract). The streaming encoder is traced with the chunked attention
mask (att_context_style=chunked_limited_with_rc) baked in for a fixed
[left, chunk, right] context, mirroring NeMo's buffered streaming inference
(examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py
with att_context_size_as_chunk=true).

Usage:
    uv run --no-sync python convert-coreml.py --output-dir ./build/parakeet_unified_coreml
    uv run --no-sync python convert-coreml.py --skip-offline --streaming-context 70,7,7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

import nemo.collections.asr as nemo_asr

from components import (
    DecoderWrapper,
    EncoderWrapper,
    ExportSettings,
    JointDecisionSingleStep,
    JointWrapper,
    MelEncoderWrapper,
    PreprocessorWrapper,
    coreml_convert,
)

MODEL_ID = "nvidia/parakeet-unified-en-0.6b"
AUTHOR = "Fluid Inference"
DEFAULT_NEMO_PATH = Path("parakeet-unified-en-0.6b.nemo")
TRACE_AUDIO = Path(__file__).parent / "audio" / "yc_first_minute_16k_15s.wav"

SAMPLE_RATE = 16000
FEATURE_STRIDE_SAMPLES = 160  # 10 ms hop
SUBSAMPLING = 8  # encoder frame = 80 ms = 1280 samples
ENCODER_FRAME_SAMPLES = FEATURE_STRIDE_SAMPLES * SUBSAMPLING


def _tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in tensor.shape)


def _save(model: ct.models.MLModel, path: Path, description: str) -> None:
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"saved {path}")


def _load_trace_audio(num_samples: int) -> torch.Tensor:
    data, sr = sf.read(str(TRACE_AUDIO), dtype="float32")
    assert sr == SAMPLE_RATE, f"trace audio must be {SAMPLE_RATE} Hz, got {sr}"
    if data.ndim > 1:
        data = data[:, 0]
    if data.size < num_samples:
        data = np.pad(data, (0, num_samples - data.size))
    return torch.from_numpy(data[:num_samples]).unsqueeze(0).to(dtype=torch.float32)


def export_offline(
    asr_model, settings: ExportSettings, output_dir: Path, metadata: Dict[str, object]
) -> None:
    max_samples = int(settings.max_audio_seconds * SAMPLE_RATE)
    audio_tensor = _load_trace_audio(max_samples)
    audio_length = torch.tensor([max_samples], dtype=torch.int32)

    preprocessor = PreprocessorWrapper(asr_model.preprocessor.eval())
    encoder = EncoderWrapper(asr_model.encoder.eval())
    decoder = DecoderWrapper(asr_model.decoder.eval())
    joint = JointWrapper(asr_model.joint.eval())

    with torch.inference_mode():
        mel_ref, mel_length_ref = preprocessor(audio_tensor, audio_length)
        mel_length_ref = mel_length_ref.to(dtype=torch.int32)
        encoder_ref, encoder_length_ref = encoder(mel_ref, mel_length_ref)
    mel_ref = mel_ref.clone()
    mel_length_ref = mel_length_ref.clone()
    encoder_ref = encoder_ref.clone()

    vocab_size = int(asr_model.tokenizer.vocab_size)
    decoder_hidden = int(asr_model.decoder.pred_hidden)
    decoder_layers = int(asr_model.decoder.pred_rnn_layers)

    targets = torch.full((1, 1), fill_value=asr_model.decoder.blank_idx, dtype=torch.int32)
    target_lengths = torch.tensor([1], dtype=torch.int32)
    zero_state = torch.zeros(decoder_layers, 1, decoder_hidden, dtype=torch.float32)

    with torch.inference_mode():
        decoder_ref, h_ref, c_ref = decoder(targets, target_lengths, zero_state, zero_state)
    decoder_ref = decoder_ref.clone()

    print("Tracing preprocessor…")
    traced = torch.jit.trace(preprocessor, (audio_tensor, audio_length), strict=False)
    model = coreml_convert(
        traced,
        [
            ct.TensorType(name="audio_signal", shape=(1, ct.RangeDim(1, max_samples)), dtype=np.float32),
            ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
        ],
        [
            ct.TensorType(name="mel", dtype=np.float32),
            ct.TensorType(name="mel_length", dtype=np.int32),
        ],
        settings,
    )
    _save(model, output_dir / "parakeet_unified_preprocessor.mlpackage", "Parakeet-unified preprocessor (≤15 s)")

    print("Tracing offline encoder…")
    traced = torch.jit.trace(encoder, (mel_ref, mel_length_ref), strict=False)
    model = coreml_convert(
        traced,
        [
            ct.TensorType(name="mel", shape=_tensor_shape(mel_ref), dtype=np.float32),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ],
        [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
        ],
        settings,
    )
    _save(model, output_dir / "parakeet_unified_encoder.mlpackage", "Parakeet-unified offline encoder (15 s window)")

    print("Tracing fused mel+encoder…")
    mel_encoder = MelEncoderWrapper(preprocessor, encoder)
    traced = torch.jit.trace(mel_encoder, (audio_tensor, audio_length), strict=False)
    model = coreml_convert(
        traced,
        [
            ct.TensorType(name="audio_signal", shape=(1, max_samples), dtype=np.float32),
            ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
        ],
        [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
        ],
        settings,
    )
    _save(model, output_dir / "parakeet_unified_mel_encoder.mlpackage", "Parakeet-unified fused mel+encoder (15 s window)")

    print("Tracing decoder…")
    traced = torch.jit.trace(decoder, (targets, target_lengths, zero_state, zero_state), strict=False)
    model = coreml_convert(
        traced,
        [
            ct.TensorType(name="targets", shape=_tensor_shape(targets), dtype=np.int32),
            ct.TensorType(name="target_length", shape=(1,), dtype=np.int32),
            ct.TensorType(name="h_in", shape=_tensor_shape(zero_state), dtype=np.float32),
            ct.TensorType(name="c_in", shape=_tensor_shape(zero_state), dtype=np.float32),
        ],
        [
            ct.TensorType(name="decoder", dtype=np.float32),
            ct.TensorType(name="h_out", dtype=np.float32),
            ct.TensorType(name="c_out", dtype=np.float32),
        ],
        settings,
    )
    _save(model, output_dir / "parakeet_unified_decoder.mlpackage", "Parakeet-unified decoder (RNNT prediction network)")

    print("Tracing joint…")
    traced = torch.jit.trace(joint, (encoder_ref, decoder_ref), strict=False)
    model = coreml_convert(
        traced,
        [
            ct.TensorType(name="encoder", shape=_tensor_shape(encoder_ref), dtype=np.float32),
            ct.TensorType(name="decoder", shape=_tensor_shape(decoder_ref), dtype=np.float32),
        ],
        [ct.TensorType(name="logits", dtype=np.float32)],
        settings,
    )
    _save(model, output_dir / "parakeet_unified_joint.mlpackage", "Parakeet-unified joint network (RNNT)")

    print("Tracing single-step joint decision…")
    jd_single = JointDecisionSingleStep(joint, vocab_size=vocab_size)
    enc_step = encoder_ref[:, :, :1].contiguous()
    dec_step = decoder_ref[:, :, :1].contiguous()
    traced = torch.jit.trace(jd_single, (enc_step, dec_step), strict=False)
    model = coreml_convert(
        traced,
        [
            ct.TensorType(name="encoder_step", shape=(1, enc_step.shape[1], 1), dtype=np.float32),
            ct.TensorType(name="decoder_step", shape=(1, dec_step.shape[1], 1), dtype=np.float32),
        ],
        [
            ct.TensorType(name="token_id", dtype=np.int32),
            ct.TensorType(name="token_prob", dtype=np.float32),
            ct.TensorType(name="top_k_ids", dtype=np.int32),
            ct.TensorType(name="top_k_logits", dtype=np.float32),
        ],
        settings,
    )
    _save(
        model,
        output_dir / "parakeet_unified_joint_decision_single_step.mlpackage",
        "Parakeet-unified single-step joint decision",
    )

    metadata["offline"] = {
        "max_audio_seconds": settings.max_audio_seconds,
        "max_audio_samples": max_samples,
        "mel_shape": list(_tensor_shape(mel_ref)),
        "encoder_shape": list(_tensor_shape(encoder_ref)),
        "decoder_shape": list(_tensor_shape(decoder_ref)),
        "decoder_state_shape": [decoder_layers, 1, decoder_hidden],
    }


def export_streaming(
    asr_model,
    settings: ExportSettings,
    output_dir: Path,
    context: List[int],
    metadata: Dict[str, object],
) -> None:
    left, chunk, right = context
    window_enc_frames = left + chunk + right
    window_samples = window_enc_frames * ENCODER_FRAME_SAMPLES

    # Bake the chunked attention mask into the trace. The mask is generated in
    # ConformerEncoder.forward from self.att_context_size; with a fixed input
    # shape it traces to a constant. This mirrors NeMo's buffered streaming
    # inference with att_context_size_as_chunk=true.
    asr_model.encoder.set_default_att_context_size(att_context_size=[left, chunk, right])

    audio_tensor = _load_trace_audio(window_samples)
    audio_length = torch.tensor([window_samples], dtype=torch.int32)

    preprocessor = PreprocessorWrapper(asr_model.preprocessor.eval())
    encoder = EncoderWrapper(asr_model.encoder.eval())

    with torch.inference_mode():
        mel_ref, mel_length_ref = preprocessor(audio_tensor, audio_length)
        mel_length_ref = mel_length_ref.to(dtype=torch.int32)
        encoder_ref, _ = encoder(mel_ref, mel_length_ref)
    mel_ref = mel_ref.clone()
    mel_length_ref = mel_length_ref.clone()
    encoder_ref = encoder_ref.clone()

    suffix = f"{left}_{chunk}_{right}"
    print(f"Tracing streaming encoder (context [{left}, {chunk}, {right}] = {window_samples / SAMPLE_RATE:.2f} s window)…")
    traced = torch.jit.trace(encoder, (mel_ref, mel_length_ref), strict=False)
    model = coreml_convert(
        traced,
        [
            ct.TensorType(name="mel", shape=_tensor_shape(mel_ref), dtype=np.float32),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ],
        [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
        ],
        settings,
    )
    _save(
        model,
        output_dir / f"parakeet_unified_encoder_streaming_{suffix}.mlpackage",
        f"Parakeet-unified streaming encoder ([L,C,R]=[{left},{chunk},{right}] encoder frames)",
    )

    print("Tracing fused streaming mel+encoder…")
    mel_encoder = MelEncoderWrapper(preprocessor, encoder)
    traced = torch.jit.trace(mel_encoder, (audio_tensor, audio_length), strict=False)
    model = coreml_convert(
        traced,
        [
            ct.TensorType(name="audio_signal", shape=(1, window_samples), dtype=np.float32),
            ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
        ],
        [
            ct.TensorType(name="encoder", dtype=np.float32),
            ct.TensorType(name="encoder_length", dtype=np.int32),
        ],
        settings,
    )
    _save(
        model,
        output_dir / f"parakeet_unified_mel_encoder_streaming_{suffix}.mlpackage",
        f"Parakeet-unified fused streaming mel+encoder ([L,C,R]=[{left},{chunk},{right}])",
    )

    metadata.setdefault("streaming", {})[suffix] = {
        "context_encoder_frames": {"left": left, "chunk": chunk, "right": right},
        "window_samples": window_samples,
        "window_seconds": window_samples / SAMPLE_RATE,
        "encoder_frame_samples": ENCODER_FRAME_SAMPLES,
        "chunk_samples": chunk * ENCODER_FRAME_SAMPLES,
        "theoretical_latency_seconds": (chunk + right) * ENCODER_FRAME_SAMPLES / SAMPLE_RATE,
        "mel_shape": list(_tensor_shape(mel_ref)),
        "encoder_shape": list(_tensor_shape(encoder_ref)),
    }

    # Restore offline default so subsequent exports/validation see full context.
    asr_model.encoder.set_default_att_context_size(att_context_size=[-1, -1, -1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-path", type=Path, default=DEFAULT_NEMO_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("build/parakeet_unified_coreml"))
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument(
        "--streaming-context",
        type=str,
        default="70,13,13",
        help="left,chunk,right in 80 ms encoder frames (default 70,13,13 = 5.6 s / 1.04 s / 1.04 s)",
    )
    args = parser.parse_args()

    settings = ExportSettings(
        output_dir=args.output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS17,
        compute_precision=None,
        max_audio_seconds=15.0,
        max_symbol_steps=1,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.nemo_path}…")
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(args.nemo_path), map_location="cpu")
    asr_model.eval()

    decoder_export_flag = getattr(asr_model.decoder, "_rnnt_export", False)
    asr_model.decoder._rnnt_export = True

    metadata: Dict[str, object] = {
        "model_id": MODEL_ID,
        "sample_rate": SAMPLE_RATE,
        "vocab_size": int(asr_model.tokenizer.vocab_size),
        "blank_idx": int(asr_model.decoder.blank_idx),
        "subsampling_factor": SUBSAMPLING,
        # compute_precision None -> coremltools mlprogram default (FLOAT16 weights/ops)
        "coreml": {"compute_units": "CPU_ONLY", "compute_precision": "FLOAT16", "deployment_target": "iOS17"},
    }

    try:
        if not args.skip_offline:
            export_offline(asr_model, settings, args.output_dir, metadata)
        if not args.skip_streaming:
            context = [int(x) for x in args.streaming_context.split(",")]
            assert len(context) == 3, "--streaming-context must be left,chunk,right"
            export_streaming(asr_model, settings, args.output_dir, context, metadata)
    finally:
        asr_model.decoder._rnnt_export = decoder_export_flag

    metadata_path = args.output_dir / "metadata.json"
    existing = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    existing.update(metadata)
    metadata_path.write_text(json.dumps(existing, indent=2))
    print(f"Export complete. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
