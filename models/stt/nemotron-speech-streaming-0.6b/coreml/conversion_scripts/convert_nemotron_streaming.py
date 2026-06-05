#!/usr/bin/env python3
"""Export Nemotron Speech Streaming 0.6B to CoreML.

Exports 4 components for streaming RNNT inference:
1. Preprocessor: audio → mel
2. Encoder: mel + cache → encoded + new_cache
3. Decoder: token + state → decoder_out + new_state
4. Joint: encoder + decoder → logits
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import typer

import nemo.collections.asr as nemo_asr

from individual_components import (
    DecoderWrapper,
    EncoderStreamingWrapper,
    ExportSettings,
    JointWrapper,
    PreprocessorWrapper,
    _coreml_convert,
)

DEFAULT_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"

# Streaming config from model:
# chunk_size=[105, 112], pre_encode_cache_size=[0, 9], valid_out_len=14
CHUNK_MEL_FRAMES = 112
PRE_ENCODE_CACHE = 9
TOTAL_MEL_FRAMES = CHUNK_MEL_FRAMES + PRE_ENCODE_CACHE  # 121


def _tensor_shape(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(d) for d in t.shape)


def _parse_cu(name: str) -> ct.ComputeUnit:
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    return mapping.get(name.upper(), ct.ComputeUnit.CPU_ONLY)


app = typer.Typer(add_completion=False)


@app.command()
def convert(
    output_dir: Path = typer.Option(Path("nemotron_coreml"), help="Output directory"),
    encoder_cu: str = typer.Option("CPU_AND_NE", help="Encoder compute units"),
    precision: str = typer.Option("FLOAT32", help="FLOAT32 or FLOAT16"),
    chunk_mel_frames: int = typer.Option(
        CHUNK_MEL_FRAMES,
        help="Mel frames per streaming chunk. 112=1120ms (default/shipped), 224=2240ms tier. "
        "Must be a clean multiple of the trained 112 (=14 encoder frames) to keep the "
        "chunked-attention mask aligned.",
    ),
) -> None:
    """Export Nemotron Streaming to CoreML."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive chunk geometry from the requested mel-chunk size.
    if chunk_mel_frames % 112 != 0:
        typer.echo(
            f"WARNING: chunk_mel_frames={chunk_mel_frames} is not a multiple of 112 "
            "(the trained 14-encoder-frame chunk); the attention mask may misalign."
        )
    chunk_mel = chunk_mel_frames
    total_mel = chunk_mel + PRE_ENCODE_CACHE
    chunk_enc_frames = chunk_mel // 8  # subsampling factor 8
    typer.echo(
        f"Chunk geometry: chunk_mel={chunk_mel} (~{chunk_mel * 10}ms), "
        f"total_mel={total_mel}, chunk_enc_frames={chunk_enc_frames}"
    )

    typer.echo("Loading model...")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(DEFAULT_MODEL_ID, map_location="cpu")
    model.eval()

    sample_rate = int(model.cfg.preprocessor.sample_rate)
    encoder = model.encoder
    encoder.setup_streaming_params()

    # Get cache shapes
    cache_channel, cache_time, cache_len = encoder.get_initial_cache_state(batch_size=1, device="cpu")
    cache_len = cache_len.to(torch.int32)

    # Transpose to [B, L, ...] for CoreML
    cache_channel_b = cache_channel.transpose(0, 1)
    cache_time_b = cache_time.transpose(0, 1)

    typer.echo(f"Cache shapes: channel={cache_channel_b.shape}, time={cache_time_b.shape}")

    # Create wrappers
    preprocessor = PreprocessorWrapper(model.preprocessor.eval())
    encoder_streaming = EncoderStreamingWrapper(encoder.eval())
    decoder = DecoderWrapper(model.decoder.eval())
    joint = JointWrapper(model.joint.eval())

    model.decoder._rnnt_export = True

    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if precision.upper() == "FLOAT16" else ct.precision.FLOAT32,
        max_audio_seconds=30.0,
        max_symbol_steps=1,
        chunk_size_frames=chunk_enc_frames,
        cache_size=cache_channel.shape[2],
    )

    # === Preprocessor ===
    typer.echo("Exporting preprocessor...")
    max_samples = 30 * sample_rate
    audio = torch.randn(1, max_samples)
    audio_len = torch.tensor([max_samples], dtype=torch.int32)

    traced = torch.jit.trace(preprocessor, (audio, audio_len), strict=False)
    inputs = [
        ct.TensorType(name="audio", shape=(1, ct.RangeDim(1, max_samples)), dtype=np.float32),
        ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="mel", dtype=np.float32),
        ct.TensorType(name="mel_length", dtype=np.int32),
    ]
    mlmodel = _coreml_convert(traced, inputs, outputs, settings, ct.ComputeUnit.CPU_ONLY)
    mlmodel.save(str(output_dir / "preprocessor.mlpackage"))

    # === Encoder (streaming) ===
    typer.echo("Exporting encoder...")
    mel_features = int(model.cfg.preprocessor.features)  # 128 for this model
    mel = torch.randn(1, mel_features, total_mel)
    mel_len = torch.tensor([total_mel], dtype=torch.int32)

    traced = torch.jit.trace(
        encoder_streaming,
        (mel, mel_len, cache_channel_b, cache_time_b, cache_len),
        strict=False
    )
    inputs = [
        ct.TensorType(name="mel", shape=_tensor_shape(mel), dtype=np.float32),
        ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="cache_channel", shape=_tensor_shape(cache_channel_b), dtype=np.float32),
        ct.TensorType(name="cache_time", shape=_tensor_shape(cache_time_b), dtype=np.float32),
        ct.TensorType(name="cache_len", shape=(1,), dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="encoded", dtype=np.float32),
        ct.TensorType(name="encoded_length", dtype=np.int32),
        ct.TensorType(name="cache_channel_out", dtype=np.float32),
        ct.TensorType(name="cache_time_out", dtype=np.float32),
        ct.TensorType(name="cache_len_out", dtype=np.int32),
    ]
    mlmodel = _coreml_convert(traced, inputs, outputs, settings, _parse_cu(encoder_cu))
    mlmodel.save(str(output_dir / "encoder.mlpackage"))

    # === Decoder ===
    typer.echo("Exporting decoder...")
    decoder_hidden = int(model.decoder.pred_hidden)
    decoder_layers = int(model.decoder.pred_rnn_layers)

    targets = torch.tensor([[model.decoder.blank_idx]], dtype=torch.int32)
    target_len = torch.tensor([1], dtype=torch.int32)
    h = torch.zeros(decoder_layers, 1, decoder_hidden)
    c = torch.zeros(decoder_layers, 1, decoder_hidden)

    traced = torch.jit.trace(decoder, (targets, target_len, h, c), strict=False)
    inputs = [
        ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="token_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="h_in", shape=_tensor_shape(h), dtype=np.float32),
        ct.TensorType(name="c_in", shape=_tensor_shape(c), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType(name="decoder_out", dtype=np.float32),
        ct.TensorType(name="h_out", dtype=np.float32),
        ct.TensorType(name="c_out", dtype=np.float32),
    ]
    mlmodel = _coreml_convert(traced, inputs, outputs, settings, ct.ComputeUnit.CPU_ONLY)
    mlmodel.save(str(output_dir / "decoder.mlpackage"))

    # === Joint ===
    typer.echo("Exporting joint...")
    with torch.no_grad():
        mel_test, _ = preprocessor(audio[:, :sample_rate], torch.tensor([sample_rate], dtype=torch.int32))
        # Run through encoder wrapper (not model.encoder directly to avoid typed method issues)
        enc_out, _, _, _, _ = encoder_streaming(
            mel_test,
            torch.tensor([mel_test.shape[2]], dtype=torch.int32),
            cache_channel_b,
            cache_time_b,
            cache_len
        )
        dec_out, _, _ = decoder(targets, target_len, h, c)

    # Single step: [B, D, 1]
    enc_step = enc_out[:, :, :1].contiguous()
    dec_step = dec_out[:, :, :1].contiguous()

    traced = torch.jit.trace(joint, (enc_step, dec_step), strict=False)
    inputs = [
        ct.TensorType(name="encoder", shape=_tensor_shape(enc_step), dtype=np.float32),
        ct.TensorType(name="decoder", shape=_tensor_shape(dec_step), dtype=np.float32),
    ]
    outputs = [ct.TensorType(name="logits", dtype=np.float32)]
    mlmodel = _coreml_convert(traced, inputs, outputs, settings, ct.ComputeUnit.CPU_ONLY)
    mlmodel.save(str(output_dir / "joint.mlpackage"))

    # === Metadata ===
    vocab_size = int(model.tokenizer.vocab_size)
    metadata = {
        "model": DEFAULT_MODEL_ID,
        "sample_rate": sample_rate,
        "mel_features": mel_features,
        "chunk_mel_frames": chunk_mel,
        "pre_encode_cache": PRE_ENCODE_CACHE,
        "total_mel_frames": total_mel,
        "vocab_size": vocab_size,
        "blank_idx": int(model.decoder.blank_idx),
        "cache_channel_shape": list(cache_channel_b.shape),
        "cache_time_shape": list(cache_time_b.shape),
        "decoder_hidden": decoder_hidden,
        "decoder_layers": decoder_layers,
        "encoder_dim": int(enc_out.shape[1]),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Tokenizer
    tokenizer = {str(i): model.tokenizer.ids_to_tokens([i])[0] for i in range(vocab_size)}
    (output_dir / "tokenizer.json").write_text(json.dumps(tokenizer, indent=2))

    typer.echo(f"Done! Exported to {output_dir}")


if __name__ == "__main__":
    app()
