#!/usr/bin/env python3
"""Convert a Sherpa-ONNX Zipformer2 transducer checkpoint to CoreML.

Loads the PyTorch checkpoint (epoch-*.pt from icefall training), builds the
Zipformer2 encoder + stateless decoder + joiner architecture, replaces custom
training-only ops with standard ones, traces each component, and exports three
CoreML .mlpackage files (encoder, decoder, joiner) plus metadata.json.

Usage:
    uv run python convert-coreml.py \
        --checkpoint /path/to/epoch-56-avg-4.pt \
        --tokens /path/to/tokens.txt \
        --output-dir ./build/vosk-model-en-0.62-atc
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import typer

from decoder import Decoder
from icefall.utils import make_pad_mask
from joiner import Joiner
from scaling import (
    Balancer,
    Dropout3,
    ScaleGrad,
    ScheduledFloat,
    SwooshL,
    SwooshLOnnx,
    SwooshR,
    SwooshROnnx,
    Whiten,
)
from subsampling import Conv2dSubsampling
from zipformer import Zipformer2

AUTHOR = "Fluid Inference"


def _patch_coremltools_int_cast() -> None:
    """Monkey-patch coremltools _cast to handle numpy ndarray constants.

    ``aten::Int`` ops from ``tensor.size()`` produce constant numpy arrays
    in the MIL graph.  coremltools 9.0's ``_cast`` calls ``int(x.val)``
    which fails when ``x.val`` is an ndarray (even 0-d).  We replace
    ``_cast`` to call ``.item()`` first.
    """
    import coremltools.converters.mil.frontend.torch.ops as torch_ops
    from coremltools.converters.mil.mil import Builder as mb

    _orig_cast = torch_ops._cast

    def _patched_cast(context, node, dtype, dtype_name):
        inputs = torch_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError("input to cast must be either a scalar or a length 1 tensor")

        if x.can_be_folded_to_const():
            val = x.val
            if isinstance(val, np.ndarray):
                val = val.item()
            if isinstance(val, (int, float, bool, np.integer, np.floating)):
                if not isinstance(val, dtype):
                    res = mb.const(val=dtype(val), name=node.name)
                else:
                    res = x
            else:
                # Symbolic or non-scalar — use runtime cast
                if len(x.shape) > 0:
                    x = mb.squeeze(x=x, name=node.name + "_item")
                res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    torch_ops._cast = _patched_cast


_patch_coremltools_int_cast()
# T=1495 mel frames -> 744 after Conv2dSubsampling, divisible by 8 (max ds factor).
# At 16 kHz / 10 ms hop -> ~14.95 s of audio.
DEFAULT_MEL_FRAMES = 1495


def _to_int_tuple(s) -> Tuple[int, ...]:
    return tuple(map(int, str(s).split(",")))


def _parse_compute_units(name: str) -> ct.ComputeUnit:
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    key = str(name).strip().upper()
    if key not in mapping:
        raise typer.BadParameter(f"Unknown compute units '{name}'. Choose from: {', '.join(mapping)}")
    return mapping[key]


def convert_scaled_for_coreml(model: nn.Module) -> None:
    """Replace icefall training-only ops with standard modules (in-place).

    Similar to icefall's ``convert_scaled_to_non_scaled`` but avoids scripting
    ``CompactRelPositionalEncoding`` (which produces float-index slices that
    coremltools rejects).  Instead we rely on torch.jit.trace with a fixed
    input size so the positional encoding is baked in as constants.
    """
    replacements: dict[str, nn.Module] = {}
    for name, m in model.named_modules():
        if isinstance(m, (Balancer, Dropout3, ScaleGrad, Whiten)):
            replacements[name] = nn.Identity()
        elif isinstance(m, SwooshR):
            replacements[name] = SwooshROnnx()
        elif isinstance(m, SwooshL):
            replacements[name] = SwooshLOnnx()
    for k, v in replacements.items():
        if "." in k:
            parent_path, child = k.rsplit(".", maxsplit=1)
            parent = model
            for part in parent_path.split("."):
                parent = getattr(parent, part)
            setattr(parent, child, v)
        else:
            setattr(model, k, v)


# ---------------------------------------------------------------------------
# Patch Conv2dSubsampling to avoid aten::Int ops that coremltools rejects
# ---------------------------------------------------------------------------

def _freeze_rel_pos_encoding(module: nn.Module) -> None:
    """Capture positional encoding outputs during a forward pass and freeze them.

    Must be called AFTER a reference forward pass through the encoder so that
    each CompactRelPositionalEncoding has computed its output for the correct
    sequence length.  Replaces forward with a method that returns the captured
    constant, avoiding aten::Int ops from pe.size(0) // 2 indexing.
    """
    import types

    for name, m in module.named_modules():
        if type(m).__name__ != "CompactRelPositionalEncoding":
            continue

        # The reference forward pass already computed pe for the right size.
        # We need to re-run forward once to capture the output for this stack's
        # actual sequence length.  We do this by probing which pe slice it would
        # return — we already know pe is built.
        # Instead of guessing seq_len, we register a forward hook to capture output.

    # Use hooks to capture outputs during a second forward pass
    captured = {}

    def _make_hook(mod_name):
        def hook(mod, inp, out):
            captured[mod_name] = out.detach().clone()
        return hook

    hooks = []
    for name, m in module.named_modules():
        if type(m).__name__ == "CompactRelPositionalEncoding":
            hooks.append(m.register_forward_hook(_make_hook(name)))

    return captured, hooks


def _apply_frozen_pos_encoding(module: nn.Module, captured: dict, hooks: list) -> None:
    """Replace pos encoding forward methods with captured constants."""
    import types

    # Remove hooks
    for h in hooks:
        h.remove()

    for name, m in module.named_modules():
        if name in captured:
            frozen_emb = captured[name]
            m.register_buffer("_frozen_pos_emb", frozen_emb)

            def _forward_frozen(self, x: torch.Tensor, left_context_len: int = 0) -> torch.Tensor:
                return self._frozen_pos_emb

            m.forward = types.MethodType(_forward_frozen, m)


def _patch_conv2d_subsampling(module: nn.Module) -> None:
    """Replace Conv2dSubsampling.forward with a trace-friendly version.

    The original uses ``b, c, t, f = x.size()`` followed by
    ``reshape(b, t, c * f)`` which emits ``aten::Int`` ops.  We replace
    the reshape with ``flatten(2).transpose(1, 2)`` which achieves the
    same result without scalar int casts.
    """
    from subsampling import Conv2dSubsampling

    for name, m in module.named_modules():
        if isinstance(m, Conv2dSubsampling):
            _patch_single_conv2d_sub(m)


def _patch_single_conv2d_sub(m) -> None:
    """Monkey-patch a Conv2dSubsampling instance."""
    import types

    orig_conv = m.conv
    orig_convnext = m.convnext
    orig_out = m.out
    orig_out_whiten = m.out_whiten
    orig_out_norm = m.out_norm
    orig_dropout = m.dropout

    def _forward_coreml(self, x: torch.Tensor, x_lens: torch.Tensor):
        x = x.unsqueeze(1)
        x = orig_conv(x)
        x = orig_convnext(x)
        # x: (N, C, T, F) -> (N, T, C*F) without aten::Int
        x = x.permute(0, 2, 1, 3).flatten(2)
        x = orig_out(x)
        x = orig_out_whiten(x)
        x = orig_out_norm(x)
        x = orig_dropout(x)
        x_lens = (x_lens - 7) // 2
        return x, x_lens

    m.forward = types.MethodType(_forward_coreml, m)


# ---------------------------------------------------------------------------
# Wrapper modules for tracing
# ---------------------------------------------------------------------------

class EncoderForExport(nn.Module):
    """Fuses encoder_embed + zipformer encoder + joiner encoder_proj.

    Takes mel frames (T, 80) as input.
    """

    def __init__(self, encoder_embed: nn.Module, encoder: nn.Module, encoder_proj: nn.Module):
        super().__init__()
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_proj = encoder_proj

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens, x.shape[1])
        x = x.permute(1, 0, 2)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)
        encoder_out = self.encoder_proj(encoder_out)
        return encoder_out, encoder_out_lens


class FusedPreprocessorForExport(nn.Module):
    """Fuses mel extraction + encoder_embed + zipformer encoder + joiner encoder_proj.

    Takes raw audio waveform (1, num_samples) as input, like Parakeet's preprocessor.
    Produces encoder features (1, T', joiner_dim) and output lengths.
    """

    def __init__(
        self,
        fbank: nn.Module,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        encoder_proj: nn.Module,
    ):
        super().__init__()
        self.fbank = fbank
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.encoder_proj = encoder_proj

    def forward(self, audio_signal: torch.Tensor, audio_length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # audio_signal: (1, num_samples), audio_length: (1,) actual sample count
        x = self.fbank(audio_signal)  # (T, 80)
        x = x.unsqueeze(0)  # (1, T, 80)
        x_lens = torch.tensor([x.shape[1]], dtype=torch.int64)
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens, x.shape[1])
        x = x.permute(1, 0, 2)
        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)
        encoder_out = encoder_out.permute(1, 0, 2)
        encoder_out = self.encoder_proj(encoder_out)
        return encoder_out, encoder_out_lens


class DecoderForExport(nn.Module):
    """Fuses decoder embedding + joiner decoder_proj."""

    def __init__(self, decoder: nn.Module, decoder_proj: nn.Module):
        super().__init__()
        self.decoder = decoder
        self.decoder_proj = decoder_proj

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        out = self.decoder(y, need_pad=False).squeeze(1)
        return self.decoder_proj(out)


class JoinerForExport(nn.Module):
    """Joiner output linear: tanh(enc + dec) -> logits."""

    def __init__(self, output_linear: nn.Module):
        super().__init__()
        self.output_linear = output_linear

    def forward(self, encoder_out: torch.Tensor, decoder_out: torch.Tensor) -> torch.Tensor:
        return self.output_linear(torch.tanh(encoder_out + decoder_out))


# ---------------------------------------------------------------------------
# Build model from checkpoint
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path):
    """Load checkpoint and build encoder_embed, encoder, decoder, joiner."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Use "model" (not "model_avg") to match sherpa-onnx ONNX export weights.
    # model_avg has different weights that produce substantially worse WER.
    state_dict = ckpt["model"]

    encoder_embed = Conv2dSubsampling(
        in_channels=int(ckpt.get("feature_dim", 80)),
        out_channels=_to_int_tuple(ckpt["encoder_dim"])[0],
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
    )
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(ckpt["downsampling_factor"]),
        num_encoder_layers=_to_int_tuple(ckpt["num_encoder_layers"]),
        encoder_dim=_to_int_tuple(ckpt["encoder_dim"]),
        encoder_unmasked_dim=_to_int_tuple(ckpt["encoder_unmasked_dim"]),
        query_head_dim=_to_int_tuple(ckpt["query_head_dim"]),
        pos_head_dim=_to_int_tuple(ckpt["pos_head_dim"]),
        value_head_dim=_to_int_tuple(ckpt["value_head_dim"]),
        pos_dim=int(ckpt["pos_dim"]),
        num_heads=_to_int_tuple(ckpt["num_heads"]),
        feedforward_dim=_to_int_tuple(ckpt["feedforward_dim"]),
        cnn_module_kernel=_to_int_tuple(ckpt["cnn_module_kernel"]),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=bool(ckpt["causal"]),
        chunk_size=_to_int_tuple(ckpt["chunk_size"]),
        left_context_frames=_to_int_tuple(ckpt["left_context_frames"]),
    )
    decoder = Decoder(
        vocab_size=int(ckpt["vocab_size"]),
        decoder_dim=int(ckpt["decoder_dim"]),
        blank_id=int(ckpt["blank_id"]),
        context_size=int(ckpt["context_size"]),
    )
    joiner = Joiner(
        encoder_dim=max(_to_int_tuple(ckpt["encoder_dim"])),
        decoder_dim=int(ckpt["decoder_dim"]),
        joiner_dim=int(ckpt["joiner_dim"]),
        vocab_size=int(ckpt["vocab_size"]),
    )

    for name, module in [
        ("encoder_embed", encoder_embed),
        ("encoder", encoder),
        ("decoder", decoder),
        ("joiner", joiner),
    ]:
        prefix = name + "."
        sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        module.load_state_dict(sub)

    return ckpt, encoder_embed, encoder, decoder, joiner


def load_tokens(tokens_path: Path) -> dict[int, str]:
    """Parse tokens.txt (token_string token_id) into {id: token}."""
    tokens: dict[int, str] = {}
    with open(tokens_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                tokens[int(parts[1])] = parts[0]
    return tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", exists=True, resolve_path=True,
        help="Path to the icefall .pt checkpoint (e.g. epoch-56-avg-4.pt).",
    ),
    tokens: Path = typer.Option(
        ..., "--tokens", exists=True, resolve_path=True,
        help="Path to tokens.txt from the model's lang directory.",
    ),
    output_dir: Path = typer.Option(
        Path("build/sherpa-onnx-zipformer"),
        "--output-dir",
        help="Directory for the exported .mlpackage files and metadata.",
    ),
    mel_frames: int = typer.Option(
        DEFAULT_MEL_FRAMES,
        "--mel-frames",
        help="Fixed number of mel spectrogram frames for the encoder input. "
             "Must produce a frame count divisible by 8 after Conv2dSubsampling.",
    ),
    compute_units: str = typer.Option(
        "CPU_ONLY",
        "--compute-units",
        help="CoreML compute units: ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE.",
    ),
    float16: bool = typer.Option(
        False,
        "--float16",
        help="Export with FLOAT16 precision (halves model size, faster serialization).",
    ),
    fuse_mel: bool = typer.Option(
        True,
        "--fuse-mel/--no-fuse-mel",
        help="Fuse kaldi fbank mel extraction into the encoder (default: on). "
             "The resulting Preprocessor.mlpackage takes raw audio (1, num_samples) "
             "like Parakeet. Use --no-fuse-mel for standalone encoder with mel input.",
    ),
    max_audio_samples: int = typer.Option(
        239120,
        "--max-audio-samples",
        help="Maximum audio samples for fused mel mode (default: 239120 ≈ 14.95s at 16kHz, "
             "produces 1495 mel frames for encoder compatibility).",
    ),
) -> None:
    """Export Sherpa-ONNX Zipformer2 transducer to CoreML."""
    cu = _parse_compute_units(compute_units)
    precision = ct.precision.FLOAT16 if float16 else ct.precision.FLOAT32
    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading checkpoint: {checkpoint}")
    ckpt, encoder_embed, encoder, decoder, joiner = load_model(checkpoint)
    context_size = int(ckpt["context_size"])
    vocab_size = int(ckpt["vocab_size"])
    joiner_dim = int(ckpt["joiner_dim"])
    blank_id = int(ckpt["blank_id"])

    # Replace training-only ops and patch trace-unfriendly patterns
    for module in [encoder_embed, encoder, decoder, joiner]:
        convert_scaled_for_coreml(module)
    _patch_conv2d_subsampling(encoder_embed)

    # ---------------------------------------------------------------
    # Encoder (or Fused Preprocessor)
    # ---------------------------------------------------------------
    import shutil

    if fuse_mel:
        # Fused mode: mel extraction + encoder in one model (like Parakeet's Preprocessor)
        from fused_fbank import KaldiFbank

        typer.echo(f"Tracing fused preprocessor (max_audio_samples={max_audio_samples})...")
        fbank = KaldiFbank().eval()
        fused = FusedPreprocessorForExport(
            fbank, encoder_embed, encoder, joiner.encoder_proj
        ).eval()

        audio_in = torch.randn(1, max_audio_samples)
        audio_len = torch.tensor([max_audio_samples], dtype=torch.int64)

        # Capture pos-encoding outputs during reference pass, then freeze them
        captured, hooks = _freeze_rel_pos_encoding(encoder)
        with torch.no_grad():
            ref_enc_out, ref_enc_lens = fused(audio_in, audio_len)
        _apply_frozen_pos_encoding(encoder, captured, hooks)

        with torch.no_grad():
            traced_enc = torch.jit.trace(fused, (audio_in, audio_len), strict=False)

        typer.echo("Converting fused preprocessor to CoreML...")
        enc_ml = ct.convert(
            traced_enc,
            inputs=[
                ct.TensorType(name="audio_signal", shape=(1, max_audio_samples), dtype=np.float32),
                ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="encoder_out", dtype=np.float32),
                ct.TensorType(name="encoder_out_lens", dtype=np.int32),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_units=cu,
            compute_precision=precision,
            skip_model_load=True,
        )
        typer.echo("CoreML conversion done. Saving preprocessor...")
        enc_path = output_dir / "Preprocessor.mlpackage"
        enc_ml.short_description = f"Zipformer2 Fused Preprocessor ({max_audio_samples} samples)"
        enc_ml.author = AUTHOR
        if enc_path.exists():
            shutil.rmtree(enc_path)
        enc_ml.save(str(enc_path))
        typer.echo(f"  -> {enc_path}")

    else:
        # Mel-frames mode: encoder takes mel spectrogram (1, T, 80)
        typer.echo(f"Tracing encoder (mel_frames={mel_frames})...")
        enc = EncoderForExport(encoder_embed, encoder, joiner.encoder_proj).eval()
        T = mel_frames
        x = torch.randn(1, T, 80)
        x_lens = torch.tensor([T], dtype=torch.int64)

        # Capture pos-encoding outputs during reference pass, then freeze them
        captured, hooks = _freeze_rel_pos_encoding(encoder)
        with torch.no_grad():
            ref_enc_out, ref_enc_lens = enc(x, x_lens)
        _apply_frozen_pos_encoding(encoder, captured, hooks)

        with torch.no_grad():
            traced_enc = torch.jit.trace(enc, (x, x_lens), strict=False)

        typer.echo("Converting encoder to CoreML...")
        enc_ml = ct.convert(
            traced_enc,
            inputs=[
                ct.TensorType(name="x", shape=(1, T, 80), dtype=np.float32),
                ct.TensorType(name="x_lens", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="encoder_out", dtype=np.float32),
                ct.TensorType(name="encoder_out_lens", dtype=np.int32),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS18,
            compute_units=cu,
            compute_precision=precision,
            skip_model_load=True,
        )
        typer.echo("CoreML conversion done. Saving encoder...")
        enc_path = output_dir / "encoder.mlpackage"
        enc_ml.short_description = f"Zipformer2 Encoder ({T} mel frames)"
        enc_ml.author = AUTHOR
        if enc_path.exists():
            shutil.rmtree(enc_path)
        enc_ml.save(str(enc_path))
        typer.echo(f"  -> {enc_path}")

    # ---------------------------------------------------------------
    # Decoder
    # ---------------------------------------------------------------
    typer.echo("Tracing decoder...")
    dec = DecoderForExport(decoder, joiner.decoder_proj).eval()
    y = torch.zeros(1, context_size, dtype=torch.int64)

    with torch.no_grad():
        traced_dec = torch.jit.trace(dec, (y,), strict=False)

    typer.echo("Converting decoder to CoreML...")
    dec_ml = ct.convert(
        traced_dec,
        inputs=[ct.TensorType(name="y", shape=(1, context_size), dtype=np.int32)],
        outputs=[ct.TensorType(name="decoder_out", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=cu,
        compute_precision=precision,
        skip_model_load=True,
    )
    dec_path = output_dir / "decoder.mlpackage"
    dec_ml.short_description = f"Zipformer2 Stateless Decoder (context_size={context_size})"
    dec_ml.author = AUTHOR
    dec_ml.save(str(dec_path))
    typer.echo(f"  -> {dec_path}")

    # ---------------------------------------------------------------
    # Joiner
    # ---------------------------------------------------------------
    typer.echo("Tracing joiner...")
    join = JoinerForExport(joiner.output_linear).eval()

    with torch.no_grad():
        traced_join = torch.jit.trace(
            join,
            (torch.randn(1, joiner_dim), torch.randn(1, joiner_dim)),
            strict=False,
        )

    typer.echo("Converting joiner to CoreML...")
    join_ml = ct.convert(
        traced_join,
        inputs=[
            ct.TensorType(name="encoder_out", shape=(1, joiner_dim), dtype=np.float32),
            ct.TensorType(name="decoder_out", shape=(1, joiner_dim), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logit", dtype=np.float32)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=cu,
        compute_precision=precision,
        skip_model_load=True,
    )
    join_path = output_dir / "joiner.mlpackage"
    join_ml.short_description = "Zipformer2 Joiner (encoder_out + decoder_out -> logits)"
    join_ml.author = AUTHOR
    join_ml.save(str(join_path))
    typer.echo(f"  -> {join_path}")

    # ---------------------------------------------------------------
    # Vocabulary
    # ---------------------------------------------------------------
    token_map = load_tokens(tokens)
    vocab = [token_map.get(i, "") for i in range(vocab_size)]
    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False))
    typer.echo(f"  -> {vocab_path} ({len(vocab)} tokens)")

    # ---------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------
    # Build encoder/preprocessor component metadata
    if fuse_mel:
        encoder_component = {
            "path": enc_path.name,
            "inputs": {
                "audio_signal": [1, max_audio_samples],
                "audio_length": [1],
            },
            "outputs": {
                "encoder_out": list(ref_enc_out.shape),
                "encoder_out_lens": list(ref_enc_lens.shape),
            },
        }
    else:
        encoder_component = {
            "path": enc_path.name,
            "inputs": {"x": [1, mel_frames, 80], "x_lens": [1]},
            "outputs": {
                "encoder_out": list(ref_enc_out.shape),
                "encoder_out_lens": list(ref_enc_lens.shape),
            },
        }

    metadata = {
        "model_type": "zipformer2_transducer",
        "fused_mel": fuse_mel,
        "source": "sherpa-onnx / icefall",
        "checkpoint": str(checkpoint.name),
        "sample_rate": 16000,
        "feature_dim": int(ckpt.get("feature_dim", 80)),
        "max_audio_samples": max_audio_samples if fuse_mel else None,
        "mel_frames": None if fuse_mel else mel_frames,
        "subsampling_factor": int(ckpt.get("subsampling_factor", 4)),
        "vocab_size": vocab_size,
        "blank_id": blank_id,
        "context_size": context_size,
        "joiner_dim": joiner_dim,
        "encoder_dim": str(ckpt["encoder_dim"]),
        "coreml": {
            "compute_units": cu.name,
            "compute_precision": "FLOAT16" if float16 else "FLOAT32",
            "deployment_target": "iOS18",
        },
        "components": {
            "preprocessor" if fuse_mel else "encoder": encoder_component,
            "decoder": {
                "path": dec_path.name,
                "inputs": {"y": [1, context_size]},
                "outputs": {"decoder_out": [1, joiner_dim]},
            },
            "joiner": {
                "path": join_path.name,
                "inputs": {
                    "encoder_out": [1, joiner_dim],
                    "decoder_out": [1, joiner_dim],
                },
                "outputs": {"logit": [1, vocab_size]},
            },
        },
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    typer.echo(f"  -> {meta_path}")
    typer.echo("Export complete.")


if __name__ == "__main__":
    app()
