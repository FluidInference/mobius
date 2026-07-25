"""Convert Inflect v2 (Micro/Nano) VITS TTS to fixed-shape CoreML models.

The VITS inference graph is split into two deterministic CoreML models:
  1. encoder: tokens + mask -> m_p, logs_p, logw          [T_text fixed]
  2. synthesizer: z_p + mask -> waveform                  [T_frames fixed]

Everything stochastic or dynamically shaped runs on the host:
  - duration ceil + length-scale, expansion of m_p/logs_p to frame rate
  - z_p = m_p + randn * exp(logs_p) * noise_scale
Waveform is trimmed to y_length * hop_length (256) samples on the host.
"""

import argparse
import contextlib
import io
import sys
import types
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent


def load_synthesizer(checkpoint_dir: Path):
    runtime = checkpoint_dir / "runtime"
    for entry in (str(runtime), str(checkpoint_dir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import utils
    from models import SynthesizerTrn
    from text.symbols import symbols

    hps = utils.get_hparams_from_file(str(checkpoint_dir / "config.json"))
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).eval()
    utils.load_checkpoint(str(checkpoint_dir / "model.pth"), model, None)
    with contextlib.redirect_stdout(io.StringIO()):
        model.dec.remove_weight_norm()
    for flow in model.flow.flows:
        encoder = getattr(flow, "enc", None)
        if encoder is not None and hasattr(encoder, "remove_weight_norm"):
            encoder.remove_weight_norm()
    return model, hps


def _patch_fused_activation():
    """Replace the IntTensor channel split in the WN gated activation with a
    Python-int slice so tracing emits static ops."""
    import commons

    def fused(input_a, input_b, n_channels):
        n = int(n_channels[0])
        in_act = input_a + input_b
        return torch.tanh(in_act[:, :n, :]) * torch.sigmoid(in_act[:, n:, :])

    commons.fused_add_tanh_sigmoid_multiply = fused
    import modules

    modules.commons.fused_add_tanh_sigmoid_multiply = fused


def _static_relative_attention(layer):
    """Coerce traced sequence lengths to Python ints so relative-attention
    pads and slices become compile-time constants (shapes are fixed)."""
    import torch.nn.functional as F
    import commons

    def _get_relative_embeddings(self, relative_embeddings, length):
        length = int(length)
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start = max((self.window_size + 1) - length, 0)
        if pad_length > 0:
            relative_embeddings = F.pad(
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        return relative_embeddings[:, slice_start : slice_start + 2 * length - 1]

    def _rel_to_abs(self, x):
        batch, heads, length, _ = (int(v) for v in x.size())
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1 :]

    def _abs_to_rel(self, x):
        batch, heads, length, _ = (int(v) for v in x.size())
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length * length + length * (length - 1)])
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

    def _attention(self, query, key, value, mask=None):
        b, d, t_s = (int(v) for v in key.size())
        t_t = int(query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        scores = torch.matmul(query / self.k_channels**0.5, key.transpose(-2, -1))
        key_rel = self._get_relative_embeddings(self.emb_rel_k, t_s)
        rel_logits = self._matmul_with_relative_keys(query / self.k_channels**0.5, key_rel)
        scores = scores + self._relative_position_to_absolute_position(rel_logits)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(p_attn, value)
        relative_weights = self._absolute_position_to_relative_position(p_attn)
        value_rel = self._get_relative_embeddings(self.emb_rel_v, t_s)
        output = output + self._matmul_with_relative_values(relative_weights, value_rel)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    layer._get_relative_embeddings = types.MethodType(_get_relative_embeddings, layer)
    layer._relative_position_to_absolute_position = types.MethodType(_rel_to_abs, layer)
    layer._absolute_position_to_relative_position = types.MethodType(_abs_to_rel, layer)
    layer.attention = types.MethodType(_attention, layer)


class EncoderWrapper(nn.Module):
    """TextEncoder + DurationPredictor with a host-provided padding mask."""

    def __init__(self, model):
        super().__init__()
        self.enc_p = model.enc_p
        for layer in self.enc_p.encoder.attn_layers:
            _static_relative_attention(layer)
        self.dp = model.dp
        self.scale = model.enc_p.hidden_channels ** 0.5
        self.out_channels = model.enc_p.out_channels

    def forward(self, tokens, x_mask):
        x = self.enc_p.emb(tokens) * self.scale  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x = self.enc_p.encoder(x * x_mask, x_mask)
        stats = self.enc_p.proj(x) * x_mask
        m_p, logs_p = torch.split(stats, self.out_channels, dim=1)
        logw = self.dp(x, x_mask)
        return m_p, logs_p, logw


class SynthesizerWrapper(nn.Module):
    """Reverse coupling flow + HiFiGAN decoder on frame-rate latents."""

    def __init__(self, model):
        super().__init__()
        _patch_fused_activation()
        self.flow = model.flow
        self.dec = model.dec

    def forward(self, z_p, y_mask):
        z = self.flow(z_p, y_mask, reverse=True)
        return self.dec(z * y_mask)


def convert(traced, inputs, outputs, precision):
    return ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        convert_to="mlprogram",
    )


def convert_variant(
    checkpoint_dir: Path, output_dir: Path, t_text: int, t_frames: int, precision, io_fp16: bool = False
):
    model, hps = load_synthesizer(checkpoint_dir)
    inter = hps.model.inter_channels
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = EncoderWrapper(model).eval()
    tokens = torch.zeros(1, t_text, dtype=torch.int64)
    tokens[0, : t_text // 2] = torch.randint(1, 100, (t_text // 2,))
    x_mask = torch.zeros(1, 1, t_text)
    x_mask[:, :, : t_text // 2] = 1.0
    with torch.no_grad():
        traced = torch.jit.trace(encoder, (tokens, x_mask))
    ml_encoder = convert(
        traced,
        inputs=[
            ct.TensorType(name="tokens", shape=(1, t_text), dtype=np.int32),
            ct.TensorType(name="x_mask", shape=(1, 1, t_text), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="m_p"),
            ct.TensorType(name="logs_p"),
            ct.TensorType(name="logw"),
        ],
        precision=precision,
    )
    encoder_path = output_dir / "encoder.mlpackage"
    ml_encoder.save(str(encoder_path))
    print(f"saved {encoder_path}")

    synthesizer = SynthesizerWrapper(model).eval()
    z_p = torch.randn(1, inter, t_frames)
    y_mask = torch.zeros(1, 1, t_frames)
    y_mask[:, :, : t_frames // 2] = 1.0
    io_dtype = np.float16 if io_fp16 else np.float32
    with torch.no_grad():
        traced = torch.jit.trace(synthesizer, (z_p, y_mask))
    ml_synth = convert(
        traced,
        inputs=[
            ct.TensorType(name="z_p", shape=(1, inter, t_frames), dtype=io_dtype),
            ct.TensorType(name="y_mask", shape=(1, 1, t_frames), dtype=io_dtype),
        ],
        outputs=[ct.TensorType(name="audio", dtype=io_dtype)],
        precision=precision,
    )
    synth_path = output_dir / "synthesizer.mlpackage"
    ml_synth.save(str(synth_path))
    print(f"saved {synth_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["micro", "nano", "all"], default="all")
    parser.add_argument("--t-text", type=int, default=256, help="fixed interspersed token length")
    parser.add_argument("--t-frames", type=int, default=1024, help="fixed mel-frame length (256 samples each)")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--io-fp16", action="store_true", help="fp16 model I/O (skips fp32<->fp16 casts)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()

    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    variants = ["micro", "nano"] if args.variant == "all" else [args.variant]
    for variant in variants:
        checkpoint_dir = ROOT / "checkpoints" / f"inflect-{variant}-v2"
        suffix = "-io16" if args.io_fp16 else ""
        out = args.output_dir / f"inflect-{variant}-v2-{args.precision}-t{args.t_text}-f{args.t_frames}{suffix}"
        print(f"=== converting {variant} (t_text={args.t_text}, t_frames={args.t_frames}, {args.precision}{suffix}) ===")
        convert_variant(checkpoint_dir, out, args.t_text, args.t_frames, precision, io_fp16=args.io_fp16)


if __name__ == "__main__":
    main()
