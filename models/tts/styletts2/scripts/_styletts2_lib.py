"""Shared loader for the StyleTTS2-LibriTTS inference modules.

All export scripts (01–04) call `load_inference_modules()` to get the same set
of `nn.Module`s in eval mode with their parameters loaded from the stripped
checkpoint produced by `00_fetch_weights.py`.

Why a shared helper:
  - The upstream `models.build_model(...)` insists on building discriminators
    and a JDC pitch extractor and an ASR aligner that we don't need at
    inference. We only need: text_encoder, predictor (incl. F0Ntrain),
    style_encoder, predictor_encoder, bert, bert_encoder, decoder, diffusion.
  - The upstream config lives in YAML, but the inference-relevant fields fit
    in a compact dataclass — no need to drag in `munch`/`yaml` semantics.
  - All forwards used at inference must be traceable. We provide thin wrappers
    that drop `pack_padded_sequence` (no-op for B=1 + lengths==seq_len) and
    avoid `.cpu().numpy()` calls.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Vendor path setup --------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PKG_DIR = THIS_DIR.parent  # models/tts/styletts2
VENDOR_DIR = PKG_DIR / "vendor" / "StyleTTS2"
DEFAULT_CHECKPOINT = PKG_DIR / "checkpoints" / "styletts2_libritts_inference.pt"
COREML_DIR = PKG_DIR / "coreml"
PLBERT_DIR = VENDOR_DIR / "Utils" / "PLBERT"

if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))


def register_coreml_op_shims() -> None:
    """Register torch op handlers for ops coremltools 8.x doesn't natively map.

    Currently:
      - aten::multiply  → coremltools `mb.mul` (used by SineGen in hifigan.py).

    Idempotent: register once per process.
    """
    try:
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
        from coremltools.converters.mil.frontend.torch.torch_op_registry import (
            register_torch_op,
        )
        from coremltools.converters.mil.mil import Builder as mb
    except ImportError:
        return

    if getattr(register_coreml_op_shims, "_done", False):
        return

    @register_torch_op
    def multiply(context, node):
        inputs = _get_inputs(context, node, expected=2)
        res = mb.mul(x=inputs[0], y=inputs[1], name=node.name)
        context.add(res)

    register_coreml_op_shims._done = True  # type: ignore[attr-defined]


# --- Compact LibriTTS config (mirrors Configs/config_libritts.yml) ------------


@dataclass(frozen=True)
class LibriTTSConfig:
    sample_rate: int = 24000
    n_mels: int = 80
    n_fft: int = 2048
    hop_length: int = 300
    win_length: int = 1200

    multispeaker: bool = True
    dim_in: int = 64
    hidden_dim: int = 512
    max_conv_dim: int = 512
    n_layer: int = 3
    n_token: int = 178
    max_dur: int = 50
    style_dim: int = 128
    dropout: float = 0.2

    decoder_type: str = "hifigan"
    resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11)
    upsample_rates: Tuple[int, ...] = (10, 5, 3, 2)
    upsample_initial_channel: int = 512
    resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    upsample_kernel_sizes: Tuple[int, ...] = (20, 10, 6, 4)

    diffusion_dist_mean: float = -3.0
    diffusion_dist_std: float = 1.0
    diffusion_sigma_data: float = 0.2

    transformer_num_layers: int = 3
    transformer_num_heads: int = 8
    transformer_head_features: int = 64
    transformer_multiplier: int = 2

    @property
    def hop_factor(self) -> int:
        f = 1
        for r in self.upsample_rates:
            f *= r
        return f


# --- Module construction ------------------------------------------------------


def _build_modules(cfg: LibriTTSConfig):
    """Build the inference-relevant nn.Modules. Imports are local to keep the
    vendor path side-effects scoped."""

    # PL-BERT (HuggingFace AlbertModel). The upstream loader lives in
    # Utils/PLBERT/util.py — `load_plbert(log_dir)`.
    sys.path.insert(0, str(PLBERT_DIR))
    from util import load_plbert  # type: ignore[import-not-found]

    bert = load_plbert(str(PLBERT_DIR))

    # Now bring in upstream model.py
    from models import (  # type: ignore[import-not-found]
        ProsodyPredictor,
        StyleEncoder,
        TextEncoder,
    )
    from Modules.diffusion.modules import (  # type: ignore
        AttentionBase,
        StyleTransformer1d,
    )
    from Modules.diffusion.diffusion import AudioDiffusionConditional  # type: ignore
    from Modules.diffusion.sampler import KDiffusion, LogNormalDistribution  # type: ignore

    # `AttentionBase.forward` uses einsum with leading `...` (e.g.
    # "... n d, ... m d -> ... n m"). coremltools 8.x routes this through
    # `solve_diagonal_einsum`, which then asks the MIL transpose op to permute
    # a 4-D tensor with a 5-element perm — fails with "perm should have the
    # same length as rank(x)". Patch the forward to use plain matmul.
    from einops import rearrange  # type: ignore

    def _attention_forward_matmul(self, q, k, v):
        # q, k, v: (b, n, h*d)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        sim = torch.matmul(q, k.transpose(-1, -2))  # (b, h, n, m)
        if self.use_rel_pos:
            sim = sim + self.rel_pos(*sim.shape[-2:])
        sim = sim * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.matmul(attn, v)  # (b, h, n, d)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

    AttentionBase.forward = _attention_forward_matmul

    # `SineGen._f02sine` (Modules/hifigan.py) uses `F.interpolate(..., mode="linear",
    # scale_factor=1/300)` which trips coremltools' upsample_linear1d translator
    # ("recompute_scale_factor=False, align_corners=False with float output size
    # is not supported"). Set align_corners=True — quality impact is negligible
    # and CoreML's upsample_bilinear supports this configuration.
    import numpy as _np
    from Modules.hifigan import SineGen  # type: ignore

    def _f02sine_align_corners(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        if not self.flag_for_pulse:
            rad_values = F.interpolate(
                rad_values.transpose(1, 2),
                scale_factor=1.0 / self.upsample_scale,
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * _np.pi
            phase = F.interpolate(
                phase.transpose(1, 2) * self.upsample_scale,
                scale_factor=float(self.upsample_scale),
                mode="linear",
                align_corners=True,
            ).transpose(1, 2)
            sines = torch.sin(phase)
        else:
            # Pulse-train branch is never used in inference (flag_for_pulse=False
            # in Generator). Keep upstream behavior if anyone flips it.
            sines = SineGen._f02sine_original(self, f0_values)  # type: ignore[attr-defined]
        return sines

    if not hasattr(SineGen, "_f02sine_original"):
        SineGen._f02sine_original = SineGen._f02sine  # type: ignore[attr-defined]
    SineGen._f02sine = _f02sine_align_corners

    if cfg.decoder_type == "hifigan":
        from Modules.hifigan import Decoder  # type: ignore
        decoder = Decoder(
            dim_in=cfg.hidden_dim,
            style_dim=cfg.style_dim,
            dim_out=cfg.n_mels,
            resblock_kernel_sizes=list(cfg.resblock_kernel_sizes),
            upsample_rates=list(cfg.upsample_rates),
            upsample_initial_channel=cfg.upsample_initial_channel,
            resblock_dilation_sizes=[list(d) for d in cfg.resblock_dilation_sizes],
            upsample_kernel_sizes=list(cfg.upsample_kernel_sizes),
        )
    else:
        raise ValueError(f"Only hifigan decoder is supported here (got {cfg.decoder_type!r}).")

    text_encoder = TextEncoder(
        channels=cfg.hidden_dim,
        kernel_size=5,
        depth=cfg.n_layer,
        n_symbols=cfg.n_token,
    )
    predictor = ProsodyPredictor(
        style_dim=cfg.style_dim,
        d_hid=cfg.hidden_dim,
        nlayers=cfg.n_layer,
        max_dur=cfg.max_dur,
        dropout=cfg.dropout,
    )
    style_encoder = StyleEncoder(
        dim_in=cfg.dim_in,
        style_dim=cfg.style_dim,
        max_conv_dim=cfg.hidden_dim,
    )
    predictor_encoder = StyleEncoder(
        dim_in=cfg.dim_in,
        style_dim=cfg.style_dim,
        max_conv_dim=cfg.hidden_dim,
    )
    bert_encoder = nn.Linear(bert.config.hidden_size, cfg.hidden_dim)

    transformer = StyleTransformer1d(
        channels=cfg.style_dim * 2,
        context_embedding_features=bert.config.hidden_size,
        context_features=cfg.style_dim * 2,
        num_layers=cfg.transformer_num_layers,
        num_heads=cfg.transformer_num_heads,
        head_features=cfg.transformer_head_features,
        multiplier=cfg.transformer_multiplier,
    )
    diffusion = AudioDiffusionConditional(
        in_channels=1,
        embedding_max_length=bert.config.max_position_embeddings,
        embedding_features=bert.config.hidden_size,
        embedding_mask_proba=0.0,
        channels=cfg.style_dim * 2,
        context_features=cfg.style_dim * 2,
    )
    diffusion.diffusion = KDiffusion(
        net=diffusion.unet,
        sigma_distribution=LogNormalDistribution(
            mean=cfg.diffusion_dist_mean, std=cfg.diffusion_dist_std
        ),
        sigma_data=cfg.diffusion_sigma_data,
        dynamic_threshold=0.0,
    )
    diffusion.diffusion.net = transformer
    diffusion.unet = transformer

    return {
        "bert": bert,
        "bert_encoder": bert_encoder,
        "text_encoder": text_encoder,
        "predictor": predictor,
        "style_encoder": style_encoder,
        "predictor_encoder": predictor_encoder,
        "decoder": decoder,
        "diffusion": diffusion,
    }


def load_inference_modules(checkpoint: Path = DEFAULT_CHECKPOINT, cfg: LibriTTSConfig | None = None):
    """Build modules and load weights from the stripped checkpoint.

    Returns a dict of `nn.Module` in eval() mode on CPU.
    """
    cfg = cfg or LibriTTSConfig()
    modules = _build_modules(cfg)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Inference checkpoint {checkpoint} not found. "
            f"Run scripts/00_fetch_weights.py first."
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for name, module in modules.items():
        sd = state.get(name)
        if sd is None:
            print(f"[lib] skip: no state for {name!r}")
            continue
        missing, unexpected = module.load_state_dict(sd, strict=False)
        if missing:
            print(f"[lib] {name}: {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"[lib] {name}: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
        module.eval()

    return modules, cfg


# --- Traceable wrappers -------------------------------------------------------
#
# Every wrapper below assumes batch_size == 1 and lengths == full T_tok / T_mel.
# This is the inference reality (we always pad to a bucket and feed full
# tensors). Pack/pad and length-mask logic from training is dropped.


class TextEncoderTraceable(nn.Module):
    """`models.TextEncoder` without pack_padded_sequence and without masks."""

    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.embedding = text_encoder.embedding
        self.cnn = text_encoder.cnn  # ModuleList of Sequential(Conv1d, LayerNorm, actv, Dropout)
        self.lstm = text_encoder.lstm

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (1, T_tok) int64
        x = self.embedding(tokens)             # (1, T_tok, channels)
        x = x.transpose(1, 2)                  # (1, channels, T_tok)
        for c in self.cnn:
            x = c(x)
        x = x.transpose(1, 2)                  # (1, T_tok, channels)
        x, _ = self.lstm(x)                    # (1, T_tok, channels)  (bidir, /2 hidden ×2)
        x = x.transpose(-1, -2)                # (1, channels, T_tok)
        return x


class DurationEncoderTraceable(nn.Module):
    """`models.DurationEncoder` for batch=1, no masking, no packing."""

    def __init__(self, duration_encoder: nn.Module):
        super().__init__()
        self.lstms = duration_encoder.lstms
        self.d_model = duration_encoder.d_model
        self.sty_dim = duration_encoder.sty_dim

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x: (1, hidden_dim, T_tok)  — d_en from bert_encoder.transpose
        # style: (1, style_dim)
        # Returns: (1, T_tok, hidden_dim + style_dim)  — same shape as upstream
        # (look at upstream final transpose: returns x.transpose(-1,-2))

        # Match upstream layout: bring T to front-of-time, expand style.
        x = x.permute(2, 0, 1)                 # (T_tok, 1, hidden_dim)
        s = style.expand(x.shape[0], x.shape[1], -1)  # (T_tok, 1, style_dim)
        x = torch.cat([x, s], axis=-1)         # (T_tok, 1, hidden_dim + style_dim)

        x = x.transpose(0, 1)                  # (1, T_tok, h+s)
        x = x.transpose(-1, -2)                # (1, h+s, T_tok)

        # Imports here to avoid leaking the vendor path at module import time.
        from models import AdaLayerNorm  # type: ignore

        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)  # cat style as channels
            else:
                x = x.transpose(-1, -2)        # (1, T, ch)
                x, _ = block(x)                # direct LSTM, no pack_padded
                x = x.transpose(-1, -2)        # (1, ch, T)

        return x.transpose(-1, -2)             # (1, T_tok, h+s)


class TextPredictorTraceable(nn.Module):
    """Package A: tokens → (t_en, d_en, d, pred_dur, fixed_embedding).

    Outputs:
      t_en:             (1, hidden_dim, T_tok)
      d_en:             (1, hidden_dim, T_tok)  — bert_encoder(bert(tokens)).transpose
      d:                (1, T_tok, h+s)         — DurationEncoder hidden
      pred_dur_log:     (1, T_tok, max_dur)     — pre-sigmoid duration logits
      fixed_embedding:  (1, T_tok, bert_dim)    — for CFG uncond branch in Swift
    """

    def __init__(self, modules: dict):
        super().__init__()
        self.bert = modules["bert"]
        self.bert_encoder = modules["bert_encoder"]
        self.text_encoder = TextEncoderTraceable(modules["text_encoder"])
        predictor = modules["predictor"]
        self.duration_encoder = DurationEncoderTraceable(predictor.text_encoder)
        self.lstm = predictor.lstm
        self.duration_proj = predictor.duration_proj
        # Hold a ref to the diffusion's fixed_embedding for CFG uncond.
        self.fixed_embedding = modules["diffusion"].unet.fixed_embedding

    def forward(self, tokens: torch.Tensor, style: torch.Tensor):
        # tokens: (1, T_tok) int64
        # style:  (1, style_dim)
        t_en = self.text_encoder(tokens)                      # (1, h, T)

        # `CustomAlbert.forward` (Utils/PLBERT/util.py) overrides AlbertModel.forward
        # to return last_hidden_state directly as a Tensor.
        bert_dur = self.bert(tokens)
        # bert_dur: (1, T_tok, bert_dim)

        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)  # (1, h, T)

        d = self.duration_encoder(d_en, style)                # (1, T, h+s)

        x, _ = self.lstm(d)                                   # (1, T, h)
        pred_dur_log = self.duration_proj(x)                  # (1, T, max_dur)

        fixed_embedding = self.fixed_embedding(bert_dur)      # (1, T, bert_dim)

        return t_en, d_en, d, pred_dur_log, fixed_embedding, bert_dur


class DiffusionDenoiseTraceable(nn.Module):
    """Package B: single ADPM2 / Karras denoising step of the style UNet.

    Wraps `KDiffusion.denoise_fn` minus the kwargs plumbing. The sampler loop
    and CFG combination live in Swift.

    Inputs:
      x_noisy:   (1, 1, style_dim*2)
      sigma:     (1,)                   scalar noise level
      embedding: (1, T_tok, bert_dim)   either real bert_dur or fixed_embedding
      features:  (1, style_dim*2)       ref_s

    Output:
      denoised:  (1, 1, style_dim*2)
    """

    def __init__(self, modules: dict, sigma_data: float):
        super().__init__()
        self.transformer = modules["diffusion"].unet  # StyleTransformer1d
        self.sigma_data = sigma_data

    def forward(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        embedding: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        # KDiffusion.get_scale_weights — inlined for tracing.
        s = sigma.view(-1, 1, 1)                             # (1,1,1)
        sd = self.sigma_data
        c_skip = (sd * sd) / (s * s + sd * sd)
        c_out = s * sd / torch.sqrt(s * s + sd * sd)
        c_in = 1.0 / torch.sqrt(s * s + sd * sd)
        c_noise = torch.log(sigma) * 0.25                    # (1,)

        # StyleTransformer1d.run bypasses the `forward()` kwargs/CFG path.
        x_pred = self.transformer.run(
            c_in * x_noisy,
            c_noise,
            embedding=embedding,
            features=features,
        )
        return c_skip * x_noisy + c_out * x_pred


class F0NEnergyTraceable(nn.Module):
    """Package C: predictor.F0Ntrain — (en, s) → (F0, N).

    Pure conv + bidirectional LSTM (`predictor.shared`). The shared LSTM has a
    fixed seq dim (T_mel) so it traces cleanly.
    """

    def __init__(self, modules: dict):
        super().__init__()
        predictor = modules["predictor"]
        self.shared = predictor.shared
        self.F0 = predictor.F0
        self.N = predictor.N
        self.F0_proj = predictor.F0_proj
        self.N_proj = predictor.N_proj

    def forward(self, en: torch.Tensor, s: torch.Tensor):
        # en: (1, hidden_dim + style_dim, T_mel) — upstream feeds
        #     `d.transpose(-1,-2) @ alignment` where d is the DurationEncoder
        #     output with channel dim h+s (640 for LibriTTS).
        # s:  (1, style_dim)
        x, _ = self.shared(en.transpose(-1, -2))             # (1, T_mel, h)

        F0 = x.transpose(-1, -2)                             # (1, h, T_mel)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0).squeeze(1)                     # (1, T_mel)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N).squeeze(1)                        # (1, T_mel)

        return F0, N


class HifiGanDecoderTraceable(nn.Module):
    """Package D: HiFi-GAN Decoder.forward but with eval-only branches.

    Identical in eval mode to `Modules.hifigan.Decoder.forward`; we inline it
    here to drop the `if self.training` branch entirely so the trace is clean.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.encode = decoder.encode
        self.decode = decoder.decode
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.asr_res = decoder.asr_res
        self.generator = decoder.generator

    def forward(
        self,
        asr: torch.Tensor,
        F0_curve: torch.Tensor,
        N: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        F0 = self.F0_conv(F0_curve.unsqueeze(1))             # (1, 1, T_mel/2)
        Nc = self.N_conv(N.unsqueeze(1))                      # (1, 1, T_mel/2)

        x = torch.cat([asr, F0, Nc], dim=1)                   # (1, h+2, T_mel/2)
        x = self.encode(x, s)                                 # (1, 1024, T_mel/2)

        asr_res = self.asr_res(asr)                           # (1, 64, T_mel/2)

        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, Nc], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False

        return self.generator(x, s, F0_curve)                 # (1, T_mel * hop_factor)
