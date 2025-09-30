#!/usr/bin/env python3
"""Convert pyannote community speaker diarization components to Core ML."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchaudio.compliance import kaldi

from pyannote.audio import Model
from pyannote.audio.utils.receptive_field import conv1d_num_frames

from embedding_io import EMBEDDING_SAMPLES, SEGMENTATION_FRAMES
from plda_module import load_plda_module_from_npz
def _patch_sincnet_encoder_for_tracing(model: nn.Module) -> None:
    """Replace SincNet encoder forward with a trace-friendly variant."""

    if not hasattr(model, "sincnet"):
        return

    encoder = getattr(model.sincnet, "conv1d", [None])[0]

    if encoder is None:
        return

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        filters = self.get_filters()
        waveform = self.filterbank.pre_analysis(waveform)
        spec = F.conv1d(waveform, filters, stride=self.stride, padding=self.padding)
        return self.filterbank.post_analysis(spec)

    encoder.forward = forward.__get__(encoder, encoder.__class__)

COREML_AUTHOR = "Fluid Inference"
COREML_LICENSE = "CC-BY-4.0"
COMMUNITY_VERSION = "pyannote-speaker-diarization-community-1"
SEGMENTATION_SAMPLES = 160_000  # 10 s @ 16 kHz
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parent / COMMUNITY_VERSION


class TraceableFbank(nn.Module):
    """Filterbank front-end built from conv/matmul to match Kaldi."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.sample_rate = hparams.sample_rate
        self.num_mel_bins = hparams.num_mel_bins
        self.frame_length = int(self.sample_rate * hparams.frame_length * 0.001)
        self.frame_shift = int(self.sample_rate * hparams.frame_shift * 0.001)
        self.round_to_power_of_two = hparams.round_to_power_of_two
        self.dither = hparams.dither
        self.remove_dc_offset = True
        self.preemphasis = 0.97
        self.raw_energy = True
        self.use_energy = hparams.use_energy

        self.padded_window = (
            1 << (self.frame_length - 1).bit_length()
            if self.round_to_power_of_two
            else self.frame_length
        )
        self.pad_amount = self.padded_window - self.frame_length
        self.num_fft_bins = self.padded_window // 2 + 1

        eye = torch.eye(self.frame_length).view(self.frame_length, 1, self.frame_length)
        self.register_buffer("frame_kernel", eye)

        n = torch.arange(self.frame_length, dtype=torch.float32)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (max(self.frame_length - 1, 1)))
        self.register_buffer("window", window.unsqueeze(0))

        time = torch.arange(self.padded_window, dtype=torch.float32)
        freqs = torch.arange(self.num_fft_bins, dtype=torch.float32).unsqueeze(1)
        angles = 2.0 * math.pi * freqs * time / self.padded_window
        dft_real = torch.cos(angles)
        dft_imag = -torch.sin(angles)
        self.register_buffer(
            "dft_real_weight",
            dft_real.contiguous().view(self.num_fft_bins, 1, self.padded_window),
        )
        self.register_buffer(
            "dft_imag_weight",
            dft_imag.contiguous().view(self.num_fft_bins, 1, self.padded_window),
        )

        mel_basis, _ = kaldi.get_mel_banks(
            self.num_mel_bins,
            self.padded_window,
            float(self.sample_rate),
            20.0,
            0.0,
            100.0,
            -500.0,
            1.0,
        )
        mel_basis = F.pad(mel_basis.unsqueeze(0), (0, 1), value=0.0).squeeze(0)
        self.register_buffer(
            "mel_weight", mel_basis.contiguous().view(self.num_mel_bins, self.num_fft_bins, 1)
        )
        # FP16-friendly epsilon to avoid underflow when logs are computed on-device.
        self.register_buffer("eps", torch.tensor(1e-6, dtype=torch.float32))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute FBANK features for a batch of waveforms."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() == 2:
            signal = waveform.unsqueeze(1)  # (batch, 1, samples)
        elif waveform.dim() == 3 and waveform.shape[1] == 1:
            signal = waveform
        else:
            raise ValueError(f"Unexpected waveform shape {tuple(waveform.shape)}")

        batch = signal.shape[0]
        frames = F.conv1d(signal, self.frame_kernel, stride=self.frame_shift)
        # frames shape: (batch, frame_length, num_frames)
        if frames.numel() == 0:
            return frames.new_zeros((batch, 0, self.num_mel_bins))

        frames = frames.transpose(1, 2)  # (batch, num_frames, frame_length)
        batch_frames, num_frames, frame_len = frames.shape
        frames = frames.reshape(batch_frames * num_frames, frame_len)

        if self.dither != 0.0:
            frames = frames + torch.randn_like(frames) * self.dither

        if self.remove_dc_offset:
            frames = frames - frames.mean(dim=1, keepdim=True)

        if self.raw_energy:
            log_energy = torch.clamp(frames.pow(2).sum(dim=1), min=self.eps).log()
        else:
            log_energy = None

        if self.preemphasis != 0.0:
            padded = F.pad(frames.unsqueeze(1), (1, 0), mode="replicate").squeeze(1)
            frames = frames - self.preemphasis * padded[:, :-1]

        frames = frames * self.window

        if self.pad_amount > 0:
            frames = F.pad(frames.unsqueeze(1), (0, self.pad_amount), value=0.0).squeeze(1)

        if not self.raw_energy:
            log_energy = torch.clamp(frames.pow(2).sum(dim=1), min=self.eps).log()

        conv_input = frames.unsqueeze(1)
        real = F.conv1d(conv_input, self.dft_real_weight).squeeze(-1)
        imag = F.conv1d(conv_input, self.dft_imag_weight).squeeze(-1)
        power = real.pow(2) + imag.pow(2)

        if self.use_energy and log_energy is not None:
            power[:, 0] = log_energy.exp()

        mel = F.conv1d(power.unsqueeze(-1), self.mel_weight).squeeze(-1)
        mel = mel + self.eps
        mel = torch.log(torch.clamp(mel, min=self.eps))

        if self.use_energy and log_energy is not None:
            mel[:, 0] = log_energy

        return mel.view(batch, num_frames, self.num_mel_bins)


class TraceableStatsPool(nn.Module):
    """TorchScript-friendly stats pooling with unbiased variance."""

    def forward(self, features: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        if features.dim() == 4:
            batch, dimension, channel, frames = features.shape
            sequences = features.reshape(batch, dimension * channel, frames)
        else:
            sequences = features

        original_dtype = sequences.dtype
        sequences_fp32 = sequences.to(torch.float32)
        weights_fp32: Optional[torch.Tensor]
        if weights is not None:
            weights_fp32 = weights.to(torch.float32)
        else:
            weights_fp32 = None

        if weights is None:
            mean = sequences_fp32.mean(dim=-1)
            centered = sequences_fp32 - mean.unsqueeze(-1)
            sum_sq = (centered * centered).sum(dim=-1)
            frames = sequences_fp32.shape[-1]
            frame_count = sum_sq.new_full((1,), frames, dtype=sequences_fp32.dtype)
            adjusted = frame_count - 1.0
            denom = torch.where(adjusted > 0.0, adjusted, torch.ones_like(adjusted))
            var = sum_sq / denom
            std = torch.sqrt(torch.clamp(var, min=1e-6))
            output = torch.cat([mean, std], dim=-1)
            return output.to(original_dtype)

        # Weighted pooling path
        if weights.dim() == 2:
            weights_fp32 = weights_fp32.unsqueeze(1)  # type: ignore[union-attr]
            has_speaker_dimension = False
        else:
            has_speaker_dimension = True

        # Note: We assume weights are already interpolated to match num_frames
        # This avoids dynamic interpolation which CoreML can't trace
        weights_expanded = weights_fp32.unsqueeze(2)  # type: ignore[union-attr]
        v1 = weights_expanded.sum(dim=-1) + 1e-4
        weighted = sequences_fp32.unsqueeze(1) * weights_expanded
        mean = weighted.sum(dim=-1) / v1
        diff = sequences_fp32.unsqueeze(1) - mean.unsqueeze(-1)
        v2 = (weights_expanded * weights_expanded).sum(dim=-1)
        denom = v1 - v2 / v1 + 1e-4
        var = (diff * diff * weights_expanded).sum(dim=-1) / denom
        std = torch.sqrt(torch.clamp(var, min=1e-6))
        output = torch.cat([mean, std], dim=-1)

        if not has_speaker_dimension:  # type: ignore[truthy-bool]
            return output.squeeze(1).to(original_dtype)

        return output.to(original_dtype)


def skip_sensitive_ops_segmentation(op) -> bool:
    """Keep operations critical for log probability accuracy in FP32."""
    sensitive_ops = [
        "log",           # Critical - outputs log probabilities
        "softmax",       # If present
        "layer_norm",    # Normalization
        "batch_norm",    # Normalization
        "reduce_mean",   # Statistics computation
        "reduce_sum",    # Statistics computation
        "div",           # Division operations
    ]
    return op.op_type not in sensitive_ops


def skip_sensitive_ops_embedding(op) -> bool:
    """Keep operations critical for embedding quality in FP32.

    Optimized for ANE: Only keeps the final stats pooling and L2 normalization
    operations in FP32. The entire ResNet backbone (Conv2d + BatchNorm + ReLU)
    can run in FP16 on ANE for maximum performance.
    """
    name_candidates: list[str] = []
    for attr in ("name", "op_name"):
        candidate = getattr(op, attr, None)
        if isinstance(candidate, str):
            name_candidates.append(candidate)
    for tensor in getattr(op, "outputs", []) or []:
        tensor_name = getattr(tensor, "name", "")
        if tensor_name:
            name_candidates.append(tensor_name)

    normalized_names = [candidate.lower() for candidate in name_candidates if candidate]

    # More surgical approach: Only match actual stats pooling operations
    # Remove "norm" keyword which was matching ALL BatchNorm layers
    stats_pooling_keywords = (
        "pool",      # Stats pooling layer
        "weighted",  # Weighted pooling operations
    )

    # Final embedding normalization keywords (for L2 norm at the end)
    final_norm_keywords = (
        "reduce_l2",  # L2 normalization
        "l2_norm",    # L2 normalization
    )

    def _matches_stats_pooling() -> bool:
        """Check if this is part of the stats pooling layer."""
        return any(
            any(keyword in candidate for keyword in stats_pooling_keywords)
            for candidate in normalized_names
        )

    def _matches_final_norm() -> bool:
        """Check if this is part of the final L2 normalization."""
        return any(
            any(keyword in candidate for keyword in final_norm_keywords)
            for candidate in normalized_names
        )

    # Keep stats pooling operations in FP32 for numerical stability
    stats_sensitive_types = {
        "reduce_mean",   # Mean calculation in stats pooling
        "reduce_sum",    # Sum for weighted pooling
        "sqrt",          # Std deviation calculation
        "div",           # Division in variance calculation
        "real_div",      # Division operations
        "sub",           # Centering operations
    }

    if op.op_type in stats_sensitive_types and _matches_stats_pooling():
        return False

    # Keep final L2 normalization in FP32
    if op.op_type == "reduce_l2_norm" or _matches_final_norm():
        return False

    # Keep constants that feed stats pooling in FP32
    if op.op_type == "const" and (_matches_stats_pooling() or _matches_final_norm()):
        return False

    # IMPORTANT: Allow ALL other operations (including BatchNorm, Conv2d, ReLU,
    # and most arithmetic) to run in FP16 on ANE. This is the key optimization
    # that increases ANE utilization from ~30% to 70-90%.
    return True


class WeSpeakerTraceableFrontend(nn.Module):
    """Trace-friendly frontend that produces centered FBANK features."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.hparams = base_model.hparams
        self._fbank = TraceableFbank(self.hparams)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.dim() != 3 or waveforms.shape[1] != 1:
            raise ValueError(f"Expected waveforms with shape (batch, 1, samples), got {tuple(waveforms.shape)}")

        waveforms = waveforms * (1 << 15)
        device = waveforms.device
        fft_device = torch.device("cpu") if device.type == "mps" else device

        batch_features = self._fbank(waveforms.to(fft_device).squeeze(1)).to(device)

        if self.hparams.fbank_centering_span is None:
            centered = torch.mean(batch_features, dim=1, keepdim=True)
        else:
            window_size = int(self.hparams.sample_rate * self.hparams.frame_length * 0.001)
            step_size = int(self.hparams.sample_rate * self.hparams.frame_shift * 0.001)
            kernel_size = conv1d_num_frames(
                num_samples=int(self.hparams.fbank_centering_span * self.hparams.sample_rate),
                kernel_size=window_size,
                stride=step_size,
                padding=0,
                dilation=1,
            )
            centered = F.avg_pool1d(
                batch_features.transpose(1, 2),
                kernel_size=2 * (kernel_size // 2) + 1,
                stride=1,
                padding=kernel_size // 2,
                count_include_pad=False,
            ).transpose(1, 2)

        features = batch_features - centered
        return features.permute(0, 2, 1).unsqueeze(1)


class WeSpeakerTraceableBackend(nn.Module):
    """Trace-friendly backend that consumes FBANK features and segmentation weights."""

    def __init__(self, base_model: nn.Module, target_frames: int) -> None:
        super().__init__()
        self.resnet = copy.deepcopy(base_model.resnet).eval()
        self.resnet.pool = TraceableStatsPool()
        self._convert_linear_head_to_conv()
        self._patch_resnet_forward()
        self.hparams = base_model.hparams
        self.target_frames = target_frames
        self.source_frames = SEGMENTATION_FRAMES

        positions = torch.linspace(
            0, self.source_frames - 1, self.target_frames, dtype=torch.float32
        )
        left_idx = torch.floor(positions).to(torch.long)
        right_idx = torch.clamp(left_idx + 1, max=self.source_frames - 1)
        right_w = positions - left_idx.to(torch.float32)
        left_w = 1.0 - right_w

        kernel = torch.zeros(self.target_frames, 1, self.source_frames, dtype=torch.float32)
        kernel[torch.arange(self.target_frames), 0, left_idx] = left_w
        kernel[torch.arange(self.target_frames), 0, right_idx] += right_w
        self.register_buffer("_interp_kernel", kernel)

    def _convert_linear_head_to_conv(self) -> None:
        """Swap the projection head to a 1x1 conv to keep a 4D layout."""
        linear = getattr(self.resnet, "seg_1", None)
        if not isinstance(linear, nn.Linear):
            return

        conv = nn.Conv2d(
            in_channels=linear.in_features,
            out_channels=linear.out_features,
            kernel_size=1,
            bias=linear.bias is not None,
        )
        conv.weight.data.copy_(linear.weight.data.view(linear.out_features, linear.in_features, 1, 1))
        if linear.bias is not None:
            conv.bias.data.copy_(linear.bias.data)

        self.resnet.seg_1 = conv

    def _patch_resnet_forward(self) -> None:
        """Patch ResNet forward to accept channels-first FBANK input."""

        def forward(resnet_self, fbank: torch.Tensor, weights: Optional[torch.Tensor] = None):
            if fbank.dim() == 4:
                features = fbank
            elif fbank.dim() == 3:
                features = fbank.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected fbank shape {tuple(fbank.shape)}")

            out = F.relu(resnet_self.bn1(resnet_self.conv1(features)))
            out = resnet_self.layer1(out)
            out = resnet_self.layer2(out)
            out = resnet_self.layer3(out)
            out = resnet_self.layer4(out)

            stats = resnet_self.pool(out, weights=weights)
            stats_4d = stats.unsqueeze(-1).unsqueeze(-1)

            embed_a = resnet_self.seg_1(stats_4d).flatten(1)
            if resnet_self.two_emb_layer:
                out = F.relu(embed_a)
                out = resnet_self.seg_bn_1(out)
                embed_b = resnet_self.seg_2(out)
                return embed_a, embed_b

            zero = torch.tensor(0.0, device=embed_a.device, dtype=embed_a.dtype)
            return zero, embed_a

        self.resnet.forward = forward.__get__(self.resnet, self.resnet.__class__)

    def interpolate_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Interpolate segmentation weights using a fixed 1D convolution kernel."""
        if weights.shape[-1] == self.source_frames:
            kernel = self._interp_kernel.to(device=weights.device, dtype=weights.dtype)
        else:
            source_frames = weights.shape[-1]
            positions = torch.linspace(
                0,
                source_frames - 1,
                self.target_frames,
                dtype=weights.dtype,
                device=weights.device,
            )
            left_idx = torch.floor(positions).to(torch.long)
            right_idx = torch.clamp(left_idx + 1, max=source_frames - 1)
            right_w = positions - left_idx.to(positions.dtype)
            left_w = 1.0 - right_w

            kernel = torch.zeros(
                self.target_frames, 1, source_frames, dtype=weights.dtype, device=weights.device
            )
            kernel[torch.arange(self.target_frames, device=weights.device), 0, left_idx] = left_w
            kernel[torch.arange(self.target_frames, device=weights.device), 0, right_idx] += right_w

        weights_1d = weights.unsqueeze(1)
        interpolated = F.conv1d(weights_1d, kernel, bias=None)
        return interpolated.squeeze(-1)

    def forward(self, fbank: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if fbank.dim() not in (3, 4):
            raise ValueError(f"Unexpected fbank feature shape {tuple(fbank.shape)}")
        if weights.dim() != 2:
            raise ValueError(f"Unexpected weights shape {tuple(weights.shape)}")

        if fbank.dim() == 3:
            features = fbank.unsqueeze(1)
        else:
            features = fbank

        if features.shape[1] != 1:
            raise ValueError(f"FBANK features must be channels-first with shape (batch, 1, mel, frames), got {tuple(features.shape)}")

        weights = self.interpolate_weights(weights)

        embeddings = self.resnet(features, weights=weights)[1]

        norms = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        norms = torch.clamp(norms, min=1e-4)
        return embeddings / norms


def trace_segmentation(model_root: Path, device: torch.device) -> torch.jit.ScriptModule:
    checkpoint_dir = model_root / "segmentation"
    model = Model.from_pretrained(str(checkpoint_dir)).to(device).eval()
    _patch_sincnet_encoder_for_tracing(model)
    # Trace with batch size 32 (most common) - will support 1..32 via EnumeratedShapes
    example = torch.zeros(32, 1, SEGMENTATION_SAMPLES, dtype=torch.float32, device=device)
    with torch.inference_mode():
        traced = model.to_torchscript(example_inputs=example, method="trace")
    return traced.to("cpu")


def trace_fbank(model_root: Path, device: torch.device) -> torch.jit.ScriptModule:
    checkpoint_dir = model_root / "embedding"
    base_model = Model.from_pretrained(str(checkpoint_dir)).to(device).eval()
    frontend = WeSpeakerTraceableFrontend(base_model).to(device).eval()
    example_audio = torch.zeros(1, 1, EMBEDDING_SAMPLES, dtype=torch.float32, device=device)

    with torch.inference_mode():
        traced = torch.jit.trace(frontend, example_audio)

    return traced.to("cpu")


def _detect_backend_target_frames(base_model: nn.Module, features: torch.Tensor) -> int:
    resnet = copy.deepcopy(base_model.resnet).eval()
    with torch.no_grad():
        out = F.relu(resnet.bn1(resnet.conv1(features)))
        out = resnet.layer1(out)
        out = resnet.layer2(out)
        out = resnet.layer3(out)
        out = resnet.layer4(out)
    return out.shape[-1]


def trace_embedding(model_root: Path, device: torch.device) -> torch.jit.ScriptModule:
    checkpoint_dir = model_root / "embedding"
    base_model = Model.from_pretrained(str(checkpoint_dir)).to(device).eval()

    frontend = WeSpeakerTraceableFrontend(base_model).to(device).eval()
    example_audio = torch.zeros(1, 1, EMBEDDING_SAMPLES, dtype=torch.float32, device=device)
    example_weights = torch.ones(1, SEGMENTATION_FRAMES, dtype=torch.float32, device=device)

    with torch.no_grad():
        example_features = frontend(example_audio)

    num_frames = _detect_backend_target_frames(base_model, example_features.clone())
    print(f"Detected {num_frames} frames at pooling layer for {EMBEDDING_SAMPLES} samples")

    backend = WeSpeakerTraceableBackend(base_model, target_frames=num_frames).to(device).eval()

    with torch.inference_mode():
        traced = torch.jit.trace(backend, (example_features, example_weights))
    return traced.to("cpu")


def trace_plda(model_root: Path, device: torch.device) -> torch.jit.ScriptModule:
    """Trace PLDA transformation module."""
    plda_module = load_plda_module_from_npz(model_root, lda_dim=128).to(device).eval()
    # Use batch size 32 as default
    example_embeddings = torch.zeros(32, 256, dtype=torch.float32, device=device)

    with torch.inference_mode():
        traced = torch.jit.trace(plda_module, example_embeddings)
    return traced.to("cpu")


def trace_plda_rho(model_root: Path, device: torch.device) -> torch.jit.ScriptModule:
    """Trace PLDA rho module (features scaled by sqrt(phi) for VBx)."""
    from plda_module import load_plda_rho_module_from_npz

    plda_rho_module = load_plda_rho_module_from_npz(model_root, lda_dim=128).to(device).eval()
    # Use batch size 32 as default
    example_embeddings = torch.zeros(32, 256, dtype=torch.float32, device=device)

    with torch.inference_mode():
        traced = torch.jit.trace(plda_rho_module, example_embeddings)
    return traced.to("cpu")


@dataclass(frozen=True)
class InputSpec:
    name: str
    shape: tuple[int, ...]  # Default shape (used for tracing and as default in EnumeratedShapes)
    enumerated_shapes: Optional[tuple[tuple[int, ...], ...]] = None  # Additional shapes for ANE optimization
    range_dim_indices: Optional[tuple[int, ...]] = None  # Indices of dimensions to make flexible with RangeDim
    range_dim_bounds: Optional[dict[int, tuple[int, int]]] = None  # Explicit (min, max) for RangeDim indices


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    trace_fn: Callable[[Path, torch.device], torch.jit.ScriptModule]
    inputs: tuple[InputSpec, ...]
    output_name: Optional[str]
    description: str
    output_shape: Optional[tuple[int, ...]] = None

    # Backward compatibility
    @property
    def input_name(self) -> str:
        return self.inputs[0].name

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.inputs[0].shape


COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        name="segmentation",
        trace_fn=trace_segmentation,
        inputs=(InputSpec(
            name="audio",
            shape=(32, 1, SEGMENTATION_SAMPLES),
            enumerated_shapes=tuple(
                (batch, 1, SEGMENTATION_SAMPLES)
                for batch in range(1, 33)
            ),
        ),),
        output_name="log_probs",
        description="pyannote community-1 segmentation (10 s powerset diarization, batch 1-32)",
    ),
    ComponentSpec(
        name="fbank",
        trace_fn=trace_fbank,
        inputs=(InputSpec(
            name="audio",
            shape=(1, 1, EMBEDDING_SAMPLES),
            enumerated_shapes=tuple(
                (batch, 1, EMBEDDING_SAMPLES)
                for batch in range(1, 33)
            ),
        ),),
        output_name="fbank_features",
        description="pyannote community-1 FBANK frontend (10 s audio preprocessing to 80×998 features, batch 1-32, CPU preferred)",
    ),
    ComponentSpec(
        name="embedding",
        trace_fn=trace_embedding,
        inputs=(
            InputSpec(
                name="fbank_features",
                shape=(1, 1, 80, 998),
                range_dim_indices=(0,),
                range_dim_bounds={0: (1, 32)},
            ),
            InputSpec(
                name="weights",
                shape=(1, SEGMENTATION_FRAMES),
                range_dim_indices=(0,),
                range_dim_bounds={0: (1, 32)},
            ),
        ),
        output_name="embedding",
        description="pyannote community-1 speaker embedding backend (WeSpeaker ResNet34 consuming 80×998 FBANK features + 589-frame weights, interpolates weights to 125-frame pooling layer internally)",
    ),
    ComponentSpec(
        name="plda",
        trace_fn=trace_plda,
        inputs=(InputSpec(
            name="embeddings",
            shape=(32, 256),
            enumerated_shapes=tuple(
                (batch, 256)
                for batch in range(1, 33)
            ),
        ),),
        output_name="plda_features",
        description="pyannote community-1 PLDA transform (x-vector whitening + LDA projection, batch 1-32)",
    ),
    ComponentSpec(
        name="plda_rho",
        trace_fn=trace_plda_rho,
        inputs=(InputSpec(
            name="embeddings",
            shape=(32, 256),
            enumerated_shapes=tuple(
                (batch, 256)
                for batch in range(1, 33)
            ),
        ),),
        output_name="rho",
        description="pyannote community-1 PLDA rho (features scaled by sqrt(phi) for VBx clustering, batch 1-32)",
    ),
)


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    relative_path: Path
    output_name: str
    description: str


RESOURCES: tuple[ResourceSpec, ...] = (
    ResourceSpec(
        name="plda_parameters",
        relative_path=Path("plda/plda.npz"),
        output_name="plda-parameters.json",
        description="PLDA mean, transform, and psi parameters for VBx clustering",
    ),
    ResourceSpec(
        name="xvec_transform",
        relative_path=Path("plda/xvec_transform.npz"),
        output_name="xvector-transform.json",
        description="x-vector whitening/centering transform used before PLDA scoring",
    ),
)


def convert_resource(model_root: Path, output_dir: Path, spec: ResourceSpec) -> Path:
    input_path = model_root / spec.relative_path
    if not input_path.exists():
        raise FileNotFoundError(f"Resource not found: {input_path}")

    print(f"Serializing {spec.name} tensors...")
    loaded = np.load(input_path)
    tensors: dict[str, dict[str, object]] = {}
    for key in loaded.files:
        array = np.asarray(loaded[key])
        dtype_str = str(array.dtype)
        tensors[key] = {
            "shape": list(array.shape),
            "dtype": dtype_str,
            "layout": "row-major",
            "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }

    payload = {
        "name": spec.name,
        "description": spec.description,
        "license": COREML_LICENSE,
        "source": spec.relative_path.as_posix(),
        "version": COMMUNITY_VERSION,
        "tensors": tensors,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / spec.output_name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"✓ {spec.name} saved to {output_path}")
    return output_path


def convert_component(
    model_root: Path,
    output_dir: Path,
    device: torch.device,
    spec: ComponentSpec,
    compute_precision: object,
    precision_label: str,
) -> Path:
    print(f"Tracing {spec.name} model...")
    traced = spec.trace_fn(model_root, device)
    output_path = output_dir / f"{spec.name}-community-1.mlpackage"

    print(f"Converting {spec.name} to Core ML ({precision_label})...")
    # Build inputs with EnumeratedShapes or RangeDim if specified
    inputs = []
    for input_spec in spec.inputs:
        if input_spec.enumerated_shapes is not None:
            # Use EnumeratedShapes for ANE optimization
            tensor_type = ct.TensorType(
                name=input_spec.name,
                shape=ct.EnumeratedShapes(
                    shapes=list(input_spec.enumerated_shapes),
                    default=input_spec.shape
                ),
                dtype=np.float32
            )
        elif input_spec.range_dim_indices is not None:
            # Use RangeDim for specified dimensions
            shape_list = list(input_spec.shape)
            bounds = input_spec.range_dim_bounds or {}
            for idx in input_spec.range_dim_indices:
                min_val, max_val = bounds.get(idx, (1, shape_list[idx]))
                shape_list[idx] = ct.RangeDim(min_val, max_val)
            tensor_type = ct.TensorType(
                name=input_spec.name,
                shape=tuple(shape_list),
                dtype=np.float32
            )
        else:
            # Use fixed shape
            tensor_type = ct.TensorType(
                name=input_spec.name,
                shape=input_spec.shape,
                dtype=np.float32
            )
        inputs.append(tensor_type)
    outputs = None
    if spec.output_name is not None:
        if spec.output_shape is not None:
            outputs = [
                ct.TensorType(name=spec.output_name, shape=spec.output_shape, dtype=np.float32)
            ]
        else:
            outputs = [ct.TensorType(name=spec.output_name, dtype=np.float32)]


    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=outputs,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=compute_precision,
    )

    mlmodel.author = COREML_AUTHOR
    mlmodel.short_description = spec.description
    mlmodel.version = COMMUNITY_VERSION
    mlmodel.license = COREML_LICENSE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    print(f"✓ {spec.name.capitalize()} model saved to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pyannote community speaker diarization components to Core ML",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Path to the downloaded pyannote-speaker-diarization-community-1 repository",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./coreml_models"),
        help="Directory where generated .mlpackage files will be stored",
    )
    parser.add_argument(
        "--selective-fp16",
        action="store_true",
        help=(
            "Convert models with selective FP16 precision, keeping sensitive operations in FP32"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    print(f"Using Torch device: {device}")
    model_root = args.model_root
    if not model_root.exists():
        raise FileNotFoundError(f"Model root not found: {model_root}")
    generated = []
    selective_fp16 = bool(args.selective_fp16)
    for spec in COMPONENTS:
        if spec.name == "fbank":
            compute_precision = ct.precision.FLOAT32
            precision_label = "FP32 (frontend forced to CPU)"
        elif selective_fp16:
            if spec.name == "segmentation":
                compute_precision = ct.transform.FP16ComputePrecision(
                    op_selector=skip_sensitive_ops_segmentation
                )
            elif spec.name == "embedding":
                compute_precision = ct.transform.FP16ComputePrecision(
                    op_selector=skip_sensitive_ops_embedding
                )
            else:
                compute_precision = ct.transform.FP16ComputePrecision()
            precision_label = "selective FP16 (sensitive ops in FP32)"
        else:
            compute_precision = ct.precision.FLOAT32
            precision_label = "FP32"

        generated.append(
            convert_component(
                model_root,
                args.output_dir,
                device,
                spec,
                compute_precision,
                precision_label,
            )
        )

    resource_output_dir = args.output_dir / "resources"
    resource_paths = []
    for spec in RESOURCES:
        resource_paths.append(convert_resource(model_root, resource_output_dir, spec))

    print("Conversion complete:")
    for path in generated:
        print(f"  - {path}")
    for path in resource_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
