#!/usr/bin/env python3
"""Wrapper modules for Qwen3-ASR-0.6B CoreML export.

Each wrapper normalises I/O shapes so that torch.jit.trace produces a clean
graph suitable for coremltools conversion.

Architecture overview (from modeling_qwen3_asr.py):
  Qwen3ASRForConditionalGeneration
    └── thinker: Qwen3ASRThinkerForConditionalGeneration
          ├── audio_tower: Qwen3ASRAudioEncoder
          ├── model: Qwen3ASRThinkerTextModel (28-layer Qwen3 LLM)
          └── lm_head: Linear(1024, 151936)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import coremltools as ct
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnemllRMSNorm(nn.Module):
    """ANEMLL-style RMSNorm that uses native LayerNorm for better CoreML precision.

    Standard RMSNorm is decomposed by coremltools into pow → mean → rsqrt → mul,
    each step losing FP16 precision. This trick concatenates [x, -x] so the mean
    is forced to zero, making LayerNorm mathematically equivalent to RMSNorm.
    The native layer_norm op preserves precision on GPU/ANE.

    Reference: https://huggingface.co/blog/anemll/anemll-style-rms-ane
    """

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps
        self.dim = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(doubled, [doubled.shape[-1]], eps=self.eps)
        normed = normed[..., : self.dim]
        return normed * self.weight


def patch_rms_norms(module: nn.Module) -> None:
    """Replace all Qwen3RMSNorm instances in a module tree with AnemllRMSNorm."""
    for name, child in list(module.named_children()):
        class_name = type(child).__name__
        if class_name == "AnemllRMSNorm":
            continue  # already patched
        if "RMSNorm" in class_name and hasattr(child, "weight"):
            eps = getattr(child, "variance_epsilon", getattr(child, "eps", 1e-6))
            replacement = AnemllRMSNorm(child.weight.data, eps=eps)
            setattr(module, name, replacement)
        else:
            patch_rms_norms(child)


@dataclass
class ExportSettings:
    output_dir: Path
    compute_units: ct.ComputeUnit
    deployment_target: Optional[ct.target]
    compute_precision: Optional[ct.precision]
    max_audio_seconds: float
    max_seq_length: int  # max token sequence length for LLM decoder


# ---------------------------------------------------------------------------
# Audio Encoder Wrappers
# ---------------------------------------------------------------------------

class AudioEncoderConvWrapper(nn.Module):
    """Conv2D frontend of the audio encoder.

    Takes a padded mel chunk batch and returns downsampled + projected features.

    Input:
      - mel_chunks: [B, 1, 128, T]  (padded mel spectrogram chunks)

    Output:
      - features: [B, T', 896]  (after conv downsampling + linear projection)
    """

    def __init__(self, audio_encoder: nn.Module) -> None:
        super().__init__()
        self.conv2d1 = audio_encoder.conv2d1
        self.conv2d2 = audio_encoder.conv2d2
        self.conv2d3 = audio_encoder.conv2d3
        self.conv_out = audio_encoder.conv_out

    def forward(self, mel_chunks: torch.Tensor) -> torch.Tensor:
        # mel_chunks: [B, 1, 128, T]
        x = F.gelu(self.conv2d1(mel_chunks))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        # x: [B, 480, F', T'] where F' = ceil(128/8) = 17, T' = ceil(T/8)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.conv_out(x)  # [B, T', 896]
        return x


class AudioEncoderTransformerWrapper(nn.Module):
    """Transformer layers of the audio encoder (without preprocessing/chunking).

    Processes a single fixed-size sequence through the 18 transformer layers
    with an explicit attention mask (replacing cu_seqlens-based windowed attention).

    Input:
      - hidden_states: [1, T, 896]  (features after conv + positional embedding)
      - attention_mask: [1, 1, T, T]  (block-diagonal mask for windowed attention)

    Output:
      - features: [1, T, 1024]  (after transformer layers + projection)
    """

    def __init__(self, audio_encoder: nn.Module) -> None:
        super().__init__()
        self.layers = audio_encoder.layers
        self.ln_post = audio_encoder.ln_post
        self.proj1 = audio_encoder.proj1
        self.proj2 = audio_encoder.proj2

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states: [1, T, 896]
        # Flatten to [T, 896] for the encoder layers (they expect unbatched input)
        seq_len = hidden_states.shape[1]
        hs = hidden_states.squeeze(0)  # [T, 896]

        for layer in self.layers:
            # The original uses cu_seqlens; we pass an explicit mask instead.
            # We need to adapt the layer forward to use attention_mask.
            # Qwen3ASRAudioEncoderLayer.forward expects (hidden_states, cu_seqlens, attention_mask)
            # We pass a dummy cu_seqlens and the real mask
            layer_outputs = layer(
                hs,
                cu_seqlens=torch.tensor([0, seq_len], dtype=torch.int32, device=hs.device),
                attention_mask=attention_mask,
            )
            hs = layer_outputs[0]

        hs = self.ln_post(hs)
        hs = self.proj1(hs)
        hs = F.gelu(hs)
        hs = self.proj2(hs)  # [T, 1024]
        return hs.unsqueeze(0)  # [1, T, 1024]


class AudioEncoderFullWrapper(nn.Module):
    """Full audio encoder forward pass for CoreML, with fixed-size input.

    Replaces the data-dependent chunking in the original forward method.
    Preprocessing (mel spectrogram, chunking into windows) must be done
    in Swift before calling this model.

    This wrapper processes a single mel spectrogram (no chunking) through:
    1. Conv2D downsampling (3 layers, stride 2 each → 8x reduction)
    2. Linear projection (8160 → 896)
    3. Sinusoidal positional embedding
    4. 18 transformer encoder layers (with explicit attention mask)
    5. LayerNorm → proj1(896→896) → GELU → proj2(896→1024)

    Input:
      - mel_input: [1, 128, T]  mel spectrogram (128 bins, T frames)

    Output:
      - features: [1, T', 1024]  where T' = output frames after conv downsampling
    """

    def __init__(self, audio_encoder: nn.Module) -> None:
        super().__init__()
        self.conv2d1 = audio_encoder.conv2d1
        self.conv2d2 = audio_encoder.conv2d2
        self.conv2d3 = audio_encoder.conv2d3
        self.conv_out = audio_encoder.conv_out
        self.positional_embedding = audio_encoder.positional_embedding
        self.layers = audio_encoder.layers
        self.ln_post = audio_encoder.ln_post
        self.proj1 = audio_encoder.proj1
        self.proj2 = audio_encoder.proj2

    def forward(self, mel_input: torch.Tensor) -> torch.Tensor:
        # mel_input: [1, 128, T]
        x = mel_input.unsqueeze(1)  # [1, 1, 128, T]

        # Conv downsampling
        x = F.gelu(self.conv2d1(x))
        x = F.gelu(self.conv2d2(x))
        x = F.gelu(self.conv2d3(x))
        # x: [1, 480, F', T'] where F'=17, T'=ceil(T/8)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        x = self.conv_out(x)  # [1, T', 896]

        # Positional embedding
        pos_emb = self.positional_embedding(t)  # [T', 896]
        x = x + pos_emb.unsqueeze(0).to(x.dtype)

        # Transformer layers — process as flat sequence with identity mask
        # For a single contiguous chunk, attention is full (no windowing needed)
        hs = x.squeeze(0)  # [T', 896]
        seq_len = hs.shape[0]
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=hs.device)

        for layer in self.layers:
            layer_outputs = layer(hs, cu_seqlens=cu_seqlens)
            hs = layer_outputs[0]

        hs = self.ln_post(hs)
        hs = self.proj1(hs)
        hs = F.gelu(hs)
        hs = self.proj2(hs)  # [T', 1024]
        return hs.unsqueeze(0)  # [1, T', 1024]


# ---------------------------------------------------------------------------
# LLM Decoder Wrappers
# ---------------------------------------------------------------------------

class TextEmbeddingWrapper(nn.Module):
    """Token embedding layer.

    Input:
      - input_ids: [1, seq_len] int32

    Output:
      - embeddings: [1, seq_len, 1024]
    """

    def __init__(self, text_model: nn.Module) -> None:
        super().__init__()
        self.embed_tokens = text_model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids.long())


class LMHeadWrapper(nn.Module):
    """Language model head: hidden states → logits.

    Input:
      - hidden_states: [1, seq_len, 1024]

    Output:
      - logits: [1, seq_len, 151936]
    """

    def __init__(self, lm_head: nn.Module, norm: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hs = self.norm(hidden_states)
        return self.lm_head(hs)


class DecoderLayerWrapper(nn.Module):
    """Single transformer decoder layer for CoreML export.

    Wraps one Qwen3ASRThinkerTextDecoderLayer with explicit KV-cache I/O.

    Input:
      - hidden_states: [1, 1, 1024]  (current token embedding)
      - k_cache: [1, 8, seq_len, 128]  (key cache for this layer)
      - v_cache: [1, 8, seq_len, 128]  (value cache for this layer)
      - position_cos: [1, 1, 128]  (RoPE cos for current position)
      - position_sin: [1, 1, 128]  (RoPE sin for current position)
      - attention_mask: [1, 1, 1, seq_len+1]  (causal mask)

    Output:
      - hidden_states: [1, 1, 1024]
      - k_cache_out: [1, 8, seq_len+1, 128]
      - v_cache_out: [1, 8, seq_len+1, 128]
    """

    def __init__(self, layer: nn.Module, layer_idx: int) -> None:
        super().__init__()
        self.layer = layer
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = self.layer.self_attn

        residual = hidden_states
        hs = self.layer.input_layernorm(hidden_states)

        # Attention with explicit KV cache
        input_shape = hs.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q = attn.q_norm(attn.q_proj(hs).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hs).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(hs).view(hidden_shape).transpose(1, 2)

        # Apply RoPE
        cos = position_cos.unsqueeze(1)  # [1, 1, 1, 128]
        sin = position_sin.unsqueeze(1)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)

        # Append to KV cache
        k_out = torch.cat([k_cache, k], dim=2)
        v_out = torch.cat([v_cache, v], dim=2)

        # Expand KV for GQA (8 KV heads → 16 query heads)
        # Use repeat_interleave instead of expand+reshape to avoid a CoreML
        # conversion bug where cache lengths near HEAD_DIM (128) cause
        # catastrophic numerical errors in the MIL graph.
        num_groups = attn.num_key_value_groups
        k_expanded = k_out.repeat_interleave(num_groups, dim=1)
        v_expanded = v_out.repeat_interleave(num_groups, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * attn.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_expanded)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        attn_output = attn.o_proj(attn_output)

        hs = residual + attn_output

        # MLP
        residual = hs
        hs = self.layer.post_attention_layernorm(hs)
        hs = self.layer.mlp(hs)
        hs = residual + hs

        return hs, k_out, v_out

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class DecoderStackWrapper(nn.Module):
    """All 28 decoder layers in a single module with stacked KV caches.

    Replaces 28 separate DecoderLayerWrapper models with one consolidated model.
    The loop is unrolled by torch.jit.trace into a single graph.

    Used for single-token autoregressive decode (seq_len=1, RangeDim on cache).

    Input:
      - hidden_states: [1, 1, 1024]  (current token embedding)
      - k_caches: [28, 8, cache_len, 128]  (all K caches stacked by layer)
      - v_caches: [28, 8, cache_len, 128]  (all V caches stacked by layer)
      - position_cos: [1, 1, 128]
      - position_sin: [1, 1, 128]
      - attention_mask: [1, 1, 1, cache_len+1]  (causal mask)

    Output:
      - output_hidden: [1, 1, 1024]
      - k_caches_out: [28, 8, cache_len+1, 128]
      - v_caches_out: [28, 8, cache_len+1, 128]
    """

    def __init__(self, text_model: nn.Module) -> None:
        super().__init__()
        self.layers = text_model.layers
        self.num_layers = len(self.layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cos = position_cos.unsqueeze(1)  # [1, 1, 1, 128]
        sin = position_sin.unsqueeze(1)

        k_outs: list[torch.Tensor] = []
        v_outs: list[torch.Tensor] = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            k_cache = k_caches[i]  # [8, cache_len, 128]
            v_cache = v_caches[i]  # [8, cache_len, 128]

            # Add batch dim back: [1, 8, cache_len, 128]
            k_cache = k_cache.unsqueeze(0)
            v_cache = v_cache.unsqueeze(0)

            hidden_states, k_out, v_out = self._layer_forward(
                layer, hidden_states, k_cache, v_cache, cos, sin, attention_mask
            )
            # Strip batch dim for stacking: [8, cache_len+1, 128]
            k_outs.append(k_out.squeeze(0))
            v_outs.append(v_out.squeeze(0))

        k_caches_out = torch.stack(k_outs, dim=0)  # [28, 8, cache_len+1, 128]
        v_caches_out = torch.stack(v_outs, dim=0)  # [28, 8, cache_len+1, 128]
        return hidden_states, k_caches_out, v_caches_out

    @staticmethod
    def _layer_forward(
        layer: nn.Module,
        hidden_states: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = layer.self_attn

        residual = hidden_states
        hs = layer.input_layernorm(hidden_states)

        input_shape = hs.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        q = attn.q_norm(attn.q_proj(hs).view(hidden_shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hs).view(hidden_shape)).transpose(1, 2)
        v = attn.v_proj(hs).view(hidden_shape).transpose(1, 2)

        # RoPE
        q = (q * cos) + (DecoderStackWrapper._rotate_half(q) * sin)
        k = (k * cos) + (DecoderStackWrapper._rotate_half(k) * sin)

        # Append to KV cache
        k_out = torch.cat([k_cache, k], dim=2)
        v_out = torch.cat([v_cache, v], dim=2)

        # Expand KV for GQA (8 KV heads → 16 query heads)
        # Use repeat_interleave instead of expand+reshape to avoid a CoreML
        # conversion bug where cache lengths near HEAD_DIM (128) cause
        # catastrophic numerical errors in the MIL graph.
        num_groups = attn.num_key_value_groups  # 2
        k_expanded = k_out.repeat_interleave(num_groups, dim=1)
        v_expanded = v_out.repeat_interleave(num_groups, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k_expanded.transpose(2, 3)) * attn.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_expanded)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1)
        attn_output = attn.o_proj(attn_output)

        hs = residual + attn_output

        # MLP
        residual = hs
        hs = layer.post_attention_layernorm(hs)
        hs = layer.mlp(hs)
        hs = residual + hs

        return hs, k_out, v_out

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class DecoderPrefillWrapper(nn.Module):
    """Decoder stack for batched prefill with ALL FIXED shapes.

    Processes the full prompt in a single call with fixed seq_len=PREFILL_SEQ_LEN
    and cache_len=1 (just the dummy entry). The causal mask is baked in as a
    constant buffer to avoid dynamic shape issues with CoreML's GPU planner.

    Prompts shorter than PREFILL_SEQ_LEN are padded with zeros by the caller.
    After inference, the caller extracts the real tokens' hidden states and
    trims the KV cache to discard padded entries.

    Input:
      - hidden_states: [1, PREFILL_SEQ_LEN, 1024]
      - k_caches: [28, 8, 1, 128]  (dummy entry only)
      - v_caches: [28, 8, 1, 128]
      - position_cos: [1, PREFILL_SEQ_LEN, 128]
      - position_sin: [1, PREFILL_SEQ_LEN, 128]

    Output:
      - output_hidden: [1, PREFILL_SEQ_LEN, 1024]
      - k_caches_out: [28, 8, 1+PREFILL_SEQ_LEN, 128]
      - v_caches_out: [28, 8, 1+PREFILL_SEQ_LEN, 128]
    """

    PREFILL_SEQ_LEN = 512

    def __init__(self, text_model: nn.Module) -> None:
        super().__init__()
        self.layers = text_model.layers
        self.num_layers = len(self.layers)

        # Pre-compute causal mask: [1, 1, N, 1+N] where N = PREFILL_SEQ_LEN
        # Cache has 1 dummy entry at position 0 (always masked).
        # Query position i attends to key positions 1..i+1 (past + self).
        N = self.PREFILL_SEQ_LEN
        total_keys = 1 + N  # dummy + N new tokens
        mask = torch.full((N, total_keys), -1e9, dtype=torch.float32)
        for i in range(N):
            mask[i, 1:i + 2] = 0.0  # attend to keys 1..i+1, skip dummy at 0
        self.register_buffer("_causal_mask", mask.view(1, 1, N, total_keys))

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_caches: torch.Tensor,
        v_caches: torch.Tensor,
        position_cos: torch.Tensor,
        position_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cos = position_cos.unsqueeze(1)  # [1, 1, N, 128]
        sin = position_sin.unsqueeze(1)
        attention_mask = self._causal_mask

        k_outs: list[torch.Tensor] = []
        v_outs: list[torch.Tensor] = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            k_cache = k_caches[i].unsqueeze(0)  # [1, 8, 1, 128]
            v_cache = v_caches[i].unsqueeze(0)

            hidden_states, k_out, v_out = DecoderStackWrapper._layer_forward(
                layer, hidden_states, k_cache, v_cache, cos, sin, attention_mask
            )
            k_outs.append(k_out.squeeze(0))
            v_outs.append(v_out.squeeze(0))

        k_caches_out = torch.stack(k_outs, dim=0)
        v_caches_out = torch.stack(v_outs, dim=0)
        return hidden_states, k_caches_out, v_caches_out


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------

def _coreml_convert(
    traced: torch.jit.ScriptModule,
    inputs,
    outputs,
    settings: ExportSettings,
    compute_units_override: Optional[ct.ComputeUnit] = None,
    compute_precision_override: Optional[ct.precision] = None,
    no_optimize: bool = False,
) -> ct.models.MLModel:
    cu = compute_units_override if compute_units_override is not None else settings.compute_units
    cp = compute_precision_override if compute_precision_override is not None else settings.compute_precision
    kwargs = {
        "convert_to": "mlprogram",
        "inputs": inputs,
        "outputs": outputs,
        "compute_units": cu,
    }
    print(f"Converting with compute_units={cu}, compute_precision={cp}, no_optimize={no_optimize}")
    if settings.deployment_target is not None:
        kwargs["minimum_deployment_target"] = settings.deployment_target
    if cp is not None:
        kwargs["compute_precision"] = cp
    if no_optimize:
        # Minimal pipeline: only essential passes for a valid model
        # Skip all fusion/optimization passes that can cause numerical issues
        minimal_passes = [
            "common::sanitize_input_output_names",
            "common::dedup_op_and_var_names",
            "common::dead_code_elimination",
            "common::const_elimination",
            "common::noop_elimination",
            "common::update_output_dtypes",
            "common::topological_reorder",
            "common::canonicalize_inplace_pattern",
        ]
        kwargs["pass_pipeline"] = ct.PassPipeline(
            pass_names=minimal_passes, pipeline_name="minimal"
        )
    return ct.convert(traced, **kwargs)
