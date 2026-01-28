"""Traceable Mimi decoder for CoreML conversion.

This module creates a single traceable forward function that handles
all state explicitly as tensor inputs/outputs.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TraceableMimiDecoder(nn.Module):
    """Single traceable module for Mimi decoder with explicit state.

    Takes latent input and all state tensors, returns audio and updated states.

    State tensors (all float32 for CoreML compatibility):
    - upsample_partial: [B, 512, 16]
    - attn0_cache: [2, B, 8, 256, 64]
    - attn0_offset: [B] (as float, cast to int internally)
    - attn0_end_offset: [B] (as float, cast to int internally)
    - attn1_cache: [2, B, 8, 256, 64]
    - attn1_offset: [B]
    - attn1_end_offset: [B]
    - conv0_prev: [B, 512, 6], conv0_first: [B]
    - convtr0_partial: [B, 256, 6]
    - res0_conv0_prev: [B, 256, 2], res0_conv0_first: [B]
    - res0_conv1_prev: [B, 128, 0], res0_conv1_first: [B]
    - convtr1_partial: [B, 128, 5]
    - res1_conv0_prev: [B, 128, 2], res1_conv0_first: [B]
    - res1_conv1_prev: [B, 64, 0], res1_conv1_first: [B]
    - convtr2_partial: [B, 64, 4]
    - res2_conv0_prev: [B, 64, 2], res2_conv0_first: [B]
    - res2_conv1_prev: [B, 32, 0], res2_conv1_first: [B]
    - conv_final_prev: [B, 64, 2], conv_final_first: [B]
    """

    def __init__(self):
        super().__init__()

        # Upsample (groups=512 depthwise convtr)
        self.upsample_convtr = nn.ConvTranspose1d(512, 512, 32, stride=16, groups=512, bias=False)

        # Transformer layers (2 layers)
        # Each layer: norm1, attn, norm2, ffn
        self.num_transformer_layers = 2
        self.embed_dim = 512
        self.num_heads = 8
        self.head_dim = 64
        self.capacity = 256
        self.context = 250
        self.rope_max_period = 10000.0

        # Transformer components
        for i in range(self.num_transformer_layers):
            setattr(self, f'attn{i}_in_proj', nn.Linear(512, 3 * 512, bias=False))
            setattr(self, f'attn{i}_out_proj', nn.Linear(512, 512, bias=False))
            setattr(self, f'norm{i}_1', nn.LayerNorm(512, eps=1e-5))
            setattr(self, f'norm{i}_2', nn.LayerNorm(512, eps=1e-5))
            setattr(self, f'linear{i}_1', nn.Linear(512, 2048, bias=False))
            setattr(self, f'linear{i}_2', nn.Linear(2048, 512, bias=False))
            setattr(self, f'gamma{i}_1', nn.Parameter(torch.full((512,), 0.01)))
            setattr(self, f'gamma{i}_2', nn.Parameter(torch.full((512,), 0.01)))

        # SEANet decoder layers
        # Conv0: 512→512, k=7
        self.conv0 = nn.Conv1d(512, 512, 7, bias=True)
        self.conv0_prev_size = 6  # k-1

        # ConvTr0: 512→256, k=12, s=6
        self.convtr0 = nn.ConvTranspose1d(512, 256, 12, stride=6, bias=True)
        self.convtr0_partial_size = 6  # k-s

        # ResBlock0: 256→128→256
        self.res0_conv0 = nn.Conv1d(256, 128, 3, dilation=1, bias=True)
        self.res0_conv1 = nn.Conv1d(128, 256, 1, bias=True)
        self.res0_conv0_prev_size = 2  # (k-1)*d
        self.res0_conv1_prev_size = 0

        # ConvTr1: 256→128, k=10, s=5
        self.convtr1 = nn.ConvTranspose1d(256, 128, 10, stride=5, bias=True)
        self.convtr1_partial_size = 5

        # ResBlock1: 128→64→128
        self.res1_conv0 = nn.Conv1d(128, 64, 3, dilation=1, bias=True)
        self.res1_conv1 = nn.Conv1d(64, 128, 1, bias=True)
        self.res1_conv0_prev_size = 2
        self.res1_conv1_prev_size = 0

        # ConvTr2: 128→64, k=8, s=4
        self.convtr2 = nn.ConvTranspose1d(128, 64, 8, stride=4, bias=True)
        self.convtr2_partial_size = 4

        # ResBlock2: 64→32→64
        self.res2_conv0 = nn.Conv1d(64, 32, 3, dilation=1, bias=True)
        self.res2_conv1 = nn.Conv1d(32, 64, 1, bias=True)
        self.res2_conv0_prev_size = 2
        self.res2_conv1_prev_size = 0

        # Final conv: 64→1, k=3
        self.conv_final = nn.Conv1d(64, 1, 3, bias=True)
        self.conv_final_prev_size = 2

    @classmethod
    def from_mimi(cls, mimi_model) -> "TraceableMimiDecoder":
        """Create traceable decoder from original Mimi model."""
        wrapper = cls()

        # Copy upsample weights
        wrapper.upsample_convtr.weight.data.copy_(mimi_model.upsample.convtr.convtr.weight.data)

        # Copy transformer weights
        transformer = mimi_model.decoder_transformer.transformer
        for i, layer in enumerate(transformer.layers):
            getattr(wrapper, f'attn{i}_in_proj').weight.data.copy_(layer.self_attn.in_proj.weight.data)
            getattr(wrapper, f'attn{i}_out_proj').weight.data.copy_(layer.self_attn.out_proj.weight.data)
            getattr(wrapper, f'norm{i}_1').weight.data.copy_(layer.norm1.weight.data)
            getattr(wrapper, f'norm{i}_1').bias.data.copy_(layer.norm1.bias.data)
            getattr(wrapper, f'norm{i}_2').weight.data.copy_(layer.norm2.weight.data)
            getattr(wrapper, f'norm{i}_2').bias.data.copy_(layer.norm2.bias.data)
            getattr(wrapper, f'linear{i}_1').weight.data.copy_(layer.linear1.weight.data)
            getattr(wrapper, f'linear{i}_2').weight.data.copy_(layer.linear2.weight.data)
            # Layer scale
            getattr(wrapper, f'gamma{i}_1').data.copy_(layer.layer_scale_1.scale.data)
            getattr(wrapper, f'gamma{i}_2').data.copy_(layer.layer_scale_2.scale.data)

        # Copy SEANet decoder weights
        decoder = mimi_model.decoder
        # Conv0
        wrapper.conv0.weight.data.copy_(decoder.model[0].conv.weight.data)
        wrapper.conv0.bias.data.copy_(decoder.model[0].conv.bias.data)

        # ConvTr0
        wrapper.convtr0.weight.data.copy_(decoder.model[2].convtr.weight.data)
        wrapper.convtr0.bias.data.copy_(decoder.model[2].convtr.bias.data)

        # ResBlock0
        wrapper.res0_conv0.weight.data.copy_(decoder.model[3].block[1].conv.weight.data)
        wrapper.res0_conv0.bias.data.copy_(decoder.model[3].block[1].conv.bias.data)
        wrapper.res0_conv1.weight.data.copy_(decoder.model[3].block[3].conv.weight.data)
        wrapper.res0_conv1.bias.data.copy_(decoder.model[3].block[3].conv.bias.data)

        # ConvTr1
        wrapper.convtr1.weight.data.copy_(decoder.model[5].convtr.weight.data)
        wrapper.convtr1.bias.data.copy_(decoder.model[5].convtr.bias.data)

        # ResBlock1
        wrapper.res1_conv0.weight.data.copy_(decoder.model[6].block[1].conv.weight.data)
        wrapper.res1_conv0.bias.data.copy_(decoder.model[6].block[1].conv.bias.data)
        wrapper.res1_conv1.weight.data.copy_(decoder.model[6].block[3].conv.weight.data)
        wrapper.res1_conv1.bias.data.copy_(decoder.model[6].block[3].conv.bias.data)

        # ConvTr2
        wrapper.convtr2.weight.data.copy_(decoder.model[8].convtr.weight.data)
        wrapper.convtr2.bias.data.copy_(decoder.model[8].convtr.bias.data)

        # ResBlock2
        wrapper.res2_conv0.weight.data.copy_(decoder.model[9].block[1].conv.weight.data)
        wrapper.res2_conv0.bias.data.copy_(decoder.model[9].block[1].conv.bias.data)
        wrapper.res2_conv1.weight.data.copy_(decoder.model[9].block[3].conv.weight.data)
        wrapper.res2_conv1.bias.data.copy_(decoder.model[9].block[3].conv.bias.data)

        # Final conv
        wrapper.conv_final.weight.data.copy_(decoder.model[11].conv.weight.data)
        wrapper.conv_final.bias.data.copy_(decoder.model[11].conv.bias.data)

        return wrapper

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings (matching original implementation).

        Original uses interleaved pairs: (q[..., 0], q[..., 1]), (q[..., 2], q[..., 3]), etc.
        Each pair is treated as (real, imag) of a complex number.
        """
        import math
        B, T, H, D = q.shape
        Bk, Tk, Hk, Dk = k.shape
        # Use fixed dimensions to avoid dynamic shapes in tracing
        D_float = float(self.head_dim)  # 64
        half_d = self.head_dim // 2  # 32

        # Compute RoPE frequencies (matching original formula)
        ds = torch.arange(half_d, device=q.device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(self.rope_max_period) * 2.0 / D_float))

        # Position indices (offset is already float for CoreML compatibility)
        ts = torch.arange(T, device=q.device, dtype=torch.float32)
        offset_f = offset.float() if offset.dtype != torch.float32 else offset
        ts = ts + offset_f.view(B, 1)  # [B, T]
        ts = ts.view(B, T, 1, 1)  # [B, T, 1, 1]

        # View as interleaved pairs [B, T, H, D//2, 2]
        q_complex = q.view(B, T, H, half_d, 2)
        k_complex = k.view(Bk, Tk, Hk, half_d, 2)

        # Extract real and imaginary parts
        qr = q_complex[..., 0].float()
        qi = q_complex[..., 1].float()
        kr = k_complex[..., 0].float()
        ki = k_complex[..., 1].float()

        # Compute rotation angles
        rotr = torch.cos(freqs * ts)  # [B, T, 1, D//2]
        roti = torch.sin(freqs * ts)

        # Apply complex rotation
        qor = qr * rotr - qi * roti
        qoi = qr * roti + qi * rotr
        kor = kr * rotr - ki * roti
        koi = kr * roti + ki * rotr

        # Stack back to original shape
        dtype = q.dtype
        qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
        ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

        return qo.view(B, T, H, D), ko.view(Bk, Tk, Hk, Dk)

    def _streaming_conv(
        self,
        x: torch.Tensor,
        conv: nn.Conv1d,
        prev: torch.Tensor,
        first: torch.Tensor,
        prev_size: int,
        pad_mode: str = "constant"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CoreML-compatible streaming conv1d."""
        B, C, T = x.shape

        if prev_size > 0:
            if pad_mode == "replicate":
                init_val = x[..., :1].expand(-1, -1, prev_size)
                first_expanded = first.view(B, 1, 1).expand(-1, C, prev_size)
                prev = torch.where(first_expanded > 0.5, init_val, prev)

            x_padded = torch.cat([prev, x], dim=-1)
        else:
            x_padded = x

        y = conv(x_padded)

        if prev_size > 0:
            new_prev = x_padded[..., -prev_size:].clone()
        else:
            new_prev = prev

        new_first = torch.zeros_like(first)
        return y, new_prev, new_first

    def _streaming_convtr(
        self,
        x: torch.Tensor,
        convtr: nn.ConvTranspose1d,
        partial: torch.Tensor,
        partial_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CoreML-compatible streaming convtranspose1d (no in-place ops)."""
        y = convtr(x)

        if partial_size > 0:
            # Overlap-add (avoid in-place operations)
            y_start = y[..., :partial_size] + partial
            y_middle = y[..., partial_size:-partial_size]
            y_end = y[..., -partial_size:]

            # New partial (subtract bias for next overlap)
            new_partial = y_end.clone()
            if convtr.bias is not None:
                new_partial = new_partial - convtr.bias.view(1, -1, 1)

            # Output is everything except the last partial_size
            y = torch.cat([y_start, y_middle], dim=-1)
        else:
            new_partial = partial

        return y, new_partial

    def _streaming_attention(
        self,
        x: torch.Tensor,
        in_proj: nn.Linear,
        out_proj: nn.Linear,
        cache: torch.Tensor,
        offset: torch.Tensor,
        end_offset: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """CoreML-compatible streaming attention."""
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim
        capacity = self.capacity
        T_float = float(T)
        capacity_float = float(capacity)

        # Project to Q, K, V
        qkv = in_proj(x).reshape(B, T, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Apply RoPE (offset as float, will be converted inside)
        q, k = self._apply_rope(q, k, offset)

        # Transpose for attention
        q = q.permute(0, 2, 1, 3)  # [B, H, T, D]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Update KV cache (non-inplace) - use float for modulo, then cast to long for scatter
        # Avoid fmod (not supported in CoreML) - use floor-based modulo: x % y = x - floor(x/y)*y
        write_indices_f = torch.arange(T, device=x.device, dtype=torch.float32)
        write_sum = write_indices_f.unsqueeze(0) + end_offset.unsqueeze(1)
        write_indices_f = write_sum - torch.floor(write_sum / capacity_float) * capacity_float
        write_indices = write_indices_f.long()

        scatter_indices = write_indices.view(B, 1, T, 1).expand(B, H, T, D)

        new_cache = cache.clone()
        new_cache[0] = new_cache[0].scatter(2, scatter_indices, k)
        new_cache[1] = new_cache[1].scatter(2, scatter_indices, v)

        # Update offsets (keep as float)
        new_end_offset = end_offset + T_float
        new_offset = offset + T_float

        # Get full K, V from cache
        keys = new_cache[0]
        values = new_cache[1]

        # Compute attention mask (all in float, cast to long only for comparisons)
        all_indices_f = torch.arange(capacity, device=x.device, dtype=torch.float32)
        last_write_pos = new_end_offset.unsqueeze(1) - 1.0
        # Avoid fmod - use floor-based modulo
        last_write_idx = last_write_pos - torch.floor(last_write_pos / capacity_float) * capacity_float

        delta = all_indices_f.unsqueeze(0) - last_write_idx
        pos_k = torch.where(delta <= 0, last_write_pos + delta, last_write_pos + delta - capacity_float)

        invalid = all_indices_f.unsqueeze(0) >= new_end_offset.unsqueeze(1)
        pos_k = torch.where(invalid, torch.full_like(pos_k, -10000.0), pos_k)

        pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=x.device, dtype=torch.float32).view(1, -1, 1)
        pos_k_expanded = pos_k.unsqueeze(1)

        delta_qk = pos_q - pos_k_expanded
        context_float = float(self.context)
        attn_mask = (pos_k_expanded >= 0) & (delta_qk >= 0) & (delta_qk < context_float)
        attn_mask = attn_mask.unsqueeze(1)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, keys, values, attn_mask=attn_mask, dropout_p=0.0)

        # Reshape and project
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        output = out_proj(attn_output)

        return output, new_cache, new_offset, new_end_offset

    def forward(
        self,
        latent: torch.Tensor,  # [B, C, T] or [B, T, C] depending on convention
        # Upsample state
        upsample_partial: torch.Tensor,
        # Transformer states (layer 0)
        attn0_cache: torch.Tensor,
        attn0_offset: torch.Tensor,
        attn0_end_offset: torch.Tensor,
        # Transformer states (layer 1)
        attn1_cache: torch.Tensor,
        attn1_offset: torch.Tensor,
        attn1_end_offset: torch.Tensor,
        # SEANet decoder states
        conv0_prev: torch.Tensor,
        conv0_first: torch.Tensor,
        convtr0_partial: torch.Tensor,
        res0_conv0_prev: torch.Tensor,
        res0_conv0_first: torch.Tensor,
        res0_conv1_prev: torch.Tensor,
        res0_conv1_first: torch.Tensor,
        convtr1_partial: torch.Tensor,
        res1_conv0_prev: torch.Tensor,
        res1_conv0_first: torch.Tensor,
        res1_conv1_prev: torch.Tensor,
        res1_conv1_first: torch.Tensor,
        convtr2_partial: torch.Tensor,
        res2_conv0_prev: torch.Tensor,
        res2_conv0_first: torch.Tensor,
        res2_conv1_prev: torch.Tensor,
        res2_conv1_first: torch.Tensor,
        conv_final_prev: torch.Tensor,
        conv_final_first: torch.Tensor,
    ):
        """Forward pass with explicit state I/O.

        Args:
            latent: [B, C, T] latent representation (C=512)
            All state tensors as described in class docstring

        Returns:
            audio: [B, 1, T_out] output audio
            All updated state tensors
        """
        # ============= UPSAMPLE =============
        x, new_upsample_partial = self._streaming_convtr(
            latent, self.upsample_convtr, upsample_partial, 16
        )

        # ============= TRANSFORMER =============
        # Transpose to [B, T, C] for transformer
        x = x.transpose(1, 2)

        # Layer 0
        residual = x
        x_norm = self.norm0_1(x)
        attn_out, new_attn0_cache, new_attn0_offset, new_attn0_end_offset = self._streaming_attention(
            x_norm, self.attn0_in_proj, self.attn0_out_proj,
            attn0_cache, attn0_offset, attn0_end_offset
        )
        x = residual + attn_out * self.gamma0_1

        residual = x
        x_norm = self.norm0_2(x)
        ffn_out = self.linear0_2(F.gelu(self.linear0_1(x_norm)))
        x = residual + ffn_out * self.gamma0_2

        # Layer 1
        residual = x
        x_norm = self.norm1_1(x)
        attn_out, new_attn1_cache, new_attn1_offset, new_attn1_end_offset = self._streaming_attention(
            x_norm, self.attn1_in_proj, self.attn1_out_proj,
            attn1_cache, attn1_offset, attn1_end_offset
        )
        x = residual + attn_out * self.gamma1_1

        residual = x
        x_norm = self.norm1_2(x)
        ffn_out = self.linear1_2(F.gelu(self.linear1_1(x_norm)))
        x = residual + ffn_out * self.gamma1_2

        # Transpose back to [B, C, T] for SEANet
        x = x.transpose(1, 2)

        # ============= SEANET DECODER =============
        # Conv0
        x, new_conv0_prev, new_conv0_first = self._streaming_conv(
            x, self.conv0, conv0_prev, conv0_first, self.conv0_prev_size, "replicate"
        )

        # ELU + ConvTr0
        x = F.elu(x, alpha=1.0)
        x, new_convtr0_partial = self._streaming_convtr(
            x, self.convtr0, convtr0_partial, self.convtr0_partial_size
        )

        # ResBlock0
        residual = x
        x_res = F.elu(x, alpha=1.0)
        x_res, new_res0_conv0_prev, new_res0_conv0_first = self._streaming_conv(
            x_res, self.res0_conv0, res0_conv0_prev, res0_conv0_first, self.res0_conv0_prev_size, "replicate"
        )
        x_res = F.elu(x_res, alpha=1.0)
        x_res, new_res0_conv1_prev, new_res0_conv1_first = self._streaming_conv(
            x_res, self.res0_conv1, res0_conv1_prev, res0_conv1_first, self.res0_conv1_prev_size, "replicate"
        )
        x = residual + x_res

        # ELU + ConvTr1
        x = F.elu(x, alpha=1.0)
        x, new_convtr1_partial = self._streaming_convtr(
            x, self.convtr1, convtr1_partial, self.convtr1_partial_size
        )

        # ResBlock1
        residual = x
        x_res = F.elu(x, alpha=1.0)
        x_res, new_res1_conv0_prev, new_res1_conv0_first = self._streaming_conv(
            x_res, self.res1_conv0, res1_conv0_prev, res1_conv0_first, self.res1_conv0_prev_size, "replicate"
        )
        x_res = F.elu(x_res, alpha=1.0)
        x_res, new_res1_conv1_prev, new_res1_conv1_first = self._streaming_conv(
            x_res, self.res1_conv1, res1_conv1_prev, res1_conv1_first, self.res1_conv1_prev_size, "replicate"
        )
        x = residual + x_res

        # ELU + ConvTr2
        x = F.elu(x, alpha=1.0)
        x, new_convtr2_partial = self._streaming_convtr(
            x, self.convtr2, convtr2_partial, self.convtr2_partial_size
        )

        # ResBlock2
        residual = x
        x_res = F.elu(x, alpha=1.0)
        x_res, new_res2_conv0_prev, new_res2_conv0_first = self._streaming_conv(
            x_res, self.res2_conv0, res2_conv0_prev, res2_conv0_first, self.res2_conv0_prev_size, "replicate"
        )
        x_res = F.elu(x_res, alpha=1.0)
        x_res, new_res2_conv1_prev, new_res2_conv1_first = self._streaming_conv(
            x_res, self.res2_conv1, res2_conv1_prev, res2_conv1_first, self.res2_conv1_prev_size, "replicate"
        )
        x = residual + x_res

        # Final ELU + Conv
        x = F.elu(x, alpha=1.0)
        audio, new_conv_final_prev, new_conv_final_first = self._streaming_conv(
            x, self.conv_final, conv_final_prev, conv_final_first, self.conv_final_prev_size, "replicate"
        )

        return (
            audio,
            new_upsample_partial,
            new_attn0_cache, new_attn0_offset, new_attn0_end_offset,
            new_attn1_cache, new_attn1_offset, new_attn1_end_offset,
            new_conv0_prev, new_conv0_first,
            new_convtr0_partial,
            new_res0_conv0_prev, new_res0_conv0_first,
            new_res0_conv1_prev, new_res0_conv1_first,
            new_convtr1_partial,
            new_res1_conv0_prev, new_res1_conv0_first,
            new_res1_conv1_prev, new_res1_conv1_first,
            new_convtr2_partial,
            new_res2_conv0_prev, new_res2_conv0_first,
            new_res2_conv1_prev, new_res2_conv1_first,
            new_conv_final_prev, new_conv_final_first,
        )

    def init_state(self, batch_size: int = 1):
        """Initialize all state tensors."""
        state = {}

        # Upsample
        state['upsample_partial'] = torch.zeros(batch_size, 512, 16)

        # Transformer caches
        for i in range(self.num_transformer_layers):
            state[f'attn{i}_cache'] = torch.zeros(2, batch_size, self.num_heads, self.capacity, self.head_dim)
            state[f'attn{i}_offset'] = torch.zeros(batch_size)
            state[f'attn{i}_end_offset'] = torch.zeros(batch_size)

        # SEANet decoder
        state['conv0_prev'] = torch.zeros(batch_size, 512, self.conv0_prev_size)
        state['conv0_first'] = torch.ones(batch_size)

        state['convtr0_partial'] = torch.zeros(batch_size, 256, self.convtr0_partial_size)

        state['res0_conv0_prev'] = torch.zeros(batch_size, 256, self.res0_conv0_prev_size)
        state['res0_conv0_first'] = torch.ones(batch_size)
        state['res0_conv1_prev'] = torch.zeros(batch_size, 128, self.res0_conv1_prev_size)
        state['res0_conv1_first'] = torch.ones(batch_size)

        state['convtr1_partial'] = torch.zeros(batch_size, 128, self.convtr1_partial_size)

        state['res1_conv0_prev'] = torch.zeros(batch_size, 128, self.res1_conv0_prev_size)
        state['res1_conv0_first'] = torch.ones(batch_size)
        state['res1_conv1_prev'] = torch.zeros(batch_size, 64, self.res1_conv1_prev_size)
        state['res1_conv1_first'] = torch.ones(batch_size)

        state['convtr2_partial'] = torch.zeros(batch_size, 64, self.convtr2_partial_size)

        state['res2_conv0_prev'] = torch.zeros(batch_size, 64, self.res2_conv0_prev_size)
        state['res2_conv0_first'] = torch.ones(batch_size)
        state['res2_conv1_prev'] = torch.zeros(batch_size, 32, self.res2_conv1_prev_size)
        state['res2_conv1_first'] = torch.ones(batch_size)

        state['conv_final_prev'] = torch.zeros(batch_size, 64, self.conv_final_prev_size)
        state['conv_final_first'] = torch.ones(batch_size)

        return state


def test_traceable_decoder():
    """Test the traceable decoder."""
    print("Loading original PocketTTS model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    mimi = model.mimi

    print("Creating traceable decoder from original...")
    decoder = TraceableMimiDecoder.from_mimi(mimi)
    decoder.eval()

    print("Initializing state...")
    state = decoder.init_state(batch_size=1)

    print("\nState tensors:")
    for k, v in state.items():
        print(f"  {k}: {v.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    latent = torch.randn(1, 512, 1)  # Single frame

    with torch.no_grad():
        outputs = decoder(
            latent,
            state['upsample_partial'],
            state['attn0_cache'], state['attn0_offset'], state['attn0_end_offset'],
            state['attn1_cache'], state['attn1_offset'], state['attn1_end_offset'],
            state['conv0_prev'], state['conv0_first'],
            state['convtr0_partial'],
            state['res0_conv0_prev'], state['res0_conv0_first'],
            state['res0_conv1_prev'], state['res0_conv1_first'],
            state['convtr1_partial'],
            state['res1_conv0_prev'], state['res1_conv0_first'],
            state['res1_conv1_prev'], state['res1_conv1_first'],
            state['convtr2_partial'],
            state['res2_conv0_prev'], state['res2_conv0_first'],
            state['res2_conv1_prev'], state['res2_conv1_first'],
            state['conv_final_prev'], state['conv_final_first'],
        )

    audio = outputs[0]
    print(f"Output audio shape: {audio.shape}")
    print("Forward pass successful!")

    return decoder, state


if __name__ == "__main__":
    test_traceable_decoder()
