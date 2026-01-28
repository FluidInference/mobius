"""CoreML-compatible versions of PocketTTS streaming modules.

These modules avoid in-place operations and use explicit state inputs/outputs
instead of dict-based state management.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List
import math


class CoreMLStreamingConv1d(nn.Module):
    """Conv1d with explicit state input/output for CoreML compatibility.

    State layout:
    - previous: [B, C, kernel-stride] float tensor for causal padding
    - first: [B] float tensor (1.0 = first frame, 0.0 = not first)
    """

    def __init__(self, conv: nn.Conv1d, pad_mode: str = "constant"):
        super().__init__()
        self.conv = conv
        self.pad_mode = pad_mode

        # Extract conv parameters
        self.stride = conv.stride[0]
        self.kernel_size = conv.kernel_size[0]
        self.dilation = conv.dilation[0]
        self.in_channels = conv.in_channels

        # Effective kernel size with dilation
        self.effective_kernel = (self.kernel_size - 1) * self.dilation + 1
        self.prev_size = self.effective_kernel - self.stride

    def get_state_size(self, batch_size: int = 1) -> Tuple[List[int], List[int]]:
        """Return shapes for (previous, first) state tensors."""
        return (
            [batch_size, self.in_channels, self.prev_size],
            [batch_size]
        )

    def init_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create initial state tensors."""
        previous = torch.zeros(batch_size, self.in_channels, self.prev_size)
        first = torch.ones(batch_size)  # Use float instead of bool for CoreML
        return previous, first

    def forward(
        self,
        x: torch.Tensor,
        state_previous: torch.Tensor,
        state_first: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, T] input tensor
            state_previous: [B, C, prev_size] previous samples
            state_first: [B] first frame flag (1.0 or 0.0)

        Returns:
            y: [B, C_out, T_out] output tensor
            new_previous: [B, C, prev_size] updated previous samples
            new_first: [B] updated first flag (always 0.0 after first call)
        """
        B, C, T = x.shape

        # Handle replicate padding for first frame
        if self.prev_size > 0 and self.pad_mode == "replicate":
            # Use first sample for padding on first frame
            init_val = x[..., :1].expand(-1, -1, self.prev_size)
            # Blend based on first flag: first=1 uses init_val, first=0 uses state_previous
            first_expanded = state_first.view(B, 1, 1).expand(-1, C, self.prev_size)
            state_previous = torch.where(first_expanded > 0.5, init_val, state_previous)

        # Concatenate previous samples with current input
        if self.prev_size > 0:
            x_padded = torch.cat([state_previous, x], dim=-1)
        else:
            x_padded = x

        # Run convolution
        y = self.conv(x_padded)

        # Update state - take last prev_size samples as new previous
        if self.prev_size > 0:
            new_previous = x_padded[..., -self.prev_size:].clone()
        else:
            new_previous = state_previous  # No change

        # First flag becomes 0 after first call
        new_first = torch.zeros_like(state_first)

        return y, new_previous, new_first


class CoreMLStreamingConvTranspose1d(nn.Module):
    """ConvTranspose1d with explicit state input/output for CoreML compatibility.

    State layout:
    - partial: [B, C_out, kernel-stride] float tensor for overlap-add
    """

    def __init__(self, convtr: nn.ConvTranspose1d):
        super().__init__()
        self.convtr = convtr

        self.stride = convtr.stride[0]
        self.kernel_size = convtr.kernel_size[0]
        self.out_channels = convtr.out_channels
        self.partial_size = self.kernel_size - self.stride

    def get_state_size(self, batch_size: int = 1) -> List[int]:
        """Return shape for partial state tensor."""
        return [batch_size, self.out_channels, self.partial_size]

    def init_state(self, batch_size: int = 1) -> torch.Tensor:
        """Create initial state tensor."""
        return torch.zeros(batch_size, self.out_channels, self.partial_size)

    def forward(
        self,
        x: torch.Tensor,
        state_partial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C_in, T] input tensor
            state_partial: [B, C_out, partial_size] partial output from previous call

        Returns:
            y: [B, C_out, T_out] output tensor (without overlap region)
            new_partial: [B, C_out, partial_size] new partial for next call
        """
        # Run transposed convolution
        y = self.convtr(x)

        PT = self.partial_size
        if PT > 0:
            # Overlap-add: add previous partial to the start of output
            # (must avoid in-place operations for CoreML)
            y_with_overlap = torch.cat([
                y[..., :PT] + state_partial,
                y[..., PT:]
            ], dim=-1)

            # Extract new partial from the end (BEFORE trimming)
            # This includes the overlapped region
            new_partial = y_with_overlap[..., -PT:].clone()

            # Remove bias from partial (will be re-added in next convtr call)
            if self.convtr.bias is not None:
                new_partial = new_partial - self.convtr.bias.view(1, -1, 1)

            # Output is everything except the last partial_size
            y = y_with_overlap[..., :-PT]
        else:
            new_partial = state_partial

        return y, new_partial


class CoreMLStreamingAttention(nn.Module):
    """CoreML-compatible streaming multi-head attention with explicit KV cache.

    State:
    - cache: [2, B, H, capacity, D] float tensor (keys and values)
    - offset: [B] int64 tensor (query position offset)
    - end_offset: [B] int64 tensor (cache write position)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        capacity: int,
        context: int,
        rope_max_period: float = 10000.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.capacity = capacity
        self.context = context

        # Projections
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # RoPE parameters
        self.rope_max_period = rope_max_period

    def init_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create initial state tensors."""
        cache = torch.zeros(2, batch_size, self.num_heads, self.capacity, self.head_dim)
        offset = torch.zeros(batch_size, dtype=torch.long)
        end_offset = torch.zeros(batch_size, dtype=torch.long)
        return cache, offset, end_offset

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings."""
        B, T, H, D = q.shape

        # Generate position indices
        positions = offset.view(-1, 1) + torch.arange(T, device=q.device, dtype=torch.long)  # [B, T]

        # Compute RoPE frequencies
        half_d = D // 2
        freqs = torch.arange(half_d, device=q.device, dtype=torch.float32)
        freqs = self.rope_max_period ** (-freqs / half_d)  # [D/2]

        # Compute angles
        angles = positions.float().unsqueeze(-1) * freqs  # [B, T, D/2]

        cos = torch.cos(angles).unsqueeze(2)  # [B, T, 1, D/2]
        sin = torch.sin(angles).unsqueeze(2)  # [B, T, 1, D/2]

        # Apply rotation
        def rotate(x, cos, sin):
            x1, x2 = x[..., :half_d], x[..., half_d:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

        q_rot = rotate(q, cos, sin)
        k_rot = rotate(k, cos, sin)

        return q_rot, k_rot

    def forward(
        self,
        x: torch.Tensor,  # [B, T, embed_dim]
        cache: torch.Tensor,  # [2, B, H, capacity, D]
        offset: torch.Tensor,  # [B]
        end_offset: torch.Tensor  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, T, embed_dim]
            cache: KV cache [2, B, H, capacity, D]
            offset: Query position offset [B]
            end_offset: Cache write position [B]

        Returns:
            output: Attention output [B, T, embed_dim]
            new_cache: Updated cache
            new_offset: Updated offset (offset + T)
            new_end_offset: Updated end_offset
        """
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim
        capacity = self.capacity

        # Project to Q, K, V
        qkv = self.in_proj(x)  # [B, T, 3*embed_dim]
        qkv = qkv.reshape(B, T, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # Each [B, T, H, D]

        # Apply RoPE
        q, k = self._apply_rope(q, k, offset)

        # Transpose for attention: [B, H, T, D]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Update KV cache (non-inplace)
        write_indices = torch.arange(T, device=end_offset.device, dtype=torch.long)
        write_indices = (write_indices.unsqueeze(0) + end_offset.unsqueeze(1)) % capacity  # [B, T]

        scatter_indices = write_indices.view(B, 1, T, 1).expand(B, H, T, D)

        new_cache = cache.clone()
        new_cache[0] = new_cache[0].scatter(2, scatter_indices, k)
        new_cache[1] = new_cache[1].scatter(2, scatter_indices, v)

        # Update offsets
        new_end_offset = end_offset + T
        new_offset = offset + T

        # Get full K, V from cache
        keys = new_cache[0]  # [B, H, capacity, D]
        values = new_cache[1]

        # Compute position indices for attention masking
        all_indices = torch.arange(capacity, device=end_offset.device, dtype=torch.long)
        last_write_pos = new_end_offset.unsqueeze(1) - 1
        last_write_idx = last_write_pos % capacity

        delta = all_indices.unsqueeze(0) - last_write_idx
        pos_k = torch.where(delta <= 0, last_write_pos + delta, last_write_pos + delta - capacity)

        # Mark invalid positions
        invalid = all_indices.unsqueeze(0) >= new_end_offset.unsqueeze(1)
        pos_k = torch.where(invalid, torch.full_like(pos_k, -10000), pos_k)  # Use large negative for masking

        # Compute attention mask
        pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device, dtype=torch.long).view(1, -1, 1)
        pos_k_expanded = pos_k.unsqueeze(1)  # [B, 1, capacity]

        delta_qk = pos_q - pos_k_expanded  # [B, T, capacity]

        # Valid positions: pos_k >= 0, delta >= 0, delta < context
        attn_mask = (pos_k_expanded >= 0) & (delta_qk >= 0) & (delta_qk < self.context)
        attn_mask = attn_mask.unsqueeze(1)  # [B, 1, T, capacity]

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, keys, values, attn_mask=attn_mask, dropout_p=0.0
        )

        # Reshape and project output
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        output = self.out_proj(attn_output)

        return output, new_cache, new_offset, new_end_offset


class CoreMLTransformerLayer(nn.Module):
    """CoreML-compatible transformer layer with explicit state."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        capacity: int,
        context: int,
        rope_max_period: float = 10000.0,
        layer_scale: float = None
    ):
        super().__init__()
        self.self_attn = CoreMLStreamingAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            capacity=capacity,
            context=context,
            rope_max_period=rope_max_period
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        # Layer scale (optional)
        self.layer_scale = layer_scale
        if layer_scale is not None:
            self.gamma_1 = nn.Parameter(torch.full((d_model,), layer_scale))
            self.gamma_2 = nn.Parameter(torch.full((d_model,), layer_scale))

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        offset: torch.Tensor,
        end_offset: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Self-attention block
        residual = x
        x_norm = self.norm1(x)
        attn_out, new_cache, new_offset, new_end_offset = self.self_attn(
            x_norm, cache, offset, end_offset
        )
        if self.layer_scale is not None:
            attn_out = attn_out * self.gamma_1
        x = residual + attn_out

        # FFN block
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.linear2(F.gelu(self.linear1(x_norm)))
        if self.layer_scale is not None:
            ffn_out = ffn_out * self.gamma_2
        x = residual + ffn_out

        return x, new_cache, new_offset, new_end_offset


class CoreMLSEANetDecoder(nn.Module):
    """CoreML-compatible SEANet decoder with explicit state.

    State tensors are passed as a flat list for each layer type:
    - conv_states: List of (previous, first) tuples for each StreamingConv1d
    - convtr_states: List of partial tensors for each StreamingConvTranspose1d
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: List[int] = [8, 5, 4, 2],
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        pad_mode: str = "reflect",
        compress: int = 2,
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.ratios = ratios

        # Build layer structure - mirrors SEANetDecoder
        mult = int(2 ** len(ratios))

        # Track which modules are streaming
        self.conv_layers = nn.ModuleList()
        self.convtr_layers = nn.ModuleList()
        self.other_layers = nn.ModuleList()
        self.layer_order = []  # Track order: ('conv', idx), ('convtr', idx), ('other', idx)

        # First conv
        conv = nn.Conv1d(dimension, mult * n_filters, kernel_size, bias=True)
        self.conv_layers.append(CoreMLStreamingConv1d(conv, pad_mode))
        self.layer_order.append(('conv', len(self.conv_layers) - 1))

        for ratio in ratios:
            # ELU before upsampling
            self.other_layers.append(nn.ELU(alpha=1.0))
            self.layer_order.append(('other', len(self.other_layers) - 1))

            # Upsampling (transposed conv)
            convtr = nn.ConvTranspose1d(
                mult * n_filters, mult * n_filters // 2,
                kernel_size=ratio * 2, stride=ratio
            )
            self.convtr_layers.append(CoreMLStreamingConvTranspose1d(convtr))
            self.layer_order.append(('convtr', len(self.convtr_layers) - 1))

            # Residual blocks
            for j in range(n_residual_layers):
                dilation = dilation_base ** j
                hidden = (mult * n_filters // 2) // compress

                # ResBlock: ELU -> Conv (dilated) -> ELU -> Conv (1x1)
                self.other_layers.append(nn.ELU(alpha=1.0))
                self.layer_order.append(('other', len(self.other_layers) - 1))

                conv1 = nn.Conv1d(mult * n_filters // 2, hidden, residual_kernel_size, dilation=dilation)
                self.conv_layers.append(CoreMLStreamingConv1d(conv1, pad_mode))
                self.layer_order.append(('conv', len(self.conv_layers) - 1))

                self.other_layers.append(nn.ELU(alpha=1.0))
                self.layer_order.append(('other', len(self.other_layers) - 1))

                conv2 = nn.Conv1d(hidden, mult * n_filters // 2, 1)
                self.conv_layers.append(CoreMLStreamingConv1d(conv2, pad_mode))
                self.layer_order.append(('conv', len(self.conv_layers) - 1))

                # Mark residual connection point
                self.layer_order.append(('residual_end', None))

            mult //= 2

        # Final layers
        self.other_layers.append(nn.ELU(alpha=1.0))
        self.layer_order.append(('other', len(self.other_layers) - 1))

        conv_final = nn.Conv1d(n_filters, channels, last_kernel_size)
        self.conv_layers.append(CoreMLStreamingConv1d(conv_final, pad_mode))
        self.layer_order.append(('conv', len(self.conv_layers) - 1))

    def get_state_shapes(self, batch_size: int = 1):
        """Get shapes for all state tensors."""
        conv_shapes = []
        for conv in self.conv_layers:
            prev_shape, first_shape = conv.get_state_size(batch_size)
            conv_shapes.append((prev_shape, first_shape))

        convtr_shapes = []
        for convtr in self.convtr_layers:
            convtr_shapes.append(convtr.get_state_size(batch_size))

        return conv_shapes, convtr_shapes


class CoreMLMimiDecoder(nn.Module):
    """CoreML-compatible Mimi decoder with flattened state.

    This module rebuilds the Mimi decoder with explicit state I/O.
    All state tensors are passed as separate inputs and outputs.
    """

    def __init__(
        self,
        dimension: int = 512,
        n_filters: int = 32,
        ratios: List[int] = [8, 5, 4, 2],
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        capacity: int = 256,
        context: int = 256,
        upsample_stride: int = 8,
        layer_scale: float = 0.01,
    ):
        super().__init__()
        self.dimension = dimension

        # Upsample (ConvTrUpsample1d)
        convtr = nn.ConvTranspose1d(
            dimension, dimension,
            kernel_size=2 * upsample_stride,
            stride=upsample_stride,
            groups=dimension,
            bias=False
        )
        self.upsample = CoreMLStreamingConvTranspose1d(convtr)

        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_transformer_layers):
            self.transformer_layers.append(CoreMLTransformerLayer(
                d_model=dimension,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                capacity=capacity,
                context=context,
                layer_scale=layer_scale
            ))

        # Input/output projections for transformer (if needed)
        # In PocketTTS, dimension matches so these are identity

        # SEANet decoder
        # Note: For simplicity, we use the original SEANetDecoder structure
        # but replace streaming layers with CoreML-compatible versions
        self.seanet = CoreMLSEANetDecoder(
            channels=1,
            dimension=dimension,
            n_filters=n_filters,
            ratios=ratios
        )


# Helper functions for state packing/unpacking

def pack_conv_states(conv_states: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack multiple conv states into flat tensors.

    Args:
        conv_states: List of (previous, first) tuples

    Returns:
        flat_previous: Concatenated previous buffers
        flat_first: Concatenated first flags
    """
    all_previous = [s[0].flatten() for s in conv_states]
    all_first = [s[1] for s in conv_states]

    return torch.cat(all_previous), torch.cat(all_first)


def unpack_conv_states(
    flat_previous: torch.Tensor,
    flat_first: torch.Tensor,
    shapes: List[Tuple[List[int], List[int]]]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Unpack flat tensors into conv states.

    Args:
        flat_previous: Concatenated previous buffers
        flat_first: Concatenated first flags
        shapes: List of (previous_shape, first_shape) for each conv

    Returns:
        List of (previous, first) tuples
    """
    states = []
    prev_offset = 0
    first_offset = 0

    for prev_shape, first_shape in shapes:
        prev_numel = 1
        for s in prev_shape:
            prev_numel *= s
        first_numel = first_shape[0]

        previous = flat_previous[prev_offset:prev_offset + prev_numel].reshape(prev_shape)
        first = flat_first[first_offset:first_offset + first_numel]

        states.append((previous, first))
        prev_offset += prev_numel
        first_offset += first_numel

    return states
