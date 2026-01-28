"""Traceable FlowLM backbone for CoreML conversion.

This creates a traceable forward function for the FlowLM transformer backbone
with explicit KV cache state.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TraceableFlowLMBackbone(nn.Module):
    """Traceable FlowLM backbone with explicit KV cache state.

    The FlowLM transformer has 6 layers with:
    - embed_dim=1024, num_heads=16, head_dim=64
    - KV cache: [2, B, max_seq_len, H, D] per layer

    State tensors (per layer):
    - cache_i: [2, B, max_seq_len, 16, 64] KV cache
    - position_i: [B] current position (as float for CoreML)
    """

    def __init__(self, max_seq_len: int = 1000):
        super().__init__()

        self.num_layers = 6
        self.embed_dim = 1024
        self.num_heads = 16
        self.head_dim = 64
        self.max_seq_len = max_seq_len
        self.rope_max_period = 10000.0

        # Input projection (ldim=32 -> dim=1024)
        self.input_linear = nn.Linear(32, 1024, bias=False)

        # Transformer layers
        for i in range(self.num_layers):
            # Attention
            setattr(self, f'attn{i}_in_proj', nn.Linear(1024, 3 * 1024, bias=False))
            setattr(self, f'attn{i}_out_proj', nn.Linear(1024, 1024, bias=False))

            # Norms
            setattr(self, f'norm{i}_1', nn.LayerNorm(1024, eps=1e-5))
            setattr(self, f'norm{i}_2', nn.LayerNorm(1024, eps=1e-5))

            # FFN
            hidden_dim = 4096  # From actual model
            setattr(self, f'linear{i}_1', nn.Linear(1024, hidden_dim, bias=False))
            setattr(self, f'linear{i}_2', nn.Linear(hidden_dim, 1024, bias=False))

        # Output norm
        self.out_norm = nn.LayerNorm(1024, eps=1e-5)

        # EOS prediction
        self.out_eos = nn.Linear(1024, 1)

    @classmethod
    def from_flowlm(cls, flow_lm_model, max_seq_len: int = 1000) -> "TraceableFlowLMBackbone":
        """Create traceable backbone from original FlowLM model."""
        wrapper = cls(max_seq_len)

        # Copy input linear
        wrapper.input_linear.weight.data.copy_(flow_lm_model.input_linear.weight.data)

        # Copy transformer layers
        for i, layer in enumerate(flow_lm_model.transformer.layers):
            # Attention
            getattr(wrapper, f'attn{i}_in_proj').weight.data.copy_(layer.self_attn.in_proj.weight.data)
            getattr(wrapper, f'attn{i}_out_proj').weight.data.copy_(layer.self_attn.out_proj.weight.data)

            # Norms
            getattr(wrapper, f'norm{i}_1').weight.data.copy_(layer.norm1.weight.data)
            getattr(wrapper, f'norm{i}_1').bias.data.copy_(layer.norm1.bias.data)
            getattr(wrapper, f'norm{i}_2').weight.data.copy_(layer.norm2.weight.data)
            getattr(wrapper, f'norm{i}_2').bias.data.copy_(layer.norm2.bias.data)

            # FFN
            getattr(wrapper, f'linear{i}_1').weight.data.copy_(layer.linear1.weight.data)
            getattr(wrapper, f'linear{i}_2').weight.data.copy_(layer.linear2.weight.data)

        # Output norm
        wrapper.out_norm.weight.data.copy_(flow_lm_model.out_norm.weight.data)
        wrapper.out_norm.bias.data.copy_(flow_lm_model.out_norm.bias.data)

        # EOS
        wrapper.out_eos.weight.data.copy_(flow_lm_model.out_eos.weight.data)
        wrapper.out_eos.bias.data.copy_(flow_lm_model.out_eos.bias.data)

        return wrapper

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings.

        Args:
            q, k: [B, T, H, D]
            offset: position offset (scalar)
        """
        B, T, H, D = q.shape
        half_d = D // 2

        # Generate position indices
        positions = offset + torch.arange(T, device=q.device, dtype=torch.long)

        # Compute RoPE frequencies
        freqs = torch.arange(half_d, device=q.device, dtype=torch.float32)
        freqs = self.rope_max_period ** (-freqs / half_d)

        # Compute angles
        angles = positions.float().unsqueeze(-1) * freqs  # [T, D/2]

        cos = torch.cos(angles).unsqueeze(0).unsqueeze(2)  # [1, T, 1, D/2]
        sin = torch.sin(angles).unsqueeze(0).unsqueeze(2)

        # Apply rotation
        q1, q2 = q[..., :half_d], q[..., half_d:]
        k1, k2 = k[..., :half_d], k[..., half_d:]

        q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

        return q_rot, k_rot

    def _causal_mask(self, q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        # Shape: [q_len, kv_len]
        # Position i in query can attend to positions 0...(kv_len - q_len + i)
        mask = torch.ones(q_len, kv_len, device=device, dtype=torch.float32)
        mask = torch.tril(mask, diagonal=kv_len - q_len)
        mask = torch.log(mask)  # -inf for masked positions
        return mask

    def _streaming_attention_fixed_kv(
        self,
        x: torch.Tensor,  # [B, T, D]
        in_proj: nn.Linear,
        out_proj: nn.Linear,
        cache: torch.Tensor,  # [2, B, max_seq_len, H, head_dim]
        position: torch.Tensor,  # [B] as float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Streaming attention with fixed-size KV cache for CoreML.

        Uses float operations throughout for CoreML compatibility.
        """
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim
        max_len = cache.shape[2]
        max_len_float = float(max_len)

        # Keep position as float for CoreML (avoid .long())
        pos_float = position.float() if position.dtype != torch.float32 else position

        # Project to Q, K, V
        qkv = in_proj(x).reshape(B, T, 3, H, D)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, T, H, D]

        # Apply RoPE (using float position)
        q, k = self._apply_rope_tensor(q, k, pos_float)

        # Update cache using scatter with long indices (required by scatter)
        new_cache = cache.clone()
        # Compute write indices using float, then convert to long for scatter
        write_base_float = pos_float.view(B, 1, 1, 1)
        write_offsets_float = torch.arange(T, device=x.device, dtype=torch.float32).view(1, T, 1, 1)
        write_indices_float = write_base_float + write_offsets_float
        # Floor-based modulo for circular buffer: x % y = x - floor(x/y)*y
        write_indices_float = write_indices_float - torch.floor(write_indices_float / max_len_float) * max_len_float
        write_indices = write_indices_float.long().expand(B, T, H, D)

        new_cache[0] = new_cache[0].scatter(1, write_indices, k)
        new_cache[1] = new_cache[1].scatter(1, write_indices, v)

        # Use full cache with masking
        keys = new_cache[0]  # [B, max_len, H, D]
        values = new_cache[1]

        # Transpose for attention
        q = q.permute(0, 2, 1, 3)  # [B, H, T, D]
        keys = keys.permute(0, 2, 1, 3)  # [B, H, max_len, D]
        values = values.permute(0, 2, 1, 3)

        # Create attention mask using float operations
        # Query positions: pos, pos+1, ..., pos+T-1
        # Key positions: 0, 1, ..., max_len-1
        q_offsets = torch.arange(T, device=x.device, dtype=torch.float32).view(1, T, 1)
        q_positions = pos_float.view(B, 1, 1) + q_offsets  # [B, T, 1]
        k_positions = torch.arange(max_len, device=x.device, dtype=torch.float32).view(1, 1, max_len)  # [1, 1, max_len]

        # Valid keys: position < pos + T (i.e., written to cache)
        valid_len = pos_float.view(B, 1, 1) + float(T)
        valid_mask = k_positions < valid_len  # [B, 1, max_len]

        # Causal mask: key_pos <= query_pos
        causal_mask = k_positions <= q_positions  # [B, T, max_len]

        # Combined mask
        attn_mask = valid_mask & causal_mask  # [B, T, max_len]
        attn_mask = attn_mask.unsqueeze(1)  # [B, 1, T, max_len]

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, keys, values, attn_mask=attn_mask, dropout_p=0.0)

        # Reshape and project
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, self.embed_dim)
        output = out_proj(attn_output)

        # Update position (keep as float)
        new_position = pos_float + float(T)

        return output, new_cache, new_position

    def _apply_rope_tensor(self, q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings with tensor offset (matching original implementation).

        Original uses interleaved pairs: (q[..., 0], q[..., 1]), (q[..., 2], q[..., 3]), etc.
        Each pair is treated as (real, imag) of a complex number.
        """
        import math
        B, T, H, D = q.shape
        Bk, Tk, Hk, Dk = k.shape
        # Use fixed dimensions to avoid dynamic shapes
        D_float = float(self.head_dim)  # 64
        half_d = self.head_dim // 2  # 32

        # Compute RoPE frequencies (matching original formula)
        ds = torch.arange(half_d, device=q.device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(self.rope_max_period) * 2.0 / D_float))

        # Position indices (offset is float for CoreML compatibility)
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

    def forward(
        self,
        sequence: torch.Tensor,  # [B, T, 32] input latents (with BOS as NaN)
        text_embeddings: torch.Tensor,  # [B, T_text, 1024] pre-computed text conditioning
        bos_emb: torch.Tensor,  # [32] BOS embedding
        # KV cache states for each layer
        cache0: torch.Tensor, position0: torch.Tensor,
        cache1: torch.Tensor, position1: torch.Tensor,
        cache2: torch.Tensor, position2: torch.Tensor,
        cache3: torch.Tensor, position3: torch.Tensor,
        cache4: torch.Tensor, position4: torch.Tensor,
        cache5: torch.Tensor, position5: torch.Tensor,
    ):
        """Forward pass through FlowLM backbone.

        Args:
            sequence: [B, T, 32] input latents (NaN values signal BOS)
            text_embeddings: [B, T_text, 1024] text conditioning
            bos_emb: [32] BOS embedding to replace NaN values
            cache0-5: [2, B, max_seq_len, 16, 64] KV caches
            position0-5: [B] current positions

        Returns:
            transformer_out: [B, T, 1024] output
            is_eos: [B, T, 1] EOS predictions
            new_cache0-5: updated caches
            new_position0-5: updated positions
        """
        # Replace NaN values with BOS embedding
        sequence = torch.where(torch.isnan(sequence), bos_emb, sequence)

        # Project input
        input_ = self.input_linear(sequence)  # [B, T, 1024]

        # Concatenate text embeddings (prepended)
        x = torch.cat([text_embeddings, input_], dim=1)  # [B, T_text + T, 1024]

        # Store caches/positions for iteration
        caches = [cache0, cache1, cache2, cache3, cache4, cache5]
        positions = [position0, position1, position2, position3, position4, position5]
        new_caches = []
        new_positions = []

        # Transformer layers
        for i in range(self.num_layers):
            # Self-attention block
            residual = x
            x_norm = getattr(self, f'norm{i}_1')(x)
            attn_out, new_cache, new_pos = self._streaming_attention_fixed_kv(
                x_norm,
                getattr(self, f'attn{i}_in_proj'),
                getattr(self, f'attn{i}_out_proj'),
                caches[i],
                positions[i]
            )
            x = residual + attn_out

            # FFN block
            residual = x
            x_norm = getattr(self, f'norm{i}_2')(x)
            ffn_out = getattr(self, f'linear{i}_2')(F.gelu(getattr(self, f'linear{i}_1')(x_norm)))
            x = residual + ffn_out

            new_caches.append(new_cache)
            new_positions.append(new_pos)

        # Output norm
        x = self.out_norm(x)

        # Remove text prefix from output
        seq_len = sequence.shape[1]
        transformer_out = x[:, -seq_len:]

        # EOS prediction
        is_eos = self.out_eos(transformer_out)

        return (
            transformer_out,
            is_eos,
            new_caches[0], new_positions[0],
            new_caches[1], new_positions[1],
            new_caches[2], new_positions[2],
            new_caches[3], new_positions[3],
            new_caches[4], new_positions[4],
            new_caches[5], new_positions[5],
        )

    def init_state(self, batch_size: int = 1):
        """Initialize all state tensors."""
        state = {}
        for i in range(self.num_layers):
            state[f'cache{i}'] = torch.full(
                (2, batch_size, self.max_seq_len, self.num_heads, self.head_dim),
                float('nan')
            )
            state[f'position{i}'] = torch.zeros(batch_size)
        return state


def test_traceable_flowlm():
    """Test the traceable FlowLM backbone."""
    print("Loading original PocketTTS model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model()
    flow_lm = model.flow_lm

    print("Creating traceable FlowLM backbone...")
    backbone = TraceableFlowLMBackbone.from_flowlm(flow_lm, max_seq_len=100)
    backbone.eval()

    print("Initializing state...")
    state = backbone.init_state(batch_size=1)

    print("\nState tensors:")
    for k, v in state.items():
        print(f"  {k}: {v.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    sequence = torch.randn(1, 1, 32)  # Single latent frame
    text_embeddings = torch.randn(1, 5, 1024)  # 5 text tokens
    bos_emb = flow_lm.bos_emb.data

    with torch.no_grad():
        outputs = backbone(
            sequence,
            text_embeddings,
            bos_emb,
            state['cache0'], state['position0'],
            state['cache1'], state['position1'],
            state['cache2'], state['position2'],
            state['cache3'], state['position3'],
            state['cache4'], state['position4'],
            state['cache5'], state['position5'],
        )

    transformer_out = outputs[0]
    is_eos = outputs[1]
    print(f"Transformer output shape: {transformer_out.shape}")
    print(f"EOS output shape: {is_eos.shape}")
    print("Forward pass successful!")

    return backbone, state


if __name__ == "__main__":
    test_traceable_flowlm()
