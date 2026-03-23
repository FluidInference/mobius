"""Traceable decoder prefill model for CoreML conversion.

Processes all speaker context tokens in a single forward pass instead of
110 sequential decoder_step calls. Outputs populated KV caches that can
be fed directly into decoder_step for autoregressive generation.

Each layer:
1. Causal self-attention over all T_ctx tokens (dense, no cache)
2. Cross-attention to encoder output
3. FFN
4. Write K,V into output cache format matching decoder_step

Usage from convert/:
    python convert/convert_decoder_prefill.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrefillCausalSelfAttention(nn.Module):
    """Dense causal self-attention for prefill (no KV cache needed)."""

    def __init__(self, d_model, n_heads, d_head=None):
        super().__init__()
        self.d_head = d_head or d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_head ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * n_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, T, d_model)
        Returns:
            output: (B, T, d_model)
            k: (B, T, H, D) - keys for cache
            v: (B, T, H, D) - values for cache
        """
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.transpose(1, 2)  # (B, H, T, D)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        attn = torch.matmul(q, k_t.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Causal mask: lower triangular
        causal = torch.tril(torch.ones(T, T, dtype=x.dtype, device=x.device))
        attn = attn + (1.0 - causal) * (-1e9)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v_t)  # (B, H, T, D)
        out = out.transpose(1, 2).reshape(B, T, -1)
        out = self.o_proj(out)

        return out, k, v


class PrefillCrossAttention(nn.Module):
    """Cross-attention to encoder output (identical to step version)."""

    def __init__(self, d_model, n_heads, d_memory, d_head=None):
        super().__init__()
        self.d_head = d_head or d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.kv_proj = nn.Linear(d_memory, 2 * n_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, memory, memory_mask=None):
        B, T_q, _ = x.shape
        T_m = memory.shape[1]
        q = self.q_proj(x).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(memory).view(B, T_m, 2, self.n_heads, self.d_head)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if memory_mask is not None:
            attn_mask = memory_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T_q, -1)
        return self.o_proj(out)


class PrefillFFN(nn.Module):
    """Positionwise feed-forward (identical to step version)."""

    def __init__(self, d_model, d_ffn, kernel_size=1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ffn, kernel_size, padding=0, bias=False)
        self.conv2 = nn.Conv1d(d_ffn, d_model, kernel_size, padding=0, bias=False)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x.transpose(1, 2)


class PrefillDecoderLayer(nn.Module):
    """Single decoder layer for prefill: self-attn + cross-attn + FFN."""

    def __init__(self, d_model, d_ffn, sa_n_heads, xa_n_heads, xa_d_memory, kernel_size=1, xa_d_head=None):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model, bias=False)
        self.self_attn = PrefillCausalSelfAttention(d_model, sa_n_heads)

        self.has_xattn = xa_n_heads is not None
        if self.has_xattn:
            self.norm_xa_query = nn.LayerNorm(d_model, bias=False)
            self.norm_xa_memory = nn.LayerNorm(xa_d_memory, bias=False)
            self.cross_attn = PrefillCrossAttention(d_model, xa_n_heads, xa_d_memory, xa_d_head)

        self.norm_ff = nn.LayerNorm(d_model, bias=False)
        self.ffn = PrefillFFN(d_model, d_ffn, kernel_size)

    def forward(self, x, encoder_output=None, encoder_mask=None):
        """
        Returns:
            x: (B, T, d_model)
            k: (B, T, H, D) - self-attn keys
            v: (B, T, H, D) - self-attn values
        """
        residual = x
        x_norm = self.norm_sa(x)
        sa_out, k, v = self.self_attn(x_norm)
        x = residual + sa_out

        if self.has_xattn and encoder_output is not None:
            residual = x
            q_norm = self.norm_xa_query(x)
            m_norm = self.norm_xa_memory(encoder_output)
            xa_out = self.cross_attn(q_norm, m_norm, encoder_mask)
            x = residual + xa_out

        residual = x
        x = self.norm_ff(x)
        x = self.ffn(x)
        x = residual + x

        return x, k, v


class TraceableDecoderPrefill(nn.Module):
    """Batched prefill decoder for CoreML.

    Processes T_ctx speaker context tokens in one pass and outputs
    KV caches compatible with decoder_step for subsequent generation.

    Inputs:
        audio_embed: (B, T_ctx, d_model) - all context embeddings
        encoder_output: (B, T_enc, d_model) - text encoder output
        encoder_mask: (B, T_enc) - text mask

    Outputs:
        cache{i}: (2, B, max_seq_len, H, D) - KV cache per layer
            First T_ctx positions populated, rest zero-padded.
    """

    def __init__(self, n_layers, d_model, d_ffn, sa_n_heads, xa_n_heads, xa_d_memory,
                 kernel_size=1, xa_d_head=None, max_seq_len=512,
                 use_pos_emb=False, max_pos=2048, t_ctx=110):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.t_ctx = t_ctx
        self.use_pos_emb = use_pos_emb
        self.n_heads = sa_n_heads
        self.d_head = d_model // sa_n_heads

        if use_pos_emb:
            self.position_embeddings = nn.Embedding(max_pos, d_model)

        self.layers = nn.ModuleList([
            PrefillDecoderLayer(
                d_model, d_ffn, sa_n_heads, xa_n_heads, xa_d_memory, kernel_size, xa_d_head
            )
            for _ in range(n_layers)
        ])

    def forward(self, audio_embed, encoder_output, encoder_mask):
        H, D = self.n_heads, self.d_head
        max_seq = self.max_seq_len
        # Use concrete Python int for pad_len so CoreML tracing doesn't
        # produce a symbolic tensor that fails in F.pad / _int cast.
        pad_len = max_seq - self.t_ctx

        x = audio_embed

        if self.use_pos_emb:
            pos_ids = torch.arange(self.t_ctx, device=x.device).unsqueeze(0)
            x = x + self.position_embeddings(pos_ids)

        # Collect KV caches from all layers
        all_caches = []
        for layer in self.layers:
            x, k, v = layer(x, encoder_output, encoder_mask)
            # k, v: (B, T_ctx, H, D) -> pad to (B, max_seq, H, D)
            k_padded = F.pad(k, (0, 0, 0, 0, 0, pad_len))  # (B, max_seq, H, D)
            v_padded = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            # Stack to (2, B, max_seq, H, D)
            cache = torch.stack([k_padded, v_padded], dim=0)
            all_caches.append(cache)

        # Return flat caches (matching decoder_step format)
        return tuple(all_caches)

    @classmethod
    def from_magpie(cls, model, t_ctx=None):
        """Create from a loaded MagpieTTSModel, copying decoder weights."""
        cfg = model.cfg
        dec_cfg = dict(cfg.decoder)

        # Determine T_ctx from baked speaker embeddings (if not overridden)
        if t_ctx is None:
            t_ctx = 110  # default
            if model.has_baked_context_embedding and model._baked_embedding_T is not None:
                t_ctx = int(model._baked_embedding_T.item())

        wrapper = cls(
            n_layers=dec_cfg["n_layers"],
            d_model=dec_cfg["d_model"],
            d_ffn=dec_cfg["d_ffn"],
            sa_n_heads=dec_cfg["sa_n_heads"],
            xa_n_heads=dec_cfg.get("xa_n_heads"),
            xa_d_memory=dec_cfg.get("xa_d_memory"),
            kernel_size=dec_cfg.get("kernel_size", 1),
            xa_d_head=dec_cfg.get("xa_d_head"),
            max_seq_len=512,
            use_pos_emb=dec_cfg.get("use_learnable_pos_emb", False),
            max_pos=dec_cfg.get("max_length_causal_mask", 2048),
            t_ctx=t_ctx,
        )

        # Copy positional embeddings
        if wrapper.use_pos_emb and model.decoder.position_embeddings is not None:
            wrapper.position_embeddings.weight.data.copy_(model.decoder.position_embeddings.weight.data)

        # Copy decoder layers (same weight names as traceable_decoder_step)
        for src_layer, dst_layer in zip(model.decoder.layers, wrapper.layers):
            # Self-attention
            dst_layer.self_attn.qkv_proj.weight.data.copy_(src_layer.self_attention.qkv_net.weight.data)
            dst_layer.self_attn.o_proj.weight.data.copy_(src_layer.self_attention.o_net.weight.data)
            dst_layer.norm_sa.weight.data.copy_(src_layer.norm_self.weight.data)

            # Cross-attention
            if dst_layer.has_xattn and hasattr(src_layer, "cross_attention"):
                dst_layer.cross_attn.q_proj.weight.data.copy_(src_layer.cross_attention.q_net.weight.data)
                dst_layer.cross_attn.kv_proj.weight.data.copy_(src_layer.cross_attention.kv_net.weight.data)
                dst_layer.cross_attn.o_proj.weight.data.copy_(src_layer.cross_attention.o_net.weight.data)
                dst_layer.norm_xa_query.weight.data.copy_(src_layer.norm_xattn_query.weight.data)
                dst_layer.norm_xa_memory.weight.data.copy_(src_layer.norm_xattn_memory.weight.data)

            # FFN
            dst_layer.norm_ff.weight.data.copy_(src_layer.norm_pos_ff.weight.data)
            dst_layer.ffn.conv1.weight.data.copy_(src_layer.pos_ff.proj.conv.weight.data)
            dst_layer.ffn.conv2.weight.data.copy_(src_layer.pos_ff.o_net.conv.weight.data)

        return wrapper
