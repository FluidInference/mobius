"""Traceable decoder step wrapper for CoreML conversion.

The decoder is a causal transformer with cross-attention to the encoder output.
For CoreML, we implement it as a single-step model with explicit KV cache I/O,
following the PocketTTS pattern.

Each step:
1. Takes one audio embedding token + encoder output
2. Runs through all decoder layers with causal self-attention + cross-attention
3. Returns logits for next token + updated KV caches
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List


class TraceableCausalSelfAttention(nn.Module):
    """Single-step causal self-attention with KV cache."""

    def __init__(self, d_model, n_heads, d_head=None):
        super().__init__()
        self.d_head = d_head or d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_head ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * n_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, kv_cache, position):
        """
        Args:
            x: (B, 1, d_model) - single token embedding
            kv_cache: (2, B, max_seq, H, D) - [key, value] cache
            position: (1,) - current write position in cache
        Returns:
            output: (B, 1, d_model)
            new_kv_cache: (2, B, max_seq, H, D) - updated cache
            new_position: (1,) - incremented position
        """
        B, T, _ = x.shape  # T=1 for single step
        max_seq = kv_cache.shape[2]

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Write new k,v to cache using scatter (CoreML-compatible)
        # Build one-hot mask for position to avoid advanced indexing
        pos_idx = position.to(torch.long)
        one_hot = torch.zeros(max_seq, dtype=x.dtype, device=x.device)
        one_hot[pos_idx] = 1.0
        # one_hot: (max_seq,) → broadcast to (1, B, max_seq, H, D)
        mask = one_hot.view(1, 1, max_seq, 1, 1)

        k_new = k.squeeze(1).unsqueeze(0).unsqueeze(2)  # (1, B, 1, H, D) → broadcast to (1, B, max_seq, H, D)
        v_new = v.squeeze(1).unsqueeze(0).unsqueeze(2)

        # new_cache = (1-mask)*old_cache + mask*new_kv
        new_cache_k = kv_cache[0:1] * (1.0 - mask) + k_new * mask  # (1, B, max_seq, H, D)
        new_cache_v = kv_cache[1:2] * (1.0 - mask) + v_new * mask
        new_cache = torch.cat([new_cache_k, new_cache_v], dim=0)  # (2, B, max_seq, H, D)

        # Attend to all positions with a causal mask (positions > pos_idx are masked out)
        # Build mask: 1 for positions <= pos_idx, 0 for positions > pos_idx
        positions_range = torch.arange(max_seq, dtype=x.dtype, device=x.device)
        causal_mask = (positions_range <= position).float()  # (max_seq,)
        causal_mask = causal_mask.view(1, 1, 1, max_seq)  # (1, 1, 1, max_seq)

        q = q.transpose(1, 2)  # (B, H, 1, D)
        k_full = new_cache[0].transpose(1, 2)  # (B, H, max_seq, D)
        v_full = new_cache[1].transpose(1, 2)

        attn = torch.matmul(q, k_full.transpose(-2, -1)) * self.scale  # (B, H, 1, max_seq)
        attn = attn + (1.0 - causal_mask) * (-1e9)  # mask future positions
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_full)

        out = out.transpose(1, 2).reshape(B, 1, -1)
        out = self.o_proj(out)

        new_position = position + 1.0
        return out, new_cache, new_position


class TraceableCrossAttention(nn.Module):
    """Cross-attention to encoder output (non-causal, full attention)."""

    def __init__(self, d_model, n_heads, d_memory, d_head=None):
        super().__init__()
        self.d_head = d_head or d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.kv_proj = nn.Linear(d_memory, 2 * n_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, memory, memory_mask=None):
        """
        Args:
            x: (B, 1, d_model) - query
            memory: (B, T_enc, d_memory) - encoder output
            memory_mask: (B, T_enc) bool - True=keep
        Returns:
            output: (B, 1, d_model)
        """
        B, T_q, _ = x.shape
        T_m = memory.shape[1]

        q = self.q_proj(x).view(B, T_q, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(memory).view(B, T_m, 2, self.n_heads, self.d_head)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if memory_mask is not None:
            attn_mask = memory_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_m)
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T_q, -1)
        return self.o_proj(out)


class TraceableFFN(nn.Module):
    """Positionwise feed-forward for decoder."""

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


class TraceableDecoderLayer(nn.Module):
    """Single decoder transformer layer with self-attn, cross-attn, and FFN."""

    def __init__(self, d_model, d_ffn, sa_n_heads, xa_n_heads, xa_d_memory, kernel_size=1, xa_d_head=None):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model, bias=False)
        self.self_attn = TraceableCausalSelfAttention(d_model, sa_n_heads)

        self.has_xattn = xa_n_heads is not None
        if self.has_xattn:
            self.norm_xa_query = nn.LayerNorm(d_model, bias=False)
            self.norm_xa_memory = nn.LayerNorm(xa_d_memory, bias=False)
            self.cross_attn = TraceableCrossAttention(d_model, xa_n_heads, xa_d_memory, xa_d_head)

        self.norm_ff = nn.LayerNorm(d_model, bias=False)
        self.ffn = TraceableFFN(d_model, d_ffn, kernel_size)

    def forward(self, x, kv_cache, position, encoder_output=None, encoder_mask=None):
        """
        Returns:
            x: (B, 1, d_model)
            new_kv_cache: updated cache
            new_position: incremented position
        """
        # Self-attention
        residual = x
        x_norm = self.norm_sa(x)
        sa_out, new_kv_cache, new_position = self.self_attn(x_norm, kv_cache, position)
        x = residual + sa_out

        # Cross-attention
        if self.has_xattn and encoder_output is not None:
            residual = x
            q_norm = self.norm_xa_query(x)
            m_norm = self.norm_xa_memory(encoder_output)
            xa_out = self.cross_attn(q_norm, m_norm, encoder_mask)
            x = residual + xa_out

        # FFN
        residual = x
        x = self.norm_ff(x)
        x = self.ffn(x)
        x = residual + x

        return x, new_kv_cache, new_position


class TraceableDecoderStep(nn.Module):
    """Complete single-step decoder for CoreML.

    Takes one audio token embedding, runs through all decoder layers with
    KV cache, and outputs logits for next codebook tokens.

    The KV caches are passed as flat arguments (not lists) for torch.jit.trace.
    """

    def __init__(self, n_layers, d_model, d_ffn, sa_n_heads, xa_n_heads, xa_d_memory,
                 kernel_size=1, xa_d_head=None, max_seq_len=512,
                 use_pos_emb=False, max_pos=2048,
                 num_codebooks=8, num_tokens_per_codebook=2024, frame_stacking_factor=1):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_pos_emb = use_pos_emb
        self.num_codebooks = num_codebooks
        self.num_tokens_per_codebook = num_tokens_per_codebook
        self.frame_stacking_factor = frame_stacking_factor
        self.d_head = d_model // sa_n_heads

        if use_pos_emb:
            self.position_embeddings = nn.Embedding(max_pos, d_model)

        self.layers = nn.ModuleList([
            TraceableDecoderLayer(
                d_model, d_ffn, sa_n_heads, xa_n_heads, xa_d_memory, kernel_size, xa_d_head
            )
            for _ in range(n_layers)
        ])

        self.norm_out = nn.Identity()  # May be replaced if model uses apply_norm_out

        # Final projection: decoder hidden → codebook logits
        self.final_proj = nn.Linear(
            d_model,
            num_codebooks * num_tokens_per_codebook * frame_stacking_factor,
        )

    def forward(self, audio_embed, encoder_output, encoder_mask,
                # Flat KV cache args (one pair per layer)
                cache0, pos0, cache1, pos1, cache2, pos2,
                cache3, pos3, cache4, pos4, cache5, pos5,
                cache6, pos6, cache7, pos7, cache8, pos8,
                cache9, pos9, cache10, pos10, cache11, pos11):
        """
        Args:
            audio_embed: (B, 1, d_model) - embedded audio token(s)
            encoder_output: (B, T_enc, d_model) - text encoder output
            encoder_mask: (B, T_enc) - text mask
            cache{i}: (2, B, max_seq, H, D) - KV cache per layer
            pos{i}: (1,) - current position per layer

        Returns:
            logits: (B, 1, num_codebooks * num_tokens * frame_stacking)
            decoder_hidden: (B, 1, d_model) - for local transformer
            new_cache{i}: updated caches
            new_pos{i}: updated positions
        """
        caches = [cache0, cache1, cache2, cache3, cache4, cache5,
                  cache6, cache7, cache8, cache9, cache10, cache11]
        positions = [pos0, pos1, pos2, pos3, pos4, pos5,
                     pos6, pos7, pos8, pos9, pos10, pos11]

        x = audio_embed

        # Add positional embedding
        if self.use_pos_emb:
            pos_idx = positions[0].to(torch.long)
            x = x + self.position_embeddings(pos_idx).unsqueeze(0)

        new_caches = []
        new_positions = []

        for i, layer in enumerate(self.layers):
            x, new_cache, new_pos = layer(
                x, caches[i], positions[i],
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
            )
            new_caches.append(new_cache)
            new_positions.append(new_pos)

        x = self.norm_out(x)
        decoder_hidden = x

        logits = self.final_proj(x)

        return (logits, decoder_hidden,
                new_caches[0], new_positions[0],
                new_caches[1], new_positions[1],
                new_caches[2], new_positions[2],
                new_caches[3], new_positions[3],
                new_caches[4], new_positions[4],
                new_caches[5], new_positions[5],
                new_caches[6], new_positions[6],
                new_caches[7], new_positions[7],
                new_caches[8], new_positions[8],
                new_caches[9], new_positions[9],
                new_caches[10], new_positions[10],
                new_caches[11], new_positions[11])

    @classmethod
    def from_magpie(cls, model):
        """Create from a loaded MagpieTTSModel."""
        cfg = model.cfg
        dec_cfg = dict(cfg.decoder)

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
            num_codebooks=model.num_audio_codebooks,
            num_tokens_per_codebook=model.num_all_tokens_per_codebook,
            frame_stacking_factor=model.frame_stacking_factor,
        )

        # Copy positional embeddings
        if wrapper.use_pos_emb and model.decoder.position_embeddings is not None:
            wrapper.position_embeddings.weight.data.copy_(model.decoder.position_embeddings.weight.data)

        # Copy decoder layers
        # NeMo TransformerLayer attr names:
        #   self_attention (SelfAttention) with qkv_net, o_net
        #   cross_attention (CrossAttention) with q_net, kv_net, o_net
        #   norm_self, norm_xattn_query, norm_xattn_memory, norm_pos_ff (LayerNorm, bias=False)
        #   pos_ff (PositionwiseConvFF) with proj.conv, o_net.conv (Conv1d)
        for i, (src_layer, dst_layer) in enumerate(zip(model.decoder.layers, wrapper.layers)):
            # Self-attention
            dst_layer.self_attn.qkv_proj.weight.data.copy_(src_layer.self_attention.qkv_net.weight.data)
            dst_layer.self_attn.o_proj.weight.data.copy_(src_layer.self_attention.o_net.weight.data)

            # Self-attn norm (bias=False in NeMo)
            dst_layer.norm_sa.weight.data.copy_(src_layer.norm_self.weight.data)

            # Cross-attention (if present)
            if dst_layer.has_xattn and hasattr(src_layer, "cross_attention"):
                dst_layer.cross_attn.q_proj.weight.data.copy_(src_layer.cross_attention.q_net.weight.data)
                dst_layer.cross_attn.kv_proj.weight.data.copy_(src_layer.cross_attention.kv_net.weight.data)
                dst_layer.cross_attn.o_proj.weight.data.copy_(src_layer.cross_attention.o_net.weight.data)

                dst_layer.norm_xa_query.weight.data.copy_(src_layer.norm_xattn_query.weight.data)
                dst_layer.norm_xa_memory.weight.data.copy_(src_layer.norm_xattn_memory.weight.data)

            # FFN norm (bias=False in NeMo)
            dst_layer.norm_ff.weight.data.copy_(src_layer.norm_pos_ff.weight.data)

            # FFN (Conv1d via PositionwiseConvFF, bias=False)
            dst_layer.ffn.conv1.weight.data.copy_(src_layer.pos_ff.proj.conv.weight.data)
            dst_layer.ffn.conv2.weight.data.copy_(src_layer.pos_ff.o_net.conv.weight.data)

        # Output norm
        if hasattr(model.decoder, "norm_out") and isinstance(model.decoder.norm_out, nn.LayerNorm):
            wrapper.norm_out = nn.LayerNorm(dec_cfg["d_model"], bias=False)
            wrapper.norm_out.weight.data.copy_(model.decoder.norm_out.weight.data)

        # Final projection
        wrapper.final_proj.weight.data.copy_(model.final_proj.weight.data)
        wrapper.final_proj.bias.data.copy_(model.final_proj.bias.data)

        return wrapper
