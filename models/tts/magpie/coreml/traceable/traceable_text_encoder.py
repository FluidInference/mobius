"""Traceable text encoder wrapper for CoreML conversion.

The text encoder is a causal transformer (transformer_2501.Transformer) that
encodes text tokens into conditioning for the decoder's cross-attention.

Architecture (verified from model inspection):
- Causal self-attention (lower-triangular mask)
- Causal Conv1d FFN (kernel_size=3, left-padded)
- LayerNorm with bias=False
- Learnable positional embeddings
- Output LayerNorm

This wrapper:
- Handles text embedding + learnable positional encoding
- Runs the encoder transformer stack with causal masking
- Applies causal padding for Conv1d FFN layers
- Produces fixed-shape outputs suitable for torch.jit.trace
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TraceableSelfAttention(nn.Module):
    """Causal self-attention for encoder."""

    def __init__(self, d_model, n_heads, d_head=None):
        super().__init__()
        self.d_head = d_head or d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_head ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * n_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask (lower-triangular)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
        attn = attn * causal_mask.unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        # Padding mask
        if mask is not None:
            # mask: (B, T) where True = keep, False = pad
            attn_mask = mask.unsqueeze(1).unsqueeze(2).float()  # (B, 1, 1, T)
            attn = attn * attn_mask
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN from all-inf rows (padded positions) with 0
        attn = attn.masked_fill(attn != attn, 0.0)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class TraceableFFN(nn.Module):
    """Positionwise feed-forward network with causal Conv1d.

    NeMo uses ConvolutionLayer which applies causal left-padding manually:
    F.pad(signal, (kernel_size - 1, 0)) before Conv1d(padding=0).
    """

    def __init__(self, d_model, d_ffn, kernel_size=1, is_causal=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.is_causal = is_causal
        # Conv1d with no built-in padding (bias=False matching NeMo) — we handle padding manually
        self.conv1 = nn.Conv1d(d_model, d_ffn, kernel_size, padding=0, bias=False)
        self.conv2 = nn.Conv1d(d_ffn, d_model, kernel_size, padding=0, bias=False)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x, mask=None):
        # x: (B, T, C) -> transpose -> (B, C, T) for conv1d
        x = x.transpose(1, 2)

        if mask is not None:
            x = x * mask.unsqueeze(1).float()

        if self.is_causal and self.kernel_size > 1:
            x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.act(self.conv1(x))

        if mask is not None:
            x = x * mask.unsqueeze(1).float()

        if self.is_causal and self.kernel_size > 1:
            x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv2(x)

        if mask is not None:
            x = x * mask.unsqueeze(1).float()

        return x.transpose(1, 2)


class TraceableEncoderLayer(nn.Module):
    """Single encoder transformer layer (causal)."""

    def __init__(self, d_model, d_ffn, n_heads, kernel_size=1, is_causal=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, bias=False)
        self.self_attn = TraceableSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model, bias=False)
        self.ffn = TraceableFFN(d_model, d_ffn, kernel_size, is_causal)

    def forward(self, x, mask=None):
        # Pre-norm causal self-attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = residual + x

        # Pre-norm FFN with causal conv
        residual = x
        x = self.norm2(x)
        x = self.ffn(x, mask)
        x = residual + x

        return x


class TraceableTextEncoder(nn.Module):
    """Complete text encoder for CoreML conversion.

    Architecture (verified from NVIDIA Magpie TTS 357M):
    - Text embedding lookup (vocab=2362, d_model=768)
    - Learnable positional encoding (max_pos=2048)
    - 6 causal transformer layers (causal self-attention + causal Conv1d FFN)
    - Output LayerNorm (bias=False)
    """

    def __init__(self, n_layers, d_model, d_ffn, n_heads, kernel_size=1,
                 vocab_size=None, use_pos_emb=False, max_pos=2048,
                 apply_norm_out=False, is_causal=True):
        super().__init__()
        self.d_model = d_model
        self.use_pos_emb = use_pos_emb

        if vocab_size is not None:
            self.text_embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.text_embedding = None

        if use_pos_emb:
            self.position_embeddings = nn.Embedding(max_pos, d_model)

        self.layers = nn.ModuleList([
            TraceableEncoderLayer(d_model, d_ffn, n_heads, kernel_size, is_causal)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model, bias=False) if apply_norm_out else nn.Identity()

    def forward(self, text_tokens, text_mask):
        """
        Args:
            text_tokens: (B, T) int32 - token IDs
            text_mask: (B, T) bool - True where valid, False for padding
        Returns:
            encoder_output: (B, T, d_model) float32
        """
        if self.text_embedding is not None:
            x = self.text_embedding(text_tokens)
        else:
            x = text_tokens  # already embedded

        if self.use_pos_emb:
            T = x.shape[1]
            positions = torch.arange(T, device=x.device)
            x = x + self.position_embeddings(positions)

        # Mask padding
        x = x * text_mask.unsqueeze(-1).float()

        for layer in self.layers:
            x = layer(x, mask=text_mask)

        x = self.norm_out(x)
        return x

    @classmethod
    def from_magpie(cls, model, include_text_embedding=True):
        """Create from a loaded MagpieTTSModel.

        Args:
            model: MagpieTTSModel instance
            include_text_embedding: Whether to include text embedding in the encoder
        """
        cfg = model.cfg
        enc_cfg = dict(cfg.encoder)

        wrapper = cls(
            n_layers=enc_cfg["n_layers"],
            d_model=enc_cfg["d_model"],
            d_ffn=enc_cfg["d_ffn"],
            n_heads=enc_cfg["sa_n_heads"],
            kernel_size=enc_cfg.get("kernel_size", 1),
            vocab_size=model.text_embedding.weight.shape[0] if include_text_embedding and hasattr(model, "text_embedding") else None,
            use_pos_emb=enc_cfg.get("use_learnable_pos_emb", False),
            max_pos=enc_cfg.get("max_length_causal_mask", 2048),
            apply_norm_out=enc_cfg.get("apply_norm_out", False),
            is_causal=enc_cfg.get("is_causal", True),
        )

        # Copy weights
        if include_text_embedding and hasattr(model, "text_embedding"):
            wrapper.text_embedding.weight.data.copy_(model.text_embedding.weight.data)

        if wrapper.use_pos_emb and model.encoder.position_embeddings is not None:
            wrapper.position_embeddings.weight.data.copy_(model.encoder.position_embeddings.weight.data)

        # Copy transformer layers
        # NeMo TransformerLayer attr names (verified):
        #   self_attention (SelfAttention) with qkv_net, o_net
        #   norm_self, norm_pos_ff (LayerNorm, bias=False)
        #   pos_ff (PositionwiseConvFF) with proj.conv, o_net.conv (Conv1d)
        for i, (src_layer, dst_layer) in enumerate(zip(model.encoder.layers, wrapper.layers)):
            # Self-attention
            dst_layer.self_attn.qkv_proj.weight.data.copy_(src_layer.self_attention.qkv_net.weight.data)
            dst_layer.self_attn.o_proj.weight.data.copy_(src_layer.self_attention.o_net.weight.data)

            # Norms (bias=False in NeMo)
            dst_layer.norm1.weight.data.copy_(src_layer.norm_self.weight.data)
            dst_layer.norm2.weight.data.copy_(src_layer.norm_pos_ff.weight.data)

            # FFN (Conv1d weights via PositionwiseConvFF, bias=False)
            dst_layer.ffn.conv1.weight.data.copy_(src_layer.pos_ff.proj.conv.weight.data)
            dst_layer.ffn.conv2.weight.data.copy_(src_layer.pos_ff.o_net.conv.weight.data)

        # Output norm
        if enc_cfg.get("apply_norm_out", False):
            wrapper.norm_out.weight.data.copy_(model.encoder.norm_out.weight.data)

        return wrapper
