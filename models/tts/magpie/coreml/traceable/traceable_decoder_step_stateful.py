"""EXPERIMENTAL — DO NOT USE IN PRODUCTION.

Stateful (MLState) variant of ``traceable_decoder_step.py``. Kept as a
documented dead-end so future agents don't repeat the experiment.

Benchmark result (Apple M2, macOS 26.5, 146-step real loop):
  rank-4 production (this file's non-stateful sibling): ~96 ms/step (97.3% ANE)
  this stateful variant (CPU_AND_GPU only):            ~212 ms/step
  → 2.2× regression. Rejected.

Why it loses for Magpie (vs CosyVoice3 where MLState gave ~3× speedup):
  Magpie's rank-4 decoder_step already lands 97.3% of cost on ANE. MLState
  graphs are ANE-incompatible, so they force CPU_AND_GPU. The IO-marshaling
  savings from collapsing 39 inputs / 38 outputs to 4 / 2 are dwarfed by the
  loss of ANE acceleration.

Variant of ``traceable_decoder_step.py`` that uses CoreML ``MLState`` (stateful
buffers) instead of passing 36 KV+position tensors through the model interface
on every step.

Differences vs. ``traceable_decoder_step.TraceableDecoderStep``:
  * Per-layer K and V caches are ``register_buffer``-ed (24 buffers total) and
    mutated in place via slice assignment.
  * Forward signature shrinks to 4 inputs: (audio_embed, encoder_output,
    encoder_mask, position). Position is a single shared scalar — all layers
    advance in lockstep so we don't statefy 12 copies of it.
  * Outputs shrink to 2: (logits, decoder_hidden). Cache updates are side
    effects on the state buffers.
  * Cross-attention path and fp16-safe ``MASK_NEG`` constant are unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# fp16 max is ±65504; -3e4 is safely representable and gives ~exp(-30000) ≈ 0
# after softmax. Identical numerical behaviour to -1e9 without the overflow.
MASK_NEG = -3.0e4


class StatefulCausalSelfAttention(nn.Module):
    """Single-step causal self-attention with state-buffer K/V caches.

    The K and V caches are owned by the parent ``StatefulDecoderLayer`` (so all
    buffers live on a single module for clean ``ct.StateType`` registration).
    This module receives the buffers by reference and mutates them in place.
    """

    def __init__(self, d_model, n_heads, d_head=None):
        super().__init__()
        self.d_head = d_head or d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_head ** -0.5
        self.qkv_proj = nn.Linear(d_model, 3 * n_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def forward(self, x, k_cache, v_cache, position):
        """
        x:        (B, 1, d_model)
        k_cache:  (B, max_seq, H, D)  — mutated in place
        v_cache:  (B, max_seq, H, D)  — mutated in place
        position: (1,) scalar — current write index (also used for causal mask)
        """
        B, T, _ = x.shape  # T = 1
        max_seq = k_cache.shape[1]

        qkv = self.qkv_proj(x)
        qkv = qkv.view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, 1, H, D)

        # In-place slice write — pure indexed_update (no scatter_nd).
        # Cast position to int via clamp for use as a slice bound.
        pos_int = position.to(torch.int32)
        # Slice bounds need to be Python ints during tracing; we materialize via
        # ``.item()``-equivalent through a 1-element tensor. CoreML's tracer will
        # capture the dynamic write index as a runtime variable.
        start = pos_int[0]
        end = start + 1

        # Cast new K/V to match buffer dtype (fp16 for the converted graph).
        k_cache[:, start:end, :, :] = k.to(k_cache.dtype)
        v_cache[:, start:end, :, :] = v.to(v_cache.dtype)

        # Reshape for batched matmul.
        q4 = q.transpose(1, 2)                    # (B, H, 1, D)
        k4 = k_cache.permute(0, 2, 1, 3)          # (B, H, max_seq, D)
        v4 = v_cache.permute(0, 2, 1, 3)          # (B, H, max_seq, D)

        # Causal mask: keep positions ≤ current `position`, drop the rest.
        positions_range = torch.arange(max_seq, dtype=x.dtype, device=x.device)
        causal_mask = (positions_range <= position).to(x.dtype).view(1, 1, 1, max_seq)

        attn = torch.matmul(q4, k4.to(x.dtype).transpose(-2, -1)) * self.scale
        attn = attn + (1.0 - causal_mask) * MASK_NEG
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v4.to(x.dtype))  # (B, H, 1, D)

        out = out.transpose(1, 2).reshape(B, 1, -1)
        out = self.o_proj(out)
        return out


class StatefulCrossAttention(nn.Module):
    """Cross-attention to encoder output (non-causal, stateless)."""

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
            mem_mask_f = memory_mask.to(x.dtype).unsqueeze(1).unsqueeze(2)
            attn = attn + (1.0 - mem_mask_f) * MASK_NEG

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T_q, -1)
        return self.o_proj(out)


class StatefulFFN(nn.Module):
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


class StatefulDecoderLayer(nn.Module):
    """One decoder layer; owns its k_cache / v_cache as registered buffers."""

    def __init__(self, layer_idx, d_model, d_ffn, sa_n_heads, xa_n_heads, xa_d_memory,
                 max_seq_len, kernel_size=1, xa_d_head=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_head = d_model // sa_n_heads
        self.n_heads = sa_n_heads

        self.norm_sa = nn.LayerNorm(d_model, bias=False)
        self.self_attn = StatefulCausalSelfAttention(d_model, sa_n_heads)

        self.has_xattn = xa_n_heads is not None
        if self.has_xattn:
            self.norm_xa_query = nn.LayerNorm(d_model, bias=False)
            self.norm_xa_memory = nn.LayerNorm(xa_d_memory, bias=False)
            self.cross_attn = StatefulCrossAttention(d_model, xa_n_heads, xa_d_memory, xa_d_head)

        self.norm_ff = nn.LayerNorm(d_model, bias=False)
        self.ffn = StatefulFFN(d_model, d_ffn, kernel_size)

        # Register cache buffers (fp16 to match converted-graph precision).
        # Persistent=False so they don't appear in state_dict and won't trip
        # weight-load checks.
        self.register_buffer(
            "k_cache",
            torch.zeros(1, max_seq_len, sa_n_heads, self.d_head, dtype=torch.float16),
            persistent=False,
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(1, max_seq_len, sa_n_heads, self.d_head, dtype=torch.float16),
            persistent=False,
        )

    def forward(self, x, position, encoder_output=None, encoder_mask=None):
        # Self-attention (mutates self.k_cache / self.v_cache in place).
        residual = x
        x_norm = self.norm_sa(x)
        sa_out = self.self_attn(x_norm, self.k_cache, self.v_cache, position)
        x = residual + sa_out

        # Cross-attention.
        if self.has_xattn and encoder_output is not None:
            residual = x
            q_norm = self.norm_xa_query(x)
            m_norm = self.norm_xa_memory(encoder_output)
            xa_out = self.cross_attn(q_norm, m_norm, encoder_mask)
            x = residual + xa_out

        # FFN.
        residual = x
        x = self.norm_ff(x)
        x = self.ffn(x)
        x = residual + x
        return x


class StatefulDecoderStep(nn.Module):
    """Stateful single-step decoder. K/V caches live as buffers on each layer.

    Forward inputs (4):
        audio_embed:    (B, 1, d_model)
        encoder_output: (B, T_enc, d_model)
        encoder_mask:   (B, T_enc) bool
        position:       (1,) scalar — write index for this step (shared across layers)

    Forward outputs (2):
        logits:         (B, 1, num_codebooks * tokens_per_codebook * frame_stack)
        decoder_hidden: (B, 1, d_model)

    State (24 buffers; named ``k_cache_{i}``, ``v_cache_{i}`` for i in 0..n-1
    after ``flatten_state_buffers`` is called):
        k_cache_{i}: (1, max_seq, H, D) fp16
        v_cache_{i}: (1, max_seq, H, D) fp16
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
        self.sa_n_heads = sa_n_heads

        if use_pos_emb:
            self.position_embeddings = nn.Embedding(max_pos, d_model)

        self.layers = nn.ModuleList([
            StatefulDecoderLayer(
                i, d_model, d_ffn, sa_n_heads, xa_n_heads, xa_d_memory,
                max_seq_len, kernel_size, xa_d_head,
            )
            for i in range(n_layers)
        ])

        self.norm_out = nn.Identity()
        self.final_proj = nn.Linear(
            d_model, num_codebooks * num_tokens_per_codebook * frame_stacking_factor)

        # Promote per-layer buffers to top-level names so coremltools can pick
        # them up via ``ct.StateType(name="k_cache_{i}")``.
        self.flatten_state_buffers()

    def flatten_state_buffers(self):
        """Re-register each layer's k_cache / v_cache as top-level buffers.

        coremltools' ``ct.StateType(name=...)`` matches the buffer name on the
        traced module. Layer-nested buffers come through as
        ``layers.{i}.k_cache``; we mirror them at the top level under the
        flatter ``k_cache_{i}`` / ``v_cache_{i}`` names that downstream code
        (and other mobius converters) expect.
        """
        for i, layer in enumerate(self.layers):
            self.register_buffer(f"k_cache_{i}", layer.k_cache, persistent=False)
            self.register_buffer(f"v_cache_{i}", layer.v_cache, persistent=False)

    def reset_state(self):
        """Zero all KV caches in place (host side, before make_state)."""
        for layer in self.layers:
            layer.k_cache.zero_()
            layer.v_cache.zero_()

    def forward(self, audio_embed, encoder_output, encoder_mask, position):
        x = audio_embed
        if self.use_pos_emb:
            pos_idx = position.to(torch.long)
            x = x + self.position_embeddings(pos_idx).unsqueeze(0)

        for layer in self.layers:
            x = layer(
                x, position,
                encoder_output=encoder_output,
                encoder_mask=encoder_mask,
            )

        decoder_hidden = self.norm_out(x)
        logits = self.final_proj(decoder_hidden)
        return logits, decoder_hidden

    @classmethod
    def from_magpie(cls, model):
        """Create from a loaded MagpieTTSModel and copy over weights."""
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

        if wrapper.use_pos_emb and model.decoder.position_embeddings is not None:
            wrapper.position_embeddings.weight.data.copy_(
                model.decoder.position_embeddings.weight.data)

        for src_layer, dst_layer in zip(model.decoder.layers, wrapper.layers):
            # Self-attention.
            dst_layer.self_attn.qkv_proj.weight.data.copy_(src_layer.self_attention.qkv_net.weight.data)
            dst_layer.self_attn.o_proj.weight.data.copy_(src_layer.self_attention.o_net.weight.data)
            dst_layer.norm_sa.weight.data.copy_(src_layer.norm_self.weight.data)

            # Cross-attention.
            if dst_layer.has_xattn and hasattr(src_layer, "cross_attention"):
                dst_layer.cross_attn.q_proj.weight.data.copy_(src_layer.cross_attention.q_net.weight.data)
                dst_layer.cross_attn.kv_proj.weight.data.copy_(src_layer.cross_attention.kv_net.weight.data)
                dst_layer.cross_attn.o_proj.weight.data.copy_(src_layer.cross_attention.o_net.weight.data)
                dst_layer.norm_xa_query.weight.data.copy_(src_layer.norm_xattn_query.weight.data)
                dst_layer.norm_xa_memory.weight.data.copy_(src_layer.norm_xattn_memory.weight.data)

            # FFN.
            dst_layer.norm_ff.weight.data.copy_(src_layer.norm_pos_ff.weight.data)
            dst_layer.ffn.conv1.weight.data.copy_(src_layer.pos_ff.proj.conv.weight.data)
            dst_layer.ffn.conv2.weight.data.copy_(src_layer.pos_ff.o_net.conv.weight.data)

        # Optional output norm.
        if hasattr(model.decoder, "norm_out") and isinstance(model.decoder.norm_out, nn.LayerNorm):
            wrapper.norm_out = nn.LayerNorm(dec_cfg["d_model"], bias=False)
            wrapper.norm_out.weight.data.copy_(model.decoder.norm_out.weight.data)

        # Final projection.
        wrapper.final_proj.weight.data.copy_(model.final_proj.weight.data)
        wrapper.final_proj.bias.data.copy_(model.final_proj.bias.data)

        # Re-flatten buffers in case eager copies replaced them.
        wrapper.flatten_state_buffers()
        return wrapper
