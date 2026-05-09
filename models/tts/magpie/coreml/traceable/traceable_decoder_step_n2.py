"""N=2 unrolled decoder_step + LT + audio_embed lookup, traced as one graph.

Fuses 2 AR iterations into a single CoreML submission to amortize the
per-dispatch boundary cost (~4 ms each on M2). Internal flow per call:

    iter 1:
        hidden_1, state_1 = decoder_step(audio_embed_in, enc, mask, state_0)
        codes_1           = local_transformer(hidden_1, uniforms_1, ...)
    audio_embed_2         = sum_cb(audio_embedding[cb][codes_1[cb]]) / 8
    iter 2:
        hidden_2, state_2 = decoder_step(audio_embed_2, enc, mask, state_1)
        codes_2           = local_transformer(hidden_2, uniforms_2, ...)
    return codes_1, codes_2, state_2

Audio embed lookup uses the same mask-multiply pattern as
``FusedLocalTransformer`` (one-hot @ table) instead of ``gather`` so the
graph stays on ANE.

Outputs:
    codes_1, codes_2 : (8,) int32 each
    new_ck{i}, new_cv{i}, new_p{i} : final state from iter 2
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .traceable_decoder_step import TraceableDecoderStep


# Constants — must match Magpie + FusedLocalTransformer.
D_MODEL = 768
LOCAL_DIM = 256
FFN_DIM = 1024
NUM_CODEBOOKS = 8
NUM_CODES_PER_CODEBOOK = 2024
DEFAULT_TOP_K = 80
EOS_ID = 2017
ALWAYS_FORBIDDEN = (2016, 2018, 2019, 2020, 2021, 2022, 2023)
NEG_INF = -1e4


def _layer_norm_no_bias(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-5) * weight


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    s = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(s * (x + 0.044715 * x.pow(3))))


class _LTLayer(nn.Module):
    """Single 1-layer pre-norm causal SA + pre-norm FFN. Mirrors LT script."""

    def __init__(self, w: Dict[str, np.ndarray]):
        super().__init__()
        self.register_buffer("norm1_w", torch.from_numpy(w["norm1_weight"]).float())
        self.register_buffer("norm2_w", torch.from_numpy(w["norm2_weight"]).float())
        self.register_buffer("qkv_w", torch.from_numpy(w["sa_qkv_weight"]).float())
        self.register_buffer("o_w", torch.from_numpy(w["sa_o_weight"]).float())
        ffn1 = w["ffn_conv1_weight"]
        if ffn1.ndim == 3:
            ffn1 = ffn1.squeeze(-1)
        ffn2 = w["ffn_conv2_weight"]
        if ffn2.ndim == 3:
            ffn2 = ffn2.squeeze(-1)
        self.register_buffer("ffn_w1", torch.from_numpy(ffn1).float())
        self.register_buffer("ffn_w2", torch.from_numpy(ffn2).float())
        self.register_buffer("pos_emb", torch.from_numpy(w["pos_emb"]).float())
        self._scale = 1.0 / math.sqrt(LOCAL_DIM)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        T = seq.shape[-2]
        x = seq + self.pos_emb[:T]
        x_norm = _layer_norm_no_bias(x, self.norm1_w)
        qkv = x_norm @ self.qkv_w.t()
        q, k, v = qkv.split(LOCAL_DIM, dim=-1)
        attn = (q @ k.t()) * self._scale
        causal = torch.tril(torch.ones(T, T, dtype=attn.dtype, device=attn.device))
        attn = attn + (1.0 - causal) * NEG_INF
        attn = attn.softmax(dim=-1)
        sa_out = attn @ v
        sa_out = sa_out @ self.o_w.t()
        x = x + sa_out
        x_norm = _layer_norm_no_bias(x, self.norm2_w)
        h = x_norm @ self.ffn_w1.t()
        h = _gelu_tanh(h)
        h = h @ self.ffn_w2.t()
        return x + h


class _FusedLT(nn.Module):
    """Internal LT + 8-codebook sampler — same logic as FusedLocalTransformer.

    Returns codes (8,) int32; the audio_embed lookup is done by the parent
    module (which has the full 768-dim audio_emb tables).
    """

    def __init__(
        self,
        lt_w: Dict[str, np.ndarray],
        out_proj_weights: List[np.ndarray],
        out_proj_biases: List[np.ndarray],
        proj_audio_emb: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
    ):
        super().__init__()
        self.layer = _LTLayer(lt_w)
        self.top_k = int(top_k)

        self.register_buffer(
            "in_proj_w", torch.from_numpy(lt_w["in_proj_weight"]).float())
        self.register_buffer(
            "in_proj_b", torch.from_numpy(lt_w["in_proj_bias"]).float())

        out_w = np.stack(out_proj_weights, axis=0)
        out_b = np.stack(out_proj_biases, axis=0)
        self.register_buffer("out_w", torch.from_numpy(out_w).float())
        self.register_buffer("out_b", torch.from_numpy(out_b).float())
        self.register_buffer(
            "proj_audio_emb", torch.from_numpy(proj_audio_emb).float())

        always = np.zeros(NUM_CODES_PER_CODEBOOK, dtype=np.float32)
        for tok in ALWAYS_FORBIDDEN:
            always[tok] = NEG_INF
        eos = np.zeros(NUM_CODES_PER_CODEBOOK, dtype=np.float32)
        eos[EOS_ID] = NEG_INF
        self.register_buffer("always_addend", torch.from_numpy(always))
        self.register_buffer("eos_addend", torch.from_numpy(eos))
        self.register_buffer(
            "arange_codes",
            torch.arange(NUM_CODES_PER_CODEBOOK, dtype=torch.int32))

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        uniforms: torch.Tensor,
        forbid_eos: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        first = decoder_hidden.reshape(D_MODEL) @ self.in_proj_w.t() + self.in_proj_b
        seq = first.unsqueeze(0)
        temp_safe = torch.clamp(temperature, min=1e-8)
        eos_scale = forbid_eos.reshape(())

        codes: List[torch.Tensor] = []
        for cb in range(NUM_CODEBOOKS):
            out = self.layer(seq)
            last = out[cb]
            logits = last @ self.out_w[cb].t() + self.out_b[cb]
            logits = logits + self.always_addend
            logits = logits + self.eos_addend * eos_scale
            logits = logits / temp_safe.reshape(())
            top_v, top_i = torch.topk(logits, self.top_k, dim=-1)
            probs = top_v.softmax(dim=-1)
            cdf = probs.cumsum(dim=-1)
            u = uniforms[cb].reshape(())
            ge_u = (cdf >= u).to(torch.int32)
            ge_cum = ge_u.cumsum(dim=-1)
            slot_mask = (ge_cum == 1).to(top_i.dtype)
            code64 = (top_i * slot_mask).sum()
            code32 = code64.to(torch.int32)
            codes.append(code32)
            code_onehot = (self.arange_codes == code32).to(self.proj_audio_emb.dtype)
            next_in = code_onehot @ self.proj_audio_emb[cb]
            seq = torch.cat([seq, next_in.unsqueeze(0)], dim=0)

        return torch.stack(codes, dim=0)


class FusedDecoderN2(nn.Module):
    """N=2 unrolled fused decoder + LT + audio_embed lookup.

    Reuses an existing TraceableDecoderStep instance for both iterations
    (parameters shared, state passed through). Audio embed for iter 2 is
    looked up via mask-multiply (no gather).
    """

    def __init__(
        self,
        decoder: TraceableDecoderStep,
        lt_w: Dict[str, np.ndarray],
        out_proj_weights: List[np.ndarray],
        out_proj_biases: List[np.ndarray],
        proj_audio_emb: np.ndarray,
        audio_emb_full: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
    ):
        super().__init__()
        self.decoder = decoder
        self.lt = _FusedLT(lt_w, out_proj_weights, out_proj_biases,
                           proj_audio_emb, top_k=top_k)

        # Full 768-dim audio embeddings for the inter-step lookup.
        # Shape: (8, 2024, 768).
        assert audio_emb_full.shape == (NUM_CODEBOOKS, NUM_CODES_PER_CODEBOOK, D_MODEL)
        self.register_buffer(
            "audio_emb_full", torch.from_numpy(audio_emb_full).float())
        self.register_buffer(
            "arange_codes",
            torch.arange(NUM_CODES_PER_CODEBOOK, dtype=torch.int32))

        # Pre-computed mean divisor (1/8) for audio_embed averaging.
        self._inv_cb = 1.0 / float(NUM_CODEBOOKS)

    def _audio_embed_lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (8,) int32 → audio_embed (1, 1, 768) fp32.

        Uses (arange == code) one-hot @ table to keep on ANE.
        Mirrors Swift's fillAudioEmbed: mean over 8 cb of table[cb][code].
        """
        accum = torch.zeros(D_MODEL, dtype=self.audio_emb_full.dtype,
                            device=codes.device)
        for cb in range(NUM_CODEBOOKS):
            code = codes[cb]
            onehot = (self.arange_codes == code).to(self.audio_emb_full.dtype)
            row = onehot @ self.audio_emb_full[cb]  # (768,)
            accum = accum + row
        accum = accum * self._inv_cb
        return accum.view(1, 1, D_MODEL)

    def forward(
        self,
        audio_embed_in,
        encoder_output, encoder_mask,
        # Iter 1 KV state in (12 layers × 3).
        ck0, cv0, p0, ck1, cv1, p1, ck2, cv2, p2,
        ck3, cv3, p3, ck4, cv4, p4, ck5, cv5, p5,
        ck6, cv6, p6, ck7, cv7, p7, ck8, cv8, p8,
        ck9, cv9, p9, ck10, cv10, p10, ck11, cv11, p11,
        # LT inputs for both iterations.
        uniforms_1, uniforms_2,
        forbid_eos_1, forbid_eos_2,
        temperature,
    ):
        # ---------------- ITER 1 ----------------
        out1 = self.decoder(
            audio_embed_in, encoder_output, encoder_mask,
            ck0, cv0, p0, ck1, cv1, p1, ck2, cv2, p2,
            ck3, cv3, p3, ck4, cv4, p4, ck5, cv5, p5,
            ck6, cv6, p6, ck7, cv7, p7, ck8, cv8, p8,
            ck9, cv9, p9, ck10, cv10, p10, ck11, cv11, p11,
        )
        # out1 = (logits, decoder_hidden, *36_state_tensors)
        hidden_1 = out1[1]
        state_1 = list(out1[2:])  # 36 tensors: nk0, nv0, np0, ..., nk11, nv11, np11

        codes_1 = self.lt(hidden_1, uniforms_1, forbid_eos_1, temperature)

        # ---------------- AUDIO EMBED LOOKUP ----------------
        audio_embed_2 = self._audio_embed_lookup(codes_1)

        # ---------------- ITER 2 ----------------
        out2 = self.decoder(
            audio_embed_2, encoder_output, encoder_mask,
            *state_1,
        )
        hidden_2 = out2[1]
        state_2 = list(out2[2:])

        codes_2 = self.lt(hidden_2, uniforms_2, forbid_eos_2, temperature)

        return (codes_1, codes_2, *state_2)
