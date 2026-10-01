"""Trace-friendly wrappers around MossTTSNanoForCausalLM for CoreML export.

Three graphs:
  MossPrefill  – prompt rows [1,T,17] → last hidden [1,768] + zero-padded KV [L,1,H,M,D]
  MossStep     – one row [1,1,17] + KV + cur_len → hidden [1,768] + updated KV
  MossFrame    – global hidden [1,768] → text stop decision + 16 audio tokens, with
                 temperature / top-k / top-p / repetition-penalty sampling done in-graph
                 from host-supplied uniform randoms (inverse-CDF; greedy flag → argmax).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1.0e4  # finite stand-in for -inf that survives fp16


def _rope_tables(positions: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """positions [T] float → cos, sin [1, T, 1, D] laid out as (c0,c0,c1,c1,...) like repeat_interleave."""
    freqs = positions[:, None] * inv_freq[None, :]  # [T, D/2]
    cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).reshape(freqs.shape[0], -1)
    sin = torch.stack((freqs.sin(), freqs.sin()), dim=-1).reshape(freqs.shape[0], -1)
    return cos[None, :, None, :], sin[None, :, None, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape(x.shape)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


class _GPT2Core(nn.Module):
    """Explicit-KV re-implementation of MossTTSNanoGPT2Model.forward (eager attention)."""

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.tr = transformer
        attn0 = transformer.h[0].attn
        self.n_head = attn0.num_heads
        self.head_dim = attn0.head_dim
        self.scale = 1.0 / (self.head_dim**0.5)
        self.register_buffer("inv_freq", attn0.rotary_emb.inv_freq.clone(), persistent=False)

    def qkv(self, block: nn.Module, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """x [B,T,C] → q,k,v [B,H,T,D] with RoPE applied to q,k."""
        B, T, _ = x.shape
        qkv = block.attn.c_attn(block.ln_1(x))
        q, k, v = qkv.split(self.n_head * self.head_dim, dim=-1)
        q = _apply_rope(q.view(B, T, self.n_head, self.head_dim), cos, sin).transpose(1, 2)
        k = _apply_rope(k.view(B, T, self.n_head, self.head_dim), cos, sin).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        return q, k, v

    def finish_block(self, block: nn.Module, x: torch.Tensor, q, k, v, bias: torch.Tensor) -> torch.Tensor:
        """Attention over (k,v) with additive bias [.., Tq, Tk], then residual + MLP."""
        B, _, T, _ = q.shape
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale + bias
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(B, T, self.n_head * self.head_dim)
        x = x + block.attn.c_proj(out)
        return x + block.mlp(block.ln_2(x))


class MossEmbedder(nn.Module):
    """Row [.., 17] int → summed text + audio embeddings [.., 768] (pad-masked, as upstream)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.wte = model.transformer.wte
        self.audio_embeddings = model.audio_embeddings
        self.audio_pad = int(model.config.audio_pad_token_id)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids = input_ids.long()
        x = self.wte(ids[..., 0])
        for c, emb in enumerate(self.audio_embeddings):
            ch = ids[..., c + 1]
            valid = ch.ne(self.audio_pad)
            safe = torch.where(valid, ch, torch.zeros_like(ch))
            x = x + emb(safe) * valid.unsqueeze(-1).to(x.dtype)
        return x


class MossPrefill(nn.Module):
    def __init__(self, model: nn.Module, max_len: int) -> None:
        super().__init__()
        self.embed = MossEmbedder(model)
        self.core = _GPT2Core(model.transformer)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, input_len: torch.Tensor):
        x = self.embed(input_ids)  # [1,T,C]
        T = x.shape[1]
        ar = torch.arange(T, device=x.device)
        cos, sin = _rope_tables(ar.float(), self.core.inv_freq)
        key_valid = ar[None, :] < input_len.long()[:, None]  # [1,T]
        causal = ar[None, :] <= ar[:, None]  # [T,T]
        mask = causal[None, None] & key_valid[:, None, None, :]
        bias = torch.where(mask, torch.zeros((), dtype=x.dtype), torch.full((), NEG, dtype=x.dtype))
        kv_keep = key_valid.to(x.dtype)[:, None, :, None]  # [1,1,T,1]
        ks, vs = [], []
        for block in self.core.tr.h:
            q, k, v = self.core.qkv(block, x, cos, sin)
            k = k * kv_keep
            v = v * kv_keep
            x = self.core.finish_block(block, x, q, k, v, bias)
            ks.append(k)
            vs.append(v)
        x = self.core.tr.ln_f(x)
        onehot = (ar[None, :] == (input_len.long() - 1)[:, None]).to(x.dtype)  # [1,T]
        hidden_last = (x * onehot[..., None]).sum(dim=1)  # [1,C]
        pad = self.max_len - T
        kv_k = F.pad(torch.stack(ks, dim=0), (0, 0, 0, pad))  # [L,1,H,M,D]
        kv_v = F.pad(torch.stack(vs, dim=0), (0, 0, 0, pad))
        return hidden_last, kv_k, kv_v


class MossStep(nn.Module):
    def __init__(self, model: nn.Module, max_len: int) -> None:
        super().__init__()
        self.embed = MossEmbedder(model)
        self.core = _GPT2Core(model.transformer)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, kv_k: torch.Tensor, kv_v: torch.Tensor, cur_len: torch.Tensor):
        x = self.embed(input_ids)  # [1,1,C]
        pos = cur_len.long()
        cos, sin = _rope_tables(pos.float(), self.core.inv_freq)  # [1,1,1,D]
        j = torch.arange(self.max_len, device=x.device)
        key_ok = j[None, :] <= pos[:, None]  # [1,M]
        bias = torch.where(key_ok, torch.zeros((), dtype=x.dtype), torch.full((), NEG, dtype=x.dtype))
        bias = bias[:, None, None, :]  # [1,1,1,M]
        onehot = (j[None, :] == pos[:, None]).to(x.dtype)[:, None, :, None]  # [1,1,M,1]
        ks, vs = [], []
        for layer, block in enumerate(self.core.tr.h):
            q, k_new, v_new = self.core.qkv(block, x, cos, sin)  # [1,H,1,D]
            k = kv_k[layer] * (1.0 - onehot) + k_new * onehot
            v = kv_v[layer] * (1.0 - onehot) + v_new * onehot
            x = self.core.finish_block(block, x, q, k, v, bias)
            ks.append(k)
            vs.append(v)
        hidden = self.core.tr.ln_f(x)[:, 0, :]
        return hidden, torch.stack(ks, dim=0), torch.stack(vs, dim=0)


def _sample(
    scores: torch.Tensor,
    u: torch.Tensor,
    temperature: torch.Tensor,
    top_k: int,
    top_p: torch.Tensor,
    greedy: torch.Tensor,
) -> torch.Tensor:
    """scores [1,V] → token [1] (int64). Mirrors upstream _sample_next_token; multinomial → inverse CDF."""
    greedy_tok = scores.argmax(dim=-1)
    s = scores / temperature
    V = s.shape[-1]
    if top_k < V:
        thr = torch.topk(s, top_k, dim=-1).values[..., -1:]
        s = torch.where(s < thr, torch.full_like(s, NEG), s)
    sorted_s, sorted_idx = torch.sort(s, dim=-1, descending=True)
    probs = torch.softmax(sorted_s, dim=-1)
    excl = torch.cumsum(probs, dim=-1) - probs  # mass strictly before each entry
    keep = excl <= top_p  # first entry always kept
    sorted_s = torch.where(keep, sorted_s, torch.full_like(sorted_s, NEG))
    probs = torch.softmax(sorted_s, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    pick = (cdf < u[:, None]).to(torch.int32).sum(dim=-1)  # [1]
    pick = torch.minimum(pick, torch.full_like(pick, V - 1))
    sampled = torch.gather(sorted_idx, 1, pick.long()[:, None])[:, 0]
    return torch.where(greedy > 0.5, greedy_tok, sampled)


class MossFrame(nn.Module):
    TEXT_TOP_K = 50
    AUDIO_TOP_K = 25

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        cfg = model.config
        self.local = _GPT2Core(model.local_transformer)
        self.wte = model.transformer.wte
        self.audio_embeddings = model.audio_embeddings
        self.audio_heads = model.audio_lm_heads
        self.n_vq = int(cfg.n_vq)
        self.slot_id = int(cfg.audio_assistant_slot_token_id)
        self.end_id = int(cfg.audio_end_token_id)
        self.register_buffer("cand_ids", torch.tensor([self.slot_id, self.end_id]), persistent=False)

    def _local_last(self, embeds: torch.Tensor) -> torch.Tensor:
        T = embeds.shape[1]
        ar = torch.arange(T, device=embeds.device)
        cos, sin = _rope_tables(ar.float(), self.local.inv_freq)
        causal = ar[None, :] <= ar[:, None]
        bias = torch.where(causal, torch.zeros((), dtype=embeds.dtype), torch.full((), NEG, dtype=embeds.dtype))
        bias = bias[None, None]
        x = embeds
        for block in self.local.tr.h:
            q, k, v = self.local.qkv(block, x, cos, sin)
            x = self.local.finish_block(block, x, q, k, v, bias)
        return self.local.tr.ln_f(x)[:, -1, :]

    def forward(
        self,
        global_hidden: torch.Tensor,  # [1,768]
        text_u: torch.Tensor,  # [1]
        audio_u: torch.Tensor,  # [1,16]
        text_temperature: torch.Tensor,  # [1]
        audio_temperature: torch.Tensor,  # [1]
        audio_top_p: torch.Tensor,  # [1]
        repetition_penalty: torch.Tensor,  # [1]
        seen: torch.Tensor,  # [1,16,1024] 0/1 float
        greedy: torch.Tensor,  # [1] 0/1 float
    ):
        embeds = global_hidden[:, None, :]
        h = self._local_last(embeds)
        cand_logits = F.linear(h, self.wte.weight[self.cand_ids])  # [1,2]
        one = torch.ones_like(text_temperature)
        cand = _sample(cand_logits, text_u, text_temperature, self.TEXT_TOP_K, one, greedy)  # 0 → continue
        should_continue = (cand == 0).to(torch.int32)
        text_tok = torch.where(cand == 0, torch.full_like(cand, self.slot_id), torch.full_like(cand, self.end_id))
        cur = self.wte(text_tok)  # [1,768]
        toks = []
        for c in range(self.n_vq):
            embeds = torch.cat((embeds, cur[:, None, :]), dim=1)
            h = self._local_last(embeds)
            logits = self.audio_heads[c](h)  # [1,1024]
            pen = torch.where(logits < 0, logits * repetition_penalty, logits / repetition_penalty)
            logits = torch.where(seen[:, c, :] > 0.5, pen, logits)
            tok = _sample(logits, audio_u[:, c], audio_temperature, self.AUDIO_TOP_K, audio_top_p, greedy)
            toks.append(tok)
            cur = self.audio_embeddings[c](tok)
        frame = torch.stack(toks, dim=-1).to(torch.int32)  # [1,16]
        return should_continue, frame
