"""Two-stage Kev export: read the state once, then answer every question from its cache.

Kev's server does this with a KV + recurrent-state cache; a row-per-question export re-reads the state for every
question. Stage A (`StatePass`) runs the state tokens and returns, per layer, what a continuation needs:

- full-attention layers: keys (after norm and RoPE) and values of the state positions;
- Gated DeltaNet layers: the recurrent state after the last real token, and the last `conv_kernel - 1` pre-convolution
  inputs (the causal conv's history).

The state is right-padded to a bucket. Padded positions get beta = 0 and g = 0 in the delta rule, which makes them
exact no-ops on the recurrent state (the WY inverse row of a beta = 0 position is the identity row), so the state
comes out as of the last real token. Attention keys of padded positions are masked in stage B.

Stage B (`QuestionPass`) runs B question branches at once, each right-padded to Q tokens and positioned right after
the state, on top of that cache, and applies Kev's pointer head at <decide> and each </opt>.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qwen35_export import DecoderChunk, RMSNorm, TextConfig, rotate_half

MASK_VALUE = -1e4
MASKED_INVERSE = True


def _rope(x, cos, sin, r):
    return torch.cat([x[..., :r] * cos + rotate_half(x[..., :r]) * sin, x[..., r:]], dim=-1)


def _delta_chunk_terms(net, q, k, v, beta, g):
    """Chunked delta-rule terms for tensors shaped [G, N, C, D] (beta [G, N, C, 1], g [G, N, C])."""
    g_col = g.unsqueeze(-1)
    cum = torch.matmul(net.prefix_sum_t, g_col).squeeze(-1)
    to_end = torch.matmul(net.suffix_sum_t, g_col).squeeze(-1)
    lead = g.shape[:2]
    pair = torch.matmul(net.pair_sum_t, g_col).reshape(*lead, net.chunk, net.chunk)
    pair_decay = torch.exp(pair) * net.lower_incl
    k_beta, v_beta = k * beta, v * beta
    kkt = torch.matmul(k_beta, k.transpose(-1, -2)) * pair_decay
    attn_intra = torch.matmul(q, k.transpose(-1, -2)) * pair_decay
    inv = _unit_lower_inverse(net.eye + kkt * net.strict_lower, net.chunk) if MASKED_INVERSE else \
        net.unit_lower_inverse(net.eye + kkt * net.strict_lower)
    new_v = torch.matmul(inv, v_beta)
    k_cumdecay = torch.matmul(inv, k_beta * torch.exp(cum)[..., None])
    q_dec = q * torch.exp(cum)[..., None]
    k_dec = k * torch.exp(to_end)[..., None]
    chunk_decay = torch.exp(cum[..., -1:])[..., None]
    return new_v, k_cumdecay, q_dec, k_dec, attn_intra, chunk_decay


def _delta_inputs(net, x, conv_input_prefix=None):
    """Projections of a DeltaNet layer for x [B, L, D]. Returns q, k, v (heads expanded), z, beta, g, and the
    pre-convolution qkv [B, L, conv]."""
    cfg = net.cfg
    B, L = x.shape[0], x.shape[1]
    H, Dk, Dv, kh = cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim, cfg.lin_k_heads
    rep = H // kh
    pre = net.in_proj_qkv(x)  # [B, L, conv]
    seq = pre.transpose(1, 2)  # [B, conv, L]
    if conv_input_prefix is None:
        seq = F.pad(seq, (cfg.conv_kernel - 1, 0))
    else:
        seq = torch.cat([conv_input_prefix, seq], dim=-1)
    qkv = F.silu(net.conv1d(seq)).transpose(1, 2)  # [B, L, conv]
    q = qkv[..., : net.key_dim].reshape(B, L, kh, Dk)
    k = qkv[..., net.key_dim : 2 * net.key_dim].reshape(B, L, kh, Dk)
    v = qkv[..., 2 * net.key_dim :].reshape(B, L, H, Dv)
    z = net.in_proj_z(x).reshape(B, L, H, Dv)
    beta = torch.sigmoid(net.in_proj_b(x))  # [B, L, H]
    g = -torch.exp(net.A_log) * F.softplus(net.in_proj_a(x) + net.dt_bias)  # [B, L, H]
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q[:, :, :, None].expand(B, L, kh, rep, Dk).reshape(B, L, H, Dk) * (Dk**-0.5)
    k = k[:, :, :, None].expand(B, L, kh, rep, Dk).reshape(B, L, H, Dk)
    return q, k, v, z, beta, g, pre


class StatePass(nn.Module):
    def __init__(self, cfg: TextConfig, seq_len: int, chunk_size: int = 128):
        super().__init__()
        self.cfg, self.S = cfg, seq_len
        self.decoder = DecoderChunk(cfg, 0, cfg.num_layers, seq_len, chunk_size=chunk_size, with_head=False)

    def _attention(self, attn, x, cos, sin):
        cfg, S = self.cfg, self.S
        h, kv, d, r = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim
        qg = attn.q_proj(x).view(1, S, h, 2 * d)
        q, gate = qg[..., :d], qg[..., d:].reshape(1, S, h * d)
        q = attn.q_norm(q).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(1, S, kv, d)).transpose(1, 2)
        v = attn.v_proj(x).view(1, S, kv, d).transpose(1, 2)
        c, s = cos[None, None], sin[None, None]
        q, k = _rope(q, c, s, r), _rope(k, c, s, r)
        rep = h // kv
        kr = k[:, :, None].expand(1, kv, rep, S, d).reshape(1, h, S, d)
        vr = v[:, :, None].expand(1, kv, rep, S, d).reshape(1, h, S, d)
        scores = torch.matmul(q, kr.transpose(-1, -2)) * (d**-0.5) + self.decoder.mask
        out = torch.matmul(torch.softmax(scores, dim=-1), vr).transpose(1, 2).reshape(1, S, h * d)
        return attn.o_proj(out * torch.sigmoid(gate)), k[0], v[0]

    def _delta(self, net, x, valid, tail_onehot):
        cfg = self.cfg
        L, C, N = self.S, net.chunk, net.num_chunks
        H, Dk, Dv = cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim
        q, k, v, z, beta, g, pre = _delta_inputs(net, x)
        tail = torch.matmul(tail_onehot, pre[0]).transpose(0, 1)  # [conv, kernel-1]
        mask = valid[None, :, None]
        beta, g = beta * mask, g * mask  # padded positions: no write, no decay
        q = q[0].permute(1, 0, 2).reshape(H, N, C, Dk)
        k = k[0].permute(1, 0, 2).reshape(H, N, C, Dk)
        v = v[0].permute(1, 0, 2).reshape(H, N, C, Dv)
        beta = beta[0].T.reshape(H, N, C, 1)
        g = g[0].T.reshape(H, N, C)
        new_v, k_cumdecay, q_dec, k_dec, attn_intra, chunk_decay = _delta_chunk_terms(net, q, k, v, beta, g)
        state = torch.zeros(H, Dk, Dv, dtype=x.dtype)
        outs = []
        for i in range(N):
            v_new = new_v[:, i] - torch.matmul(k_cumdecay[:, i], state)
            outs.append(torch.matmul(q_dec[:, i], state) + torch.matmul(attn_intra[:, i], v_new))
            state = state * chunk_decay[:, i] + torch.matmul(k_dec[:, i].transpose(-1, -2), v_new)
        core = torch.stack(outs, dim=1).reshape(H, L, Dv).permute(1, 0, 2)
        core = net.norm(core, z[0])
        return net.out_proj(core.reshape(1, L, H * Dv)), state, tail

    def forward(self, hidden, cos, sin, valid, tail_onehot):
        """hidden [1, S, D], cos/sin [S, R], valid [S] (1 = real token), tail_onehot [kernel-1, S] selecting the last
        real positions in order -> keys, values [A, kv, S, d]; delta states [Ld, H, Dk, Dv]; conv tails [Ld, conv, k-1]."""
        keys, values, states, tails = [], [], [], []
        x = hidden
        for layer in self.decoder.layers:
            h = layer.input_layernorm(x)
            if layer.is_linear:
                out, state, tail = self._delta(layer.linear_attn, h, valid, tail_onehot)
                states.append(state)
                tails.append(tail)
            else:
                out, key, value = self._attention(layer.self_attn, h, cos, sin)
                keys.append(key)
                values.append(value)
            x = x + out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return torch.stack(keys), torch.stack(values), torch.stack(states), torch.stack(tails)


class QuestionPass(nn.Module):
    def __init__(self, cfg: TextConfig, state_len: int, question_len: int, batch: int, max_options: int,
                 pointer_dim: int = 256):
        super().__init__()
        self.cfg, self.S, self.Q, self.B, self.K = cfg, state_len, question_len, batch, max_options
        # chunk = the whole question: one delta-rule chunk on top of the cached state
        self.decoder = DecoderChunk(cfg, 0, cfg.num_layers, question_len, chunk_size=question_len, with_head=False)
        self.norm = RMSNorm(cfg.hidden_size, cfg.eps)
        self.q = nn.Linear(cfg.hidden_size, pointer_dim)
        self.k = nn.Linear(cfg.hidden_size, pointer_dim)
        self.register_buffer("pointer_scale", torch.tensor([pointer_dim**-0.5]))
        self.register_buffer("inverse_temperature", torch.tensor([1.0]))
        causal = torch.full((question_len, question_len), MASK_VALUE).triu(1)
        self.register_buffer("causal", causal)

    def _attention(self, attn, x, cos, sin, key_cache, value_cache, cache_bias):
        cfg, B, Q, S = self.cfg, self.B, self.Q, self.S
        h, kv, d, r = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim
        qg = attn.q_proj(x).view(B, Q, h, 2 * d)
        q, gate = qg[..., :d], qg[..., d:].reshape(B, Q, h * d)
        q = attn.q_norm(q).transpose(1, 2)  # [B, h, Q, d]
        k = attn.k_norm(attn.k_proj(x).view(B, Q, kv, d)).transpose(1, 2)
        v = attn.v_proj(x).view(B, Q, kv, d).transpose(1, 2)
        c, s = cos[None, None], sin[None, None]
        q, k = _rope(q, c, s, r), _rope(k, c, s, r)
        k_all = torch.cat([key_cache[None].expand(B, kv, S, d), k], dim=2)  # [B, kv, S+Q, d]
        v_all = torch.cat([value_cache[None].expand(B, kv, S, d), v], dim=2)
        rep, T = h // kv, S + Q
        k_all = k_all[:, :, None].expand(B, kv, rep, T, d).reshape(B, h, T, d)
        v_all = v_all[:, :, None].expand(B, kv, rep, T, d).reshape(B, h, T, d)
        bias = torch.cat([cache_bias[None, :].expand(Q, S), self.causal], dim=1)  # [Q, S+Q]
        scores = torch.matmul(q, k_all.transpose(-1, -2)) * (d**-0.5) + bias
        out = torch.matmul(torch.softmax(scores, dim=-1), v_all).transpose(1, 2).reshape(B, Q, h * d)
        return attn.o_proj(out * torch.sigmoid(gate))

    def _delta(self, net, x, state, tail):
        cfg, B, Q = self.cfg, self.B, self.Q
        H, Dk, Dv = cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim
        q, k, v, z, beta, g, _ = _delta_inputs(net, x, conv_input_prefix=tail[None].expand(B, -1, -1))
        G = B * H
        q = q.permute(0, 2, 1, 3).reshape(G, 1, Q, Dk)
        k = k.permute(0, 2, 1, 3).reshape(G, 1, Q, Dk)
        v = v.permute(0, 2, 1, 3).reshape(G, 1, Q, Dv)
        beta = beta.permute(0, 2, 1).reshape(G, 1, Q, 1)
        g = g.permute(0, 2, 1).reshape(G, 1, Q)
        new_v, k_cumdecay, q_dec, _, attn_intra, _ = _delta_chunk_terms(net, q, k, v, beta, g)
        s0 = state[None].expand(B, H, Dk, Dv).reshape(G, Dk, Dv)
        v_new = new_v[:, 0] - torch.matmul(k_cumdecay[:, 0], s0)
        core = torch.matmul(q_dec[:, 0], s0) + torch.matmul(attn_intra[:, 0], v_new)  # [G, Q, Dv]
        core = core.reshape(B, H, Q, Dv).permute(0, 2, 1, 3)  # [B, Q, H, Dv]
        core = net.norm(core, z)
        return net.out_proj(core.reshape(B, Q, H * Dv))

    def forward(self, hidden, cos, sin, keys, values, cache_valid, states, tails, decide_onehot, option_onehot,
                option_mask):
        """hidden [B, Q, D]; cos/sin [Q, R] (positions n .. n+Q-1); keys/values [A, kv, S, d]; cache_valid [S];
        states [Ld, H, Dk, Dv]; tails [Ld, conv, k-1]; decide_onehot [B, Q]; option_onehot [B, K, Q];
        option_mask [B, K] -> logits [B, K] (temperature applied), probabilities [B, K]."""
        cache_bias = (1.0 - cache_valid) * MASK_VALUE
        x = hidden
        a = d = 0
        for layer in self.decoder.layers:
            h = layer.input_layernorm(x)
            if layer.is_linear:
                out = self._delta(layer.linear_attn, h, states[d], tails[d])
                d += 1
            else:
                out = self._attention(layer.self_attn, h, cos, sin, keys[a], values[a], cache_bias)
                a += 1
            x = x + out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        decide = self.norm(torch.matmul(decide_onehot[:, None, :], x)[:, 0])  # [B, D]
        options = self.norm(torch.matmul(option_onehot, x))  # [B, K, D]
        logits = (self.k(options) * self.q(decide)[:, None, :]).sum(-1) * self.pointer_scale * self.inverse_temperature
        logits = torch.where(option_mask > 0.5, logits, torch.full_like(logits, MASK_VALUE))
        return logits, torch.softmax(logits, dim=-1)


def load_stages(merged, state_len: int, question_len: int, batch: int, max_options: int, chunk_size: int = 128):
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    merged = Path(merged)
    meta = json.loads((merged / "meta.json").read_text())
    cfg = TextConfig(meta["text_config"])
    state = load_file(str(merged / "text.safetensors"))
    head = load_file(str(merged / "head.safetensors"))
    first = StatePass(cfg, state_len, chunk_size)
    first.decoder.load_merged(state)
    second = QuestionPass(cfg, state_len, question_len, batch, max_options)
    second.decoder.load_merged(state)
    second.norm.load_state_dict({"weight": state["norm.weight"]})
    second.q.load_state_dict({"weight": head["q.weight"], "bias": head["q.bias"]})
    second.k.load_state_dict({"weight": head["k.weight"], "bias": head["k.bias"]})
    second.pointer_scale.fill_(meta["pointer_scale"])
    second.inverse_temperature.fill_(1.0 / meta["temperature"])
    return first.eval(), second.eval(), cfg, meta, state["embed_tokens.weight"]


def stage_inputs(cfg, embed, state_ids, question_rows, state_len, question_len, batch, max_options, pad_id):
    """question_rows: [(branch ids, decide offset, option offsets)] with offsets inside the branch."""
    from qwen35_export import rope_cos_sin

    n = len(state_ids)
    if n > state_len or len(question_rows) > batch:
        raise ValueError(f"state {n} > {state_len} or {len(question_rows)} questions > {batch}")
    ids = torch.full((state_len,), pad_id, dtype=torch.long)
    ids[:n] = torch.tensor(state_ids)
    positions = torch.arange(state_len)
    cos, sin = rope_cos_sin(cfg, positions.unsqueeze(0).expand(3, -1))
    valid = torch.zeros(state_len)
    valid[:n] = 1
    kernel = cfg.conv_kernel - 1
    tail = torch.zeros(kernel, state_len)
    for j in range(kernel):
        p = n - kernel + j
        if p >= 0:
            tail[j, p] = 1
    state_in = (embed[ids].unsqueeze(0), cos, sin, valid, tail)

    q_ids = torch.full((batch, question_len), pad_id, dtype=torch.long)
    decide = torch.zeros(batch, question_len)
    options = torch.zeros(batch, max_options, question_len)
    mask = torch.zeros(batch, max_options)
    for b, (branch, d, opts) in enumerate(question_rows):
        if len(branch) > question_len or len(opts) > max_options:
            raise ValueError("question exceeds its bucket")
        q_ids[b, : len(branch)] = torch.tensor(branch)
        decide[b, d] = 1
        for i, o in enumerate(opts):
            options[b, i, o] = 1
            mask[b, i] = 1
    q_pos = torch.arange(n, n + question_len)
    q_cos, q_sin = rope_cos_sin(cfg, q_pos.unsqueeze(0).expand(3, -1))
    return state_in, (embed[q_ids], q_cos, q_sin, valid, decide, options, mask)


class PackedQuestionPass(nn.Module):
    """Stage B with the questions packed end to end in one sequence of P tokens instead of B padded rows.

    `segment` [P, P] is 1 where j <= i in the same question (padding positions are their own one-token segments):
    it is the attention mask over the question tokens, the segment-local cumulative-decay operator of the delta rule
    (every question restarts from the cached state), and its strict part is the WY triangle. A question's first
    conv_kernel - 1 positions read the state's conv tail instead of the previous question: `lag_keep` [k-1, P] keeps
    in-segment lags and `lag_tail` [k-1, P, k-1] selects tail entries for the rest.
    """

    def __init__(self, cfg: TextConfig, state_len: int, packed_len: int, batch: int, max_options: int,
                 pointer_dim: int = 256):
        super().__init__()
        self.cfg, self.S, self.P, self.B, self.K = cfg, state_len, packed_len, batch, max_options
        self.decoder = DecoderChunk(cfg, 0, cfg.num_layers, packed_len, chunk_size=packed_len, with_head=False)
        self.norm = RMSNorm(cfg.hidden_size, cfg.eps)
        self.q = nn.Linear(cfg.hidden_size, pointer_dim)
        self.k = nn.Linear(cfg.hidden_size, pointer_dim)
        self.register_buffer("pointer_scale", torch.tensor([pointer_dim**-0.5]))
        self.register_buffer("inverse_temperature", torch.tensor([1.0]))
        self.register_buffer("eye", torch.eye(packed_len))

    def _attention(self, attn, x, cos, sin, key_cache, value_cache, bias):
        cfg, P, S = self.cfg, self.P, self.S
        h, kv, d, r = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim
        qg = attn.q_proj(x).view(1, P, h, 2 * d)
        q, gate = qg[..., :d], qg[..., d:].reshape(1, P, h * d)
        q = attn.q_norm(q).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(1, P, kv, d)).transpose(1, 2)
        v = attn.v_proj(x).view(1, P, kv, d).transpose(1, 2)
        c, s = cos[None, None], sin[None, None]
        q, k = _rope(q, c, s, r), _rope(k, c, s, r)
        k_all = torch.cat([key_cache[None], k], dim=2)  # [1, kv, S+P, d]
        v_all = torch.cat([value_cache[None], v], dim=2)
        rep, T = h // kv, S + P
        k_all = k_all[:, :, None].expand(1, kv, rep, T, d).reshape(1, h, T, d)
        v_all = v_all[:, :, None].expand(1, kv, rep, T, d).reshape(1, h, T, d)
        scores = torch.matmul(q, k_all.transpose(-1, -2)) * (d**-0.5) + bias
        out = torch.matmul(torch.softmax(scores, dim=-1), v_all).transpose(1, 2).reshape(1, P, h * d)
        return attn.o_proj(out * torch.sigmoid(gate))

    def _delta(self, net, x, state, tail, segment, lag_keep, lag_tail):
        cfg, P = self.cfg, self.P
        H, Dk, Dv, kh = cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim, cfg.lin_k_heads
        rep = H // kh
        pre = net.in_proj_qkv(x)[0]  # [P, conv]
        weight = net.conv1d.weight[:, 0]  # [conv, kernel]
        lags = weight.shape[1] - 1
        conv = pre * weight[:, lags]
        for s in range(1, lags + 1):
            shifted = F.pad(pre[: P - s], (0, 0, s, 0)) * lag_keep[s - 1][:, None]
            shifted = shifted + torch.matmul(lag_tail[s - 1], tail.transpose(0, 1))
            conv = conv + shifted * weight[:, lags - s]
        qkv = F.silu(conv)
        q = qkv[:, : net.key_dim].reshape(P, kh, Dk)
        k = qkv[:, net.key_dim : 2 * net.key_dim].reshape(P, kh, Dk)
        v = qkv[:, 2 * net.key_dim :].reshape(P, H, Dv).transpose(0, 1)  # [H, P, Dv]
        z = net.in_proj_z(x).reshape(1, P, H, Dv)
        beta = torch.sigmoid(net.in_proj_b(x))[0].transpose(0, 1)[..., None]  # [H, P, 1]
        g = (-torch.exp(net.A_log) * F.softplus(net.in_proj_a(x) + net.dt_bias))[0].transpose(0, 1)  # [H, P]
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = (q[:, :, None].expand(P, kh, rep, Dk).reshape(P, H, Dk) * (Dk**-0.5)).transpose(0, 1)  # [H, P, Dk]
        k = k[:, :, None].expand(P, kh, rep, Dk).reshape(P, H, Dk).transpose(0, 1)
        cum = torch.matmul(g, segment.transpose(0, 1))  # [H, P]: decay since the segment start
        pair = (cum[:, :, None] - cum[:, None, :]) * segment + (segment - 1.0) * 1e4
        pair_decay = torch.exp(pair)
        k_beta, v_beta = k * beta, v * beta
        kkt = torch.matmul(k_beta, k.transpose(-1, -2)) * pair_decay
        attn_intra = torch.matmul(q, k.transpose(-1, -2)) * pair_decay
        inv = _unit_lower_inverse(self.eye + kkt * (segment - self.eye), P)
        decay = torch.exp(cum)[..., None]
        v_new = torch.matmul(inv, v_beta) - torch.matmul(torch.matmul(inv, k_beta * decay), state)
        core = torch.matmul(q * decay, state) + torch.matmul(attn_intra, v_new)  # [H, P, Dv]
        core = net.norm(core.transpose(0, 1)[None], z)
        return net.out_proj(core.reshape(1, P, H * Dv))

    def forward(self, hidden, cos, sin, keys, values, cache_valid, states, tails, segment, lag_keep, lag_tail,
                decide_onehot, option_onehot, option_mask):
        """hidden [1, P, D]; cos/sin [P, R] (each question at positions n, n+1, ...); keys/values [A, kv, S, d];
        cache_valid [S]; states [Ld, H, Dk, Dv]; tails [Ld, conv, k-1]; segment [P, P]; lag_keep [k-1, P];
        lag_tail [k-1, P, k-1]; decide_onehot [B, P]; option_onehot [B, K, P]; option_mask [B, K]
        -> logits [B, K] (temperature applied), probabilities [B, K]."""
        bias = torch.cat([((1.0 - cache_valid) * MASK_VALUE)[None, :].expand(self.P, self.S),
                          (1.0 - segment) * MASK_VALUE], dim=1)
        x = hidden
        a = d = 0
        for layer in self.decoder.layers:
            h = layer.input_layernorm(x)
            if layer.is_linear:
                out = self._delta(layer.linear_attn, h, states[d], tails[d], segment, lag_keep, lag_tail)
                d += 1
            else:
                out = self._attention(layer.self_attn, h, cos, sin, keys[a], values[a], bias)
                a += 1
            x = x + out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        decide = self.norm(torch.matmul(decide_onehot, x[0]))  # [B, D]
        options = self.norm(torch.matmul(option_onehot, x[0]))  # [B, K, D]
        logits = (self.k(options) * self.q(decide)[:, None, :]).sum(-1) * self.pointer_scale * self.inverse_temperature
        logits = torch.where(option_mask > 0.5, logits, torch.full_like(logits, MASK_VALUE))
        return logits, torch.softmax(logits, dim=-1)


def load_packed(merged, state_len: int, packed_len: int, batch: int, max_options: int):
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    merged = Path(merged)
    meta = json.loads((merged / "meta.json").read_text())
    cfg = TextConfig(meta["text_config"])
    state = load_file(str(merged / "text.safetensors"))
    head = load_file(str(merged / "head.safetensors"))
    model = PackedQuestionPass(cfg, state_len, packed_len, batch, max_options)
    model.decoder.load_merged(state)
    model.norm.load_state_dict({"weight": state["norm.weight"]})
    model.q.load_state_dict({"weight": head["q.weight"], "bias": head["q.bias"]})
    model.k.load_state_dict({"weight": head["k.weight"], "bias": head["k.bias"]})
    model.pointer_scale.fill_(meta["pointer_scale"])
    model.inverse_temperature.fill_(1.0 / meta["temperature"])
    return model.eval()


def packed_inputs(cfg, embed, state_count, question_rows, packed_len, batch, max_options, pad_id, lags=3, lane=None):
    """question_rows: [(branch ids, decide offset, option offsets)] -> PackedQuestionPass inputs after the cache.
    With `lane`, a question that would cross a lane boundary starts at the next lane."""
    from qwen35_export import rope_cos_sin

    P = packed_len
    ids = torch.full((P,), pad_id, dtype=torch.long)
    positions = torch.full((P,), state_count, dtype=torch.long)
    segment = torch.eye(P)
    lag_keep = torch.zeros(lags, P)
    lag_tail = torch.zeros(lags, P, lags)
    decide = torch.zeros(batch, P)
    options = torch.zeros(batch, max_options, P)
    mask = torch.zeros(batch, max_options)
    start = 0
    for b, (branch, d, opts) in enumerate(question_rows):
        n = len(branch)
        if lane and start // lane != (start + n - 1) // lane:
            start = (start // lane + 1) * lane
        if start + n > P or b >= batch or len(opts) > max_options:
            raise ValueError("questions exceed the packed bucket")
        ids[start : start + n] = torch.tensor(branch)
        positions[start : start + n] = torch.arange(state_count, state_count + n)
        segment[start : start + n, start : start + n] = torch.ones(n, n).tril()
        for p in range(n):
            for s in range(1, lags + 1):
                if p >= s:
                    lag_keep[s - 1, start + p] = 1
                else:
                    lag_tail[s - 1, start + p, lags + p - s] = 1
        decide[b, start + d] = 1
        for i, o in enumerate(opts):
            options[b, i, start + o] = 1
            mask[b, i] = 1
        start += n
    cos, sin = rope_cos_sin(cfg, positions.unsqueeze(0).expand(3, -1))
    return embed[ids].unsqueeze(0), cos, sin, segment, lag_keep, lag_tail, decide, options, mask


def _masked_block_inverse(t, C, block=None):
    """Same blocking as GatedDeltaNet.unit_lower_inverse, kept as one [.., C, C] matrix: with D the inverse of t's
    diagonal s-blocks and L = t's lower-left s-blocks inside each diagonal 2s-block, the 2s-block inverse is
    D - D L D (L D L = 0). Four ops per level instead of block extraction, slicing and concatenation."""
    idx = torch.arange(C)
    inv = None
    s = 1
    while s < min(C, block or C):
        same_pair = (idx[:, None] // (2 * s)) == (idx[None, :] // (2 * s))
        lower_left = same_pair & ((idx[:, None] // s) % 2 == 1) & ((idx[None, :] // s) % 2 == 0)
        mask = lower_left.to(t.dtype)
        if inv is None:
            inv = torch.eye(C, dtype=t.dtype) - t * mask
        else:
            inv = inv - torch.matmul(torch.matmul(inv, t * mask), inv)
        s *= 2
    return inv


def _unit_lower_inverse(t, C, block=None):
    """GatedDeltaNet.unit_lower_inverse for a [G, C, C] unit-lower-triangular batch of any power-of-two C. `block`:
    t is known block-diagonal in blocks of that size (packed questions never cross a lane)."""
    if MASKED_INVERSE:
        return _masked_block_inverse(t, C, block)
    b = t.shape[0]
    inv = torch.ones(b, C, 1, 1, dtype=t.dtype)
    s = 1
    while s < C:
        n2 = C // (2 * s)
        blocks = (t.reshape(b, n2, 2 * s, n2, 2 * s) * torch.eye(n2, dtype=t.dtype)[:, None, :, None]).sum(-2)
        x = blocks[..., s:, :s]
        pair = inv.reshape(b, n2, 2, s, s)
        a_inv, b_inv = pair[..., 0, :, :], pair[..., 1, :, :]
        lower = -torch.matmul(b_inv, torch.matmul(x, a_inv))
        inv = torch.cat([torch.cat([a_inv, torch.zeros_like(a_inv)], dim=-1), torch.cat([lower, b_inv], dim=-1)], dim=-2)
        s *= 2
    return inv.reshape(b, C, C)


class FusedPass(nn.Module):
    """State pass and packed question pass in one call: sequence = S state tokens (right-padded) + P packed question
    tokens. Projections, norms, MLPs and attention run once over all S + P positions (attention under one combined
    mask); only the delta rule splits: chunked over the state, then one segment-masked chunk for the questions on top
    of the state's final recurrent state and conv tail."""

    def __init__(self, cfg: TextConfig, state_len: int, packed_len: int, batch: int, max_options: int,
                 chunk_size: int = 64, pointer_dim: int = 256, lane: int = 128):
        super().__init__()
        self.cfg, self.S, self.P, self.B, self.K = cfg, state_len, packed_len, batch, max_options
        self.lane = min(lane, packed_len)
        self.decoder = DecoderChunk(cfg, 0, cfg.num_layers, state_len, chunk_size=min(chunk_size, state_len),
                                    with_head=False)
        self.norm = RMSNorm(cfg.hidden_size, cfg.eps)
        self.q = nn.Linear(cfg.hidden_size, pointer_dim)
        self.k = nn.Linear(cfg.hidden_size, pointer_dim)
        self.register_buffer("pointer_scale", torch.tensor([pointer_dim**-0.5]))
        self.register_buffer("inverse_temperature", torch.tensor([1.0]))
        self.register_buffer("eye_p", torch.eye(packed_len))
        self.register_buffer("state_causal", torch.full((state_len, state_len), MASK_VALUE).triu(1))
        self.register_buffer("state_block", torch.full((state_len, packed_len), MASK_VALUE))

    def _attention(self, attn, x, cos, sin, bias):
        cfg, T = self.cfg, self.S + self.P
        h, kv, d, r = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.rotary_dim
        qg = attn.q_proj(x).view(1, T, h, 2 * d)
        q, gate = qg[..., :d], qg[..., d:].reshape(1, T, h * d)
        q = attn.q_norm(q).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(x).view(1, T, kv, d)).transpose(1, 2)
        v = attn.v_proj(x).view(1, T, kv, d).transpose(1, 2)
        c, s = cos[None, None], sin[None, None]
        q, k = _rope(q, c, s, r), _rope(k, c, s, r)
        rep = h // kv
        k = k[:, :, None].expand(1, kv, rep, T, d).reshape(1, h, T, d)
        v = v[:, :, None].expand(1, kv, rep, T, d).reshape(1, h, T, d)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (d**-0.5) + bias
        out = torch.matmul(torch.softmax(scores, dim=-1), v).transpose(1, 2).reshape(1, T, h * d)
        return attn.o_proj(out * torch.sigmoid(gate))

    def _delta(self, net, x, valid, tail_onehot, segment, lag_keep, lag_tail):
        cfg, S, P = self.cfg, self.S, self.P
        T = S + P
        H, Dk, Dv, kh = cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim, cfg.lin_k_heads
        rep = H // kh
        pre = net.in_proj_qkv(x)[0]  # [T, conv]
        pre_s, pre_p = pre[:S], pre[S:]
        tail = torch.matmul(tail_onehot, pre_s)  # [k-1, conv]
        weight = net.conv1d.weight[:, 0]  # [conv, kernel]
        lags = weight.shape[1] - 1
        conv_s, conv_p = pre_s * weight[:, lags], pre_p * weight[:, lags]
        for s in range(1, lags + 1):
            conv_s = conv_s + F.pad(pre_s[: S - s], (0, 0, s, 0)) * weight[:, lags - s]
            shifted = F.pad(pre_p[: P - s], (0, 0, s, 0)) * lag_keep[s - 1][:, None] + torch.matmul(lag_tail[s - 1], tail)
            conv_p = conv_p + shifted * weight[:, lags - s]
        qkv = F.silu(torch.cat([conv_s, conv_p], dim=0))  # [T, conv]
        q = qkv[:, : net.key_dim].reshape(T, kh, Dk)
        k = qkv[:, net.key_dim : 2 * net.key_dim].reshape(T, kh, Dk)
        v = qkv[:, 2 * net.key_dim :].reshape(T, H, Dv).transpose(0, 1)  # [H, T, Dv]
        z = net.in_proj_z(x).reshape(1, T, H, Dv)
        beta = torch.sigmoid(net.in_proj_b(x))[0].transpose(0, 1)  # [H, T]
        g = (-torch.exp(net.A_log) * F.softplus(net.in_proj_a(x) + net.dt_bias))[0].transpose(0, 1)  # [H, T]
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = (q[:, :, None].expand(T, kh, rep, Dk).reshape(T, H, Dk) * (Dk**-0.5)).transpose(0, 1)  # [H, T, Dk]
        k = k[:, :, None].expand(T, kh, rep, Dk).reshape(T, H, Dk).transpose(0, 1)

        # state: chunked, padded positions are no-ops
        C, N = net.chunk, net.num_chunks
        beta_s, g_s = beta[:, :S] * valid, g[:, :S] * valid
        new_v, k_cumdecay, q_dec, k_dec, attn_intra, chunk_decay = _delta_chunk_terms(
            net, q[:, :S].reshape(H, N, C, Dk), k[:, :S].reshape(H, N, C, Dk), v[:, :S].reshape(H, N, C, Dv),
            beta_s.reshape(H, N, C, 1), g_s.reshape(H, N, C))
        state = torch.zeros(H, Dk, Dv, dtype=x.dtype)
        outs = []
        for i in range(N):
            v_new = new_v[:, i] - torch.matmul(k_cumdecay[:, i], state)
            outs.append(torch.matmul(q_dec[:, i], state) + torch.matmul(attn_intra[:, i], v_new))
            state = state * chunk_decay[:, i] + torch.matmul(k_dec[:, i].transpose(-1, -2), v_new)

        # questions: one segment-masked chunk from the final state
        qp, kp, vp = q[:, S:], k[:, S:], v[:, S:]
        beta_p = beta[:, S:][..., None]
        cum = torch.matmul(g[:, S:], segment.transpose(0, 1))  # [H, P]
        pair_decay = torch.exp((cum[:, :, None] - cum[:, None, :]) * segment + (segment - 1.0) * 1e4)
        k_beta = kp * beta_p
        kkt = torch.matmul(k_beta, kp.transpose(-1, -2)) * pair_decay
        attn_p = torch.matmul(qp, kp.transpose(-1, -2)) * pair_decay
        inv = _unit_lower_inverse(self.eye_p + kkt * (segment - self.eye_p), P, self.lane)
        decay = torch.exp(cum)[..., None]
        v_new = torch.matmul(inv, vp * beta_p) - torch.matmul(torch.matmul(inv, k_beta * decay), state)
        outs.append(torch.matmul(qp * decay, state) + torch.matmul(attn_p, v_new))

        core = torch.cat(outs, dim=1).transpose(0, 1)[None]  # [1, T, H, Dv]
        core = net.norm(core, z)
        return net.out_proj(core.reshape(1, T, H * Dv))

    def forward(self, hidden, cos, sin, valid, tail_onehot, segment, lag_keep, lag_tail, decide_onehot, option_onehot,
                option_mask):
        """hidden [1, S+P, D]; cos/sin [S+P, R]; valid [S]; tail_onehot [k-1, S]; segment [P, P]; lag_keep [k-1, P];
        lag_tail [k-1, P, k-1]; decide_onehot [B, P]; option_onehot [B, K, P]; option_mask [B, K]
        -> logits [B, K] (temperature applied), probabilities [B, K]."""
        S, P = self.S, self.P
        bias = torch.cat([
            torch.cat([self.state_causal, self.state_block], dim=1),
            torch.cat([((1.0 - valid) * MASK_VALUE)[None, :].expand(P, S), (1.0 - segment) * MASK_VALUE], dim=1),
        ], dim=0)
        x = hidden
        for layer in self.decoder.layers:
            h = layer.input_layernorm(x)
            if layer.is_linear:
                out = self._delta(layer.linear_attn, h, valid, tail_onehot, segment, lag_keep, lag_tail)
            else:
                out = self._attention(layer.self_attn, h, cos, sin, bias)
            x = x + out
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        xq = x[0, S:]
        decide = self.norm(torch.matmul(decide_onehot, xq))
        options = self.norm(torch.matmul(option_onehot, xq))
        logits = (self.k(options) * self.q(decide)[:, None, :]).sum(-1) * self.pointer_scale * self.inverse_temperature
        logits = torch.where(option_mask > 0.5, logits, torch.full_like(logits, MASK_VALUE))
        return logits, torch.softmax(logits, dim=-1)


def load_fused(merged, state_len: int, packed_len: int, batch: int, max_options: int, chunk_size: int = 64,
               lane: int = 128):
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    merged = Path(merged)
    meta = json.loads((merged / "meta.json").read_text())
    cfg = TextConfig(meta["text_config"])
    state = load_file(str(merged / "text.safetensors"))
    head = load_file(str(merged / "head.safetensors"))
    model = FusedPass(cfg, state_len, packed_len, batch, max_options, chunk_size, lane=lane)
    model.decoder.load_merged(state)
    model.norm.load_state_dict({"weight": state["norm.weight"]})
    model.q.load_state_dict({"weight": head["q.weight"], "bias": head["q.bias"]})
    model.k.load_state_dict({"weight": head["k.weight"], "bias": head["k.bias"]})
    model.pointer_scale.fill_(meta["pointer_scale"])
    model.inverse_temperature.fill_(1.0 / meta["temperature"])
    return model.eval()


def fused_inputs(cfg, embed, state_ids, question_rows, state_len, packed_len, batch, max_options, pad_id, lane=128):
    state_in, _ = stage_inputs(cfg, embed, state_ids, [], state_len, 32, 2, max_options, pad_id)
    s_hidden, s_cos, s_sin, valid, tail = state_in
    p_hidden, p_cos, p_sin, segment, keep, lag_tail, decide, options, mask = packed_inputs(
        cfg, embed, len(state_ids), question_rows, packed_len, batch, max_options, pad_id, lane=min(lane, packed_len))
    return (torch.cat([s_hidden, p_hidden], dim=1), torch.cat([s_cos, p_cos]), torch.cat([s_sin, p_sin]), valid, tail,
            segment, keep, lag_tail, decide, options, mask)


def pack_groups(lengths, packed_len, batch, lane=None):
    """Greedy packing (the order is kept) of question lengths into calls of `packed_len` tokens and `batch` readouts,
    no question crossing a lane. Returns lists of question indices."""
    groups, current, start = [], [], 0
    for i, n in enumerate(lengths):
        if n > (lane or packed_len):
            raise ValueError(f"question of {n} tokens exceeds the lane")
        placed = start
        if lane and placed // lane != (placed + n - 1) // lane:
            placed = (placed // lane + 1) * lane
        if current and (placed + n > packed_len or len(current) == batch):
            groups.append(current)
            current, placed = [], 0
        current.append(i)
        start = placed + n
    if current:
        groups.append(current)
    return groups
