"""ANE-canonical rewrite of one Zipformer2EncoderLayer.

Layout: x is (1, C, 1, S) fp32, S fixed. Every nn.Linear becomes a 1x1
Conv2d, norms/bypass operate on the channel axis, attention runs per-head
with S on the last axis of the softmax input, and the rel->abs positional
conversion is a constant-shape pad/reshape/slice skew (no gather).

Weights are imported (never re-initialized) from the POST
convert_scaled_to_non_scaled(is_onnx=True) module tree, so Balancer /
Whiten / Dropout are already Identity and the activations are the
SwooshL/ROnnx formulas.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _logaddexp0(x: Tensor) -> Tensor:
    # logaddexp(0, x) in the numerically-stable form used by
    # scaling.logaddexp_onnx: max(0,x) + log1p(exp(-|x|)).
    return torch.relu(x) + torch.log1p(torch.exp(-torch.abs(x)))


def swoosh_l(x: Tensor) -> Tensor:
    """SwooshL: logaddexp(0, x-4) - 0.08x - 0.035 (scaling.SwooshLOnnx)."""
    return _logaddexp0(x - 4.0) - 0.08 * x - 0.035


def swoosh_r(x: Tensor) -> Tensor:
    """SwooshR: logaddexp(0, x-1) - 0.08x - 0.313261687 (scaling.SwooshROnnx)."""
    return _logaddexp0(x - 1.0) - 0.08 * x - 0.313261687


def linear_to_conv2d(linear: nn.Linear) -> nn.Conv2d:
    """nn.Linear(in, out) -> nn.Conv2d(in, out, 1) with identical weights."""
    out_f, in_f = linear.weight.shape
    conv = nn.Conv2d(in_f, out_f, 1, bias=linear.bias is not None)
    with torch.no_grad():
        conv.weight.copy_(linear.weight.reshape(out_f, in_f, 1, 1))
        if linear.bias is not None:
            conv.bias.copy_(linear.bias)
    return conv


def _adl_to_conv2d(adl) -> nn.Conv2d:
    """ActivationDropoutAndLinear's linear part -> 1x1 Conv2d."""
    out_f, in_f = adl.weight.shape
    conv = nn.Conv2d(in_f, out_f, 1, bias=adl.bias is not None)
    with torch.no_grad():
        conv.weight.copy_(adl.weight.reshape(out_f, in_f, 1, 1))
        if adl.bias is not None:
            conv.bias.copy_(adl.bias)
    return conv


class AneBiasNorm(nn.Module):
    """scaling.BiasNorm over the channel axis of (1, C, 1, S)."""

    def __init__(self, norm):
        super().__init__()
        c = norm.bias.numel()
        self.register_buffer("bias", norm.bias.detach().clone().reshape(1, c, 1, 1))
        self.register_buffer("scale", norm.log_scale.detach().exp().clone())

    def forward(self, x: Tensor) -> Tensor:
        # rsqrt + explicit square: ANE has no pow op.
        d = x - self.bias
        scales = torch.rsqrt(torch.mean(d * d, dim=1, keepdim=True))
        return x * scales * self.scale


class AneBypass(nn.Module):
    """scaling BypassModule (eval path): src_orig + (src - src_orig) * scale."""

    def __init__(self, bypass):
        super().__init__()
        c = bypass.bypass_scale.numel()
        self.register_buffer(
            "bypass_scale", bypass.bypass_scale.detach().clone().reshape(1, c, 1, 1)
        )

    def forward(self, src_orig: Tensor, src: Tensor) -> Tensor:
        return src_orig + (src - src_orig) * self.bypass_scale


class AneFeedforward(nn.Module):
    """FeedforwardModule: 1x1 conv -> SwooshL -> 1x1 conv."""

    def __init__(self, ff):
        super().__init__()
        self.in_proj = linear_to_conv2d(ff.in_proj)
        assert ff.out_proj.activation == "SwooshL"
        self.out_proj = _adl_to_conv2d(ff.out_proj)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_proj(swoosh_l(self.in_proj(x)))


class AneAttentionWeights(nn.Module):
    """RelPositionMultiheadAttentionWeights on (1, C, 1, S).

    Returns per-head attention weights of shape (H, S_q, 1, S_k), softmax
    over the last (S_k) axis.

    Rel->abs positional handling: linear_pos(pos_emb) is folded eagerly and
    the relative->absolute reindexing (out[q, j] = rel[q, S-1-q+j]) is baked
    into a constant buffer pos_abs[h, c, q, j] = pos_proj[h, c, S-1-q+j],
    so at runtime pos_scores = sum_c p[h,c,q] * pos_abs[h,c,q,j] — one
    broadcast multiply + channel reduce, no gather/as_strided and no >16K
    flatten (which the ANE cannot tile; the pad+reshape+slice skew trick
    needs a 2*S*S flat axis and falls back to CPU).
    """

    def __init__(self, saw, pos_emb: Tensor, seq_len: int):
        super().__init__()
        self.num_heads = saw.num_heads
        self.query_head_dim = saw.query_head_dim
        self.pos_head_dim = saw.pos_head_dim
        self.seq_len = seq_len
        self.in_proj = linear_to_conv2d(saw.in_proj)

        # pos_emb: (1, 2S-1, pos_dim). Fold linear_pos eagerly:
        # (1, 2S-1, H*phd) -> (H, phd, 2S-1), feature index = h*phd + d.
        with torch.no_grad():
            pe = F.linear(pos_emb, saw.linear_pos.weight, saw.linear_pos.bias)
        n = 2 * seq_len - 1
        assert pe.shape == (1, n, self.num_heads * self.pos_head_dim)
        pe = pe.reshape(n, self.num_heads, self.pos_head_dim).permute(1, 2, 0)
        # Absolute-indexed constant: (H, phd, S_q, S_k).
        idx = (seq_len - 1) - torch.arange(seq_len).unsqueeze(1) + torch.arange(seq_len)
        self.register_buffer("pos_abs", pe[:, :, idx].contiguous())

    def forward(self, x: Tensor) -> Tensor:
        h = self.num_heads
        qd = self.query_head_dim
        pd = self.pos_head_dim
        s = self.seq_len

        proj = self.in_proj(x)  # (1, (2*qd+pd)*H, 1, S)
        q = proj[:, : h * qd].reshape(h, qd, 1, s)
        k = proj[:, h * qd : 2 * h * qd].reshape(h, qd, 1, s)
        p = proj[:, 2 * h * qd :].reshape(h, pd, 1, s)

        # (H, S_q, 1, S_k): contract channel axis, keep S on the last axis.
        attn_scores = torch.einsum("hcoq,hcok->hqok", q, k)
        # p: (H, phd, 1, S_q) -> (H, phd, S_q, 1); broadcast over S_k, reduce phd.
        pos_scores = (p.permute(0, 1, 3, 2) * self.pos_abs).sum(dim=1, keepdim=True)
        # (H, 1, S_q, S_k) -> (H, S_q, 1, S_k): layout-preserving reshape.
        attn_scores = attn_scores + pos_scores.reshape(h, s, 1, s)
        return attn_scores.softmax(dim=-1)


class AneSelfAttention(nn.Module):
    """SelfAttention (apply precomputed weights) on (1, C, 1, S)."""

    def __init__(self, sa, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.in_proj = linear_to_conv2d(sa.in_proj)
        self.out_proj = linear_to_conv2d(sa.out_proj)
        self.value_head_dim = sa.in_proj.out_features // num_heads

    def forward(self, x: Tensor, attn_weights: Tensor) -> Tensor:
        h = self.num_heads
        vd = self.value_head_dim
        s = x.shape[-1]
        v = self.in_proj(x).reshape(h, vd, 1, s)  # feature index = h*vd + c
        out = torch.einsum("hqok,hcok->hcoq", attn_weights, v)
        out = out.reshape(1, h * vd, 1, s)
        return self.out_proj(out)


class AneNonlinAttention(nn.Module):
    """NonlinAttention on (1, C, 1, S); consumes head-0 attention weights."""

    def __init__(self, na):
        super().__init__()
        self.hidden_channels = na.hidden_channels
        self.in_proj = linear_to_conv2d(na.in_proj)
        self.out_proj = linear_to_conv2d(na.out_proj)

    def forward(self, x: Tensor, attn_weights: Tensor) -> Tensor:
        # attn_weights: (1, S, 1, S) — head 0 only.
        hc = self.hidden_channels
        x = self.in_proj(x)  # (1, 3*hc, 1, S)
        s_gate = torch.tanh(x[:, :hc])
        v = x[:, hc : 2 * hc] * s_gate
        y = x[:, 2 * hc :]
        v = torch.einsum("bqok,bcok->bcoq", attn_weights, v)
        return self.out_proj(v * y)


class AneConvolutionModule(nn.Module):
    """ConvolutionModule: 1x1 conv -> GLU-ish gate -> depthwise (1,31) conv
    -> SwooshR -> 1x1 conv, all on (1, C, 1, S).

    The ANE rejects grouped convs with kernel width > 15 ("large kernel
    size"; empirically 16 falls back, 15 places). The kernel-31 depthwise
    conv is split exactly into taps 0..14 + 15..29 (kernel 15 each) + tap
    30 (kernel 1):
        y[t] = sum_{k=0..30} w[k] x[t+k-15] = yA[t] + yB[t] + yC[t]
    Asymmetric offsets are realized with the convs' own symmetric padding
    plus output slices (standalone pad ops are not ANE-placeable and drag
    the convs onto CPU with them).
    """

    def __init__(self, cm):
        super().__init__()
        self.in_proj = linear_to_conv2d(cm.in_proj)
        dw = cm.depthwise_conv  # Conv1d(C, C, 31, groups=C, padding=15)
        c, _, k = dw.weight.shape
        assert k == 31
        self.channels = c
        self.dw_a = nn.Conv2d(
            c, c, (1, 15), groups=c, padding=(0, 15), bias=dw.bias is not None
        )
        self.dw_b = nn.Conv2d(c, c, (1, 15), groups=c, padding=(0, 15), bias=False)
        self.dw_c = nn.Conv2d(c, c, (1, 1), groups=c, padding=(0, 15), bias=False)
        with torch.no_grad():
            self.dw_a.weight.copy_(dw.weight[:, :, :15].reshape(c, 1, 1, 15))
            self.dw_b.weight.copy_(dw.weight[:, :, 15:30].reshape(c, 1, 1, 15))
            self.dw_c.weight.copy_(dw.weight[:, :, 30:].reshape(c, 1, 1, 1))
            if dw.bias is not None:
                self.dw_a.bias.copy_(dw.bias)
        assert cm.out_proj.activation == "SwooshR"
        self.out_proj = _adl_to_conv2d(cm.out_proj)

    def forward(self, x: Tensor) -> Tensor:
        c = self.channels
        s = x.shape[-1]
        x = self.in_proj(x)  # (1, 2C, 1, S)
        x = x[:, :c] * torch.sigmoid(x[:, c:])
        # With padding p, output index i covers window starting at x[i-p].
        # A taps x[t-15..t-1]: start t-15 => i=t       => [:S]
        # B taps x[t..t+14]:   start t    => i=t+15    => [15:15+S]
        # C tap  x[t+15]:      start t+15 => i=t+30    => [30:30+S]
        ya = self.dw_a(x)[..., :s]
        yb = self.dw_b(x)[..., 15 : 15 + s]
        yc = self.dw_c(x)[..., 30 : 30 + s]
        return self.out_proj(swoosh_r(ya + yb + yc))


class AneZipformerLayer(nn.Module):
    """Numerically-exact ANE-canonical Zipformer2EncoderLayer.

    forward(x, time_emb): x (1, C, 1, S), time_emb (1, C, 1, 1) — the
    already-projected per-timestep embedding the original layer receives
    (added to src at three points, broadcast over S).
    """

    def __init__(self, layer, pos_emb: Tensor, seq_len: int):
        super().__init__()
        self.self_attn_weights = AneAttentionWeights(
            layer.self_attn_weights, pos_emb, seq_len
        )
        h = layer.self_attn_weights.num_heads
        self.self_attn1 = AneSelfAttention(layer.self_attn1, h)
        self.self_attn2 = AneSelfAttention(layer.self_attn2, h)
        self.feed_forward1 = AneFeedforward(layer.feed_forward1)
        self.feed_forward2 = AneFeedforward(layer.feed_forward2)
        self.feed_forward3 = AneFeedforward(layer.feed_forward3)
        self.nonlin_attention = AneNonlinAttention(layer.nonlin_attention)
        self.conv_module1 = AneConvolutionModule(layer.conv_module1)
        self.conv_module2 = AneConvolutionModule(layer.conv_module2)
        self.norm = AneBiasNorm(layer.norm)
        self.bypass = AneBypass(layer.bypass)
        self.bypass_mid = AneBypass(layer.bypass_mid)

    def forward(self, x: Tensor, time_emb: Tensor) -> Tensor:
        src_orig = x
        attn_weights = self.self_attn_weights(x)  # (H, S, 1, S)

        x = x + time_emb
        x = x + self.feed_forward1(x)
        x = x + self.nonlin_attention(x, attn_weights[0:1])
        x = x + self.self_attn1(x, attn_weights)

        x = x + time_emb
        x = x + self.conv_module1(x)
        x = x + self.feed_forward2(x)
        x = self.bypass_mid(src_orig, x)

        x = x + self.self_attn2(x, attn_weights)
        x = x + time_emb
        x = x + self.conv_module2(x)
        x = x + self.feed_forward3(x)

        x = self.norm(x)
        return self.bypass(src_orig, x)


def tbc_to_ane(x: Tensor) -> Tensor:
    """(T, B=1, C) -> (1, C, 1, S=T)."""
    return x.permute(1, 2, 0).unsqueeze(2).contiguous()


def ane_to_tbc(x: Tensor) -> Tensor:
    """(1, C, 1, S) -> (S, 1, C)."""
    return x.squeeze(2).permute(2, 0, 1).contiguous()
