"""ANE-canonical rewrite of the full TTSZipformer fm_decoder.

Layout: activations are (1, C, 1, S) fp32, S fixed. Reuses the submodules of
coreml.ane.layer (1x1-conv linears, channel-axis BiasNorm/Bypass, per-head
attention with S on the softmax axis, split depthwise convs) and adds the
whole-decoder plumbing: in/out projections, the t + guidance_scale embedding
path, per-stack time projections, SimpleDownsample/SimpleUpsample in
channel-first form, stack out-combiners, and float padding-mask support
(additive -1000 attention bias + zeroing before the depthwise convs, exactly
mirroring upstream masked_fill semantics).

pos_abs sharing: linear_pos weights are NOT shared across layers, so the
folded per-layer (H, phd, S, S) constant of the single-layer trial would cost
~260 MB decoder-wide. Instead, every layer's posproj = PE @ W_l^T lies in the
column space of the (2S-1, 48) positional encoding PE, and the SVD of the
concatenated per-layer posproj matrices shows numerical rank ~27. We build
ONE orthonormal basis U_R (R=32, reconstruction ~1e-7 relative) per distinct
sequence length, bake pos_basis[r, q, j] = U_R[S-1-q+j, r] as a shared
constant, and fold the per-layer basis coefficients A_l = U_R^T posproj_l
into that layer's attention in_proj (p block becomes H*R channels). This
keeps the no-gather broadcast-mul+reduce form of the layer trial while
sharing 3 buffers across all 16 layers (~88 MB fp16 total).

Weights are imported from the POST convert_scaled_to_non_scaled module tree.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from coreml.ane.layer import (
    AneBiasNorm,
    AneBypass,
    AneFeedforward,
    AneNonlinAttention,
    AneSelfAttention,
    _adl_to_conv2d,
    linear_to_conv2d,
    swoosh_r,
)

POS_RANK = 32


def build_pos_basis(pos_emb: Tensor, layers, rank: int = POS_RANK):
    """Shared rel->abs positional basis for all layers at one seq len.

    pos_emb: (1, 2S-1, pos_dim) from the (parameter-free) encoder_pos.
    layers: every Zipformer2EncoderLayer that runs at this seq len.

    Returns (U (2S-1, R) float64, coeffs {id(layer): A_l (R, H*phd) float64}).
    posproj_l ~= U @ A_l to ~1e-7 relative; U is rescaled so that basis and
    coefficient magnitudes match (fp16 balance in the converted model).
    """
    pe = pos_emb.squeeze(0).double()  # (2S-1, pos_dim)
    projs = [
        F.linear(pe, l.self_attn_weights.linear_pos.weight.double()) for l in layers
    ]
    M = torch.cat(projs, dim=1)  # (2S-1, 16*L)
    U, _, _ = torch.linalg.svd(M, full_matrices=False)
    U = U[:, :rank]  # (2S-1, R)
    coeffs = {id(l): U.T @ p for l, p in zip(layers, projs)}  # (R, H*phd)
    # Balance magnitudes: |c*U| == |A/c| at the max.
    amax = max(a.abs().max().item() for a in coeffs.values())
    umax = U.abs().max().item()
    c = (amax / umax) ** 0.5
    U = U * c
    coeffs = {k: a / c for k, a in coeffs.items()}
    # Report reconstruction error (should be ~1e-7 relative).
    rec = torch.cat([U @ coeffs[id(l)] for l in layers], dim=1)
    err = (M - rec).abs().max().item() / M.abs().max().item()
    return U, coeffs, err


def pos_basis_buffer(U: Tensor, seq_len: int) -> Tensor:
    """pos_basis[1, r, q, j] = U[S-1-q+j, r] (rel->abs baked, no gather)."""
    idx = (seq_len - 1) - torch.arange(seq_len).unsqueeze(1) + torch.arange(seq_len)
    return U[idx].permute(2, 0, 1).unsqueeze(0).contiguous().float()  # (1, R, S, S)


class AneSharedPosAttentionWeights(nn.Module):
    """RelPositionMultiheadAttentionWeights with the shared positional basis.

    The layer's own linear_pos is folded away: the p block of in_proj is
    replaced by H*R channels producing ptilde[h, r, q] = sum_c A[r, h*phd+c]
    * p[h, c, q], and pos_scores[h, q, j] = sum_r ptilde[h, r, q]
    * pos_basis[r, q, j]. pos_basis is passed in at forward time so the
    traced graph holds ONE constant shared by every layer at that seq len.
    """

    def __init__(self, saw, coeff: Tensor, seq_len: int):
        super().__init__()
        h = saw.num_heads
        qd = saw.query_head_dim
        pd = saw.pos_head_dim
        r = coeff.shape[0]
        self.num_heads = h
        self.query_head_dim = qd
        self.rank = r
        self.seq_len = seq_len

        w = saw.in_proj.weight.double()  # (2*h*qd + h*pd, E)
        b = saw.in_proj.bias.double()
        qk = 2 * h * qd
        wp = w[qk:].reshape(h, pd, -1)  # (h, pd, E)
        bp = b[qk:].reshape(h, pd)
        a = coeff.reshape(r, h, pd).permute(1, 0, 2)  # (h, R, pd)
        wpt = torch.einsum("hrc,hce->hre", a, wp).reshape(h * r, -1)
        bpt = torch.einsum("hrc,hc->hr", a, bp).reshape(h * r)

        e = w.shape[1]
        self.in_proj = nn.Conv2d(e, qk + h * r, 1, bias=True)
        with torch.no_grad():
            self.in_proj.weight.copy_(
                torch.cat([w[:qk], wpt], dim=0).float().reshape(qk + h * r, e, 1, 1)
            )
            self.in_proj.bias.copy_(torch.cat([b[:qk], bpt], dim=0).float())

    def forward(self, x: Tensor, pos_basis: Tensor, attn_bias: Tensor) -> Tensor:
        h = self.num_heads
        qd = self.query_head_dim
        r = self.rank
        s = self.seq_len

        proj = self.in_proj(x)  # (1, 2*h*qd + h*R, 1, S)
        q = proj[:, : h * qd].reshape(h, qd, 1, s)
        k = proj[:, h * qd : 2 * h * qd].reshape(h, qd, 1, s)
        pt = proj[:, 2 * h * qd :].reshape(h, r, 1, s)

        attn_scores = torch.einsum("hcoq,hcok->hqok", q, k)  # (H, S_q, 1, S_k)
        # (h, R, S_q, 1) * (1, R, S_q, S_k) -> reduce R -> (h, 1, S_q, S_k)
        pos_scores = (pt.permute(0, 1, 3, 2) * pos_basis).sum(dim=1, keepdim=True)
        attn_scores = attn_scores + pos_scores.reshape(h, s, 1, s)
        # attn_bias: (1, 1, 1, S_k) float, -1000 at padded keys (as upstream).
        attn_scores = attn_scores + attn_bias
        return attn_scores.softmax(dim=-1)


class AneConvModule(nn.Module):
    """ConvolutionModule for kernel sizes 7/15/31 with padding-mask zeroing.

    k <= 15 places directly on ANE; k=31 splits into 15+15+1 exactly as in
    coreml.ane.layer.AneConvolutionModule.
    """

    def __init__(self, cm):
        super().__init__()
        self.in_proj = linear_to_conv2d(cm.in_proj)
        dw = cm.depthwise_conv  # Conv1d(C, C, k, groups=C, padding=k//2)
        c, _, k = dw.weight.shape
        self.channels = c
        self.kernel = k
        if k <= 15:
            self.dw = nn.Conv2d(c, c, (1, k), groups=c, padding=(0, k // 2), bias=True)
            with torch.no_grad():
                self.dw.weight.copy_(dw.weight.reshape(c, 1, 1, k))
                self.dw.bias.copy_(dw.bias)
        else:
            assert k == 31, k
            self.dw_a = nn.Conv2d(c, c, (1, 15), groups=c, padding=(0, 15), bias=True)
            self.dw_b = nn.Conv2d(c, c, (1, 15), groups=c, padding=(0, 15), bias=False)
            self.dw_c = nn.Conv2d(c, c, (1, 1), groups=c, padding=(0, 15), bias=False)
            with torch.no_grad():
                self.dw_a.weight.copy_(dw.weight[:, :, :15].reshape(c, 1, 1, 15))
                self.dw_b.weight.copy_(dw.weight[:, :, 15:30].reshape(c, 1, 1, 15))
                self.dw_c.weight.copy_(dw.weight[:, :, 30:].reshape(c, 1, 1, 1))
                self.dw_a.bias.copy_(dw.bias)
        assert cm.out_proj.activation == "SwooshR"
        self.out_proj = _adl_to_conv2d(cm.out_proj)

    def forward(self, x: Tensor, conv_keep: Tensor) -> Tensor:
        c = self.channels
        s = x.shape[-1]
        x = self.in_proj(x)  # (1, 2C, 1, S)
        x = x[:, :c] * torch.sigmoid(x[:, c:])
        # Upstream zeroes padded frames before the depthwise conv.
        x = x * conv_keep
        if self.kernel <= 15:
            y = self.dw(x)
        else:
            y = (
                self.dw_a(x)[..., :s]
                + self.dw_b(x)[..., 15 : 15 + s]
                + self.dw_c(x)[..., 30 : 30 + s]
            )
        return self.out_proj(swoosh_r(y))


class AneDecoderLayer(nn.Module):
    """Zipformer2EncoderLayer with shared positional basis + mask support."""

    def __init__(self, layer, coeff: Tensor, seq_len: int):
        super().__init__()
        self.self_attn_weights = AneSharedPosAttentionWeights(
            layer.self_attn_weights, coeff, seq_len
        )
        h = layer.self_attn_weights.num_heads
        self.self_attn1 = AneSelfAttention(layer.self_attn1, h)
        self.self_attn2 = AneSelfAttention(layer.self_attn2, h)
        self.feed_forward1 = AneFeedforward(layer.feed_forward1)
        self.feed_forward2 = AneFeedforward(layer.feed_forward2)
        self.feed_forward3 = AneFeedforward(layer.feed_forward3)
        self.nonlin_attention = AneNonlinAttention(layer.nonlin_attention)
        self.conv_module1 = AneConvModule(layer.conv_module1)
        self.conv_module2 = AneConvModule(layer.conv_module2)
        self.norm = AneBiasNorm(layer.norm)
        self.bypass = AneBypass(layer.bypass)
        self.bypass_mid = AneBypass(layer.bypass_mid)

    def forward(
        self,
        x: Tensor,
        time_emb: Tensor,
        pos_basis: Tensor,
        attn_bias: Tensor,
        conv_keep: Tensor,
    ) -> Tensor:
        src_orig = x
        attn_weights = self.self_attn_weights(x, pos_basis, attn_bias)  # (H, S, 1, S)

        x = x + time_emb
        x = x + self.feed_forward1(x)
        x = x + self.nonlin_attention(x, attn_weights[0:1])
        x = x + self.self_attn1(x, attn_weights)

        x = x + time_emb
        x = x + self.conv_module1(x, conv_keep)
        x = x + self.feed_forward2(x)
        x = self.bypass_mid(src_orig, x)

        x = x + self.self_attn2(x, attn_weights)
        x = x + time_emb
        x = x + self.conv_module2(x, conv_keep)
        x = x + self.feed_forward3(x)

        x = self.norm(x)
        return self.bypass(src_orig, x)


class AneEncoderStack(nn.Module):
    """Zipformer2Encoder: per-stack time projection + layer chain."""

    def __init__(self, enc, coeffs: dict, seq_len: int):
        super().__init__()
        # enc.time_emb = Sequential(SwooshROnnx, Linear(time_embed_dim, E))
        self.time_proj = linear_to_conv2d(enc.time_emb[1])
        self.layers = nn.ModuleList(
            AneDecoderLayer(l, coeffs[id(l)], seq_len) for l in enc.layers
        )

    def forward(
        self,
        x: Tensor,
        time_emb: Tensor,
        pos_basis: Tensor,
        attn_bias: Tensor,
        conv_keep: Tensor,
    ) -> Tensor:
        te = self.time_proj(swoosh_r(time_emb))  # (1, E, 1, 1)
        for layer in self.layers:
            x = layer(x, te, pos_basis, attn_bias, conv_keep)
        return x


class AneDownsampledStack(nn.Module):
    """DownsampledZipformer2Encoder: weighted-sum downsample (depthwise
    strided conv), inner stack at S/ds, nearest-repeat upsample, bypass
    out-combiner. seq_len must be divisible by ds (1024 is, for ds in 2/4)."""

    def __init__(self, enc, coeffs: dict, seq_len: int):
        super().__init__()
        ds = enc.downsample_factor
        self.ds = ds
        c = enc.out_combiner.bypass_scale.numel()
        w = enc.downsample.bias.softmax(dim=0)  # (ds,)
        self.down = nn.Conv2d(c, c, (1, ds), stride=(1, ds), groups=c, bias=False)
        with torch.no_grad():
            self.down.weight.copy_(w.reshape(1, 1, 1, ds).expand(c, 1, 1, ds))
        self.stack = AneEncoderStack(enc.encoder, coeffs, seq_len // ds)
        self.out_combiner = AneBypass(enc.out_combiner)

    def forward(
        self,
        x: Tensor,
        time_emb: Tensor,
        pos_basis: Tensor,
        attn_bias: Tensor,
        conv_keep: Tensor,
    ) -> Tensor:
        y = self.down(x)  # (1, C, 1, S/ds)
        y = self.stack(y, time_emb, pos_basis, attn_bias, conv_keep)
        y = F.interpolate(y, scale_factor=(1.0, float(self.ds)), mode="nearest")
        return self.out_combiner(x, y)


class AneFmDecoder(nn.Module):
    """Full TTSZipformer fm_decoder, ANE-canonical.

    forward(t, x, guidance_scale, mask):
      t, guidance_scale: (1,) runtime scalars
      x: (1, in_dim=300, 1, S) pre-concatenated [xt, text_cond, speech_cond]
      mask: (1, 1, 1, S) float, 1.0 = padded
    returns velocity (1, out_dim=100, 1, S).
    """

    def __init__(self, fm, seq_len: int = 1024, rank: int = POS_RANK):
        super().__init__()
        self.seq_len = seq_len
        self.dsf = list(fm.downsampling_factor)
        self.in_proj = linear_to_conv2d(fm.in_proj)
        self.out_proj = linear_to_conv2d(fm.out_proj)

        # --- t / guidance_scale embedding (timestep_embedding + MLPs) ---
        dim = fm.time_embed_dim
        assert dim % 2 == 0 and fm.guidance_scale_embed_dim == dim
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0)) * torch.arange(half).float() / half
        )
        self.register_buffer("freqs", freqs.reshape(1, half, 1, 1))
        self.guid_proj = linear_to_conv2d(fm.guidance_scale_embed)
        self.time_lin1 = linear_to_conv2d(fm.time_embed[0])
        self.time_lin2 = linear_to_conv2d(fm.time_embed[2])

        # --- shared positional bases per distinct (downsampled) seq len ---
        by_len = {}
        for i, enc in enumerate(fm.encoders):
            sl = seq_len // self.dsf[i]
            inner = enc.encoder if hasattr(enc, "encoder") else enc
            by_len.setdefault(sl, []).extend(inner.layers)
        self.basis_errs = {}
        coeffs = {}
        for sl, layers in by_len.items():
            inner = next(
                (e.encoder if hasattr(e, "encoder") else e)
                for i, e in enumerate(fm.encoders)
                if seq_len // self.dsf[i] == sl
            )
            with torch.no_grad():
                pos_emb = inner.encoder_pos(torch.zeros(sl, 1, 2)).detach()
            u, cf, err = build_pos_basis(pos_emb, layers, rank)
            self.register_buffer(f"pos_basis_{sl}", pos_basis_buffer(u, sl))
            coeffs.update(cf)
            self.basis_errs[sl] = err

        # --- stacks ---
        stacks = []
        for i, enc in enumerate(fm.encoders):
            if self.dsf[i] == 1:
                stacks.append(AneEncoderStack(enc, coeffs, seq_len))
            else:
                stacks.append(AneDownsampledStack(enc, coeffs, seq_len))
        self.stacks = nn.ModuleList(stacks)

    def _time_embed(self, t: Tensor, guidance_scale: Tensor) -> Tensor:
        targs = t.reshape(1, 1, 1, 1) * self.freqs
        te = torch.cat([targs.cos(), targs.sin()], dim=1)  # (1, dim, 1, 1)
        gargs = guidance_scale.reshape(1, 1, 1, 1) * self.freqs
        ge = torch.cat([gargs.cos(), gargs.sin()], dim=1)
        emb = te + self.guid_proj(ge)
        return self.time_lin2(swoosh_r(self.time_lin1(emb)))

    def forward(
        self, t: Tensor, x: Tensor, guidance_scale: Tensor, mask: Tensor
    ) -> Tensor:
        time_emb = self._time_embed(t, guidance_scale)  # (1, dim, 1, 1)

        bias = {1: mask * -1000.0}
        keep = {1: 1.0 - mask}
        for ds in sorted({d for d in self.dsf if d != 1}):
            bias[ds] = bias[1][..., ::ds]
            keep[ds] = keep[1][..., ::ds]

        h = self.in_proj(x)
        for i, stack in enumerate(self.stacks):
            ds = self.dsf[i]
            pb = getattr(self, f"pos_basis_{self.seq_len // ds}")
            h = stack(h, time_emb, pb, bias[ds], keep[ds])
        return self.out_proj(h)


class AneFmDecoderIO(nn.Module):
    """Drop-in wrapper with the SAME I/O contract as the original FmDecoder:
    t (1,), x/text_condition/speech_condition (1, S, 100), guidance_scale
    (1,), padding_mask (1, S) float 1.0 = padded -> v (1, S, 100)."""

    def __init__(self, core: AneFmDecoder):
        super().__init__()
        self.core = core

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: Tensor,
        guidance_scale: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        s = self.core.seq_len
        xt = torch.cat([x, text_condition, speech_condition], dim=2)  # (1, S, 300)
        h = xt.permute(0, 2, 1).unsqueeze(2)  # (1, 300, 1, S)
        mask = padding_mask.reshape(1, 1, 1, s)
        v = self.core(t, h, guidance_scale, mask)  # (1, 100, 1, S)
        return v.squeeze(2).permute(0, 2, 1)
