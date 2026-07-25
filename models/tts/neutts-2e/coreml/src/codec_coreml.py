"""CoreML-friendly NeuCodec decoder (FSQ codes → 24 kHz audio).

Reimplements only the ops coremltools cannot convert or that bake the
sequence length at trace time; everything else reuses the loaded neucodec
modules directly:

  * FSQ dequant       — reimplemented (base-4 digit decomposition; the
                        vector-quantize-pytorch path is einops/int heavy)
  * transformer RoPE  — upstream misuses torchtune RoPE on [b, h, t, d]
                        (rotation by head index, constant over time); replicated
                        with fixed per-head buffers (see AttentionRope)
  * ISTFT             — irfft/complex → real IDFT matmul (1x1 convs) +
                        overlap-add via ConvTranspose1d, "same"-padding trim

Reused as-is: embed conv, prior/post ResnetBlocks, attention/MLP weights,
final LayerNorm, fc_post_a, ISTFTHead.out linear.

Wrapper I/O:
    codes:    [1, T] int32 NeuCodec indices
    → audio:  [1, T * 480] fp32 @ 24 kHz

T is flexible (RangeDim): every reimplemented op is length-agnostic.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HOP = 480
N_FFT = 1920
ROPE_DIM = 64  # transformer head dim (pos_meb_dim)
ROPE_BASE = 10_000


class FSQDequant(nn.Module):
    """indices [1, T] → embeddings [1, T, 2048], matching
    ``ResidualFSQ.get_output_from_indices`` for levels=[4]*8, 1 quantizer."""

    def __init__(self, rfsq: nn.Module):
        super().__init__()
        levels = [4] * 8
        # Precompute all 65536 dequantized code vectors as an embedding table.
        # Digit arithmetic (floor(code/basis) % level) is exact in fp32 but the
        # fp16 compute pass corrupts integers > 2048; a gather is precision-safe.
        basis = [1]
        for lv in levels[:-1]:
            basis.append(basis[-1] * lv)
        idx = torch.arange(65_536, dtype=torch.float64).unsqueeze(-1)
        digits = torch.floor(idx / torch.tensor(basis, dtype=torch.float64)) % torch.tensor(
            levels, dtype=torch.float64
        )
        # vector-quantize-pytorch: half_width = level // 2 (integer), so level 4
        # dequantizes digits {0..3} to {-1, -0.5, 0, 0.5}.
        half = torch.tensor([lv // 2 for lv in levels], dtype=torch.float64)
        vals = (digits - half) / half
        # ResidualFSQ per-quantizer, per-dim scales (quantizer 0 only here).
        scales = rfsq.scales[0].detach().double().reshape(1, -1) if hasattr(rfsq, "scales") \
            else torch.ones(1, len(levels), dtype=torch.float64)
        self.table = nn.Embedding(65_536, 8)
        with torch.no_grad():
            self.table.weight.copy_((vals * scales).float())
        self.project_out = nn.Linear(8, 2048)
        with torch.no_grad():
            self.project_out.weight.copy_(rfsq.project_out.weight.float())
            self.project_out.bias.copy_(rfsq.project_out.bias.float())

        self._verify(rfsq)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.project_out(self.table(codes.to(torch.long)))  # [1, T, 2048]

    @torch.no_grad()
    def _verify(self, rfsq: nn.Module) -> None:
        idx = torch.randint(0, 65_536, (1, 173))
        want = rfsq.get_output_from_indices(idx.unsqueeze(-1))  # [1, T, 2048]
        got = self.forward(idx.to(torch.int32))
        diff = (got - want.float()).abs().max().item()
        if diff > 1e-4:
            raise RuntimeError(f"FSQDequant mismatch vs vector-quantize-pytorch: {diff}")


class AttentionRope(nn.Module):
    """bs_roformer5.Attention, replicating its RoPE quirk.

    Upstream calls torchtune's RotaryPositionalEmbeddings (which expects
    ``[b, s, n_h, d]``) on tensors shaped ``[b, n_h, t, d]``, so the "position"
    that gets rotated is the HEAD INDEX — a constant per head, identical at
    every timestep. The pretrained weights bake this in, so we replicate it
    with fixed [1, H, 1, D/2] cos/sin buffers. Upshot: no time-dependent
    tables, and the sequence length stays fully flexible.
    """

    def __init__(self, hf_attn: nn.Module, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.c_attn = hf_attn.c_attn
        self.c_proj = hf_attn.c_proj

        theta = 1.0 / (
            ROPE_BASE ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        pos = torch.arange(n_heads, dtype=torch.float32)  # head index as "position"
        freqs = torch.outer(pos, theta)  # [H, D/2]
        self.register_buffer("head_cos", freqs.cos().view(1, n_heads, 1, head_dim // 2))
        self.register_buffer("head_sin", freqs.sin().view(1, n_heads, 1, head_dim // 2))

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]; rotate interleaved pairs by per-head constants.
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.stack(
            [x1 * self.head_cos - x2 * self.head_sin, x2 * self.head_cos + x1 * self.head_sin],
            dim=-1,
        )
        return out.flatten(-2)

    def forward(self, x):
        B = 1
        H, D = self.n_heads, self.head_dim
        C = H * D
        qkv = self.c_attn(x)  # [1, T, 3*C]
        q, k, v = qkv.split(C, dim=-1)
        q = q.reshape(B, -1, H, D).transpose(1, 2)
        k = k.reshape(B, -1, H, D).transpose(1, 2)
        v = v.reshape(B, -1, H, D).transpose(1, 2)
        q = self._rope(q)
        k = self._rope(k)
        y = F.scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).reshape(B, -1, C)
        return self.c_proj(y)


class TransformerBlockRope(nn.Module):
    def __init__(self, hf_block: nn.Module, n_heads: int, head_dim: int):
        super().__init__()
        self.att_norm = hf_block.att_norm
        self.ffn_norm = hf_block.ffn_norm
        self.att = AttentionRope(hf_block.att, n_heads, head_dim)
        self.mlp = hf_block.mlp

    def forward(self, x):
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class ISTFTSame(nn.Module):
    """Vocos "same"-padding ISTFT via IDFT matmul + ConvTranspose1d overlap-add."""

    def __init__(self):
        super().__init__()
        n_bins = N_FFT // 2 + 1
        n = np.arange(N_FFT)[:, None]
        k = np.arange(n_bins)[None, :]
        scale = np.ones(n_bins)
        scale[1:-1] = 2.0  # hermitian doubling except DC and Nyquist
        wr = (scale * np.cos(2 * np.pi * n * k / N_FFT) / N_FFT).astype(np.float32)
        wi = (-scale * np.sin(2 * np.pi * n * k / N_FFT) / N_FFT).astype(np.float32)
        window = torch.hann_window(N_FFT)

        # IDFT as 1x1 convs: [B, n_bins, T] → [B, N_FFT, T], window folded in.
        w = window.numpy()[:, None]
        self.register_buffer("idft_real", torch.from_numpy((wr * w))[:, :, None])
        self.register_buffer("idft_imag", torch.from_numpy((wi * w))[:, :, None])
        # Overlap-add: ConvTranspose1d, in=N_FFT, out=1, kernel=N_FFT, stride=HOP.
        # Channel c contributes its value at kernel offset c: kernel[c, 0, c] = 1.
        eye = torch.eye(N_FFT).reshape(N_FFT, 1, N_FFT)
        self.register_buffer("ola_kernel", eye)
        self.register_buffer("win_sq", (window * window).reshape(1, 1, N_FFT))
        self.pad = (N_FFT - HOP) // 2

        self._verify()

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        # real/imag: [B, n_bins, T]
        frames = F.conv1d(real, self.idft_real) + F.conv1d(imag, self.idft_imag)
        audio = F.conv_transpose1d(frames, self.ola_kernel, stride=HOP)  # [B, 1, L]
        ones = torch.ones_like(real[:, 0:1, :])
        envelope = F.conv_transpose1d(ones, self.win_sq, stride=HOP)
        audio = audio[:, 0, self.pad : -self.pad] / envelope[:, 0, self.pad : -self.pad]
        return audio  # [B, T*HOP]

    @torch.no_grad()
    def _verify(self) -> None:
        from neucodec.codec_decoder_vocos import ISTFT

        ref = ISTFT(n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, padding="same")
        t = 37
        spec = torch.randn(1, N_FFT // 2 + 1, t, dtype=torch.complex64)
        want = ref(spec)
        got = self.forward(spec.real, spec.imag)
        diff = (got - want).abs().max().item()
        if diff > 1e-3:
            raise RuntimeError(f"ISTFTSame mismatch vs neucodec ISTFT: {diff}")


class NeuCodecDecoder(nn.Module):
    """codes [1, T] → audio [1, T*480]."""

    def __init__(self, codec: nn.Module):
        super().__init__()
        gen = codec.generator
        self.fsq = FSQDequant(gen.quantizer)
        self.fc_post_a = codec.fc_post_a
        backbone = gen.backbone
        self.embed = backbone.embed
        self.prior_net = backbone.prior_net
        self.blocks = nn.ModuleList(
            [TransformerBlockRope(b, n_heads=16, head_dim=64) for b in backbone.transformers]
        )
        self.final_layer_norm = backbone.final_layer_norm
        self.post_net = backbone.post_net
        self.head_out = gen.head.out
        self.istft = ISTFTSame()

    def forward(self, codes: torch.Tensor):
        x = self.fsq(codes)  # [1, T, 2048]
        x = self.fc_post_a(x)  # [1, T, 1024]
        x = x.transpose(1, 2)  # [1, 1024, T]
        x = self.embed(x)
        x = self.prior_net(x)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        x = self.post_net(x)
        x = x.transpose(1, 2)
        x = self.final_layer_norm(x)

        x = self.head_out(x).transpose(1, 2)  # [1, N_FFT+2, T]
        mag, p = x.chunk(2, dim=1)
        mag = torch.clip(torch.exp(mag), max=1e2)
        real = mag * torch.cos(p)
        imag = mag * torch.sin(p)
        return self.istft(real, imag)  # [1, T*480]
