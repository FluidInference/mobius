"""Trace-friendly wrappers around MossAudioTokenizerModel (Nano) for CoreML export.

MossCodecDecoder: codes [16,1,T] → stereo 48 kHz audio [1,2,T*3840]   (full-utterance, flexible T)
MossCodecEncoder: audio [1,2,S] (S % 3840 == 0) → codes [16,1,S/3840]  (voice-clone prompt encode)

Channel interleave (stereo → one 96 kHz-rate mono stream) is replicated from
MossAudioTokenizerModel._flatten_channels_for_codec / _restore_channels_from_codec.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_rope(codec: nn.Module) -> None:
    """Replace the HF module's apply_rope with a version whose head-dim math is a Python constant.

    Upstream computes ``2 / D`` from a traced tensor size, which torch.jit.trace records as an
    integer division op that CoreML cannot lower (``inverse`` on int32). Only D is made static;
    the time axis stays dynamic so flexible-length export still works.
    """
    import math
    import sys

    mod = sys.modules[type(codec).__module__]

    def apply_rope_static(q, k, offset, max_period=10_000, time_before_heads=False):
        D = int(q.shape[-1])
        ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(max_period) * 2.0 / float(D)))
        T = q.shape[1] if time_before_heads else q.shape[2]
        ts = offset.float().view(-1, 1) + torch.arange(T, device=q.device, dtype=torch.float32)
        ts = ts.view(ts.shape[0], -1, 1, 1) if time_before_heads else ts.view(ts.shape[0], 1, -1, 1)
        dims = q.shape[:-1]
        q2 = q.reshape(*dims, D // 2, 2)
        k2 = k.reshape(*dims, D // 2, 2)
        qr, qi = q2[..., 0], q2[..., 1]
        kr, ki = k2[..., 0], k2[..., 1]
        rotr = torch.cos(freqs * ts)
        roti = torch.sin(freqs * ts)
        qo = torch.stack((qr * rotr - qi * roti, qr * roti + qi * rotr), dim=-1)
        ko = torch.stack((kr * rotr - ki * roti, kr * roti + ki * rotr), dim=-1)
        return qo.reshape(*dims, D), ko.reshape(*dims, D)

    mod.apply_rope = apply_rope_static


def strip_weight_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            try:
                torch.nn.utils.parametrize.remove_parametrizations(module, "weight")
            except ValueError:
                pass


class MossCodecDecoder(nn.Module):
    def __init__(self, codec: nn.Module) -> None:
        super().__init__()
        strip_weight_norm(codec)
        patch_rope(codec)
        self.quantizers = codec.quantizer.quantizers
        self.output_proj = codec.quantizer.output_proj
        self.decoder = codec.decoder
        self.n_channels = int(codec.number_channels)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.long()
        emb = None
        for i, lfq in enumerate(self.quantizers):
            z = lfq.out_proj(lfq.codebook(codes[i]).transpose(1, 2))  # [B, rvq_dim, T]
            emb = z if emb is None else emb + z
        x = self.output_proj(emb)  # [B, 768, T]
        lengths = torch.ones_like(x[:, 0, :]).sum(dim=-1).long()  # [B] = T, shape-derived for flexible T
        for module in self.decoder:
            x, lengths = module(x, lengths)
        # [B, 1, T*7680] interleaved → [B, 2, T*3840]
        return x.squeeze(1).reshape(x.shape[0], -1, self.n_channels).transpose(1, 2)


class MossCodecEncoder(nn.Module):
    def __init__(self, codec: nn.Module) -> None:
        super().__init__()
        strip_weight_norm(codec)
        patch_rope(codec)
        self.encoder = codec.encoder
        self.input_proj = codec.quantizer.input_proj
        self.quantizers = codec.quantizer.quantizers
        self.n_channels = int(codec.number_channels)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.transpose(1, 2).reshape(audio.shape[0], 1, -1)  # interleave channels
        lengths = torch.ones_like(x[:, 0, :]).sum(dim=-1).long()  # [B] = 2*S
        for module in self.encoder:
            x, lengths = module(x, lengths)
        residual = self.input_proj(x)  # [B, 512, T]
        codes = []
        for lfq in self.quantizers:
            z_e = lfq.in_proj(residual)  # [B, 8, T]
            enc = F.normalize(z_e.transpose(1, 2).reshape(-1, z_e.shape[1]), dim=-1)  # [B*T, 8]
            cb = F.normalize(lfq.codebook.weight, dim=-1)  # [1024, 8]
            sim = torch.matmul(enc, cb.t())  # cosine similarity ⇔ -‖·‖² after normalisation
            idx = sim.argmax(dim=-1)  # [B*T]
            z_q = lfq.out_proj(lfq.codebook(idx).reshape(z_e.shape[0], -1, z_e.shape[1]).transpose(1, 2))
            residual = residual - z_q
            codes.append(idx.reshape(z_e.shape[0], -1))
        return torch.stack(codes, dim=0).to(torch.int32)  # [16, B, T]


NEG = -1.0e4


def _rope_tables(positions: torch.Tensor, head_dim: int, max_period: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [len(positions), head_dim/2] for the codec's (real, imag) interleaved RoPE layout."""
    import math

    ds = torch.arange(head_dim // 2, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2.0 / head_dim))
    ang = positions.float()[:, None] * freqs[None, :]
    return ang.cos(), ang.sin()


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B,H,T,D] with pairs (re, im) interleaved along D; cos/sin broadcast to [T, D/2]."""
    B, H, T, D = x.shape
    x2 = x.reshape(B, H, T, D // 2, 2)
    re, im = x2[..., 0], x2[..., 1]
    out = torch.stack((re * cos - im * sin, re * sin + im * cos), dim=-1)
    return out.reshape(B, H, T, D)


class MossCodecStepDecoder(nn.Module):
    """One 80 ms frame per call: codes [16,1,1] → audio [1,2,3840], with explicit per-layer KV caches.

    Cache layout per transformer layer: k/v [1, heads, context, head_dim] holding the most recent
    `context` positions in chronological order (shift-append, no ring indices). Keys are cached
    *unrotated*; RoPE is applied at attention time with constant relative-position tables, which is
    mathematically identical to upstream's absolute positions and keeps all trig out of the graph.
    `frame_index` (0-based) only drives the warm-up mask for not-yet-written cache slots.
    """

    def __init__(self, codec: nn.Module) -> None:
        super().__init__()
        strip_weight_norm(codec)
        self.quantizers = codec.quantizer.quantizers
        self.output_proj = codec.quantizer.output_proj
        self.decoder = codec.decoder
        self.n_channels = int(codec.number_channels)
        # Static per-stage geometry: tokens per frame, context, layer list.
        self.stage_specs: list[dict] = []
        n = 1
        for module in codec.decoder:
            if hasattr(module, "patch_size"):
                n *= int(module.patch_size)
                continue
            tr = module.transformer
            layer0 = tr.layers[0]
            attn = layer0.self_attn
            self.stage_specs.append(
                {
                    "n": n,
                    "context": int(attn.context),
                    "heads": int(attn.num_heads),
                    "head_dim": int(attn.embed_dim // attn.num_heads),
                    "layers": len(tr.layers),
                    "max_period": float(tr.rope.max_period),
                }
            )
        for s, spec in enumerate(self.stage_specs):
            cap, n, hd = spec["context"], spec["n"], spec["head_dim"]
            k_cos, k_sin = _rope_tables(torch.arange(cap), hd, spec["max_period"])
            q_cos, q_sin = _rope_tables(torch.arange(cap - n, cap), hd, spec["max_period"])
            self.register_buffer(f"k_cos_{s}", k_cos, persistent=False)
            self.register_buffer(f"k_sin_{s}", k_sin, persistent=False)
            self.register_buffer(f"q_cos_{s}", q_cos, persistent=False)
            self.register_buffer(f"q_sin_{s}", q_sin, persistent=False)
            i = torch.arange(cap)
            j = torch.arange(n)
            delta = j[:, None] + (cap - n) - i[None, :]  # query pos − key pos, [n, cap]
            static_ok = (delta >= 0) & (delta < cap)
            self.register_buffer(f"static_ok_{s}", static_ok, persistent=False)
            self.register_buffer(f"slot_pos_{s}", (i + n - cap).float(), persistent=False)  # + frame_index*n

    def cache_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        shapes = []
        for s, spec in enumerate(self.stage_specs):
            for l in range(spec["layers"]):
                shape = (1, spec["heads"], spec["context"], spec["head_dim"])
                shapes.append((f"k{s}_{l}", shape))
                shapes.append((f"v{s}_{l}", shape))
        return shapes

    def _layer_step(self, layer, x, s: int, spec: dict, frame_index: torch.Tensor, k_cache, v_cache):
        n, H, Dh, cap = spec["n"], spec["heads"], spec["head_dim"], spec["context"]
        attn = layer.self_attn
        h = layer.norm1(x)
        proj = attn.in_proj(h).reshape(1, n, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = proj[0], proj[1], proj[2]  # [1,H,n,Dh]
        k_all = torch.cat((k_cache[:, :, n:, :], k), dim=2)  # [1,H,cap,Dh], unrotated
        v_all = torch.cat((v_cache[:, :, n:, :], v), dim=2)
        q_rot = _rotate(q, getattr(self, f"q_cos_{s}"), getattr(self, f"q_sin_{s}"))
        k_rot = _rotate(k_all, getattr(self, f"k_cos_{s}"), getattr(self, f"k_sin_{s}"))
        slot_pos = getattr(self, f"slot_pos_{s}") + frame_index.float() * float(n)  # [cap]
        ok = getattr(self, f"static_ok_{s}") & (slot_pos >= 0)[None, :]  # [n, cap]
        bias = torch.where(ok, torch.zeros((), dtype=x.dtype), torch.full((), NEG, dtype=x.dtype))[None, None]
        scores = torch.matmul(q_rot, k_rot.transpose(-1, -2)) * (Dh**-0.5) + bias
        out = torch.matmul(torch.softmax(scores, dim=-1), v_all)  # [1,H,n,Dh]
        out = out.transpose(1, 2).reshape(1, n, H * Dh)
        x = x + layer.layer_scale_1(attn.out_proj(out))
        x = x + layer.layer_scale_2(layer.ffn(layer.norm2(x)))
        return x, k_all, v_all

    def forward(self, codes: torch.Tensor, frame_index: torch.Tensor, *caches: torch.Tensor):
        codes = codes.long()
        emb = None
        for i, lfq in enumerate(self.quantizers):
            z = lfq.out_proj(lfq.codebook(codes[i]).transpose(1, 2))
            emb = z if emb is None else emb + z
        x = self.output_proj(emb)  # [1,768,1]
        lengths = torch.ones(1, dtype=torch.long, device=x.device)
        new_caches: list[torch.Tensor] = []
        ci = 0
        s = 0
        for module in self.decoder:
            if hasattr(module, "patch_size"):
                x, lengths = module(x, lengths)
                continue
            spec = self.stage_specs[s]
            x = module.input_proj(x.transpose(1, 2))  # [1,n,d_model]
            for layer in module.transformer.layers:
                x, k_all, v_all = self._layer_step(layer, x, s, spec, frame_index, caches[ci], caches[ci + 1])
                new_caches.extend((k_all, v_all))
                ci += 2
            x = module.output_proj(x).transpose(1, 2)
            s += 1
        audio = x.squeeze(1).reshape(1, -1, self.n_channels).transpose(1, 2)  # [1,2,3840]
        return (audio, *new_caches)
