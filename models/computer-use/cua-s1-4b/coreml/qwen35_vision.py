"""Qwen3.5 vision tower (+ patch merger) for Cua-S1-4B multimodal, token level.

Screenshots are smart-resized to their own grid (any multiple of 32 px), so one static
graph serves every grid up to a patch budget N: the host patchifies in the processor's
spatial-merge-window order and supplies the grid-dependent pos-embed and 2D rotary rows;
padded patches are masked out of attention (keys), which keeps real tokens exact.

The processor repeats a still image to temporal_patch_size=2, so the Conv3d patch embed
over two identical frames is a Linear over one frame with the temporal taps summed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: int):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, dim * 3)
        self.attn.proj = nn.Linear(dim, dim)
        self.mlp = nn.Module()
        self.mlp.linear_fc1 = nn.Linear(dim, mlp)
        self.mlp.linear_fc2 = nn.Linear(mlp, dim)

    def forward(self, x, cos, sin, mask):
        n, dim = x.shape
        h = self.norm1(x)
        qkv = self.attn.qkv(h).reshape(n, 3, self.heads, dim // self.heads).permute(1, 2, 0, 3)  # [3, H, n, d]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        scores = torch.matmul(q, k.transpose(-1, -2)) * (dim // self.heads) ** -0.5 + mask
        out = torch.matmul(torch.softmax(scores, dim=-1), v).permute(1, 0, 2).reshape(n, dim)
        x = x + self.attn.proj(out)
        h = self.mlp.linear_fc2(F.gelu(self.mlp.linear_fc1(self.norm2(x)), approximate="tanh"))
        return x + h


class VisionTower(nn.Module):
    """Token-level tower for a fixed patch budget `max_patches` (multiple of 4).

    Inputs (host-prepared, exact for any image grid that fits):
      patches   [N, 768]   normalised pixels per 16x16 patch (C, py, px), merge-window order
      pos_embed [N, 1024]  bilinear-resampled learned position table rows
      cos, sin  [N, 64]    2D rotary tables
      key_mask  [1, N]     0 for real patches, -1e4 for padding (padding is appended at the end)
    Output: image_embeds [N/4, 2560]; rows past (real patches / 4) are padding.
    """

    def __init__(self, vcfg: dict, max_patches: int):
        super().__init__()
        assert max_patches % 4 == 0
        self.cfg = Qwen3_5VisionConfig(**vcfg)
        c = self.cfg
        self.max_patches = max_patches
        self.merge = c.spatial_merge_size
        self.dim = c.hidden_size
        self.patch_embed = nn.Linear(c.in_channels * c.patch_size**2, c.hidden_size)
        self.blocks = nn.ModuleList([Block(c.hidden_size, c.num_heads, c.intermediate_size) for _ in range(c.depth)])
        merged = c.hidden_size * self.merge**2
        self.merger = nn.Module()
        self.merger.norm = nn.LayerNorm(c.hidden_size, eps=1e-6)
        self.merger.linear_fc1 = nn.Linear(merged, merged)
        self.merger.linear_fc2 = nn.Linear(merged, c.out_hidden_size)

    def forward(self, patches, pos_embed, cos, sin, key_mask):
        x = self.patch_embed(patches) + pos_embed
        mask = key_mask[None]  # [1, 1, N] broadcast over heads and queries
        for blk in self.blocks:
            x = blk(x, cos[None], sin[None], mask)
        x = self.merger.norm(x).reshape(-1, self.dim * self.merge**2)
        return self.merger.linear_fc2(F.gelu(self.merger.linear_fc1(x)))

    def load_merged(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Loads weights; returns the learned position table [side*side, dim] (host-side resampling)."""
        state = {k: v.float() for k, v in state.items()}
        conv = state.pop("patch_embed.proj.weight")  # [dim, C, T, p, p]; a still image repeats over T
        own = {
            "patch_embed.weight": conv.sum(2).reshape(conv.shape[0], -1),
            "patch_embed.bias": state.pop("patch_embed.proj.bias"),
        }
        table = state.pop("pos_embed.weight")
        own.update(state)
        missing, unexpected = self.load_state_dict(own, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"vision weight mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        return table


def host_inputs(cfg: Qwen3_5VisionConfig, table: torch.Tensor, grid_h: int, grid_w: int, max_patches: int):
    """pos_embed / cos / sin / key_mask for one image grid, padded to max_patches (reference for the Swift host)."""
    n = grid_h * grid_w
    assert n <= max_patches, (n, max_patches)
    grid = torch.tensor([[1, grid_h, grid_w]])
    idx, w = get_vision_interpolation_indices_and_weights(
        grid, int(cfg.num_position_embeddings**0.5), mode="bilinear", align_corners=True,
        spatial_merge_size=cfg.spatial_merge_size,
    )
    pos = (table[idx] * w[..., None].float()).sum(1)
    cos, sin = Qwen3_5VisionRotaryEmbedding(cfg)(torch.zeros(1), get_vision_position_ids(grid, cfg.spatial_merge_size))
    pad = max_patches - n
    mask = torch.zeros(1, max_patches)
    mask[0, n:] = -1e4
    return (
        F.pad(pos, (0, 0, 0, pad)),
        F.pad(cos.float(), (0, 0, 0, pad)),
        F.pad(sin.float(), (0, 0, 0, pad)),
        mask,
    )


def patches_from_pixel_values(pixel_values: torch.Tensor, max_patches: int) -> torch.Tensor:
    """Processor pixel_values [n, C*T*p*p] -> first temporal copy [max_patches, C*p*p], zero padded."""
    n = pixel_values.shape[0]
    x = pixel_values.reshape(n, 3, 2, 16, 16)[:, :, 0].reshape(n, -1)
    return F.pad(x.float(), (0, 0, 0, max_patches - n))


def pixels_from_patches(pixel_values: torch.Tensor, grid_h: int, grid_w: int, patch: int = 16, merge: int = 2):
    """Invert the processor's patch layout back to the resized image in [0, 1] (parity tests only)."""
    n = pixel_values.shape[0]
    x = pixel_values.reshape(n, 3, 2, patch, patch)[:, :, 0]  # first temporal copy
    x = x.reshape(grid_h // merge, grid_w // merge, merge, merge, 3, patch, patch)
    x = x.permute(4, 0, 2, 5, 1, 3, 6).reshape(1, 3, grid_h * patch, grid_w * patch)
    return (x + 1.0) / 2.0
