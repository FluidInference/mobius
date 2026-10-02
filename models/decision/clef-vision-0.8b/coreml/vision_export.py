"""Trace-friendly Qwen3.5 vision tower for Core ML: one image per call, patches right-padded to a fixed count.

Inputs (all fp32, host-prepared from the HF processor's ``pixel_values`` / ``image_grid_thw``):
  patches     [N, 1536]  flattened (C, T, P, P) patches, zero padded after the image's n patches
  pos_embeds  [N, 768]   bilinear-resampled learned position grid (host: HF interpolation indices/weights)
  cos, sin    [N, 64]    axial 2D rope tables (host)
  mask        [1, 1, N, N] additive key mask: 0 for real patches, -1e4 for padding
Output: merged tokens [N/4, 1024]; rows [0, n/4) are the image's tokens in the order the language model expects.

Padding is appended after the image, so with the key mask real patches never attend to it; the merger groups four
consecutive patches, and n is always a multiple of 4 (grid h, w are multiples of the merge size), so no group mixes
real and padded patches. Parity vs ``Qwen3_5VisionModel`` is checked in ``vision_parity``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

VISION_PREFIX = "model.visual."


class VisionConfig:
    def __init__(self, cfg: dict):
        self.depth = cfg["depth"]
        self.hidden = cfg["hidden_size"]
        self.heads = cfg["num_heads"]
        self.head_dim = self.hidden // self.heads
        self.intermediate = cfg["intermediate_size"]
        self.out_hidden = cfg["out_hidden_size"]
        self.patch = cfg["patch_size"]
        self.temporal = cfg["temporal_patch_size"]
        self.channels = cfg["in_channels"]
        self.merge = cfg["spatial_merge_size"]
        self.num_pos = cfg["num_position_embeddings"]
        self.act = cfg["hidden_act"]  # gelu_pytorch_tanh for the blocks; the merger uses exact GELU
        self.rope_theta = cfg.get("rope_parameters", {}).get("rope_theta", 10000.0)
        self.patch_dim = self.channels * self.temporal * self.patch * self.patch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Block(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = nn.LayerNorm(cfg.hidden, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden, eps=1e-6)
        self.qkv = nn.Linear(cfg.hidden, cfg.hidden * 3)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.fc1 = nn.Linear(cfg.hidden, cfg.intermediate)
        self.fc2 = nn.Linear(cfg.intermediate, cfg.hidden)
        self.scale = cfg.head_dim**-0.5

    def forward(self, x, cos, sin, mask):
        n = x.shape[0]
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(n, 3, self.cfg.heads, self.cfg.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # [N, H, d]
        c, s = cos[:, None, :], sin[:, None, :]
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        q = q.permute(1, 0, 2)[None]  # [1, H, N, d]
        k = k.permute(1, 0, 2)[None]
        v = v.permute(1, 0, 2)[None]
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale + mask
        attn = torch.matmul(torch.softmax(scores, dim=-1), v)  # [1, H, N, d]
        attn = attn[0].permute(1, 0, 2).reshape(n, self.cfg.hidden)
        x = x + self.proj(attn)
        h = self.norm2(x)
        act = F.gelu(self.fc1(h), approximate="tanh" if self.cfg.act == "gelu_pytorch_tanh" else "none")
        return x + self.fc2(act)


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionConfig, n_patches: int):
        super().__init__()
        assert n_patches % (cfg.merge**2) == 0
        self.cfg = cfg
        self.n_patches = n_patches
        self.patch_embed = nn.Linear(cfg.patch_dim, cfg.hidden)  # Conv3d with kernel == stride == patch
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.depth)])
        merged = cfg.hidden * cfg.merge**2
        self.merger_norm = nn.LayerNorm(cfg.hidden, eps=1e-6)
        self.merger_fc1 = nn.Linear(merged, merged)
        self.merger_fc2 = nn.Linear(merged, cfg.out_hidden)

    def forward(self, patches, pos_embeds, cos, sin, mask):
        x = self.patch_embed(patches) + pos_embeds
        for block in self.blocks:
            x = block(x, cos, sin, mask)
        x = self.merger_norm(x).reshape(self.n_patches // self.cfg.merge**2, -1)
        return self.merger_fc2(F.gelu(self.merger_fc1(x)))

    def load_hf(self, state: dict[str, torch.Tensor]) -> None:
        """``state`` holds the HF vision weights with the ``model.visual.`` prefix stripped."""
        own = {
            "patch_embed.weight": state["patch_embed.proj.weight"].reshape(self.cfg.hidden, -1),
            "patch_embed.bias": state["patch_embed.proj.bias"],
            "merger_norm.weight": state["merger.norm.weight"],
            "merger_norm.bias": state["merger.norm.bias"],
            "merger_fc1.weight": state["merger.linear_fc1.weight"],
            "merger_fc1.bias": state["merger.linear_fc1.bias"],
            "merger_fc2.weight": state["merger.linear_fc2.weight"],
            "merger_fc2.bias": state["merger.linear_fc2.bias"],
        }
        for i in range(self.cfg.depth):
            src, dst = f"blocks.{i}.", f"blocks.{i}."
            for a, b in (("norm1", "norm1"), ("norm2", "norm2"), ("attn.qkv", "qkv"), ("attn.proj", "proj"),
                         ("mlp.linear_fc1", "fc1"), ("mlp.linear_fc2", "fc2")):
                own[f"{dst}{b}.weight"] = state[f"{src}{a}.weight"]
                own[f"{dst}{b}.bias"] = state[f"{src}{a}.bias"]
        missing, unexpected = self.load_state_dict({k: v.float() for k, v in own.items()}, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"vision weight mismatch: missing={missing[:6]} unexpected={unexpected[:6]}")


# ----------------------------------------------------------------------------------------------------------------------
# Host-side preparation (numpy/torch; mirrors transformers.vision_utils + Qwen3_5VisionRotaryEmbedding)

def axial_rope(cfg: VisionConfig, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``position_ids`` [n, 2] (h, w) -> cos/sin [n, head_dim]: freqs over head_dim//4 per axis, concat (h, w), x2."""
    spatial = cfg.head_dim // 2
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, spatial, 2, dtype=torch.float32) / spatial))
    freqs = position_ids[..., None].float() * inv_freq  # [n, 2, spatial/2]
    hw = torch.cat([freqs[:, 0], freqs[:, 1]], dim=-1)
    emb = torch.cat([hw, hw], dim=-1)
    return emb.cos(), emb.sin()


def prepare_inputs(cfg: VisionConfig, pos_table: torch.Tensor, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
                   n_patches: int):
    """One image. Uses the HF helpers for position ids and pos-embed interpolation so the host matches the reference."""
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids

    n = int(pixel_values.shape[0])
    if n > n_patches:
        raise ValueError(f"image has {n} patches; bucket holds {n_patches}")
    grid = grid_thw.reshape(1, 3)
    idx, w = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=int(math.isqrt(cfg.num_pos)), mode="bilinear", align_corners=True,
        spatial_merge_size=cfg.merge, kwargs={})
    pos_embeds = (pos_table[idx] * w[:, :, None]).sum(1)  # [n, hidden]
    position_ids = get_vision_position_ids(grid, cfg.merge, kwargs={})  # [n, 2]
    cos, sin = axial_rope(cfg, position_ids)
    patches = torch.zeros(n_patches, cfg.patch_dim)
    patches[:n] = pixel_values.float()
    pe = torch.zeros(n_patches, cfg.hidden)
    pe[:n] = pos_embeds.float()
    cs = torch.zeros(n_patches, cfg.head_dim)
    sn = torch.zeros(n_patches, cfg.head_dim)
    cs[:n], sn[:n] = cos, sin
    mask = torch.zeros(1, 1, n_patches, n_patches)
    mask[..., n:] = -1e4
    return patches, pe, cs, sn, mask, n


def load_vision(merged_backbone: Path, n_patches: int):
    from safetensors.torch import load_file

    config = json.loads((merged_backbone / "config.json").read_text())
    cfg = VisionConfig(config["vision_config"])
    state = {k[len(VISION_PREFIX):]: v for k, v in load_file(str(merged_backbone / "model.safetensors")).items()
             if k.startswith(VISION_PREFIX)}
    tower = VisionTower(cfg, n_patches)
    tower.load_hf(state)
    return tower.eval(), cfg, state["pos_embed.weight"].float()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path, default=Path("build/student-r1/merged/backbone"))
    ap.add_argument("--patches", type=int, default=784, help="patch bucket (784 = 448x448 cap -> 196 tokens)")
    ap.add_argument("--precision", choices=("fp16", "fp32", "mixed"), default="fp32",
                    help="fp32 is the only exact option: fp16 (0.27 max err) and mixed with fp32 norms/softmax "
                    "(0.33) both break on this ViT's activation range; fp32 runs 35 ms on the M5 Pro GPU")
    ap.add_argument("--out", type=Path, default=Path("build/coreml"))
    args = ap.parse_args()
    import coremltools as ct

    tower, cfg, pos_table = load_vision(args.merged, args.patches)
    N = args.patches
    example = (torch.zeros(N, cfg.patch_dim), torch.zeros(N, cfg.hidden), torch.zeros(N, cfg.head_dim),
               torch.zeros(N, cfg.head_dim), torch.zeros(1, 1, N, N))
    started = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(tower, example, check_trace=False)
    names = ["patches", "pos_embeds", "cos", "sin", "mask"]
    if args.precision == "fp32":
        precision = ct.precision.FLOAT32
    elif args.precision == "fp16":
        precision = ct.precision.FLOAT16
    else:
        keep_fp32 = {"layer_norm", "softmax", "reduce_mean", "reduce_sum", "pow", "sqrt", "rsqrt", "real_div"}
        precision = ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in keep_fp32)
    model = ct.convert(
        traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[ct.TensorType(name=n, shape=tuple(t.shape), dtype=np.float32) for n, t in zip(names, example)],
        outputs=[ct.TensorType(name="tokens", dtype=np.float32)],
    )
    model.short_description = "Qwen3.5-0.8B vision tower (clef-vision student): one image, fixed patch bucket"
    model.license = "Apache-2.0"
    model.user_defined_metadata.update({"patches": str(N), "tokens": str(N // cfg.merge**2)})
    out = args.out / f"Vision_P{N}"
    out.mkdir(parents=True, exist_ok=True)
    package = out / f"VisionTower_{args.precision}.mlpackage"
    model.save(str(package))
    pos_table.numpy().astype(np.float32).tofile(out / "pos_embed.f32")  # the host resamples this per image
    (out / "config.json").write_text(json.dumps({"patches": N, "tokens": N // cfg.merge**2, "patch_dim": cfg.patch_dim,
                                                  "hidden": cfg.hidden, "head_dim": cfg.head_dim,
                                                  "out_hidden": cfg.out_hidden, "precision": args.precision},
                                                 indent=2) + "\n")
    print(f"saved {package} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
