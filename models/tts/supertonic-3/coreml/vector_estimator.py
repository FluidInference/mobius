"""vector_estimator.onnx → PyTorch port (flow-matching denoiser with CFG).

Inputs:
    noisy_latent : [B, 144, L_ttl]
    text_emb     : [B, 256, T_text]
    style_ttl    : [B, 50, 256]
    latent_mask  : [B, 1, L_ttl]
    text_mask    : [B, 1, T_text]
    current_step : [B]
    total_step   : [B]

Output:
    denoised_latent : [B, 144, L_ttl]
        = (noisy_latent + (1/total_step) * (W_COND * cond - W_UNCOND * uncond)) * latent_mask

ONNX-mirror pipeline (verified against graph trace):

  1. Tile inputs to batch=2: [cond_half, uncond_half] stacked along axis 0.
     - noisy_latent → (2B, 144, L)
     - latent_mask  → (2B, 1, L)
     - text_mask    → (2B, 1, T_text)
     - text_emb (axis=0): concat(text_emb, expand(text_special_token, (B, 256, T_text)))
     - style_value (V source for style attn): concat(style_ttl, expand(style_value_special_token))
     - style_key   (K source for style attn): concat(expand(cond_style_key_init),
                                                       expand(style_key_special_token))
     - current_step/total_step → time scalar → tile to 2B

  2. time_embed = TimeEncoder(current_step / total_step)            # (2B, 64, 1)

  3. x = proj_in(noisy_latent_tiled) * latent_mask                  # (2B, 512, L)

  4. For k in 0..3:
       block = main_blocks[k]
         convnext_0  (4 layers, dilations [1,2,4,8])
         time_cond   (Linear 64→512, added)
         convnext_1  (1 layer)
         text_attn   (rotary, K/V from text_emb 256→512, n_heads=8)  + post-LayerNorm
         convnext_2  (1 layer)
         style_attn  (Q 512→256, K from style_key 256→256, V from style_value 256→256,
                      out 256→512, n_heads=2)                       + post-LayerNorm

  5. last_convnext (4 layers) → proj_out → * latent_mask            # (2B, 144, L)

  6. CFG combine (constants from graph: W_COND=4.0, W_UNCOND=3.0):
       cond   = out[0:B]
       uncond = out[B:2B]
       step   = 1 / total_step                          # broadcast as (B, 1, 1)
       denoised_latent = (noisy_latent + step * (W_COND*cond - W_UNCOND*uncond))
                          * latent_mask

Rotary embedding (text_attn): Llama-style "rotated-half" on the full dk=64 axis.
    theta: (1, 1, 32), increments: (1, 1000, 1) – sliced to current position counts.
    angles[..., i] = position * theta[i]                            # (B, T, 32)
    cos = cos(angles)[B, 1, T, 32]; sin similar
    q_a, q_b = q[..., :32], q[..., 32:]
    rotated = concat([q_a*cos - q_b*sin, q_a*sin + q_b*cos], dim=-1)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    ConvNeXt1DBlock,
    LayerNorm1d,
    assign_param,
    load_convnext_block,
    onnx_initializers,
)


# ---------------------------------------------------------------------------
# Constants (verified from ONNX initializers + tts.json/ttl.vector_field)
# ---------------------------------------------------------------------------
LATENT_DIM_TTL = 144
FEATURE_DIM = 512
TIME_DIM = 64
TIME_MLP_HDIM = 256
TEXT_DIM = 256
STYLE_TOKENS = 50
STYLE_DIM = 256

N_BLOCKS = 4
CONVNEXT_KSZ = 5
CONVNEXT_HDIM = 2048
CONVNEXT_0_DILATIONS = [1, 2, 4, 8]
CONVNEXT_1_DILATIONS = [1]
CONVNEXT_2_DILATIONS = [1]
LAST_CONVNEXT_DILATIONS = [1, 1, 1, 1]

TEXT_ATTN_HEADS = 8
TEXT_ATTN_NUNITS = 512
TEXT_ROTARY_INCREMENTS = 1000

STYLE_ATTN_HEADS = 2
STYLE_ATTN_NUNITS = 256

# Both attention layers use a FIXED divisor of 16.0 (not sqrt(dk)).
# This matches /main_blocks.{3,5}/attn(tion)/Div_4 = /Constant_51_output_0 = 16.0.
ATTN_SCORE_DIVISOR = 16.0

# CFG weights (from graph constants /Constant_3=4, /Constant_4=3)
W_COND = 4.0
W_UNCOND = 3.0


def mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(F.softplus(x))


class _LinearWrap(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Time encoder
# ---------------------------------------------------------------------------
class TimeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("omegas", torch.zeros(1, 32), persistent=True)
        self.register_buffer("time_scale", torch.tensor(1000.0), persistent=True)
        self.mlp = nn.ModuleList([
            _LinearWrap(TIME_DIM, TIME_MLP_HDIM),
            nn.Identity(),
            _LinearWrap(TIME_MLP_HDIM, TIME_DIM),
        ])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] → [B, 64, 1]."""
        t_scaled = t.reshape(-1, 1) * self.time_scale
        phase = t_scaled * self.omegas
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        h = self.mlp[0](emb)
        h = mish(h)
        h = self.mlp[2](h)
        return h.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Time conditioning: Linear(64 → 512), added to features.
# ---------------------------------------------------------------------------
class _TimeCondLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = _LinearWrap(TIME_DIM, FEATURE_DIM)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        t = self.linear(time_emb.squeeze(-1)).unsqueeze(-1)
        return x + t


# ---------------------------------------------------------------------------
# Text cross-attention with rotary (rotated-half Llama scheme on full dk=64).
# ---------------------------------------------------------------------------
class _TextAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_query = _LinearWrap(FEATURE_DIM, TEXT_ATTN_NUNITS)
        self.W_key = _LinearWrap(TEXT_DIM, TEXT_ATTN_NUNITS)
        self.W_value = _LinearWrap(TEXT_DIM, TEXT_ATTN_NUNITS)
        self.out_fc = _LinearWrap(TEXT_ATTN_NUNITS, FEATURE_DIM)
        self.increments = nn.Parameter(torch.zeros(1, TEXT_ROTARY_INCREMENTS, 1))
        self.theta = nn.Parameter(torch.zeros(1, 1, 32))
        self.dk = TEXT_ATTN_NUNITS // TEXT_ATTN_HEADS  # 64
        self.half = 32  # = dk / 2

    def _apply_rotary(self, x: torch.Tensor, positions: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
        """x: [B, H, T, dk=64]; positions: [B, T, 1]; divisor: [B] (mask sum)."""
        scaled = positions / divisor.reshape(-1, 1, 1)  # [B, T, 1]
        angles = scaled * self.theta                    # [B, T, 32]
        cos = torch.cos(angles).unsqueeze(1)            # [B, 1, T, 32]
        sin = torch.sin(angles).unsqueeze(1)            # [B, 1, T, 32]
        x_a, x_b = x[..., : self.half], x[..., self.half :]
        rot_a = x_a * cos - x_b * sin
        rot_b = x_a * sin + x_b * cos
        return torch.cat([rot_a, rot_b], dim=-1)

    def forward(
        self,
        x: torch.Tensor,             # [2B, 512, L]
        text_emb: torch.Tensor,      # [2B, 256, T_text]
        latent_mask: torch.Tensor,   # [2B, 1, L]
        text_mask: torch.Tensor,     # [2B, 1, T_text]
    ) -> torch.Tensor:
        B, _, L = x.shape
        T_text = text_emb.shape[-1]
        H, dk = TEXT_ATTN_HEADS, self.dk

        q = self.W_query(x.transpose(1, 2)).reshape(B, L, H, dk).transpose(1, 2)
        k = self.W_key(text_emb.transpose(1, 2)).reshape(B, T_text, H, dk).transpose(1, 2)
        v = self.W_value(text_emb.transpose(1, 2)).reshape(B, T_text, H, dk).transpose(1, 2)

        q_pos = self.increments[:, :L, :].expand(B, -1, -1).to(x.dtype)
        k_pos = self.increments[:, :T_text, :].expand(B, -1, -1).to(x.dtype)
        # length-normalized rotary: angles = (pos / sum(mask)) * theta
        q_div = latent_mask.sum(dim=(1, 2))  # [B]
        k_div = text_mask.sum(dim=(1, 2))    # [B]
        q = self._apply_rotary(q, q_pos, q_div)
        k = self._apply_rotary(k, k_pos, k_div)

        scores = torch.matmul(q, k.transpose(-2, -1)) / ATTN_SCORE_DIVISOR
        mask = text_mask.unsqueeze(2)  # [2B, 1, 1, T_text]
        # ANE-friendly additive mask (float, no bool/select): invalid positions
        # get a large negative score → softmax → ~0. Equivalent to
        # masked_fill(mask==0, -inf) within FP16 tolerance.
        scores = scores - (1.0 - mask) * 1e4
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, H * dk)
        out = self.out_fc(out).transpose(1, 2)
        return out * latent_mask


# ---------------------------------------------------------------------------
# Style cross-attention. K and V come from SEPARATE sources.
# ---------------------------------------------------------------------------
class _StyleAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_query = _LinearWrap(FEATURE_DIM, STYLE_ATTN_NUNITS)
        self.W_key = _LinearWrap(STYLE_DIM, STYLE_ATTN_NUNITS)
        self.W_value = _LinearWrap(STYLE_DIM, STYLE_ATTN_NUNITS)
        self.out_fc = _LinearWrap(STYLE_ATTN_NUNITS, FEATURE_DIM)
        self.dk = STYLE_ATTN_NUNITS // STYLE_ATTN_HEADS

    def forward(
        self,
        x: torch.Tensor,             # [2B, 512, L]
        style_key: torch.Tensor,     # [2B, 50, 256] (cond half = learned cond_style_key)
        style_value: torch.Tensor,   # [2B, 50, 256] (cond half = style_ttl)
        latent_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, _, L = x.shape
        S = style_key.shape[1]
        H, dk = STYLE_ATTN_HEADS, self.dk

        q = self.W_query(x.transpose(1, 2)).reshape(B, L, H, dk).transpose(1, 2)
        k = self.W_key(style_key).reshape(B, S, H, dk).transpose(1, 2)
        v = self.W_value(style_value).reshape(B, S, H, dk).transpose(1, 2)

        # Style attention applies tanh to K before the matmul.
        scores = torch.matmul(q, torch.tanh(k).transpose(-2, -1)) / ATTN_SCORE_DIVISOR
        attn = F.softmax(scores, dim=-1)
        # Post-softmax mask on the Q-side latent mask (zero out invalid Q rows).
        # ANE-friendly multiplicative mask: float mult ≡ where(mask==0, 0, attn)
        # but avoids bool tile/select that the ANE compiler rejects.
        mask = latent_mask.transpose(1, 2).unsqueeze(1)  # [2B, 1, L, 1]
        attn = attn * mask
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, H * dk)
        out = self.out_fc(out).transpose(1, 2)
        return out * latent_mask


# ---------------------------------------------------------------------------
# A single logical MainBlock (= 6 sub-blocks in the ONNX flat list).
# ---------------------------------------------------------------------------
class _MainBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnext_0 = nn.ModuleList(
            [ConvNeXt1DBlock(FEATURE_DIM, CONVNEXT_HDIM, CONVNEXT_KSZ, dilation=d, wrap_dwconv=False, causal=False)
             for d in CONVNEXT_0_DILATIONS]
        )
        self.time_cond = _TimeCondLayer()
        self.convnext_1 = nn.ModuleList(
            [ConvNeXt1DBlock(FEATURE_DIM, CONVNEXT_HDIM, CONVNEXT_KSZ, dilation=d, wrap_dwconv=False, causal=False)
             for d in CONVNEXT_1_DILATIONS]
        )
        self.text_attn = _TextAttn()
        self.text_norm = LayerNorm1d(FEATURE_DIM)
        self.convnext_2 = nn.ModuleList(
            [ConvNeXt1DBlock(FEATURE_DIM, CONVNEXT_HDIM, CONVNEXT_KSZ, dilation=d, wrap_dwconv=False, causal=False)
             for d in CONVNEXT_2_DILATIONS]
        )
        self.style_attn = _StyleAttn()
        self.style_norm = LayerNorm1d(FEATURE_DIM)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        text_emb: torch.Tensor,
        style_key: torch.Tensor,
        style_value: torch.Tensor,
        latent_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        for blk in self.convnext_0:
            x = blk(x * latent_mask, mask=latent_mask)
        x = self.time_cond(x, time_emb) * latent_mask
        for blk in self.convnext_1:
            x = blk(x * latent_mask, mask=latent_mask)
        # text cross-attn: post-norm residual with mask on both branches.
        x_m = x * latent_mask
        x = self.text_norm(x_m + self.text_attn(x, text_emb, latent_mask, text_mask)) * latent_mask
        for blk in self.convnext_2:
            x = blk(x * latent_mask, mask=latent_mask)
        # style cross-attn: post-norm residual with mask on both branches.
        x_m = x * latent_mask
        x = self.style_norm(x_m + self.style_attn(x, style_key, style_value, latent_mask)) * latent_mask
        return x


# ---------------------------------------------------------------------------
# UncondMasker: builds batched [cond | uncond] inputs.
# ---------------------------------------------------------------------------
class _UncondMasker(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_special_token = nn.Parameter(torch.zeros(1, TEXT_DIM, 1))
        self.style_value_special_token = nn.Parameter(torch.zeros(1, STYLE_TOKENS, STYLE_DIM))
        self.style_key_special_token = nn.Parameter(torch.zeros(1, STYLE_TOKENS, STYLE_DIM))
        # Learned constant that serves as the conditional style_KEY source. This
        # is the ONNX initializer `/vector_estimator/Expand_output_0` of shape
        # (1, 50, 256). Style value (V) comes from the user-provided style_ttl;
        # style key (K) comes from this learned constant for the cond half and
        # from style_key_special_token for the uncond half.
        self.cond_style_key = nn.Parameter(torch.zeros(1, STYLE_TOKENS, STYLE_DIM))

    def forward(
        self,
        text_emb: torch.Tensor,
        style_ttl: torch.Tensor,
        text_mask_shape_t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (text_emb_batched, style_key_batched, style_value_batched), each (2B, ...)."""
        B = text_emb.shape[0]
        T_text = text_mask_shape_t

        # Uncond halves (broadcast specials to per-sample shape).
        text_uncond = self.text_special_token.expand(B, -1, T_text)         # (B, 256, T_text)
        style_key_cond = self.cond_style_key.expand(B, -1, -1)              # (B, 50, 256)
        style_key_uncond = self.style_key_special_token.expand(B, -1, -1)   # (B, 50, 256)
        style_value_uncond = self.style_value_special_token.expand(B, -1, -1)  # (B, 50, 256)

        text_batched = torch.cat([text_emb, text_uncond], dim=0)
        style_key_batched = torch.cat([style_key_cond, style_key_uncond], dim=0)
        style_value_batched = torch.cat([style_ttl, style_value_uncond], dim=0)
        return text_batched, style_key_batched, style_value_batched


# ---------------------------------------------------------------------------
# VectorEstimator top-level
# ---------------------------------------------------------------------------
class VectorEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.uncond_masker = _UncondMasker()
        self.time_encoder = TimeEncoder()
        self.proj_in = nn.Conv1d(LATENT_DIM_TTL, FEATURE_DIM, kernel_size=1, bias=False)
        self.main_blocks = nn.ModuleList([_MainBlock() for _ in range(N_BLOCKS)])
        self.last_convnext = nn.ModuleList(
            [ConvNeXt1DBlock(FEATURE_DIM, CONVNEXT_HDIM, CONVNEXT_KSZ, dilation=d, wrap_dwconv=False, causal=False)
             for d in LAST_CONVNEXT_DILATIONS]
        )
        self.proj_out = nn.Conv1d(FEATURE_DIM, LATENT_DIM_TTL, kernel_size=1, bias=False)

    def forward(
        self,
        noisy_latent: torch.Tensor,  # [B, 144, L]
        text_emb: torch.Tensor,      # [B, 256, T_text]
        style_ttl: torch.Tensor,     # [B, 50, 256]
        latent_mask: torch.Tensor,   # [B, 1, L]
        text_mask: torch.Tensor,     # [B, 1, T_text]
        current_step: torch.Tensor,  # [B]
        total_step: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        B = noisy_latent.shape[0]
        T_text = text_emb.shape[-1]

        # 1. CFG batch duplication.
        text_b, style_key_b, style_value_b = self.uncond_masker(text_emb, style_ttl, T_text)
        noisy_b = torch.cat([noisy_latent, noisy_latent], dim=0)
        latent_mask_b = torch.cat([latent_mask, latent_mask], dim=0)
        text_mask_b = torch.cat([text_mask, text_mask], dim=0)

        # 2. Time embedding (tile current/total to 2B).
        t = current_step / total_step                          # (B,)
        t_b = torch.cat([t, t], dim=0)                         # (2B,)
        time_emb = self.time_encoder(t_b)                      # (2B, 64, 1)

        # 3. proj_in.
        x = self.proj_in(noisy_b) * latent_mask_b              # (2B, 512, L)

        # 4. Main blocks.
        for block in self.main_blocks:
            x = block(x, time_emb, text_b, style_key_b, style_value_b, latent_mask_b, text_mask_b)

        # 5. last_convnext.
        for blk in self.last_convnext:
            x = blk(x * latent_mask_b, mask=latent_mask_b)

        # 6. proj_out + mask.
        v = self.proj_out(x) * latent_mask_b                   # (2B, 144, L)

        # 7. CFG combine.
        cond = v[:B]
        uncond = v[B:]
        step = (1.0 / total_step).reshape(-1, 1, 1)            # (B, 1, 1)
        denoised = (noisy_latent + step * (W_COND * cond - W_UNCOND * uncond)) * latent_mask
        return denoised


def build_vector_estimator_from_onnx(onnx_path: Path) -> VectorEstimator:
    inits = onnx_initializers(onnx_path)
    model = VectorEstimator()
    _load(model, inits)
    model.eval()
    return model


def _load(model: VectorEstimator, inits: Dict[str, np.ndarray]) -> None:
    p = "vector_estimator.tts.ttl"

    # Uncond masker tokens.
    assign_param(model.uncond_masker.text_special_token,
                 inits[f"{p}.uncond_masker.text_special_token"], name="text_special_token")
    assign_param(model.uncond_masker.style_value_special_token,
                 inits[f"{p}.uncond_masker.style_value_special_token"], name="style_value_special_token")
    assign_param(model.uncond_masker.style_key_special_token,
                 inits[f"{p}.uncond_masker.style_key_special_token"], name="style_key_special_token")
    # Learned cond_style_key: stored unnamed in the graph as Expand input.
    assign_param(model.uncond_masker.cond_style_key,
                 inits["/vector_estimator/Expand_output_0"], name="cond_style_key")

    # Time encoder MLP.
    assign_param(model.time_encoder.mlp[0].linear.weight,
                 inits[f"{p}.vector_field.time_encoder.mlp.0.linear.weight"], name="time.mlp.0.weight")
    assign_param(model.time_encoder.mlp[0].linear.bias,
                 inits[f"{p}.vector_field.time_encoder.mlp.0.linear.bias"], name="time.mlp.0.bias")
    assign_param(model.time_encoder.mlp[2].linear.weight,
                 inits[f"{p}.vector_field.time_encoder.mlp.2.linear.weight"], name="time.mlp.2.weight")
    assign_param(model.time_encoder.mlp[2].linear.bias,
                 inits[f"{p}.vector_field.time_encoder.mlp.2.linear.bias"], name="time.mlp.2.bias")

    omegas = inits["/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_3_output_0"]
    assign_param(model.time_encoder.omegas, np.asarray(omegas).reshape(1, 32), name="omegas")
    ts = inits["/vector_estimator/vector_field/time_encoder/sinusoidal/Constant_2_output_0"]
    with torch.no_grad():
        model.time_encoder.time_scale.copy_(torch.tensor(float(np.asarray(ts).item())))

    # proj_in / proj_out (wrapped, no bias).
    assign_param(model.proj_in.weight, inits[f"{p}.vector_field.proj_in.net.weight"], name="proj_in.weight")
    assign_param(model.proj_out.weight, inits[f"{p}.vector_field.proj_out.net.weight"], name="proj_out.weight")

    # Main blocks: 24 entries in flat ONNX list = 4 logical blocks × 6 sub-blocks.
    # Orphan MatMul IDs spaced 45 apart per logical block.
    matmul_base = [3384, 3390, 3391, 3392, 3399, 3405, 3406, 3407, 3408]

    for k, block in enumerate(model.main_blocks):
        for i, blk in enumerate(block.convnext_0):
            load_convnext_block(blk, inits, prefix=f"{p}.vector_field.main_blocks.{6 * k + 0}.convnext.{i}",
                                wrap_dwconv=False)

        tc_idx = 6 * k + 1
        tc_w = inits[f"onnx::MatMul_{matmul_base[0] + 45 * k}"]
        assign_param(block.time_cond.linear.linear.weight, tc_w.T, name=f"block.{k}.time_cond.weight")
        assign_param(block.time_cond.linear.linear.bias,
                     inits[f"{p}.vector_field.main_blocks.{tc_idx}.linear.linear.bias"],
                     name=f"block.{k}.time_cond.bias")

        for i, blk in enumerate(block.convnext_1):
            load_convnext_block(blk, inits, prefix=f"{p}.vector_field.main_blocks.{6 * k + 2}.convnext.{i}",
                                wrap_dwconv=False)

        ta_idx = 6 * k + 3
        ta_prefix = f"{p}.vector_field.main_blocks.{ta_idx}.attn"
        ta = block.text_attn
        for j, attr_name in enumerate(["W_query", "W_key", "W_value"]):
            w = inits[f"onnx::MatMul_{matmul_base[1 + j] + 45 * k}"]
            lin = getattr(ta, attr_name).linear
            assign_param(lin.weight, w.T, name=f"block.{k}.text_attn.{attr_name}.weight")
            assign_param(lin.bias, inits[f"{ta_prefix}.{attr_name}.linear.bias"],
                         name=f"block.{k}.text_attn.{attr_name}.bias")
        w_o = inits[f"onnx::MatMul_{matmul_base[4] + 45 * k}"]
        assign_param(ta.out_fc.linear.weight, w_o.T, name=f"block.{k}.text_attn.out_fc.weight")
        assign_param(ta.out_fc.linear.bias, inits[f"{ta_prefix}.out_fc.linear.bias"],
                     name=f"block.{k}.text_attn.out_fc.bias")
        # Rotary params live under main_blocks.3.attn only (shared at runtime).
        assign_param(ta.increments, inits[f"{p}.vector_field.main_blocks.3.attn.increments"],
                     name=f"block.{k}.text_attn.increments")
        assign_param(ta.theta, inits[f"{p}.vector_field.main_blocks.3.attn.theta"],
                     name=f"block.{k}.text_attn.theta")
        ta_norm_prefix = f"{p}.vector_field.main_blocks.{ta_idx}.norm.norm"
        assign_param(block.text_norm.norm.weight, inits[f"{ta_norm_prefix}.weight"],
                     name=f"block.{k}.text_norm.weight")
        assign_param(block.text_norm.norm.bias, inits[f"{ta_norm_prefix}.bias"],
                     name=f"block.{k}.text_norm.bias")

        for i, blk in enumerate(block.convnext_2):
            load_convnext_block(blk, inits, prefix=f"{p}.vector_field.main_blocks.{6 * k + 4}.convnext.{i}",
                                wrap_dwconv=False)

        sa_idx = 6 * k + 5
        sa_prefix = f"{p}.vector_field.main_blocks.{sa_idx}.attention"
        sa = block.style_attn
        for j, attr_name in enumerate(["W_query", "W_key", "W_value", "out_fc"]):
            w = inits[f"onnx::MatMul_{matmul_base[5 + j] + 45 * k}"]
            lin = getattr(sa, attr_name).linear
            assign_param(lin.weight, w.T, name=f"block.{k}.style_attn.{attr_name}.weight")
            assign_param(lin.bias, inits[f"{sa_prefix}.{attr_name}.linear.bias"],
                         name=f"block.{k}.style_attn.{attr_name}.bias")
        sa_norm_prefix = f"{p}.vector_field.main_blocks.{sa_idx}.norm.norm"
        assign_param(block.style_norm.norm.weight, inits[f"{sa_norm_prefix}.weight"],
                     name=f"block.{k}.style_norm.weight")
        assign_param(block.style_norm.norm.bias, inits[f"{sa_norm_prefix}.bias"],
                     name=f"block.{k}.style_norm.bias")

    for i, blk in enumerate(model.last_convnext):
        load_convnext_block(blk, inits, prefix=f"{p}.vector_field.last_convnext.convnext.{i}",
                            wrap_dwconv=False)


if __name__ == "__main__":  # pragma: no cover
    import sys

    onnx_path = Path(sys.argv[1] if len(sys.argv) > 1 else "build/supertonic-3-coreml/_onnx/vector_estimator.onnx")
    model = build_vector_estimator_from_onnx(onnx_path)
    print(f"Built VectorEstimator with {sum(p.numel() for p in model.parameters())} params")
    B, L, T = 1, 4, 8
    noisy = torch.randn(B, 144, L)
    text = torch.randn(B, 256, T)
    style = torch.randn(B, 50, 256)
    lmask = torch.ones(B, 1, L)
    tmask = torch.ones(B, 1, T)
    cur = torch.tensor([0.0])
    tot = torch.tensor([8.0])
    with torch.no_grad():
        y = model(noisy, text, style, lmask, tmask, cur, tot)
    print(f"Output shape: {tuple(y.shape)}  (expected ({B}, 144, {L}))")
