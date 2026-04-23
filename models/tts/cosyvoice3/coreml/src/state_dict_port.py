"""State dict porter: host DiT → ANE BC1S DiT.

Host parameter shapes (from `verify/CosyVoice/cosyvoice/flow/DiT/modules.py`
and `dit.py`) → ANE shapes:

    transformer_blocks.{i}.attn_norm.linear.weight : (6C, C)   → (6C, C, 1, 1)
    transformer_blocks.{i}.attn_norm.linear.bias   : (6C,)     → unchanged
    transformer_blocks.{i}.attn_norm.norm.*        : (no params, affine=False)

    transformer_blocks.{i}.attn.to_q.weight        : (C, C)    → (C, C, 1, 1)
    transformer_blocks.{i}.attn.to_q.bias          : (C,)      → unchanged
    transformer_blocks.{i}.attn.to_k.*             :           → same
    transformer_blocks.{i}.attn.to_v.*             :           → same
    transformer_blocks.{i}.attn.to_out.0.weight    : (C, C)    → (C, C, 1, 1)
    transformer_blocks.{i}.attn.to_out.0.bias      : (C,)      → unchanged

    transformer_blocks.{i}.ff_norm.*               : (no params, affine=False)

    # Host FeedForward: Sequential(Sequential(Linear, GELU), Dropout, Linear)
    # → stored as ff.ff.0.0.{weight,bias} (fc1), ff.ff.2.{weight,bias} (fc2).
    # ANE FeedForward uses flat fc1/fc2 naming.
    transformer_blocks.{i}.ff.ff.0.0.weight : (inner, C)      → .ff.fc1.weight: (inner, C, 1, 1)
    transformer_blocks.{i}.ff.ff.0.0.bias   : (inner,)        → .ff.fc1.bias
    transformer_blocks.{i}.ff.ff.2.weight   : (C, inner)      → .ff.fc2.weight: (C, inner, 1, 1)
    transformer_blocks.{i}.ff.ff.2.bias     : (C,)            → .ff.fc2.bias

    norm_out.linear.weight : (2C, C)  → (2C, C, 1, 1)
    norm_out.linear.bias   : (2C,)    → unchanged
    proj_out.weight        : (M,  C)  → (M,  C, 1, 1)
    proj_out.bias          : (M,)     → unchanged

Parameters NOT touched by this port:
    - time_embed.*  (host TimestepEmbedding reused as-is)
    - input_embed.* (host InputEmbedding reused as-is)
    - rotary_embed.freqs (x_transformers RotaryEmbedding; ANE replaces it
      with per-attention `ANERotaryEmbedding` whose cos/sin tables are
      derived from the same inv_freq formula and don't need porting)

Any parameter not covered by the rules above is passed through unchanged.
"""
from __future__ import annotations

from typing import Any, Dict

import re
import torch


def _unsqueeze_linear_weight(w: torch.Tensor) -> torch.Tensor:
    """(out, in) → (out, in, 1, 1)."""
    assert w.ndim == 2, f"expected 2D linear weight, got shape {tuple(w.shape)}"
    return w.unsqueeze(-1).unsqueeze(-1).contiguous()


def convert_state_dict_to_ane(host_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Produce a state dict compatible with `ANEDiT`.

    Expects `host_sd` keyed as for the upstream `DiT` module (e.g.
    `transformer_blocks.0.attn.to_q.weight`).

    Returns a new dict; does not mutate the input.
    """
    out: Dict[str, torch.Tensor] = {}

    # Regexes keyed by target substring → weight is 2D Linear, needs unsqueeze.
    linear_weight_patterns = [
        # per-block attention + AdaLN
        re.compile(r"^transformer_blocks\.\d+\.attn_norm\.linear\.weight$"),
        re.compile(r"^transformer_blocks\.\d+\.attn\.to_q\.weight$"),
        re.compile(r"^transformer_blocks\.\d+\.attn\.to_k\.weight$"),
        re.compile(r"^transformer_blocks\.\d+\.attn\.to_v\.weight$"),
        re.compile(r"^transformer_blocks\.\d+\.attn\.to_out\.0\.weight$"),
        # final AdaLN + proj
        re.compile(r"^norm_out\.linear\.weight$"),
        re.compile(r"^proj_out\.weight$"),
    ]

    # FF rename patterns: ff.ff.0.0 → ff.fc1, ff.ff.2 → ff.fc2
    ff_fc1 = re.compile(r"^(transformer_blocks\.\d+\.ff)\.ff\.0\.0\.(weight|bias)$")
    ff_fc2 = re.compile(r"^(transformer_blocks\.\d+\.ff)\.ff\.2\.(weight|bias)$")

    for k, v in host_sd.items():
        new_key = k
        new_val = v

        # Rename FF subpaths first.
        m1 = ff_fc1.match(k)
        m2 = ff_fc2.match(k)
        if m1 is not None:
            new_key = f"{m1.group(1)}.fc1.{m1.group(2)}"
        elif m2 is not None:
            new_key = f"{m2.group(1)}.fc2.{m2.group(2)}"

        # Then unsqueeze 2D weights on matching keys.
        # Note: the FF fc1/fc2 weight shapes also need unsqueeze — match
        # against the NEW key (after rename).
        is_ff_weight = (
            (m1 is not None and m1.group(2) == "weight")
            or (m2 is not None and m2.group(2) == "weight")
        )
        is_linear_weight = any(p.match(k) for p in linear_weight_patterns)

        if (is_linear_weight or is_ff_weight) and new_val.ndim == 2:
            new_val = _unsqueeze_linear_weight(new_val)

        out[new_key] = new_val

    return out


def strip_unused_rotary(host_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove `rotary_embed.*` entries — ANEDiT rebuilds rotary tables in
    each ANEAttention, so the host `x_transformers.RotaryEmbedding.freqs`
    buffer has no counterpart to load into.
    """
    return {k: v for k, v in host_sd.items() if not k.startswith("rotary_embed.")}


def summarize_port(host_sd: Dict[str, torch.Tensor], ane_sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Small diagnostic helper for the compare script.

    Returns counts and the list of renamed / reshaped keys so the caller
    can log a one-line summary.
    """
    renamed = [k for k in ane_sd if k not in host_sd]
    dropped = [k for k in host_sd if k not in ane_sd and not k.startswith("rotary_embed.")]
    reshaped = [
        k for k, v in ane_sd.items()
        if k in host_sd and tuple(host_sd[k].shape) != tuple(v.shape)
    ]
    return {
        "host_count": len(host_sd),
        "ane_count": len(ane_sd),
        "renamed": renamed,
        "dropped_outside_rotary": dropped,
        "reshaped": reshaped,
    }
