"""Language-model rows for Core ML: Qwen3.5-0.8B decoder (merged student) -> post-norm hidden states [1, L, D].

Reuses the Kev / Intern-Decision prefill-only decoder (``qwen35_export.DecoderChunk``). Inputs: ``hidden`` [1, L, D]
(host embedding gather with the vision tower's tokens spliced at the image-pad positions), ``cos``/``sin`` [L, R]
(host M-RoPE tables from the 3-D position ids). The row is right-padded to the bucket L (multiple of the 64-token
delta-rule chunk); the model is causal so padding cannot change real tokens. The output feeds the head export.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qwen35_export import DecoderChunk, RMSNorm, TextConfig, rope_cos_sin

LM_PREFIX = "model.language_model."


class LMRows(nn.Module):
    def __init__(self, cfg: TextConfig, seq_len: int, chunk_size: int = 64):
        super().__init__()
        self.decoder = DecoderChunk(cfg, 0, cfg.num_layers, seq_len, chunk_size=chunk_size, with_head=False)
        self.norm = RMSNorm(cfg.hidden_size, cfg.eps)

    def forward(self, hidden, cos, sin):
        return self.norm(self.decoder(hidden, cos, sin))


def load_lm(merged_backbone: Path, seq_len: int, chunk_size: int = 64):
    """Returns (LMRows, TextConfig, embed_table fp32 [V, D], config dict)."""
    from safetensors.torch import load_file

    config = json.loads((merged_backbone / "config.json").read_text())
    cfg = TextConfig(config["text_config"])
    raw = load_file(str(merged_backbone / "model.safetensors"))
    state = {k[len(LM_PREFIX):]: v for k, v in raw.items() if k.startswith(LM_PREFIX) and ".mtp." not in k
             and not k[len(LM_PREFIX):].startswith("mtp")}
    rows = LMRows(cfg, seq_len, chunk_size)
    rows.decoder.load_merged(state)
    rows.norm.load_state_dict({"weight": state["norm.weight"].float()})
    embed = state["embed_tokens.weight"].float()
    return rows.eval(), cfg, embed, config


def mrope_position_ids(hf_model, input_ids: torch.Tensor, image_grid_thw: torch.Tensor | None,
                       image_token_id: int) -> torch.Tensor:
    """[3, n] M-RoPE positions for one row, from the HF model's own ``get_rope_index`` (host side)."""
    mm = (input_ids == image_token_id).long()[None]  # 1 = image token, 0 = text (no video here)
    position_ids, _ = hf_model.model.get_rope_index(input_ids=input_ids[None], image_grid_thw=image_grid_thw,
                                                    video_grid_thw=None, attention_mask=torch.ones_like(input_ids)[None],
                                                    mm_token_type_ids=mm)
    return position_ids[:, 0, :]


def row_inputs(cfg: TextConfig, embed: torch.Tensor, input_ids: torch.Tensor, position_ids: torch.Tensor,
               vision_tokens: torch.Tensor | None, image_token_id: int, seq_len: int, pad_id: int):
    """Right-pad one record to ``seq_len``: embeddings with vision tokens spliced in, M-RoPE cos/sin."""
    n = int(input_ids.shape[0])
    if n > seq_len:
        raise ValueError(f"record needs {n} tokens; bucket holds {seq_len}")
    ids = torch.full((seq_len,), pad_id, dtype=torch.long)
    ids[:n] = input_ids
    hidden = embed[ids].clone()
    if vision_tokens is not None:
        slots = (ids == image_token_id).nonzero(as_tuple=True)[0]
        assert slots.numel() == vision_tokens.shape[0], (slots.numel(), vision_tokens.shape)
        hidden[slots] = vision_tokens.float()
    pos = torch.zeros(3, seq_len, dtype=torch.long)
    pos[:, :n] = position_ids
    tail = position_ids.max().item() + 1 + torch.arange(seq_len - n)
    pos[:, n:] = tail[None, :]
    cos, sin = rope_cos_sin(cfg, pos)
    return hidden[None], cos, sin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path, default=Path("build/student-r1/merged/backbone"))
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--out", type=Path, default=Path("build/coreml"))
    args = ap.parse_args()
    import coremltools as ct

    rows, cfg, embed, config = load_lm(args.merged, args.length, args.chunk_size)
    L = args.length
    emb_path = args.out / "embeddings.f16"
    args.out.mkdir(parents=True, exist_ok=True)
    if not emb_path.exists():
        embed.to(torch.float16).numpy().tofile(emb_path)
    example = (torch.zeros(1, L, cfg.hidden_size), torch.zeros(L, cfg.rotary_dim), torch.zeros(L, cfg.rotary_dim))
    started = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(rows, example, check_trace=False)
    model = ct.convert(
        traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[ct.TensorType(name=n, shape=tuple(t.shape), dtype=np.float32) for n, t in zip(("hidden", "cos", "sin"), example)],
        outputs=[ct.TensorType(name="states", dtype=np.float32)],
    )
    model.short_description = "clef-vision-0.8b language model rows: Qwen3.5-0.8B decoder + final norm, one prefill pass"
    model.license = "Apache-2.0"
    model.user_defined_metadata.update({"length": str(L)})
    out = args.out / f"LM_L{L}"
    out.mkdir(parents=True, exist_ok=True)
    package = out / f"LMRows_{args.precision}.mlpackage"
    model.save(str(package))
    (out / "config.json").write_text(json.dumps({
        "length": L, "hidden_size": cfg.hidden_size, "rotary_dim": cfg.rotary_dim, "rope_theta": cfg.rope_theta,
        "mrope_section": cfg.mrope_section, "vocab_size": int(embed.shape[0]), "image_token_id": config["image_token_id"],
        "pad_id": 248044 if "pad_token_id" not in config else config["pad_token_id"], "precision": args.precision,
    }, indent=2) + "\n")
    print(f"saved {package} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
