"""fp32 reference from transformers' own Qwen3.5 decoder layers, streamed one layer at a time.

The FourBModel bf16 reference is noisy (bf16 activations; near-tie argmaxes flip), and a
whole fp32 HF model does not fit a 24 GB Mac. This runs the merged weights through
`Qwen3_5DecoderLayer` modules layer by layer on the UNPADDED prompt, so it is an
independent fp32 check of both the export math and the right-padding assumption.
Writes fixtures/reference-<modality>-fp32.json (same schema as the bf16 reference).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoProcessor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionModel,
)

BASE = "Qwen/Qwen3.5-4B"


class _RopeIndex:
    """Just enough of Qwen3_5Model to call its own get_rope_index."""

    get_vision_position_ids = Qwen3_5Model.get_vision_position_ids
    get_rope_index = Qwen3_5Model.get_rope_index

    def __init__(self, config):
        self.config = config


def image_inputs(args, full_cfg, records, emb):
    """Per record: (inputs_embeds [1, n, D] with image features spliced in, position_ids [3, 1, n])."""
    from PIL import Image

    processor = AutoProcessor.from_pretrained(BASE)
    vcfg = full_cfg.vision_config
    vcfg._attn_implementation = "eager"
    vision = Qwen3_5VisionModel(vcfg).eval()
    with safe_open(str(args.build / f"merged-{args.modality}" / "vision.safetensors"), "pt") as vf:
        vision.load_state_dict({k: vf.get_tensor(k).float() for k in vf.keys()}, strict=True)
    rope = _RopeIndex(full_cfg)
    out = []
    for r in records:
        ids = torch.tensor(r["input_ids"])[None]
        shot = Image.open(args.fixtures / "screens" / r["screenshot"]).convert("RGB")
        pix = processor.image_processor(images=[shot], return_tensors="pt")
        grid = pix["image_grid_thw"]
        assert grid[0].tolist() == r["image_grid_thw"], (grid, r["image_grid_thw"])
        with torch.no_grad():
            feats = vision(pix["pixel_values"].float(), grid_thw=grid).pooler_output
        h = emb[ids[0]].float()[None].clone()
        mask = ids[0] == full_cfg.image_token_id
        assert int(mask.sum()) == feats.shape[0]
        h[0, mask] = feats
        pos, _ = rope.get_rope_index(ids, mask.int()[None], image_grid_thw=grid)
        out.append((h, pos))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="text")
    ap.add_argument("--build", type=Path, default=Path("build"))
    ap.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    args = ap.parse_args()

    full_raw = json.loads(Path(hf_hub_download(BASE, "config.json")).read_text())
    raw = full_raw["text_config"]
    cfg = Qwen3_5TextConfig(**raw)
    cfg._attn_implementation = "eager"
    ref = json.loads((args.fixtures / f"reference-{args.modality}.json").read_text())
    letter_ids = ref["letter_token_ids"]
    f = safe_open(str(args.build / f"merged-{args.modality}" / "text.safetensors"), "pt")

    emb = f.get_tensor("embed_tokens.weight")
    letter_rows = emb[letter_ids].float()
    if args.modality == "multimodal":
        pairs = image_inputs(args, Qwen3_5Config(**full_raw), ref["records"], emb)
        hidden = [h for h, _ in pairs]
        positions = [p for _, p in pairs]
    else:
        hidden = [emb[torch.tensor(r["input_ids"])].float()[None] for r in ref["records"]]
        positions = [torch.arange(h.shape[1])[None, None].expand(3, 1, h.shape[1]) for h in hidden]
    del emb

    rot = Qwen3_5TextRotaryEmbedding(cfg)
    pos_emb, masks = [], []
    for h, pos in zip(hidden, positions):
        n = h.shape[1]
        pos_emb.append(rot(h, pos))
        masks.append(torch.full((n, n), float("-inf")).triu(1)[None, None])

    for i in range(cfg.num_hidden_layers):
        layer = Qwen3_5DecoderLayer(cfg, i).eval()
        prefix = f"layers.{i}."
        state = {k[len(prefix):]: f.get_tensor(k).float() for k in f.keys() if k.startswith(prefix)}
        layer.load_state_dict(state, strict=True)
        full = cfg.layer_types[i] == "full_attention"
        with torch.no_grad():
            for j, h in enumerate(hidden):
                hidden[j] = layer(h, position_embeddings=pos_emb[j], attention_mask=masks[j] if full else None)
        print(f"layer {i}", flush=True)

    norm = Qwen3_5RMSNorm(raw["hidden_size"], raw["rms_norm_eps"])
    norm.load_state_dict({"weight": f.get_tensor("norm.weight").float()})
    records = []
    for r, h, pos in zip(ref["records"], hidden, positions):
        n = len(r["letter_logits"])
        with torch.no_grad():
            logits = (norm(h[0, -1]) @ letter_rows.T)[:n]
        rec = {**r, "letter_logits": logits.tolist(), "argmax": int(logits.argmax())}
        if args.modality == "multimodal":
            rec["position_ids"] = pos[:, 0].tolist()
        records.append(rec)
    out = {**ref, "records": records, "precision": "fp32-hf-layers"}
    (args.fixtures / f"reference-{args.modality}-fp32.json").write_text(json.dumps(out))
    flips = sum(a["argmax"] != b["argmax"] for a, b in zip(ref["records"], records))
    print(f"bf16 FourBModel vs fp32 HF layers: {flips}/{len(records)} argmax flips")


if __name__ == "__main__":
    main()
