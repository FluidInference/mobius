"""Parity: export (PyTorch fp32 or Core ML parts) vs the fp32 (default) or bf16 reference.

Runs part by part over every fixture so only one part is resident at a time.
Multimodal: the vision tower runs first (same backend) on the processor's exact patches;
its features are spliced at the image-pad tokens and the reference's M-RoPE positions are used.
Reports argmax agreement and max |Δp| of the letter softmax over each task's options.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from qwen35_export import DecoderChunk, TextConfig, rope_cos_sin

BASE = "Qwen/Qwen3.5-4B"


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def padded_positions(pos: list[list[int]] | None, n: int, length: int) -> torch.Tensor:
    """[3, length]: the prompt's (t, h, w) positions, then padding continuing after the max."""
    if pos is None:
        return torch.arange(length)[None].expand(3, length)
    real = torch.tensor(pos)
    tail = torch.arange(length - n) + int(real.max()) + 1
    return torch.cat([real, tail[None].expand(3, -1)], dim=1)


def vision_embeds(args, records, vcfg_raw) -> tuple[list[np.ndarray], float]:
    """Image features per record (processor patches -> tower), plus median ms."""
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoProcessor

    from qwen35_vision import VisionTower, host_inputs, patches_from_pixel_values

    N = args.max_patches
    processor = AutoProcessor.from_pretrained(BASE)
    tower = VisionTower(vcfg_raw, N).eval()
    with safe_open(str(args.build / "merged-multimodal" / "vision.safetensors"), "pt") as f:
        table = tower.load_merged({k: f.get_tensor(k) for k in f.keys()})
    backend = args.vision_backend or args.backend
    if backend == "coreml":
        import coremltools as ct

        mlvision = ct.models.MLModel(
            str(args.build / "multimodal" / "vision" / (f"CuaS1Vision_P{N}" + (f"-{args.vision_variant}" if args.vision_variant else "") + ".mlpackage")),
            compute_units=getattr(ct.ComputeUnit, args.compute_units),
        )
    feats, times = [], []
    for r in records:
        shot = Image.open(args.fixtures / "screens" / r["screenshot"]).convert("RGB")
        pix = processor.image_processor(images=[shot], return_tensors="pt")
        _, gh, gw = pix["image_grid_thw"][0].tolist()
        inputs = (patches_from_pixel_values(pix["pixel_values"], N), *host_inputs(tower.cfg, table, gh, gw, N))
        t0 = time.perf_counter()
        if backend == "coreml":
            names = ["patches", "pos_embed", "cos", "sin", "key_mask"]
            out = mlvision.predict({k: v.numpy().astype(np.float16) for k, v in zip(names, inputs)})["image_embeds"]
        else:
            with torch.no_grad():
                out = tower(*inputs).numpy()
        times.append((time.perf_counter() - t0) * 1000)
        feats.append(np.asarray(out, dtype=np.float32)[: gh * gw // 4])
    return feats, float(np.median(times[1:] if len(times) > 1 else times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "multimodal"], default="text")
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--backend", choices=["torch", "coreml"], default="coreml")
    ap.add_argument("--compute-units", default="ALL")
    ap.add_argument("--build", type=Path, default=Path("build"))
    ap.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--reference", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--max-patches", type=int, default=4096)
    ap.add_argument("--vision-variant", default="", help="e.g. fp32 -> CuaS1Vision_P4096-fp32")
    ap.add_argument("--vision-backend", choices=["torch", "coreml"], default=None, help="defaults to --backend")
    ap.add_argument("--variant", default="", help="quantized build suffix, e.g. w8 -> L1024-w8")
    args = ap.parse_args()

    full_cfg = json.loads(Path(hf_hub_download(BASE, "config.json")).read_text())
    raw_cfg = full_cfg["text_config"]
    cfg = TextConfig(raw_cfg)
    model_dir = args.build / args.modality / (f"L{args.length}" + (f"-{args.variant}" if args.variant else ""))
    meta = json.loads((model_dir / "config.json").read_text())
    suffix = "-fp32" if args.reference == "fp32" else ""
    ref = json.loads((args.fixtures / f"reference-{args.modality}{suffix}.json").read_text())
    L, D = args.length, cfg.hidden_size

    emb = np.memmap(model_dir.parent / "embeddings.f16", dtype=np.float16, mode="r").reshape(-1, D)
    records = [r for r in ref["records"] if len(r["input_ids"]) <= L]
    skipped = len(ref["records"]) - len(records)
    vision_ms = None
    if args.modality == "multimodal":
        feats, vision_ms = vision_embeds(args, records, full_cfg["vision_config"])
        print(f"vision: median {vision_ms:.1f} ms", flush=True)

    hidden, cos_sin, onehots = [], [], []
    for j, r in enumerate(records):
        ids = np.array(r["input_ids"])
        h = np.zeros((1, L, D), dtype=np.float32)
        h[0, : len(ids)] = emb[ids]
        if args.modality == "multimodal":
            h[0, np.flatnonzero(ids == full_cfg["image_token_id"])] = feats[j]
        hidden.append(h)
        cos, sin = rope_cos_sin(cfg, padded_positions(r.get("position_ids"), len(ids), L))
        cos_sin.append((cos.numpy(), sin.numpy()))
        oh = np.zeros((1, L), dtype=np.float32)
        oh[0, len(ids) - 1] = 1
        onehots.append(oh)

    parts = meta["parts"]
    logits = [None] * len(records)
    part_ms = []
    for p, part in enumerate(parts):
        last = p == len(parts) - 1
        t_part = []
        if args.backend == "torch":
            from safetensors import safe_open

            start, end = part["layers"]
            prefixes = tuple(f"layers.{i}." for i in range(start, end)) + (("norm.", "embed_tokens.") if last else ())
            state = {}
            with safe_open(str(args.build / f"merged-{args.modality}" / "text.safetensors"), "pt") as f:
                for k in f.keys():
                    if k.startswith(prefixes):
                        state[k] = f.get_tensor(k)
            rows = state.pop("embed_tokens.weight")[meta["letter_token_ids"]].float() if last else None
            model = DecoderChunk(cfg, start, end, L, meta["delta_chunk_size"], last, len(meta["letter_token_ids"]))
            model.load_merged(state, rows)
            model.eval()
            del state

            def run(i, model=model, last=last):
                args_ = [torch.from_numpy(hidden[i]), torch.from_numpy(cos_sin[i][0]), torch.from_numpy(cos_sin[i][1])]
                if last:
                    args_.append(torch.from_numpy(onehots[i]))
                with torch.no_grad():
                    return model(*args_).numpy()
        else:
            import coremltools as ct

            model = ct.models.MLModel(
                str(model_dir / f"CuaS1Decoder_part{p}.mlpackage"),
                compute_units=getattr(ct.ComputeUnit, args.compute_units),
            )

            def run(i, model=model, last=last):
                feed = {
                    "hidden": hidden[i].astype(np.float16),
                    "cos": cos_sin[i][0].astype(np.float16),
                    "sin": cos_sin[i][1].astype(np.float16),
                }
                if last:
                    feed["last_onehot"] = onehots[i].astype(np.float16)
                out = model.predict(feed)
                return np.asarray(out["letter_logits" if last else "hidden_out"], dtype=np.float32)

        for i in range(len(records)):
            t0 = time.perf_counter()
            out = run(i)
            t_part.append((time.perf_counter() - t0) * 1000)
            if last:
                logits[i] = out.reshape(-1)
            else:
                hidden[i] = out
        part_ms.append(float(np.median(t_part[1:] if len(t_part) > 1 else t_part)))
        print(f"part {p}: median {part_ms[-1]:.1f} ms", flush=True)
        del model

    agree, max_dp, rows = 0, 0.0, []
    for r, lg in zip(records, logits):
        n = len(r["letter_logits"])
        want = softmax(np.array(r["letter_logits"]))
        got = softmax(lg[:n])
        dp = float(np.abs(got - want).max())
        ok = int(got.argmax()) == r["argmax"]
        agree += ok
        max_dp = max(max_dp, dp)
        rows.append({"id": r["id"], "tokens": len(r["input_ids"]), "argmax_ok": ok, "max_dp": dp})
    summary = {
        "backend": args.backend,
        "variant": args.variant or "fp16",
        "reference": args.reference,
        "compute_units": args.compute_units if args.backend == "coreml" else "cpu-fp32",
        "length": L,
        "tasks": len(records),
        "skipped_too_long": skipped,
        "argmax_agree": agree,
        "max_abs_dprob": max_dp,
        "vision_median_ms": vision_ms,
        "part_median_ms": part_ms,
        "total_median_ms": sum(part_ms) + (vision_ms or 0.0),
    }
    print(json.dumps(summary, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
