"""Merge a Cua-S1-4B-0.2 LoRA into Qwen3.5-4B and record the PyTorch reference.

For one modality:
  1. load the base with the model class `cua_s1.four_b.FourBModel` uses for that
     modality, attach the PEFT adapter (bf16, as FourBModel runs it) and merge the
     LoRA deltas in fp32;
  2. save the merged language-model weights (and the vision tower for
     multimodal) to build/merged-<modality>/;
  3. run the exact FourBModel prompt/readout on fixtures/tasks.jsonl and save
     input ids, letter token ids and fp32 letter logits to
     fixtures/reference-<modality>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from cua_bench_s1.task import load_jsonl
from cua_s1.four_b import LETTERS, Option, assign_letters, build_prompt
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

BASE = "Qwen/Qwen3.5-4B"
ADAPTER = "cua-ai/cua-s1-4b-0.2"


def merged_state(peft_model) -> dict[str, torch.Tensor]:
    """Base weights with every LoRA delta added in fp32 (W + scale * B @ A), stored fp16
    (matrices) / fp32 (vectors).

    merge_and_unload() on a bf16 model rounds W + delta back to bf16, which drops
    deltas smaller than the bf16 ulp of W; merging here keeps them for fp16 export."""
    from peft.tuners.lora import LoraLayer

    state = {}
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer):
            base = module.get_base_layer()
            w = base.weight.detach().float()
            for adapter in module.active_adapters:
                a = module.lora_A[adapter].weight.detach().float()
                b = module.lora_B[adapter].weight.detach().float()
                w = w + (b @ a) * module.scaling[adapter]
            key = name.removeprefix("base_model.model.")
            state[f"{key}.weight"] = w.half()
            if base.bias is not None:
                state[f"{key}.bias"] = base.bias.detach().float()
    for key, value in peft_model.state_dict().items():
        key = key.removeprefix("base_model.model.")
        if "lora_" in key:
            continue
        key = key.replace(".base_layer.", ".")
        if key == "lm_head.weight":  # tied to embed_tokens
            continue
        if key not in state:
            state[key] = value.detach().half() if value.ndim >= 2 else value.detach().float()
    return state


def split_state(state: dict) -> tuple[dict, dict]:
    """(language-model state relative to the text model, vision state relative to the visual module)."""
    text, vision = {}, {}
    for key, value in state.items():
        value = value.contiguous()
        for prefix in ("model.language_model.", "language_model.model.", "model."):
            if key.startswith(prefix) and not key.startswith("model.visual."):
                text[key[len(prefix):]] = value
                break
        if key.startswith("model.visual."):
            vision[key[len("model.visual."):]] = value
    return text, vision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["text", "multimodal"], required=True)
    ap.add_argument("--fixtures", type=Path, default=Path("fixtures"))
    ap.add_argument("--out", type=Path, default=Path("build"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    adapter_root = Path(snapshot_download(ADAPTER))
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    processor = AutoProcessor.from_pretrained(BASE) if args.modality == "multimodal" else None
    cls = AutoModelForImageTextToText if args.modality == "multimodal" else AutoModelForCausalLM
    model = cls.from_pretrained(BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_root / args.modality)).eval()
    print(f"loaded {args.modality} adapter on {type(model.base_model.model).__name__}")

    text_state, vision_state = split_state(merged_state(model))
    out_dir = args.out / f"merged-{args.modality}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(text_state, str(out_dir / "text.safetensors"))
    if vision_state:
        save_file(vision_state, str(out_dir / "vision.safetensors"))
    counts = (len(text_state), len(vision_state))
    del text_state, vision_state
    model.to(args.device)
    letter_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        assert len(ids) == 1, letter
        letter_ids.append(ids[0])
    (out_dir / "letters.json").write_text(json.dumps({"letters": LETTERS, "token_ids": letter_ids}))
    print(f"saved {counts[0]} text / {counts[1]} vision tensors to {out_dir}")

    tasks = [t for t in load_jsonl(args.fixtures / "tasks.jsonl") if len(t.options) <= len(LETTERS)]
    if args.limit:
        tasks = tasks[: args.limit]
    records = []
    for task in tasks:
        options = [Option(o.element_id, o.role, o.label, o.action, o.entity_id) for o in task.options]
        assignment = assign_letters(options)
        n = len(assignment.letters)
        if args.modality == "text":
            messages = build_prompt(
                assignment, app=task.app, task_family=task.family, ax_tree=task.ax_tree, goal=task.goal
            )
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat, return_tensors="pt")
        else:
            from PIL import Image

            shot = args.fixtures / "screens" / task.screenshot
            messages = build_prompt(
                assignment,
                app=task.app,
                task_family=task.family,
                screenshot=str(shot),
                modality="multimodal",
                goal=task.goal,
            )
            chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[chat], images=[Image.open(shot).convert("RGB")], return_tensors="pt")
        t0 = time.perf_counter()
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float().cpu()
        dt = time.perf_counter() - t0
        letter_logits = logits[torch.tensor(letter_ids[:n])].float()
        rec = {
            "id": task.id,
            "family": task.family,
            "chat": chat,
            "input_ids": inputs["input_ids"][0].cpu().tolist(),
            "letter_logits": letter_logits.tolist(),
            "argmax": int(letter_logits.argmax()),
        }
        if args.modality == "multimodal":
            rec["image_grid_thw"] = inputs["image_grid_thw"][0].cpu().tolist()
            rec["screenshot"] = task.screenshot
        records.append(rec)
        print(f"{task.id}: len={rec['input_ids'].__len__()} argmax={rec['argmax']} {dt:.1f}s", flush=True)

    ref = {"modality": args.modality, "letter_token_ids": letter_ids, "records": records}
    (args.fixtures / f"reference-{args.modality}.json").write_text(json.dumps(ref))


if __name__ == "__main__":
    main()
