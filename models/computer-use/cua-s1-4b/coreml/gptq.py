"""Calibrated (GPTQ) mixed-precision decoder: MLP int4 per block, other linears int8 per channel.

Plain round-to-nearest int4 on the MLPs (quantize.py --mode m4) flips 6/38 fixture argmaxes. GPTQ
compensates each column's rounding error using the layer-input Hessian from calibration prompts.

Calibration data is disjoint from every evaluation set: GUI-360 *train* episodes (the benchmark uses
test) and Cua generator tasks drawn with a different seed than fixtures/. Prompts are built with
the exact FourBModel text prompt, tokenized, and packed back to back into windows of the bucket
length (no padding, the usual GPTQ recipe).

Layers are compressed in order, each calibrated on the outputs of the already-compressed layers
before it. torch 2.7's MPS graph cache segfaults after ~5 GPTQ'd layers in one process, so
`calibrate` handles a few layers per process and hands activations over on disk:

    for s in 0 4 8 ... 28; do python gptq.py calibrate --start $s --end $((s + 4)); done
    python gptq.py assemble

coremltools' GPTQ registers compression metadata that ct.convert lowers to exact constexpr
int4/int8 weights. Output: build/text/L<len>-<tag>/ (same layout as the fp16 bucket).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.optimize.torch.layerwise_compression import (
    LayerwiseCompressor,
    LayerwiseCompressorConfig,
    ModuleGPTQConfig,
)
from cua_bench_s1.datagen.generator import generate_dataset
from cua_bench_s1.datagen.gui360 import convert_episode
from cua_bench_s1.datagen.specs import EXAMPLE_APPS
from cua_s1.four_b import LETTERS, Option, assign_letters, build_prompt
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer

from qwen35_export import DecoderChunk, TextConfig, rope_cos_sin

BASE = "Qwen/Qwen3.5-4B"


def _on_cpu(fn):
    """MPS lacks the Cholesky kernels GPTQ's Hessian solve uses; run those (small) ops on CPU."""

    def wrapped(x, *args, **kwargs):
        out = fn(x.cpu(), *args, **kwargs)
        return out.to(x.device)

    return wrapped


# Python 3.12 LogRecords call asyncio.current_task(); under MPS GPTQ that lookup segfaulted after ~40 records
logging.logAsyncioTasks = False

torch.cholesky_inverse = _on_cpu(torch.cholesky_inverse)
torch.linalg.cholesky = _on_cpu(torch.linalg.cholesky)
CALIBRATION_SEED = 7_777_777  # fixtures use 20260923


def calibration_prompts(gui360_train: Path, scratch: Path) -> list[str]:
    tasks = []
    for path in sorted(gui360_train.glob("*/*/success/*.jsonl")):
        tasks += convert_episode(
            path,
            images_root=Path("/nonexistent"),
            out_dir=scratch,
            modality_available=("text",),
            hard_distractor=True,
        )
    tasks += generate_dataset(EXAMPLE_APPS, 4, CALIBRATION_SEED, ("text",), scratch)
    tok = AutoTokenizer.from_pretrained(BASE)
    prompts = []
    for task in tasks:
        if len(task.options) > len(LETTERS):
            continue
        assignment = assign_letters(
            [Option(o.element_id, o.role, o.label, o.action, o.entity_id) for o in task.options]
        )
        msgs = build_prompt(assignment, app=task.app, task_family=task.family, ax_tree=task.ax_tree, goal=task.goal)
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return prompts


def packed_windows(prompts: list[str], length: int, nsamples: int) -> list[list[int]]:
    tok = AutoTokenizer.from_pretrained(BASE)
    stream: list[int] = []
    for prompt in prompts:
        stream += tok(prompt)["input_ids"]
    windows = [stream[i : i + length] for i in range(0, len(stream) - length + 1, length)]
    return windows[:nsamples]


class CalibrationLayer(nn.Module):
    """A decoder layer with the bucket's cos/sin/mask bound, so it takes only hidden states."""

    def __init__(self, layer, cos, sin, mask):
        super().__init__()
        self.layer = layer
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        return self.layer(x, self.cos, self.sin, self.mask)


class CalibrationStack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


def load_layer_weights(cfg: TextConfig, weights: Path, layer_ids: range, L: int, chunk_size: int) -> DecoderChunk:
    """fp32 DecoderChunk for layer_ids (no head) with the merged fp16 weights."""
    prefixes = tuple(f"layers.{i}." for i in layer_ids)
    with safe_open(str(weights), "pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefixes)}
    chunk = DecoderChunk(cfg, layer_ids.start, layer_ids.stop, L, chunk_size)
    chunk.load_merged(state)
    return chunk.eval()


def restore_compressed(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a GPTQ'd state dict, re-registering coremltools' compression-metadata buffers."""
    for key, value in state.items():
        if "_COREML_" not in key:
            continue
        owner, _, name = key.rpartition(".")
        target = module.get_submodule(owner) if owner else module
        target.register_buffer(name, value.clone())
    missing, unexpected = module.load_state_dict(state, strict=False)
    missing = [m for m in missing if "mask" not in m]
    if missing or unexpected:
        raise RuntimeError(f"compressed state mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


def calibrate(args, cfg: TextConfig, meta: dict, work: Path) -> None:
    """GPTQ layers [start, end) in this process (torch 2.7's MPS graph cache crashes after ~5 layers)."""
    L = args.length
    inputs_path = work / f"hidden_{args.start}.pt"
    if args.start == 0:
        windows = packed_windows(calibration_prompts(args.gui360_train, work / "_scratch"), L, args.nsamples)
        shutil.rmtree(work / "_scratch", ignore_errors=True)
        print(f"calibration: {len(windows)} windows x {L} tokens", flush=True)
        emb = np.memmap(args.build / "text" / "embeddings.f16", dtype=np.float16, mode="r")
        emb = emb.reshape(-1, cfg.hidden_size)
        torch.save([torch.from_numpy(emb[np.array(w)].astype(np.float32))[None] for w in windows], inputs_path)
    hidden = torch.load(inputs_path)

    cos, sin = rope_cos_sin(cfg, torch.arange(L)[None].expand(3, L))
    int4 = ModuleGPTQConfig(weight_dtype="uint4", granularity="per_block", block_size=args.block)
    int8 = ModuleGPTQConfig(weight_dtype="uint8", granularity="per_channel")  # u-dtypes: bit width; symmetric
    t0 = time.time()
    chunk = load_layer_weights(
        cfg, args.build / "merged-text" / "text.safetensors", range(args.start, args.end), L, meta["delta_chunk_size"]
    )
    stack = CalibrationStack([CalibrationLayer(layer, cos, sin, chunk.mask) for layer in chunk.layers])
    names = {n: (int4 if ".mlp." in n else int8) for n, m in stack.named_modules() if isinstance(m, nn.Linear)}
    compressor = LayerwiseCompressor(
        stack,
        LayerwiseCompressorConfig(layers=stack.layers, module_name_configs=names, calibration_nsamples=len(hidden)),
    )
    with torch.no_grad():
        compressor.compress(hidden, device=args.device, inplace=True)
        stack.to(args.device)
        hidden = [stack_forward(stack, h.to(args.device)).cpu() for h in hidden]
    stack.cpu()
    for i, layer in zip(range(args.start, args.end), chunk.layers):
        torch.save(layer.state_dict(), work / f"layer_{i}.pt")
    torch.save(hidden, work / f"hidden_{args.end}.pt")
    print(f"layers {args.start}-{args.end - 1}: GPTQ {time.time() - t0:.0f}s", flush=True)


def assemble(args, cfg: TextConfig, meta: dict, work: Path, dst: Path) -> None:
    """Rebuild each part from the GPTQ'd layers and convert (compression metadata -> constexpr weights)."""
    L = args.length
    cos, sin = rope_cos_sin(cfg, torch.arange(L)[None].expand(3, L))
    weights = args.build / "merged-text" / "text.safetensors"
    parts = meta["parts"]
    for p, part in enumerate(parts):
        t0 = time.time()
        start, end = part["layers"]
        last = p == len(parts) - 1
        chunk = DecoderChunk(cfg, start, end, L, meta["delta_chunk_size"], last, len(LETTERS))
        if last:
            with safe_open(str(weights), "pt") as f:
                chunk.norm.weight.data = f.get_tensor("norm.weight").float()
                chunk.letter_head.weight.data = f.get_tensor("embed_tokens.weight")[meta["letter_token_ids"]].float()
        for local, i in enumerate(range(start, end)):
            restore_compressed(chunk.layers[local], torch.load(work / f"layer_{i}.pt"))
        chunk.eval()
        ex = [torch.zeros(1, L, cfg.hidden_size), cos, sin] + ([torch.zeros(1, L)] if last else [])
        inputs = [
            ct.TensorType("hidden", shape=(1, L, cfg.hidden_size), dtype=np.float16),
            ct.TensorType("cos", shape=(L, cfg.rotary_dim), dtype=np.float16),
            ct.TensorType("sin", shape=(L, cfg.rotary_dim), dtype=np.float16),
        ] + ([ct.TensorType("last_onehot", shape=(1, L), dtype=np.float16)] if last else [])
        with torch.no_grad():
            traced = torch.jit.trace(chunk, tuple(ex), check_trace=False)
        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=[ct.TensorType("letter_logits" if last else "hidden_out", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
        )
        mlmodel.short_description = (
            f"Cua-S1-4B-0.2 (text) layers {start}-{end - 1}, L={L}, GPTQ MLP int4/{args.block} + int8"
        )
        mlmodel.save(str(dst / f"CuaS1Decoder_part{p}.mlpackage"))
        print(f"part {p}: converted ({time.time() - t0:.0f}s)", flush=True)
        del chunk, traced, mlmodel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["calibrate", "assemble"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=4)
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--nsamples", type=int, default=32)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--tag", default="g4b16")
    ap.add_argument("--build", type=Path, default=Path("build"))
    ap.add_argument("--device", default="mps", help="calibration forwards dominate; cpu is ~10x slower")
    ap.add_argument("--gui360-train", type=Path, default=Path("data/gui360/train/data"))
    args = ap.parse_args()

    cfg = TextConfig(json.loads(Path(hf_hub_download(BASE, "config.json")).read_text())["text_config"])
    src = args.build / "text" / f"L{args.length}"
    dst = args.build / "text" / f"L{args.length}-{args.tag}"
    work = args.build / f"gptq-{args.tag}"
    dst.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "config.json").read_text())
    shutil.copy(src / "config.json", dst / "config.json")
    if args.command == "calibrate":
        calibrate(args, cfg, meta, work)
    else:
        assemble(args, cfg, meta, work, dst)


def stack_forward(stack: CalibrationStack, h: torch.Tensor) -> torch.Tensor:
    for layer in stack.layers:
        h = layer(h)
    return h


if __name__ == "__main__":
    main()
