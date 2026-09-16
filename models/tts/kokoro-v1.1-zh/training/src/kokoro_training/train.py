"""Single-speaker real-recording adaptation with resumable, bounded local runs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from .artifacts import sha256, verify_assets, write_json
from .model import CompatibleGenerator
from .supervised import AdaptationModel, SupervisedLoss


def state_hash(items):
    digest = hashlib.sha256()
    for key, value in sorted(items):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def load_data(directory):
    contract = json.loads((directory / "data-contract.json").read_text())
    if sha256(directory / "manifest.jsonl") != contract["manifest_sha256"]:
        raise ValueError("Prepared manifest checksum mismatch")
    records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    passages, sources, prepared, texts, seen_ids = {}, {}, {}, {}, set()
    for row in records:
        if row["id"] in seen_ids:
            raise ValueError("Duplicate recording ID")
        seen_ids.add(row["id"])
        if row["origin"] != "real_recording" or row["speaker_id"] != contract["speaker"]:
            raise ValueError("Expected the declared single real speaker")
        identities = [(passages, row["passage_id"]), (sources, row["source_sha256"]),
                      (prepared, row["targets_sha256"])]
        if row.get("normalized_text"):
            identities.append((texts, row["normalized_text"].strip().casefold()))
        for index, key in identities:
            if key in index and index[key] != row["split"]:
                raise ValueError("Train/evaluation leakage")
            index[key] = row["split"]
        target = directory / row["targets"]
        if not target.resolve().is_relative_to(directory.resolve()) or sha256(target) != row["targets_sha256"]:
            raise ValueError("Prepared target checksum/path mismatch")
    return records, contract


def batch_for(row, directory, device):
    return {key: value.to(device) for key, value in torch.load(
        directory / row["targets"], map_location="cpu", weights_only=True).items()}


@torch.no_grad()
def validate(model, criterion, rows, directory, device, crop_frames):
    model.eval()
    totals, by_language = {}, {}
    # Preserve the training RNG; every evaluation uses the same excitation/crops.
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith("cuda") else []):
        torch.manual_seed(123456)
        for row in rows:
            batch = batch_for(row, directory, device)
            frames = int(batch["durations"].sum())
            output = model(batch, crop_start=max(0, (frames - crop_frames) // 2), crop_frames=crop_frames)
            losses = criterion(output)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value) / len(rows)
            by_language.setdefault(row["language"], []).append(float(losses["total"]))
    model.train()
    return {**totals, "language_total": {k: sum(v) / len(v) for k, v in by_language.items()}}


def train(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    verify_assets(args.assets)
    rows, contract = load_data(args.data)
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    if args.micro:
        train_rows = [r for lang in ("en", "zh") for r in sorted(
            (r for r in train_rows if r["language"] == lang),
            key=lambda r: (r["alignment"]["mean_path_cost"], r["id"]))[:args.micro // 2]]
    if not train_rows or not dev_rows:
        raise ValueError("Training and development data both required")
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_config.update(manifest_sha256=contract["manifest_sha256"],
                      train_ids=[r["id"] for r in train_rows], dev_ids=[r["id"] for r in dev_rows])
    if args.output.exists() and not args.resume:
        raise ValueError("Refusing to overwrite a run; use --resume explicitly")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.assets / "config.json").read_text())
    generator = CompatibleGenerator(config)
    generator.load_checkpoint(torch.load(args.assets / "kokoro-v1_1-zh.pth", map_location="cpu", weights_only=True))
    voices = torch.load(args.assets / "voices/zf_001.pt", map_location="cpu", weights_only=True)
    model = AdaptationModel(generator, voices, train_style=args.train_style).to(args.device).train()
    criterion = SupervisedLoss().to(args.device)
    learned = [p for name, p in model.named_parameters() if p.requires_grad and name != "style_offset"]
    groups = [{"params": learned, "lr": args.lr}]
    if args.train_style:
        groups.append({"params": [model.style_offset], "lr": args.style_lr})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.01)
    frozen_before = state_hash((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    trainable_before = state_hash((n, p) for n, p in model.named_parameters() if p.requires_grad)
    step, best = 0, float("inf")
    sample_rng = random.Random(args.seed)
    if args.resume:
        saved = torch.load(args.output / "last.pt", map_location="cpu", weights_only=True)
        for key in ("manifest_sha256", "train_ids", "dev_ids", "lr", "style_lr", "train_style", "seed", "accumulation", "crop_frames", "device"):
            if saved["config"][key] != run_config[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"])
        if str(args.device).startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        sample_rng.setstate(saved["sample_rng"])
        step, best = saved["step"], saved["best"]
    else:
        write_json(args.output / "run-config.json", run_config)
        initial = validate(model, criterion, dev_rows, args.data, args.device, args.crop_frames)
        micro_initial = validate(model, criterion, train_rows, args.data, args.device, args.crop_frames) if args.micro else None
        write_json(args.output / "initial.json", {"development": initial, "micro_train": micro_initial,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "frozen_hash": frozen_before, "trainable_hash": trainable_before})
        print(json.dumps({"step": 0, "development": initial, "micro_train": micro_initial}), flush=True)
    start = time.monotonic()
    by_language = {lang: [r for r in train_rows if r["language"] == lang] for lang in ("en", "zh")}
    if not all(by_language.values()):
        raise ValueError("Both languages required")
    parameters = [p for p in model.parameters() if p.requires_grad]
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        total = {}
        used = []
        for _ in range(args.accumulation):
            row = sample_rng.choice(by_language[sample_rng.choice(("en", "zh"))])
            batch = batch_for(row, args.data, args.device)
            crop_start = sample_rng.randint(0, max(0, int(batch["durations"].sum()) - args.crop_frames))
            losses = criterion(model(batch, crop_start=crop_start, crop_frames=args.crop_frames))
            if not all(torch.isfinite(v) for v in losses.values()):
                raise FloatingPointError(f"Nonfinite loss at step {step}, {row['id']}")
            (losses["total"] / args.accumulation).backward()
            for key, value in losses.items():
                total[key] = total.get(key, 0.0) + float(value.detach()) / args.accumulation
            used.append(row["id"])
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        event = {"step": step, "loss": total, "gradient_norm_before_clip": float(grad_norm),
                 "recordings": used, "elapsed_seconds": time.monotonic() - start}
        if step % args.validate_every == 0 or step == args.steps:
            event["development"] = validate(model, criterion, dev_rows, args.data, args.device, args.crop_frames)
            improved = event["development"]["total"] < best
            best = min(best, event["development"]["total"])
            checkpoint = {"schema_version": 1, "step": step, "best": best, "config": run_config,
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                          "sample_rng": sample_rng.getstate()}
            atomic_save(checkpoint, args.output / "last.pt")
            if improved:
                atomic_save(checkpoint, args.output / "best.pt")
            if args.snapshot_every and step % args.snapshot_every == 0:
                atomic_save({k: checkpoint[k] for k in ("schema_version", "step", "config", "model")},
                            args.output / f"snapshot-{step}.pt")
            print(json.dumps(event), flush=True)
        elif step % 10 == 0:
            print(json.dumps({"step": step, "total": total["total"], "elapsed_seconds": event["elapsed_seconds"]}), flush=True)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
    frozen_after = state_hash((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    if frozen_before != frozen_after:
        raise AssertionError("Frozen model parameters changed")
    summary = {"completed_steps": step, "best_development_total": best, "frozen_parameters_unchanged": True,
               "trainable_parameters_changed": trainable_before != state_hash((n, p) for n, p in model.named_parameters() if p.requires_grad),
               "style_offset_rms": float(model.style_offset.detach().square().mean().sqrt()),
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "last_sha256": sha256(args.output / "last.pt"),
               "best_sha256": sha256(args.output / "best.pt"), "production_qualified": False}
    if args.micro:
        summary["micro_train_final"] = validate(model, criterion, train_rows, args.data, args.device, args.crop_frames)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path(".artifacts/baseline"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--style-lr", type=float, default=1e-3)
    parser.add_argument("--train-style", action="store_true")
    parser.add_argument("--micro", type=int, default=0)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--crop-frames", type=int, default=96)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.accumulation, args.crop_frames, args.validate_every) <= 0 or args.lr <= 0 or args.micro % 2:
        parser.error("Positive training settings and an even micro-set size required")
    train(args)


if __name__ == "__main__":
    main()
