"""Compare Core ML SigLIP2 encoders with PyTorch on ImageNet-1k zero-shot classification."""

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModel, AutoProcessor

DATASET = "clip-benchmark/wds_imagenet1k"
COMPUTE_UNITS = {
    "all": ct.ComputeUnit.ALL,
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu": ct.ComputeUnit.CPU_ONLY,
}


def iter_samples(root: Path, limit: int):
    shards = sorted((root / "test").glob("*.tar"), key=lambda p: int(p.stem))
    count = 0
    for shard in shards:
        with tarfile.open(shard) as tar:
            pending = {}
            for member in tar:
                key, _, ext = member.name.rpartition(".")
                pending.setdefault(key, {})[ext] = tar.extractfile(member).read()
                sample = pending[key]
                image_ext = next((e for e in ("jpg", "jpeg", "png", "webp") if e in sample), None)
                if image_ext and "cls" in sample:
                    del pending[key]
                    image = Image.open(io.BytesIO(sample[image_ext])).convert("RGB")
                    yield image, int(sample["cls"].decode().strip())
                    count += 1
                    if count >= limit:
                        return


def text_prompts(root: Path, template: str):
    classnames = (root / "classnames.txt").read_text().splitlines()
    return [template.format(c).lower() for c in classnames if c]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build/siglip2-base-patch16-256"))
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--compute-units", default="all", choices=COMPUTE_UNITS)
    parser.add_argument("--template", default="this is a photo of {}.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config = json.loads((args.build_dir / "config.json").read_text())
    name = config["model_id"].split("/")[-1]
    units = COMPUTE_UNITS[args.compute_units]
    image_model = ct.models.MLModel(
        str(args.build_dir / f"{name}-image-{args.precision}.mlpackage"), compute_units=units
    )
    text_model = ct.models.MLModel(
        str(args.build_dir / f"{name}-text-{args.precision}.mlpackage"), compute_units=units
    )
    model = AutoModel.from_pretrained(config["model_id"], dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(config["model_id"])
    root = Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=["test/*", "*.txt"]))

    prompts = text_prompts(root, args.template)
    tokens = processor(
        text=prompts, padding="max_length", max_length=config["text_length"], return_tensors="pt"
    ).input_ids
    with torch.no_grad():
        text_ref = model.text_model(input_ids=tokens).pooler_output
        text_ref = torch.nn.functional.normalize(text_ref, dim=-1).numpy()
    text_cml = np.concatenate(
        [
            text_model.predict({"input_ids": row[None].numpy().astype(np.int32)})["text_embeds"]
            for row in tokens
        ]
    )
    text_cos = (text_ref * text_cml).sum(-1)

    image_cos, agree, correct_ref, correct_cml, latencies = [], 0, 0, 0, []
    for image, label in iter_samples(root, args.limit):
        pixels = processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            ref = model.vision_model(pixel_values=pixels).pooler_output
            ref = torch.nn.functional.normalize(ref, dim=-1)
        ref = ref.numpy()[0]
        start = time.perf_counter()
        cml = image_model.predict({"pixel_values": pixels.numpy()})["image_embeds"][0]
        latencies.append((time.perf_counter() - start) * 1000)
        image_cos.append(float(ref @ cml))
        pred_ref = int(np.argmax(text_ref @ ref))
        pred_cml = int(np.argmax(text_cml @ cml))
        agree += pred_ref == pred_cml
        correct_ref += pred_ref == label
        correct_cml += pred_cml == label

    n = len(image_cos)
    report = {
        "model_id": config["model_id"],
        "precision": args.precision,
        "compute_units": args.compute_units,
        "template": args.template,
        "images": n,
        "text_cosine_min": float(text_cos.min()),
        "text_cosine_mean": float(text_cos.mean()),
        "image_cosine_min": float(np.min(image_cos)),
        "image_cosine_mean": float(np.mean(image_cos)),
        "top1_agreement": agree / n,
        "top1_pytorch": correct_ref / n,
        "top1_coreml": correct_cml / n,
        "image_latency_ms_median": float(np.median(latencies[5:] or latencies)),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
