"""Zero-shot Oxford-IIIT Pets (37 breeds) with the Core ML SigLIP2 encoders vs PyTorch."""

import argparse
import io
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoModel, AutoProcessor

DATASET = "timm/oxford-iiit-pet"
BREEDS = [
    "abyssinian", "american bulldog", "american pit bull terrier", "basset hound", "beagle",
    "bengal", "birman", "bombay", "boxer", "british shorthair", "chihuahua", "egyptian mau",
    "english cocker spaniel", "english setter", "german shorthaired", "great pyrenees", "havanese",
    "japanese chin", "keeshond", "leonberger", "maine coon", "miniature pinscher", "newfoundland",
    "persian", "pomeranian", "pug", "ragdoll", "russian blue", "saint bernard", "samoyed",
    "scottish terrier", "shiba inu", "siamese", "sphynx", "staffordshire bull terrier",
    "wheaten terrier", "yorkshire terrier",
]  # fmt: skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build/siglip2-base-patch16-256"))
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--template", default="a photo of a {}, a type of pet.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    config = json.loads((args.build_dir / "config.json").read_text())
    name = config["model_id"].split("/")[-1]
    units = ct.ComputeUnit.CPU_AND_NE
    image_model = ct.models.MLModel(
        str(args.build_dir / f"{name}-image-{args.precision}.mlpackage"), compute_units=units
    )
    text_model = ct.models.MLModel(
        str(args.build_dir / f"{name}-text-{args.precision}.mlpackage"), compute_units=units
    )
    model = AutoModel.from_pretrained(config["model_id"], dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(config["model_id"])

    table = pq.read_table(hf_hub_download(DATASET, "data/test-00000-of-00001.parquet", repo_type="dataset"))
    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()
    if args.limit:
        images, labels = images[: args.limit], labels[: args.limit]

    prompts = [args.template.format(b).lower() for b in BREEDS]
    tokens = processor(
        text=prompts, padding="max_length", max_length=config["text_length"], return_tensors="pt"
    ).input_ids
    with torch.no_grad():
        text_ref = torch.nn.functional.normalize(model.text_model(input_ids=tokens).pooler_output, dim=-1).numpy()
    text_cml = np.concatenate(
        [text_model.predict({"input_ids": r[None].numpy().astype(np.int32)})["text_embeds"] for r in tokens]
    )

    agree = correct_ref = correct_cml = 0
    latencies = []
    for item, label in zip(images, labels):
        image = Image.open(io.BytesIO(item["bytes"])).convert("RGB")
        pixels = processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            ref = torch.nn.functional.normalize(model.vision_model(pixel_values=pixels).pooler_output, dim=-1)
        start = time.perf_counter()
        cml = image_model.predict({"pixel_values": pixels.numpy()})["image_embeds"][0]
        latencies.append((time.perf_counter() - start) * 1000)
        pred_ref = int(np.argmax(text_ref @ ref.numpy()[0]))
        pred_cml = int(np.argmax(text_cml @ cml))
        agree += pred_ref == pred_cml
        correct_ref += pred_ref == label
        correct_cml += pred_cml == label

    n = len(labels)
    report = {
        "model_id": config["model_id"],
        "dataset": f"{DATASET} test",
        "template": args.template,
        "images": n,
        "top1_pytorch": correct_ref / n,
        "top1_coreml": correct_cml / n,
        "top1_agreement": agree / n,
        "image_latency_ms_median": float(np.median(latencies[5:] or latencies)),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
