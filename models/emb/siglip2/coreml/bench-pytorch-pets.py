"""PyTorch (transformers) baseline for the Swift ImageSortCheck: same photos, prompts, and scoring."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

BREEDS = [
    "abyssinian", "american bulldog", "american pit bull terrier", "basset hound", "beagle", "bengal", "birman",
    "bombay", "boxer", "british shorthair", "chihuahua", "egyptian mau", "english cocker spaniel", "english setter",
    "german shorthaired", "great pyrenees", "havanese", "japanese chin", "keeshond", "leonberger", "maine coon",
    "miniature pinscher", "newfoundland", "persian", "pomeranian", "pug", "ragdoll", "russian blue", "saint bernard",
    "samoyed", "scottish terrier", "shiba inu", "siamese", "sphynx", "staffordshire bull terrier", "wheaten terrier",
    "yorkshire terrier",
]  # fmt: skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()
    cache = Path(os.path.expanduser("~/Library/Caches/FluidUse/image-sort/oxford-pets-test"))
    labels = {int(k): v for k, v in json.loads((cache / "manifest.json").read_text()).items()}
    ids = sorted(labels)
    dtype = getattr(torch, args.dtype)
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-256", dtype=dtype).eval().to(args.device)
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-256")
    prompts = [f"a photo of a {b}, a type of pet." for b in BREEDS]
    with torch.no_grad():
        tokens = processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt").input_ids
        text = model.text_model(input_ids=tokens.to(args.device)).pooler_output
        text = torch.nn.functional.normalize(text.float(), dim=-1)
        # warm up
        warm = processor(images=[Image.open(cache / f"{ids[0]}.jpg").convert("RGB")], return_tensors="pt")
        model.vision_model(pixel_values=warm.pixel_values.to(args.device, dtype))
        if args.device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        correct = 0
        for offset in range(0, len(ids), args.batch):
            chunk = ids[offset : offset + args.batch]
            images = [Image.open(cache / f"{i}.jpg").convert("RGB") for i in chunk]
            pixels = processor(images=images, return_tensors="pt").pixel_values.to(args.device, dtype)
            emb = torch.nn.functional.normalize(model.vision_model(pixel_values=pixels).pooler_output.float(), dim=-1)
            pred = (emb @ text.T).argmax(-1).cpu().tolist()
            correct += sum(int(p == labels[i]) for p, i in zip(pred, chunk))
        if args.device == "mps":
            torch.mps.synchronize()
        seconds = time.perf_counter() - start
    n = len(ids)
    print(json.dumps({"device": args.device, "batch": args.batch, "dtype": args.dtype, "photos": n,
                      "seconds": round(seconds, 2), "photos_per_s": round(n / seconds, 1),
                      "ms_per_photo": round(1000 * seconds / n, 2), "accuracy": round(correct / n, 4)}))


if __name__ == "__main__":
    main()
