"""Convert a fixed-resolution SigLIP2 checkpoint to Core ML image and text encoders."""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision = model.vision_model.eval()

    def forward(self, pixel_values):
        pooled = self.vision(pixel_values=pixel_values).pooler_output
        return pooled / pooled.norm(dim=-1, keepdim=True)


class TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text = model.text_model.eval()

    def forward(self, input_ids):
        pooled = self.text(input_ids=input_ids).pooler_output
        return pooled / pooled.norm(dim=-1, keepdim=True)


def convert(module, example, input_name, input_dtype, output_name, precision):
    traced = torch.jit.trace(module, example)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=input_name, shape=example.shape, dtype=input_dtype)],
        outputs=[ct.TensorType(name=output_name, dtype=np.float32)],
        compute_precision=precision,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="google/siglip2-base-patch16-256")
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_id, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    vision_config = model.config.vision_config
    size = vision_config.image_size
    text_len = model.config.text_config.max_position_embeddings
    precision = ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    name = args.model_id.split("/")[-1]
    out = args.output_dir / name
    out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        image = convert(
            ImageEncoder(model),
            torch.zeros(1, 3, size, size),
            "pixel_values",
            np.float32,
            "image_embeds",
            precision,
        )
        text = convert(
            TextEncoder(model),
            torch.zeros(1, text_len, dtype=torch.int32),
            "input_ids",
            np.int32,
            "text_embeds",
            precision,
        )

    for package, kind in ((image, "image"), (text, "text")):
        package.author = "FluidInference (converted from Google SigLIP 2)"
        package.license = "Apache-2.0"
        package.short_description = f"{args.model_id} {kind} encoder, L2-normalized embeddings"
        package.save(str(out / f"{name}-{kind}-{args.precision}.mlpackage"))

    image_processor = processor.image_processor
    config = {
        "model_id": args.model_id,
        "image_size": size,
        "image_mean": list(image_processor.image_mean),
        "image_std": list(image_processor.image_std),
        "resample": "bilinear",
        "text_length": text_len,
        "text_padding": "max_length",
        "text_lowercase": True,
        "logit_scale": float(model.logit_scale.exp()),
        "logit_bias": float(model.logit_bias),
        "embedding_dim": vision_config.hidden_size,
        "precision": args.precision,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
