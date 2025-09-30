#!/usr/bin/env python3
"""Check the CoreML model spec to see expected weight frames."""

from pathlib import Path

import coremltools as ct

# Load the embedding CoreML model
model_path = Path(__file__).parent / "embedding-community-1.mlpackage"

if not model_path.exists():
    print(f"Model not found at {model_path}")
    print("Available .mlpackage files:")
    for f in Path(__file__).parent.glob("*.mlpackage"):
        print(f"  {f.name}")
else:
    model = ct.models.MLModel(str(model_path))
    spec = model.get_spec()

    print("=== Embedding Model Input Specs ===\n")
    for input_desc in spec.description.input:
        name = getattr(input_desc, "name", "")
        print(f"Input: {name}")

        array_type = getattr(input_desc.type, "multiArrayType", None)
        if array_type is not None:
            shape = list(getattr(array_type, "shape", []))
            print(f"  Shape: {shape}")

            if name == "weights" and shape:
                weight_frames = int(shape[-1])
                print(f"  ⚠️ Weight frames: {weight_frames}")
