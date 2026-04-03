#!/usr/bin/env python3
"""Inspect FluidInference/fleurs-full dataset structure."""
from datasets import load_dataset

print("Loading FluidInference/fleurs-full...")
dataset = load_dataset("FluidInference/fleurs-full", split="train")

print(f"\nDataset size: {len(dataset)}")
print(f"Column names: {dataset.column_names}")

# Show first few samples
print("\nFirst 5 samples:")
for i in range(min(5, len(dataset))):
    sample = dataset[i]
    print(f"\nSample {i}:")
    for key, value in sample.items():
        if key == "audio":
            print(f"  {key}: {{array shape: {value['array'].shape if hasattr(value['array'], 'shape') else len(value['array'])}, sampling_rate: {value['sampling_rate']}}}")
        else:
            print(f"  {key}: {value}")

# Try to find Japanese samples
print("\n\nSearching for Japanese samples...")
ja_count = 0
for i in range(min(100, len(dataset))):
    sample = dataset[i]
    # Check different possible fields
    sample_str = str(sample)
    if "ja_jp" in sample_str or "japanese" in sample_str.lower():
        ja_count += 1
        if ja_count <= 3:
            print(f"\nFound Japanese sample at index {i}:")
            print(f"  {sample}")

print(f"\nFound {ja_count} Japanese samples in first 100 samples")
