#!/usr/bin/env python3
"""Inspect FluidInference/fleurs-full dataset structure for Japanese."""
from datasets import load_dataset

print("Loading Japanese samples...")
dataset = load_dataset(
    "FluidInference/fleurs-full",
    data_dir="ja_jp",
    split="train[:5]",  # Just load 5 samples
    trust_remote_code=True,
)

print(f"\nDataset size: {len(dataset)}")
print(f"Column names: {dataset.column_names}")

# Show first sample with all fields
print("\nFirst sample (all fields):")
sample = dataset[0]
for key, value in sample.items():
    if key == "audio":
        print(f"  {key}: {{array: {type(value['array'])}, sampling_rate: {value['sampling_rate']}}}")
    else:
        print(f"  {key}: {value}")
