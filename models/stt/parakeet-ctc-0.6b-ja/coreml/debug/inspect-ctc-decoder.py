#!/usr/bin/env python3
"""Inspect the CTC decoder's forward method to see what it actually does."""
import inspect
import torch
import nemo.collections.asr as nemo_asr

# Load model
print("Loading NeMo model...")
asr_model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet-tdt_ctc-0.6b-ja", map_location="cpu"
)

ctc_decoder = asr_model.ctc_decoder

print(f"\n=== CTC Decoder Class: {type(ctc_decoder).__name__} ===\n")
print(f"Module: {ctc_decoder.__class__.__module__}")
print(f"Full path: {ctc_decoder.__class__}")

# Get the source code
print("\n=== CTC Decoder forward() method source ===\n")
try:
    source = inspect.getsource(ctc_decoder.forward)
    print(source)
except Exception as e:
    print(f"Could not get source: {e}")

# Get the entire class source
print("\n=== Full CTC Decoder class source ===\n")
try:
    source = inspect.getsource(ctc_decoder.__class__)
    print(source)
except Exception as e:
    print(f"Could not get source: {e}")

# List all methods
print("\n=== CTC Decoder methods ===\n")
for name in dir(ctc_decoder):
    if not name.startswith('_'):
        attr = getattr(ctc_decoder, name)
        if callable(attr):
            print(f"  {name}: {type(attr)}")

# Check for log_softmax
print("\n=== Checking attributes ===\n")
for name, value in vars(ctc_decoder).items():
    print(f"  {name}: {type(value).__name__} = {value}")
