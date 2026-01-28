# %% [markdown]
# # Qwen3-TTS 12Hz Tokenizer Decoder Exploration
#
# This script explores the decoder architecture to understand what needs
# to be converted to CoreML.

# %%
import torch
import json
from pathlib import Path

# Load the tokenizer model
from qwen_tts import Qwen3TTSTokenizer

print("Loading tokenizer...")
tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "./tokenizer_12hz",
    device_map="cpu",
)

print(f"Tokenizer loaded: {type(tokenizer)}")

# %%
# Inspect the model structure
print("\n=== Model Structure ===")
model = tokenizer.model
print(f"Model type: {type(model)}")
print(f"Model config: {model.config}")

# %%
# Get the decoder specifically
print("\n=== Decoder Architecture ===")
if hasattr(model, 'decoder'):
    decoder = model.decoder
    print(f"Decoder type: {type(decoder)}")

    # Print all modules
    print("\nDecoder modules:")
    for name, module in decoder.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            print(f"  {name}: {type(module).__name__}")
else:
    print("No 'decoder' attribute found")
    print("Available attributes:", dir(model))

# %%
# Print full model structure
print("\n=== Full Model Hierarchy ===")
def print_model_tree(model, prefix=""):
    for name, child in model.named_children():
        print(f"{prefix}{name}: {type(child).__name__}")
        print_model_tree(child, prefix + "  ")

print_model_tree(model)

# %%
# Check model size
total_params = sum(p.numel() for p in model.parameters())
decoder_params = sum(p.numel() for p in model.decoder.parameters()) if hasattr(model, 'decoder') else 0
print(f"\n=== Model Size ===")
print(f"Total parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB in FP32)")
print(f"Decoder parameters: {decoder_params:,} ({decoder_params * 4 / 1e6:.1f} MB in FP32)")

# %%
# Try to understand the forward pass
print("\n=== Forward Pass Signature ===")
if hasattr(model, 'decode'):
    import inspect
    sig = inspect.signature(model.decode)
    print(f"model.decode{sig}")

if hasattr(model, 'forward'):
    sig = inspect.signature(model.forward)
    print(f"model.forward{sig}")

# %%
# Create sample input and trace
print("\n=== Testing Decode ===")
# Create dummy codec codes: [batch, num_quantizers, time]
# From config: 16 quantizers
batch_size = 1
num_quantizers = 16
seq_len = 10  # ~0.8 seconds of audio at 12.5Hz

dummy_codes = torch.randint(0, 2048, (batch_size, num_quantizers, seq_len))
print(f"Input codes shape: {dummy_codes.shape}")

# Try decoding
try:
    with torch.no_grad():
        output = tokenizer.decode(dummy_codes)
    print(f"Output type: {type(output)}")
    if isinstance(output, tuple):
        print(f"Output shapes: {[x.shape if hasattr(x, 'shape') else type(x) for x in output]}")
    elif hasattr(output, 'shape'):
        print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"Decode failed: {e}")
    import traceback
    traceback.print_exc()
