#!/usr/bin/env python3
"""Test if cache works in PyTorch before CoreML conversion."""

import torch
from transformers import AutoModelForSpeechSeq2Seq
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
import sys
sys.path.insert(0, str(__file__).replace("test-pytorch-cache.py", ""))
# Import the wrapper class
import importlib.util
spec = importlib.util.spec_from_file_location("export_decoder", "export-decoder-cached.py")
export_decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_decoder)
MaskedCachedDecoderWrapper = export_decoder.MaskedCachedDecoderWrapper

print("="*70)
print("PyTorch Cache Test")
print("="*70)

# Load model
print("\n[1/3] Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
model.eval()

# Wrap decoder
print("\n[2/3] Wrapping decoder...")
wrapped = MaskedCachedDecoderWrapper(model, max_seq_len=108)
wrapped.eval()

# Test inputs
print("\n[3/3] Testing cache...")
input_id = torch.tensor([[13764]], dtype=torch.long)
encoder_hidden = torch.randn(1, 376, 1024)
cache_k = torch.zeros(8, 8, 108, 128)
cache_v = torch.zeros(8, 8, 108, 128)
cross_mask = torch.ones(1, 1, 1, 376)

print("\nStep 0:")
step_0 = torch.tensor([0], dtype=torch.int32)
with torch.no_grad():
    logits_0, new_cache_k_0, new_cache_v_0 = wrapped(input_id, encoder_hidden, cache_k, cache_v, step_0, cross_mask)

# Check cache
cache_k_norm = torch.sqrt(torch.sum(new_cache_k_0**2, dim=(0, 1, 3)))  # (108,)
cache_v_norm = torch.sqrt(torch.sum(new_cache_v_0**2, dim=(0, 1, 3)))  # (108,)
num_nonzero_k = torch.sum(cache_k_norm > 1e-8).item()
num_nonzero_v = torch.sum(cache_v_norm > 1e-8).item()
max_k = torch.max(cache_k_norm).item()
max_v = torch.max(cache_v_norm).item()

print(f"  Output cache_k: {num_nonzero_k} non-zero positions (max norm: {max_k:.6f})")
print(f"  Output cache_v: {num_nonzero_v} non-zero positions (max norm: {max_v:.6f})")
print(f"  Logits shape: {logits_0.shape}")

if num_nonzero_k == 0:
    print("\n❌ CACHE IS ALL ZEROS IN PYTORCH TOO!")
    print("   The export wrapper is broken, not just CoreML conversion")
else:
    print("\n✅ Cache works in PyTorch")
    print("   Issue is in CoreML conversion")

# Test step 1
print("\nStep 1:")
step_1 = torch.tensor([1], dtype=torch.int32)
next_token = torch.argmax(logits_0, dim=-1)
input_id_1 = next_token.unsqueeze(0)

with torch.no_grad():
    logits_1, new_cache_k_1, new_cache_v_1 = wrapped(input_id_1, encoder_hidden, new_cache_k_0, new_cache_v_0, step_1, cross_mask)

cache_k_norm_1 = torch.sqrt(torch.sum(new_cache_k_1**2, dim=(0, 1, 3)))
cache_v_norm_1 = torch.sqrt(torch.sum(new_cache_v_1**2, dim=(0, 1, 3)))
num_nonzero_k_1 = torch.sum(cache_k_norm_1 > 1e-8).item()
num_nonzero_v_1 = torch.sum(cache_v_norm_1 > 1e-8).item()
max_k_1 = torch.max(cache_k_norm_1).item()
max_v_1 = torch.max(cache_v_norm_1).item()

print(f"  Output cache_k: {num_nonzero_k_1} non-zero positions (max norm: {max_k_1:.6f})")
print(f"  Output cache_v: {num_nonzero_v_1} non-zero positions (max norm: {max_v_1:.6f})")
