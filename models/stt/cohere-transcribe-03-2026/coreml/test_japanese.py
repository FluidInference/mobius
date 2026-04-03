#!/usr/bin/env python3
"""Test Japanese generation with language parameter."""
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset

model_id = "CohereLabs/cohere-transcribe-03-2026"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    trust_remote_code=True
)
model.eval()

print("=" * 80)
print("TESTING JAPANESE GENERATION WITH LANGUAGE PARAMETER")
print("=" * 80)

# Load Japanese FLEURS sample
dataset = load_dataset("google/fleurs", "ja_jp", split="test", streaming=False)
sample = list(dataset)[0]

audio = sample["audio"]["array"]
reference = sample["transcription"]

print(f"\nReference text: {reference}")

# Test 1: Without language parameter (current approach)
print("\n" + "-" * 80)
print("Test 1: WITHOUT language parameter")
print("-" * 80)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
batch_size = inputs["input_features"].shape[0]
decoder_input_ids = torch.full(
    (batch_size, 1),
    model.generation_config.decoder_start_token_id,
    dtype=torch.long
)

with torch.no_grad():
    outputs = model.generate(
        input_features=inputs["input_features"],
        length=inputs.get("length"),
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=128,
    )
    hypothesis = processor.batch_decode(outputs, skip_special_tokens=True)[0]

print(f"Generated: {hypothesis}")
print(f"Matches reference: {hypothesis.strip() == reference.strip()}")

# Test 2: Check if processor supports language parameter
print("\n" + "-" * 80)
print("Test 2: Checking processor parameters")
print("-" * 80)

import inspect
sig = inspect.signature(processor.__call__)
print(f"Processor parameters: {list(sig.parameters.keys())}")

# Try passing language if supported
try:
    inputs_with_lang = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt",
        language="ja"
    )
    print("✓ Processor accepts 'language' parameter")
    
    with torch.no_grad():
        outputs = model.generate(
            input_features=inputs_with_lang["input_features"],
            length=inputs_with_lang.get("length"),
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=128,
        )
        hypothesis = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print(f"Generated with language='ja': {hypothesis}")
except TypeError as e:
    print(f"✗ Processor doesn't accept 'language' parameter: {e}")

# Test 3: Check generation config for language tokens
print("\n" + "-" * 80)
print("Test 3: Checking generation config")
print("-" * 80)

config_attrs = [attr for attr in dir(model.generation_config) if not attr.startswith('_')]
print("Generation config attributes:")
for attr in config_attrs[:15]:
    try:
        val = getattr(model.generation_config, attr)
        if not callable(val):
            print(f"  {attr}: {val}")
    except:
        pass

print("\n" + "=" * 80)
