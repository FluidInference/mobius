#!/usr/bin/env python3
"""Compare stateful vs stateless decoder on the same input.

This helps debug where the outputs diverge.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import coremltools as ct
from datasets import load_dataset
from cohere_mel_spectrogram import CohereMelSpectrogram

print("="*70)
print("Debug: Stateful vs Stateless Decoder Comparison")
print("="*70)

# Configuration
PROMPT_IDS = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
MAX_STEPS = 20  # Just test first 20 steps

# Load one sample
print("\n[1/4] Loading 1 sample from LibriSpeech...")
dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
sample = next(iter(dataset))
print(f"   ✓ Ground truth: \"{sample['text'].lower()}\"")

# Load models
print("\n[2/4] Loading CoreML models...")
stateful_decoder = ct.models.MLModel(
    "build/cohere_decoder_stateful.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
encoder = ct.models.MLModel(
    "build/cohere_encoder.mlpackage",
    compute_units=ct.ComputeUnit.CPU_AND_GPU
)
print("   ✓ Models loaded")

# Process audio
print("\n[3/4] Processing audio...")
audio = sample['audio']['array'].astype(np.float32)
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)
mel_padded = np.pad(
    mel,
    ((0, 0), (0, 0), (0, 3001 - mel.shape[2])),
    mode='constant',
    constant_values=0
)

# Encode
encoder_output = encoder.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([mel.shape[2]], dtype=np.int32)
})

encoder_hidden = None
for key, value in encoder_output.items():
    if hasattr(value, 'shape') and len(value.shape) == 3:
        encoder_hidden = value
        break

cross_attention_mask = np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16)
print(f"   ✓ Encoder output shape: {encoder_hidden.shape}")

# Run decoder twice to check self-consistency
print("\n[4/4] Testing self-consistency (running same sequence twice)...")

# Collect token sequence
token_sequence = list(PROMPT_IDS)

# Run 1: Generate tokens
state1 = stateful_decoder.make_state()
run1_tokens = []

for step in range(MAX_STEPS):
    current_token = token_sequence[step] if step < len(token_sequence) else run1_tokens[-1]

    input_dict = {
        "input_id": np.array([[current_token]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "cross_attention_mask": cross_attention_mask,
        "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float16),
        "position_ids": np.array([[step]], dtype=np.int32),
    }
    output = stateful_decoder.predict(input_dict, state=state1)
    next_token = int(np.argmax(output["logits"][0]))
    run1_tokens.append(next_token)

# Run 2: Replay EXACT same token inputs as Run 1
state2 = stateful_decoder.make_state()
run2_tokens = []

# Build full input sequence: prompt + outputs from previous steps
# Step 0-9: use prompt tokens
# Step 10+: use output from previous step (run1_tokens[step-1])
full_input_sequence = []
for step in range(MAX_STEPS):
    if step < len(PROMPT_IDS):
        full_input_sequence.append(PROMPT_IDS[step])
    else:
        # Use the output from the previous step
        full_input_sequence.append(run1_tokens[step - 1])

for step in range(MAX_STEPS):
    current_token = full_input_sequence[step]

    input_dict = {
        "input_id": np.array([[current_token]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "cross_attention_mask": cross_attention_mask,
        "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float16),
        "position_ids": np.array([[step]], dtype=np.int32),
    }
    output = stateful_decoder.predict(input_dict, state=state2)
    next_token = int(np.argmax(output["logits"][0]))
    run2_tokens.append(next_token)

print(f"\nInput tokens (prompt + run1 outputs): {full_input_sequence[:15]}...")

# Compare runs
print(f"\n{'Step':<6} {'Run1':<10} {'Run2':<10} {'Match':<8} {'Note'}")
print("-" * 70)

all_match = True
for step in range(MAX_STEPS):
    t1 = run1_tokens[step]
    t2 = run2_tokens[step]
    match = "✓" if t1 == t2 else "✗"
    note = "(prompt)" if step < len(PROMPT_IDS) else "(generated)"

    if t1 != t2:
        all_match = False

    print(f"{step:<6} {t1:<10} {t2:<10} {match:<8} {note}")

if all_match:
    print("\n✅ Model is self-consistent!")
else:
    print("\n❌ Model is NOT self-consistent - outputs diverge across runs!")

print("\n" + "="*70)
