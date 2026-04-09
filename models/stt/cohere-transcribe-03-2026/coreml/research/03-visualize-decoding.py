#!/usr/bin/env python3
"""Visualize decoder behavior over multiple steps."""

import numpy as np
import coremltools as ct
import json
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq
import torch

print("="*80)
print("EXPERIMENT 3: Visualize Decoding Over Time")
print("="*80)

# Configuration
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 160
N_FFT = 400
MAX_FRAMES = 3500
NUM_STEPS = 30  # Decode 30 tokens

# Load vocabulary
print("\n[1/6] Loading vocabulary...")
with open("f16/vocab.json") as f:
    vocab = {int(k): v for k, v in json.load(f).items()}
print(f"✓ Loaded {len(vocab)} tokens")

# Load encoder
print("\n[2/6] Loading PyTorch encoder...")
encoder_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "CohereLabs/cohere-transcribe-03-2026",
    trust_remote_code=True,
    torch_dtype=torch.float32,
)
encoder_model.eval()
print("✓ Loaded encoder")

# Load decoders
print("\n[3/6] Loading decoders...")
baseline_decoder = ct.models.MLModel(
    "hf-upload/cohere-transcribe-cache-external-coreml/cohere_decoder_cache_external.mlpackage"
)
english_decoder = ct.models.MLModel(
    "build-per-language/cohere_decoder_english.mlpackage"
)
print("✓ Loaded baseline and English decoders")

# Load a real audio sample
print("\n[4/6] Loading FLEURS English sample...")
dataset = load_dataset(
    "google/fleurs", "en_us", split="test", trust_remote_code=True
)
sample = dataset[0]
audio = sample["audio"]["array"]
sr = sample["audio"]["sampling_rate"]
reference = sample["transcription"]

print(f"✓ Loaded sample")
print(f"  Reference: {reference[:80]}...")

# Compute mel spectrogram
def compute_mel_spectrogram(audio, sr=SAMPLE_RATE):
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=8000,
    )

    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80) / 80
    mel = np.clip(mel, -1, 1)

    return mel

def pad_mel(mel, target_frames=MAX_FRAMES):
    n_mels, n_frames = mel.shape

    if n_frames >= target_frames:
        return mel[:, :target_frames], n_frames

    padded = np.zeros((n_mels, target_frames), dtype=np.float32)
    padded[:, :n_frames] = mel

    return padded, n_frames

mel = compute_mel_spectrogram(audio, sr)
mel_padded, actual_frames = pad_mel(mel)

# Encode
print("\n[5/6] Encoding audio...")
with torch.no_grad():
    input_features = torch.from_numpy(mel_padded[np.newaxis, :, :]).float()
    feature_length = torch.tensor([actual_frames], dtype=torch.int32)

    encoder_hidden, encoder_length = encoder_model.encoder(
        input_features=input_features,
        length=feature_length,
    )

    if encoder_model.encoder_decoder_proj is not None:
        encoder_hidden = encoder_model.encoder_decoder_proj(encoder_hidden)

encoder_hidden_np = encoder_hidden.numpy()
encoder_seq_len = encoder_hidden_np.shape[1]
print(f"✓ Encoder output shape: {encoder_hidden_np.shape}")

# Decode with both decoders, tracking logits
print(f"\n[6/6] Decoding {NUM_STEPS} steps...")

def decode_and_track(decoder, decoder_name):
    """Decode and track all logits."""
    k_caches = [np.zeros((1, 8, 108, 128), dtype=np.float32) for _ in range(8)]
    v_caches = [np.zeros((1, 8, 108, 128), dtype=np.float32) for _ in range(8)]

    cross_mask = np.ones((1, 1, 1, encoder_seq_len), dtype=np.float32)

    tokens = []
    all_logits = []
    current_token = 4  # START

    for step in range(NUM_STEPS):
        # Build input
        inputs = {
            "input_id": np.array([[current_token]], dtype=np.int32),
            "position_id": np.array([[step]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden_np,
            "cross_attention_mask": cross_mask,
            "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float32),
        }

        for i in range(8):
            inputs[f"k_cache_{i}"] = k_caches[i]
            inputs[f"v_cache_{i}"] = v_caches[i]

        # Run decoder
        outputs = decoder.predict(inputs)

        # Get logits
        logits = outputs["logits"][0]
        all_logits.append(logits)

        # Update caches
        for i in range(8):
            k_caches[i] = outputs[f"k_cache_{i}_out"]
            v_caches[i] = outputs[f"v_cache_{i}_out"]

        # Next token
        next_token = int(np.argmax(logits))
        tokens.append(next_token)
        current_token = next_token

        # Print progress
        if step < 10 or step % 5 == 0:
            print(
                f"  {decoder_name:10s} Step {step:2d}: "
                f"token {next_token:5d} ({vocab.get(next_token, '???'):30s})"
            )

    return tokens, np.array(all_logits)

baseline_tokens, baseline_logits = decode_and_track(baseline_decoder, "baseline")
english_tokens, english_logits = decode_and_track(english_decoder, "english")

# Visualize
print("\n" + "="*80)
print("Creating visualizations...")
print("="*80)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Token IDs over time
ax = axes[0, 0]
steps = np.arange(NUM_STEPS)
ax.plot(steps, baseline_tokens, 'o-', label='Baseline', linewidth=2, markersize=6)
ax.plot(steps, english_tokens, 's--', label='English (per-lang)', linewidth=2, markersize=6)
ax.set_xlabel('Decoding Step')
ax.set_ylabel('Token ID')
ax.set_title('Generated Token IDs Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Top-5 logits heatmap (baseline)
ax = axes[0, 1]
top_50_tokens = np.argsort(baseline_logits[0])[-50:][::-1]
logits_subset = baseline_logits[:, top_50_tokens]
im = ax.imshow(logits_subset.T, aspect='auto', cmap='hot', interpolation='nearest')
ax.set_xlabel('Decoding Step')
ax.set_ylabel('Token Rank (Top 50)')
ax.set_title('Baseline Decoder: Logit Heatmap (Top 50 Tokens)')
plt.colorbar(im, ax=ax, label='Logit value')

# 3. Top-5 logits heatmap (english)
ax = axes[1, 0]
top_50_tokens = np.argsort(english_logits[0])[-50:][::-1]
logits_subset = english_logits[:, top_50_tokens]
im = ax.imshow(logits_subset.T, aspect='auto', cmap='hot', interpolation='nearest')
ax.set_xlabel('Decoding Step')
ax.set_ylabel('Token Rank (Top 50)')
ax.set_title('English Decoder: Logit Heatmap (Top 50 Tokens)')
plt.colorbar(im, ax=ax, label='Logit value')

# 4. Token diversity (entropy over time)
ax = axes[1, 1]
def compute_entropy(logits):
    probs = np.exp(logits) / np.sum(np.exp(logits))
    # Avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))

baseline_entropy = [compute_entropy(logits) for logits in baseline_logits]
english_entropy = [compute_entropy(logits) for logits in english_logits]

ax.plot(steps, baseline_entropy, 'o-', label='Baseline', linewidth=2)
ax.plot(steps, english_entropy, 's--', label='English (per-lang)', linewidth=2)
ax.set_xlabel('Decoding Step')
ax.set_ylabel('Entropy (nats)')
ax.set_title('Logit Distribution Entropy Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = "research/decoding_visualization.png"
plt.savefig(output_file, dpi=150)
print(f"✓ Saved visualization to {output_file}")

# Text output
print("\n" + "="*80)
print("Generated Text")
print("="*80)

def detokenize(tokens):
    text = "".join([vocab.get(t, f"<unk_{t}>") for t in tokens])
    text = text.replace("▁", " ").strip()
    return text

baseline_text = detokenize(baseline_tokens)
english_text = detokenize(english_tokens)

print(f"\nReference:  {reference[:80]}...")
print(f"Baseline:   {baseline_text[:80]}...")
print(f"English:    {english_text[:80]}...")

# Check if stuck in loop
def check_loop(tokens):
    """Check if tokens are repeating."""
    if len(tokens) < 4:
        return False, None

    # Check last 10 tokens
    recent = tokens[-10:]
    if len(set(recent)) <= 2:
        return True, recent[0]

    return False, None

baseline_loop, baseline_loop_token = check_loop(baseline_tokens)
english_loop, english_loop_token = check_loop(english_tokens)

print("\n" + "="*80)
print("Loop Detection")
print("="*80)

if baseline_loop:
    print(f"✓ Baseline STUCK in loop: token {baseline_loop_token} ({vocab.get(baseline_loop_token, '???')})")
else:
    print("✗ Baseline NOT looping")

if english_loop:
    print(f"✓ English STUCK in loop: token {english_loop_token} ({vocab.get(english_loop_token, '???')})")
else:
    print("✗ English NOT looping")

print("\n" + "="*80)
print("EXPERIMENT 3 COMPLETE")
print("="*80)
