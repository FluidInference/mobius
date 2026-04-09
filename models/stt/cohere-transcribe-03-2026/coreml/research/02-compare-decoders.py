#!/usr/bin/env python3
"""Compare baseline vs per-language decoder outputs with same input."""

import numpy as np
import coremltools as ct
import json

print("="*80)
print("EXPERIMENT 2: Compare Decoder Outputs")
print("="*80)

# Load vocabulary
print("\n[1/4] Loading vocabulary...")
with open("f16/vocab.json") as f:
    vocab = {int(k): v for k, v in json.load(f).items()}
print(f"✓ Loaded {len(vocab)} tokens")

# Load decoders
print("\n[2/4] Loading decoders...")
baseline_decoder = ct.models.MLModel(
    "hf-upload/cohere-transcribe-cache-external-coreml/cohere_decoder_cache_external.mlpackage"
)
print("✓ Loaded baseline decoder")

per_lang_decoders = {}
for lang_name in ["english", "french", "spanish", "chinese"]:
    decoder = ct.models.MLModel(
        f"build-per-language/cohere_decoder_{lang_name}.mlpackage"
    )
    per_lang_decoders[lang_name] = decoder
    print(f"✓ Loaded {lang_name} decoder")

# Create shared test input
print("\n[3/4] Creating test input...")
np.random.seed(42)  # Reproducible
input_id = np.array([[4]], dtype=np.int32)  # START token
position_id = np.array([[0]], dtype=np.int32)
encoder_hidden = np.random.randn(1, 438, 1024).astype(np.float32)
cross_mask = np.ones((1, 1, 1, 438), dtype=np.float32)
attention_mask = np.zeros((1, 1, 1, 1), dtype=np.float32)

# Initialize KV caches
k_caches = [np.zeros((1, 8, 108, 128), dtype=np.float32) for _ in range(8)]
v_caches = [np.zeros((1, 8, 108, 128), dtype=np.float32) for _ in range(8)]

# Build input dict
inputs = {
    "input_id": input_id,
    "position_id": position_id,
    "encoder_hidden_states": encoder_hidden,
    "cross_attention_mask": cross_mask,
    "attention_mask": attention_mask,
}

for i in range(8):
    inputs[f"k_cache_{i}"] = k_caches[i]
    inputs[f"v_cache_{i}"] = v_caches[i]

print(f"  input_id shape: {input_id.shape}")
print(f"  encoder_hidden shape: {encoder_hidden.shape}")

# Run all decoders with same input
print("\n[4/4] Running decoders...")

results = {}

# Baseline decoder
print("\n--- Baseline Decoder ---")
baseline_output = baseline_decoder.predict(inputs)
baseline_logits = baseline_output["logits"][0]
baseline_probs = np.exp(baseline_logits) / np.sum(np.exp(baseline_logits))
baseline_top_token = int(np.argmax(baseline_logits))

print(f"Logits shape: {baseline_logits.shape}")
print(f"Top token: {baseline_top_token} ({vocab.get(baseline_top_token, '???')})")
print(f"Top 10 tokens:")
top_10_idx = np.argsort(baseline_probs)[-10:][::-1]
for idx in top_10_idx:
    print(f"  {idx:5d} ({vocab.get(idx, '???'):30s}): {baseline_probs[idx]:.6f}")

results["baseline"] = {
    "top_token": baseline_top_token,
    "top_token_text": vocab.get(baseline_top_token, "???"),
    "top_prob": float(baseline_probs[baseline_top_token]),
    "top_10": [
        {
            "token": int(idx),
            "text": vocab.get(idx, "???"),
            "prob": float(baseline_probs[idx]),
        }
        for idx in top_10_idx
    ],
}

# Per-language decoders
for lang_name, decoder in per_lang_decoders.items():
    print(f"\n--- {lang_name.capitalize()} Decoder ---")

    output = decoder.predict(inputs)
    logits = output["logits"][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    top_token = int(np.argmax(logits))

    print(f"Top token: {top_token} ({vocab.get(top_token, '???')})")
    print(f"Top 10 tokens:")
    top_10_idx = np.argsort(probs)[-10:][::-1]
    for idx in top_10_idx:
        print(f"  {idx:5d} ({vocab.get(idx, '???'):30s}): {probs[idx]:.6f}")

    results[lang_name] = {
        "top_token": top_token,
        "top_token_text": vocab.get(top_token, "???"),
        "top_prob": float(probs[top_token]),
        "top_10": [
            {
                "token": int(idx),
                "text": vocab.get(idx, "???"),
                "prob": float(probs[idx]),
            }
            for idx in top_10_idx
        ],
    }

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("\nTop token comparison:")
print(f"{'Decoder':15s} {'Token':>6s}  {'Text':30s}  {'Probability':>12s}")
print("-" * 80)
for decoder_name, result in results.items():
    print(
        f"{decoder_name:15s} {result['top_token']:>6d}  "
        f"{result['top_token_text']:30s}  {result['top_prob']:>12.6f}"
    )

# Check if per-language decoders all produce language tokens
print("\n" + "="*80)
print("Language Token Detection")
print("="*80)

language_tokens = {
    "english": 62,
    "french": 69,
    "spanish": 169,
    "chinese": 50,
    "arabic": 63,
    "polish": 120,
}

print("\nChecking if decoders output language tokens:")
for decoder_name, result in results.items():
    top_token = result["top_token"]
    is_lang_token = False
    which_lang = None

    for lang, token_id in language_tokens.items():
        if top_token == token_id:
            is_lang_token = True
            which_lang = lang
            break

    if is_lang_token:
        print(f"  {decoder_name:15s}: ✓ outputs <|{which_lang}|> (token {top_token})")
    else:
        print(f"  {decoder_name:15s}: ✗ outputs '{result['top_token_text']}'")

# Save results
print("\n" + "="*80)
print("Saving results...")
output_file = "research/decoder_comparison_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved to {output_file}")

print("\n" + "="*80)
print("EXPERIMENT 2 COMPLETE")
print("="*80)
