#!/usr/bin/env python3
"""Minimal reproduction: test decoder with controlled inputs."""

import numpy as np
import coremltools as ct
import json

print("="*80)
print("EXPERIMENT 4: Minimal Reproduction - Controlled Inputs")
print("="*80)

# Load vocabulary
print("\n[1/3] Loading vocabulary...")
with open("f16/vocab.json") as f:
    vocab = {int(k): v for k, v in json.load(f).items()}
print(f"✓ Loaded {len(vocab)} tokens")

# Load decoders
print("\n[2/3] Loading decoders...")
baseline_decoder = ct.models.MLModel(
    "hf-upload/cohere-transcribe-cache-external-coreml/cohere_decoder_cache_external.mlpackage"
)
english_decoder = ct.models.MLModel(
    "build-per-language/cohere_decoder_english.mlpackage"
)
spanish_decoder = ct.models.MLModel(
    "build-per-language/cohere_decoder_spanish.mlpackage"
)
print("✓ Loaded 3 decoders")

def decode_n_steps(decoder, decoder_name, encoder_hidden, num_steps=15):
    """Decode N steps and return tokens."""
    encoder_seq_len = encoder_hidden.shape[1]

    k_caches = [np.zeros((1, 8, 108, 128), dtype=np.float32) for _ in range(8)]
    v_caches = [np.zeros((1, 8, 108, 128), dtype=np.float32) for _ in range(8)]

    cross_mask = np.ones((1, 1, 1, encoder_seq_len), dtype=np.float32)

    tokens = []
    current_token = 4  # START

    for step in range(num_steps):
        inputs = {
            "input_id": np.array([[current_token]], dtype=np.int32),
            "position_id": np.array([[step]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden,
            "cross_attention_mask": cross_mask,
            "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float32),
        }

        for i in range(8):
            inputs[f"k_cache_{i}"] = k_caches[i]
            inputs[f"v_cache_{i}"] = v_caches[i]

        outputs = decoder.predict(inputs)

        for i in range(8):
            k_caches[i] = outputs[f"k_cache_{i}_out"]
            v_caches[i] = outputs[f"v_cache_{i}_out"]

        next_token = int(np.argmax(outputs["logits"][0]))
        tokens.append(next_token)
        current_token = next_token

    text = "".join([vocab.get(t, f"<unk_{t}>") for t in tokens])
    return tokens, text

print("\n[3/3] Testing with controlled encoder inputs...")

# Test configurations
tests = [
    ("Zeros", np.zeros((1, 438, 1024), dtype=np.float32)),
    ("Ones", np.ones((1, 438, 1024), dtype=np.float32)),
    ("Random (seed=42)", np.random.RandomState(42).randn(1, 438, 1024).astype(np.float32)),
    ("Random (seed=99)", np.random.RandomState(99).randn(1, 438, 1024).astype(np.float32)),
    ("Small values (0.01)", np.full((1, 438, 1024), 0.01, dtype=np.float32)),
    ("Large values (10.0)", np.full((1, 438, 1024), 10.0, dtype=np.float32)),
]

results = {}

for test_name, encoder_hidden in tests:
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")
    print(f"Encoder hidden shape: {encoder_hidden.shape}")
    print(f"Encoder hidden stats: min={encoder_hidden.min():.4f}, max={encoder_hidden.max():.4f}, mean={encoder_hidden.mean():.4f}")

    results[test_name] = {}

    for decoder_name, decoder in [
        ("baseline", baseline_decoder),
        ("english", english_decoder),
        ("spanish", spanish_decoder),
    ]:
        tokens, text = decode_n_steps(decoder, decoder_name, encoder_hidden, num_steps=15)

        results[test_name][decoder_name] = {
            "tokens": tokens,
            "text": text,
        }

        # Show first 10 tokens
        tokens_str = " ".join([f"{t}({vocab.get(t, '???')[:8]})" for t in tokens[:10]])
        print(f"  {decoder_name:10s}: {tokens_str}...")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Check if baseline and per-language decoders ever diverge
print("\n1. Do baseline and English decoder ever produce different outputs?")
for test_name in results.keys():
    baseline_tokens = results[test_name]["baseline"]["tokens"]
    english_tokens = results[test_name]["english"]["tokens"]

    if baseline_tokens == english_tokens:
        print(f"  {test_name:25s}: ✗ IDENTICAL")
    else:
        # Find first divergence
        for i, (b, e) in enumerate(zip(baseline_tokens, english_tokens)):
            if b != e:
                print(
                    f"  {test_name:25s}: ✓ DIVERGE at step {i}: "
                    f"baseline={b}({vocab.get(b, '???')}), english={e}({vocab.get(e, '???')})"
                )
                break

print("\n2. Do English and Spanish decoder produce different outputs?")
for test_name in results.keys():
    english_tokens = results[test_name]["english"]["tokens"]
    spanish_tokens = results[test_name]["spanish"]["tokens"]

    if english_tokens == spanish_tokens:
        print(f"  {test_name:25s}: ✗ IDENTICAL")
    else:
        for i, (e, s) in enumerate(zip(english_tokens, spanish_tokens)):
            if e != s:
                print(
                    f"  {test_name:25s}: ✓ DIVERGE at step {i}: "
                    f"english={e}({vocab.get(e, '???')}), spanish={s}({vocab.get(s, '???')})"
                )
                break

print("\n3. Does encoder input affect decoder output?")
# Compare zeros vs ones
zeros_baseline = results["Zeros"]["baseline"]["tokens"]
ones_baseline = results["Ones"]["baseline"]["tokens"]

if zeros_baseline == ones_baseline:
    print("  ✗ Zeros vs Ones: IDENTICAL (decoder ignores encoder!)")
else:
    for i, (z, o) in enumerate(zip(zeros_baseline, ones_baseline)):
        if z != o:
            print(f"  ✓ Zeros vs Ones: DIVERGE at step {i}")
            break

# Compare two random seeds
rand42_baseline = results["Random (seed=42)"]["baseline"]["tokens"]
rand99_baseline = results["Random (seed=99)"]["baseline"]["tokens"]

if rand42_baseline == rand99_baseline:
    print("  ✗ Random(42) vs Random(99): IDENTICAL (decoder ignores encoder!)")
else:
    for i, (r42, r99) in enumerate(zip(rand42_baseline, rand99_baseline)):
        if r42 != r99:
            print(f"  ✓ Random(42) vs Random(99): DIVERGE at step {i}")
            break

print("\n4. Check for language token output")
language_tokens = {
    "english": 62,
    "french": 69,
    "spanish": 169,
    "chinese": 50,
    "arabic": 63,
}

for test_name in results.keys():
    print(f"\n  {test_name}:")
    for decoder_name in ["baseline", "english", "spanish"]:
        tokens = results[test_name][decoder_name]["tokens"]

        lang_token_counts = {lang: tokens.count(token_id) for lang, token_id in language_tokens.items()}
        total_lang_tokens = sum(lang_token_counts.values())

        if total_lang_tokens > 0:
            lang_distribution = ", ".join([f"{lang}={count}" for lang, count in lang_token_counts.items() if count > 0])
            print(f"    {decoder_name:10s}: {total_lang_tokens}/{len(tokens)} language tokens ({lang_distribution})")
        else:
            print(f"    {decoder_name:10s}: No language tokens")

# Save results
output_file = "research/minimal_reproduction_results.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved to {output_file}")

print("\n" + "="*80)
print("EXPERIMENT 4 COMPLETE")
print("="*80)
