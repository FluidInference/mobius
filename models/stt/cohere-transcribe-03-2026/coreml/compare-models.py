#!/usr/bin/env python3
"""Compare our exported models against BarathwajAnandan's reference models."""

import numpy as np
import coremltools as ct
from cohere_mel_spectrogram import CohereMelSpectrogram

print("="*70)
print("Comparing Our Models vs BarathwajAnandan Reference")
print("="*70)

# Load test audio
print("\n[1/4] Loading test audio...")
try:
    import soundfile as sf
    audio, sr = sf.read(".venv/lib/python3.10/site-packages/pyannote/audio/sample/sample.wav")
    print(f"   ✓ Loaded: {len(audio)} samples ({len(audio)/sr:.2f}s)")

    # Limit to first 5 seconds to match typical inference
    audio = audio[:sr*5]
    print(f"   Using first 5 seconds: {len(audio)} samples")

except Exception as e:
    print(f"   ❌ Could not load audio: {e}")
    exit(1)

# Compute mel spectrogram
print("\n[2/4] Computing mel spectrogram...")
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)
mel_padded = np.pad(
    mel,
    ((0, 0), (0, 0), (0, 3001 - mel.shape[2])),
    mode='constant',
    constant_values=0
)
print(f"   Mel shape: {mel_padded.shape}")

# Test encoders
print("\n[3/4] Comparing encoders...")
try:
    # Our encoder
    our_encoder = ct.models.MLModel(
        "build/cohere_encoder.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    our_encoder_output = our_encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([3001], dtype=np.int32)
    })
    our_hidden = None
    for key, value in our_encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            our_hidden = value
            break

    # Reference encoder
    ref_encoder = ct.models.MLModel(
        "barathwaj-models/cohere_encoder.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    ref_encoder_output = ref_encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([3001], dtype=np.int32)
    })
    ref_hidden = None
    for key, value in ref_encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            ref_hidden = value
            break

    print(f"   Our encoder output: {our_hidden.shape}")
    print(f"   Ref encoder output: {ref_hidden.shape}")

    # Compare
    diff = np.abs(our_hidden - ref_hidden)
    print(f"\n   Encoder comparison:")
    print(f"     Max difference: {diff.max():.6f}")
    print(f"     Mean difference: {diff.mean():.6f}")
    print(f"     Std difference: {diff.std():.6f}")

    if diff.max() < 0.01:
        print(f"     ✅ Excellent match!")
    elif diff.max() < 0.1:
        print(f"     ✅ Good match")
    elif diff.max() < 1.0:
        print(f"     ⚠️  Some differences")
    else:
        print(f"     ❌ Significant differences")

    encoder_hidden = our_hidden  # Use ours for decoder test

except FileNotFoundError as e:
    print(f"   ⚠️  Reference model not found: {e}")
    print(f"   Skipping encoder comparison")
    # Just use our encoder
    encoder = ct.models.MLModel("build/cohere_encoder.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
    encoder_output = encoder.predict({
        "input_features": mel_padded.astype(np.float32),
        "feature_length": np.array([3001], dtype=np.int32)
    })
    for key, value in encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            encoder_hidden = value
            break
except Exception as e:
    print(f"   ❌ Encoder comparison error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test decoders (first 5 steps)
print("\n[4/4] Comparing decoders (first 5 steps)...")
try:
    # Our decoder
    our_decoder = ct.models.MLModel(
        "build/cohere_decoder_cached.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )

    # Reference decoder
    try:
        ref_decoder = ct.models.MLModel(
            "barathwaj-models/cohere_decoder_cached.mlpackage",
            compute_units=ct.ComputeUnit.CPU_AND_GPU
        )
        has_ref = True
    except FileNotFoundError:
        print("   ⚠️  Reference decoder not found, testing ours only")
        has_ref = False

    decoder_start_token_id = 13764
    num_steps = 5

    # Our decoder tokens
    our_tokens = [decoder_start_token_id]
    our_cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)
    our_cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

    # Reference decoder tokens
    if has_ref:
        ref_tokens = [decoder_start_token_id]
        ref_cache_k = np.zeros((8, 8, 108, 128), dtype=np.float16)  # Note: 108 not 1024
        ref_cache_v = np.zeros((8, 8, 108, 128), dtype=np.float16)

    for step in range(num_steps):
        # Our decoder
        our_input = {
            "input_id": np.array([[our_tokens[-1]]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden.astype(np.float16),
            "step": np.array([step], dtype=np.int32),
            "cross_attention_mask": np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16),
            "cache_k": our_cache_k,
            "cache_v": our_cache_v,
        }
        our_output = our_decoder.predict(our_input)

        # Extract our outputs
        our_logits = None
        for key, value in our_output.items():
            if hasattr(value, 'shape'):
                if len(value.shape) == 2 and value.shape[1] > 1000:
                    our_logits = value
                elif len(value.shape) == 4 and 'cache_k' in key.lower() or key == 'new_cache_k':
                    our_cache_k = value
                elif len(value.shape) == 4 and 'cache_v' in key.lower() or key == 'new_cache_v':
                    our_cache_v = value

        our_next_token = int(np.argmax(our_logits[0]))
        our_tokens.append(our_next_token)

        # Reference decoder
        if has_ref:
            ref_input = {
                "input_id": np.array([[ref_tokens[-1]]], dtype=np.int32),
                "encoder_hidden_states": encoder_hidden.astype(np.float16),
                "step": np.array([step], dtype=np.int32),
                "cross_attention_mask": np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16),
                "cache_k": ref_cache_k,
                "cache_v": ref_cache_v,
            }
            try:
                ref_output = ref_decoder.predict(ref_input)

                # Extract ref outputs
                ref_logits = None
                for key, value in ref_output.items():
                    if hasattr(value, 'shape'):
                        if len(value.shape) == 2 and value.shape[1] > 1000:
                            ref_logits = value
                        elif len(value.shape) == 4:
                            if ref_cache_k.shape == value.shape:
                                if 'k' in key.lower():
                                    ref_cache_k = value
                                else:
                                    ref_cache_v = value

                ref_next_token = int(np.argmax(ref_logits[0]))
                ref_tokens.append(ref_next_token)

                # Compare logits
                logits_diff = np.abs(our_logits - ref_logits)
                print(f"\n   Step {step}:")
                print(f"     Our token: {our_next_token}, Ref token: {ref_next_token}")
                print(f"     Logits diff: max={logits_diff.max():.6f}, mean={logits_diff.mean():.6f}")

            except Exception as e:
                print(f"   ⚠️  Reference decoder error at step {step}: {e}")
                has_ref = False

        else:
            print(f"   Step {step}: Our token = {our_next_token}")

    print(f"\n   Our tokens: {our_tokens}")
    if has_ref:
        print(f"   Ref tokens: {ref_tokens}")
        if our_tokens == ref_tokens:
            print(f"   ✅ Perfect match!")
        else:
            print(f"   ⚠️  Tokens differ")

except Exception as e:
    print(f"   ❌ Decoder comparison error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)
