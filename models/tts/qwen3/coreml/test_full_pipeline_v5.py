# Full TTS Pipeline Test V5
# Uses V5 prefill with correct 10-position sequence construction
import numpy as np
import coremltools as ct
import torch
import time
import soundfile as sf

MAX_CODEC_TOKENS = 125
SAMPLE_RATE = 24000

# Special token IDs
TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]  # <|im_start|>assistant\n


def test_full_pipeline():
    print("=" * 60)
    print("Full TTS Pipeline Test V5 (Correct Sequence Construction)")
    print("=" * 60)

    # Load models
    print("\n1. Loading models...")
    from qwen_tts import Qwen3TTSModel

    t0 = time.time()
    tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    processor = tts_model.processor
    talker = tts_model.model.talker
    config = talker.config
    print(f"   PyTorch (tokenizer): {time.time() - t0:.1f}s")

    t0 = time.time()
    lm_prefill = ct.models.MLModel("qwen3_tts_lm_prefill_v5.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
    lm_decode = ct.models.MLModel("qwen3_tts_lm_decode.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
    code_predictor = ct.models.MLModel("qwen3_tts_code_predictor_v3.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
    decoder = ct.models.MLModel("qwen3_tts_decoder_10s.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
    print(f"   CoreML models: {time.time() - t0:.1f}s")

    # Pre-compute TTS embeddings using PyTorch
    print("\n   Pre-computing TTS embeddings...")
    with torch.no_grad():
        tts_special_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID]])
        tts_special_embed = talker.text_projection(talker.model.text_embedding(tts_special_ids))
        tts_bos_embed = tts_special_embed[:, 0:1, :].numpy().astype(np.float32)
        tts_pad_embed = tts_special_embed[:, 1:2, :].numpy().astype(np.float32)
    print(f"   TTS BOS embed shape: {tts_bos_embed.shape}")
    print(f"   TTS PAD embed shape: {tts_pad_embed.shape}")

    # Load speaker embedding
    speaker_embed = np.load("speaker_embedding.npy").reshape(1, 1024).astype(np.float32)
    print(f"   Speaker embedding: {speaker_embed.shape}")

    text = "Hello world, this is a test of the text to speech system."
    print(f"\n2. Input text: '{text}'")

    # Tokenize text and add role prefix
    tokenizer = processor.tokenizer
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    full_ids = ROLE_PREFIX + text_ids
    print(f"   Role prefix: {ROLE_PREFIX}")
    print(f"   Text tokens: {len(text_ids)}")
    print(f"   Full sequence: {len(full_ids)} tokens")

    # Extract role_ids and first_text_id
    role_ids = np.array([full_ids[:3]], dtype=np.int32)
    first_text_id = np.array([[full_ids[3]]], dtype=np.int32)
    print(f"   Role IDs: {role_ids}")
    print(f"   First text ID: {first_text_id}")

    # === LM Generation ===
    print("\n3. LM Generation (CoreML V5)...")

    t0 = time.time()
    prefill_result = lm_prefill.predict({
        "role_ids": role_ids,
        "first_text_id": first_text_id,
        "tts_bos_embed": tts_bos_embed,
        "tts_pad_embed": tts_pad_embed,
        "speaker_embed": speaker_embed,
    })
    logits = prefill_result["logits"]
    kv_cache = prefill_result["kv_cache"]
    prefill_time = time.time() - t0
    print(f"   Prefill: {prefill_time * 1000:.1f}ms")
    print(f"   KV cache shape: {kv_cache.shape}")

    # Generate tokens with proper suppression
    EOS_TOKEN = config.codec_eos_token_id
    VOCAB_SIZE = config.vocab_size  # 3072

    # Suppress tokens 2048-3071 except EOS (2150)
    suppress_mask = np.zeros(VOCAB_SIZE, dtype=bool)
    suppress_mask[2048:] = True
    suppress_mask[EOS_TOKEN] = False  # Allow EOS

    def sample_with_suppress(logits_np):
        """Apply suppression mask and sample."""
        logits_np = logits_np.copy()
        logits_np[0, suppress_mask] = -float('inf')
        return int(np.argmax(logits_np, axis=-1)[0])

    generated_tokens = []
    current_kv = kv_cache
    position = 10  # V5 prefill produces 10 positions

    first_token = sample_with_suppress(logits)
    generated_tokens.append(first_token)
    print(f"   First token: {first_token}")

    t0 = time.time()
    while len(generated_tokens) < MAX_CODEC_TOKENS:
        result = lm_decode.predict({
            "token_id": np.array([[generated_tokens[-1]]], dtype=np.int32),
            "kv_cache": current_kv,
            "position": np.array([position], dtype=np.int32),
        })
        next_token = sample_with_suppress(result["logits"])
        generated_tokens.append(next_token)
        current_kv = result["new_kv_cache"]
        position += 1

        if next_token == EOS_TOKEN:
            print(f"   EOS at token {len(generated_tokens)}")
            break

    lm_time = time.time() - t0
    num_tokens = len(generated_tokens)
    print(f"   Generated {num_tokens} tokens in {lm_time:.2f}s ({num_tokens/lm_time:.1f} tok/s)")
    print(f"   Codebook 0: {generated_tokens[:10]}...")

    # === Code Predictor ===
    print("\n4. Code Predictor (CoreML V3)...")

    codebook0 = np.zeros((1, MAX_CODEC_TOKENS), dtype=np.int32)
    codebook0[0, :num_tokens] = generated_tokens

    t0 = time.time()
    cp_result = code_predictor.predict({"codebook0": codebook0})
    cp_time = time.time() - t0

    all_codebooks = cp_result["all_codebooks"]  # Shape: (1, 14, 125) for codebooks 1-14
    print(f"   Code Predictor: {cp_time:.2f}s")
    print(f"   All codebooks shape: {all_codebooks.shape}")
    print(f"   Codebook 1: {all_codebooks[0, 0, :5]}...")

    # Combine codes: codebook 0 from LM, 1-14 from code predictor, 15 is zeros
    codes = np.zeros((1, 16, MAX_CODEC_TOKENS), dtype=np.int32)
    codes[0, 0, :] = codebook0[0]
    codes[0, 1:15, :] = all_codebooks[0]  # Codebooks 1-14

    # === Decoder ===
    print("\n5. Decoder (CoreML)...")

    t0 = time.time()
    decoder_result = decoder.predict({"codes": codes})
    decoder_time = time.time() - t0

    audio = decoder_result["audio"]
    print(f"   Decoder: {decoder_time:.2f}s")
    print(f"   Audio shape: {audio.shape}")

    samples_per_token = SAMPLE_RATE // 12
    actual_samples = num_tokens * samples_per_token
    audio_trimmed = audio[0, 0, :actual_samples]
    duration = len(audio_trimmed) / SAMPLE_RATE

    output_file = "test_full_pipeline_v5_output.wav"
    sf.write(output_file, audio_trimmed, SAMPLE_RATE)
    print(f"   Saved: {output_file} ({duration:.2f}s)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_time = prefill_time + lm_time + cp_time + decoder_time
    print(f"LM Prefill: {prefill_time * 1000:.1f}ms")
    print(f"LM Decode: {lm_time:.2f}s ({num_tokens} tokens)")
    print(f"Code Predictor: {cp_time:.2f}s")
    print(f"Decoder: {decoder_time:.2f}s")
    print(f"Total: {total_time:.2f}s for {duration:.2f}s audio")
    print(f"RTF: {total_time / duration:.2f}x")


if __name__ == "__main__":
    test_full_pipeline()
