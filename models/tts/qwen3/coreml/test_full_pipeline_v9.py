# Full TTS Pipeline Test V9
# Uses V9 CoreML prefill + V3 PyTorch decode (with code_predictor)
import numpy as np
import coremltools as ct
import torch
import time
import soundfile as sf

MAX_TEXT_LENGTH = 128
MAX_CODEC_TOKENS = 125
SAMPLE_RATE = 24000

# Special token IDs
TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]


def test_full_pipeline():
    print("=" * 60)
    print("Full TTS Pipeline Test V9 (V9 Prefill + V3 Decode)")
    print("=" * 60)

    # Load models
    print("\n1. Loading models...")
    from qwen_tts import Qwen3TTSModel
    from convert_lm_decode_v3 import TracableDecodeV3

    t0 = time.time()
    tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    processor = tts_model.processor
    talker = tts_model.model.talker
    config = talker.config
    print(f"   PyTorch model: {time.time() - t0:.1f}s")

    t0 = time.time()
    lm_prefill = ct.models.MLModel("qwen3_tts_lm_prefill_v9.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
    code_predictor = ct.models.MLModel("qwen3_tts_code_predictor_v3.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
    decoder = ct.models.MLModel("qwen3_tts_decoder_10s.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
    print(f"   CoreML models: {time.time() - t0:.1f}s")

    # Create V3 decode wrapper (PyTorch)
    decode_wrapper = TracableDecodeV3(talker)
    decode_wrapper.eval()
    print("   V3 decode wrapper: ready")

    # Pre-compute TTS embeddings
    print("\n   Pre-computing TTS embeddings...")
    with torch.no_grad():
        tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
        tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
        tts_bos_embed = tts_embed[:, 0:1, :]
        tts_pad_embed = tts_embed[:, 1:2, :]
        tts_eos_embed = tts_embed[:, 2:3, :]

        # NumPy versions for CoreML
        tts_bos_embed_np = tts_bos_embed.numpy().astype(np.float32)
        tts_pad_embed_np = tts_pad_embed.numpy().astype(np.float32)
        tts_eos_embed_np = tts_eos_embed.numpy().astype(np.float32)

    # Load speaker embedding
    speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024).astype(np.float32)
    speaker_embed = torch.from_numpy(speaker_embed_np)

    text = "Hello world, this is a test of the text to speech system."
    print(f"\n2. Input text: '{text}'")

    # Tokenize text
    tokenizer = processor.tokenizer
    text_ids_list = tokenizer.encode(text, add_special_tokens=False)
    text_len = len(text_ids_list)

    role_ids = np.array([ROLE_PREFIX], dtype=np.int32)
    text_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
    text_ids[0, :text_len] = text_ids_list
    text_length = np.array([text_len], dtype=np.int32)

    actual_len = text_len + 11
    print(f"   Text tokens: {text_len}")
    print(f"   Actual prefill length: {actual_len}")

    # === LM Generation ===
    print("\n3. LM Generation (V9 Prefill + V3 Decode)...")

    # Prefill with CoreML V9
    t0 = time.time()
    prefill_result = lm_prefill.predict({
        "role_ids": role_ids,
        "text_ids": text_ids,
        "text_length": text_length,
        "tts_bos_embed": tts_bos_embed_np,
        "tts_pad_embed": tts_pad_embed_np,
        "tts_eos_embed": tts_eos_embed_np,
        "speaker_embed": speaker_embed_np,
    })
    logits = prefill_result["logits"]
    kv_cache_full = prefill_result["kv_cache"]
    past_hidden = prefill_result["past_hidden"]  # V9 outputs this!
    prefill_time = time.time() - t0
    print(f"   Prefill: {prefill_time * 1000:.1f}ms")
    print(f"   Full KV cache shape: {kv_cache_full.shape}")
    print(f"   Past hidden shape: {past_hidden.shape}")

    # Truncate KV cache to actual length
    kv_cache = kv_cache_full[:, :, :, :actual_len, :]
    print(f"   Truncated KV cache shape: {kv_cache.shape}")

    # Generate tokens
    EOS_TOKEN = config.codec_eos_token_id
    VOCAB_SIZE = config.vocab_size

    suppress_mask = np.zeros(VOCAB_SIZE, dtype=bool)
    suppress_mask[2048:] = True
    suppress_mask[EOS_TOKEN] = False

    def sample_with_suppress(logits_np):
        logits_np = logits_np.copy()
        logits_np[0, suppress_mask] = -float('inf')
        return int(np.argmax(logits_np, axis=-1)[0])

    generated_tokens = []
    position = actual_len

    first_token = sample_with_suppress(logits)
    generated_tokens.append(first_token)
    print(f"   First token: {first_token}")

    # Convert to PyTorch for V3 decode
    kv_cache_torch = torch.from_numpy(kv_cache).float()
    past_hidden_torch = torch.from_numpy(past_hidden).float()

    t0 = time.time()
    while len(generated_tokens) < MAX_CODEC_TOKENS:
        token_id = torch.tensor([[generated_tokens[-1]]], dtype=torch.long)
        position_tensor = torch.tensor([position], dtype=torch.long)

        with torch.no_grad():
            logits_torch, kv_cache_torch, past_hidden_torch = decode_wrapper(
                token_id, past_hidden_torch, tts_pad_embed, kv_cache_torch, position_tensor
            )

        next_token = sample_with_suppress(logits_torch.numpy())
        generated_tokens.append(next_token)
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

    all_codebooks = cp_result["all_codebooks"]
    print(f"   Code Predictor: {cp_time:.2f}s")
    print(f"   All codebooks shape: {all_codebooks.shape}")

    codes = np.zeros((1, 16, MAX_CODEC_TOKENS), dtype=np.int32)
    codes[0, 0, :] = codebook0[0]
    codes[0, 1:15, :] = all_codebooks[0]

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

    output_file = "test_full_pipeline_v9_output.wav"
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
