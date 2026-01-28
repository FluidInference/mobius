# End-to-end TTS test
# Uses CoreML for LM and Decoder, PyTorch for Code Predictor
import numpy as np
import coremltools as ct
import torch
import time
from pathlib import Path

MAX_TEXT_LENGTH = 128
SAMPLE_RATE = 24000


def test_end_to_end():
    """Test end-to-end TTS pipeline."""
    print("=" * 60)
    print("End-to-End TTS Test")
    print("=" * 60)

    # Load PyTorch model for components we haven't converted
    print("\n1. Loading models...")
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

    t0 = time.time()
    tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    tokenizer_model = Qwen3TTSTokenizer.from_pretrained("./tokenizer_12hz", device_map="cpu")
    print(f"   PyTorch models: {time.time() - t0:.1f}s")

    talker = tts_model.model.talker
    config = talker.config
    processor = tts_model.processor

    t0 = time.time()
    lm_prefill = ct.models.MLModel("qwen3_tts_lm_prefill.mlpackage")
    lm_decode = ct.models.MLModel("qwen3_tts_lm_decode.mlpackage")
    decoder = ct.models.MLModel("qwen3_tts_decoder_10s.mlpackage")
    print(f"   CoreML models: {time.time() - t0:.1f}s")

    # Test text
    text = "Hello world, this is a test of the Qwen3 TTS system."
    print(f"\n2. Input text: '{text}'")

    # Tokenize
    inputs = processor(text=text, return_tensors="pt")
    text_ids = inputs.input_ids
    text_len = text_ids.shape[1]
    print(f"   Text tokens: {text_len}")

    # === LM Generation (CoreML) ===
    print("\n3. LM Generation (CoreML)...")

    # Prepare input - pad or truncate to MAX_TEXT_LENGTH
    if text_len > MAX_TEXT_LENGTH:
        print(f"   Warning: truncating from {text_len} to {MAX_TEXT_LENGTH}")
        text_ids = text_ids[:, :MAX_TEXT_LENGTH]
        text_len = MAX_TEXT_LENGTH

    # For CoreML, we need to pad to fixed size
    # But we also need to track actual length for proper generation
    text_np = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
    text_np[0, :text_len] = text_ids[0].numpy()

    # Prefill
    t0 = time.time()
    prefill_result = lm_prefill.predict({"text_ids": text_np})
    prefill_time = time.time() - t0

    logits = prefill_result["logits"]
    kv_cache = prefill_result["kv_cache"]

    # The KV cache includes all positions including padding
    # For proper generation, we need the position after the actual content
    # But since padding affects the embeddings, let's use PyTorch for comparison

    print(f"   Prefill: {prefill_time * 1000:.1f}ms")
    print(f"   KV cache shape: {kv_cache.shape}")

    # Generate tokens
    EOS_TOKEN = config.codec_eos_token_id
    MAX_CODEC_TOKENS = 125  # ~10 seconds

    # Use greedy decoding
    generated_tokens = []
    current_kv = kv_cache
    position = kv_cache.shape[3]  # Start after prefill seq len

    first_token = int(np.argmax(logits, axis=-1)[0])
    generated_tokens.append(first_token)

    t0 = time.time()
    while len(generated_tokens) < MAX_CODEC_TOKENS:
        result = lm_decode.predict({
            "token_id": np.array([[generated_tokens[-1]]], dtype=np.int32),
            "kv_cache": current_kv,
            "position": np.array([position], dtype=np.int32),
        })
        next_token = int(np.argmax(result["logits"], axis=-1)[0])
        generated_tokens.append(next_token)
        current_kv = result["new_kv_cache"]
        position += 1

        if next_token == EOS_TOKEN:
            break

    decode_time = time.time() - t0
    print(f"   Generated {len(generated_tokens)} tokens in {decode_time:.2f}s ({len(generated_tokens)/decode_time:.1f} tok/s)")
    print(f"   First 10 tokens: {generated_tokens[:10]}")

    # === Code Predictor (PyTorch) ===
    # The code predictor needs hidden states from the LM, which we don't have from CoreML
    # For a full test, let's use PyTorch for the entire LM+CodePredictor

    print("\n4. Full generation with PyTorch (for comparison)...")

    with torch.no_grad():
        # Use the model's generate method
        t0 = time.time()

        # Prepare input for generate
        text_embed = talker.model.text_embedding(text_ids)
        text_projected = talker.text_projection(text_embed)
        lang_embed = talker.model.codec_embedding(torch.tensor([[config.codec_language_id["english"]]]))

        # Simple generation loop with code predictor
        # This is complex - let's use the high-level API instead
        try:
            wavs, sr = tts_model.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=None,
                ref_text=None,
            )
            pytorch_time = time.time() - t0
            print(f"   PyTorch generation: {pytorch_time:.2f}s")
            if wavs is not None and len(wavs) > 0:
                print(f"   Audio shape: {wavs[0].shape}")
                print(f"   Duration: {wavs[0].shape[0] / sr:.2f}s")

                # Save audio
                import soundfile as sf
                sf.write("test_pytorch_output.wav", wavs[0].numpy(), sr)
                print("   Saved: test_pytorch_output.wav")
        except Exception as e:
            print(f"   PyTorch generation failed: {e}")
            import traceback
            traceback.print_exc()

    # === Test Decoder with random codes ===
    print("\n5. Testing Decoder (CoreML)...")

    # Create random codes for testing
    test_codes = np.random.randint(0, 2048, (1, 16, 50), dtype=np.int32)
    print(f"   Input codes shape: {test_codes.shape}")

    # For the decoder, we need to pad to the expected shape
    codes_padded = np.zeros((1, 16, 125), dtype=np.int32)
    codes_padded[:, :, :50] = test_codes

    t0 = time.time()
    decoder_result = decoder.predict({"codes": codes_padded})
    decoder_time = time.time() - t0

    audio = decoder_result["audio"]
    print(f"   Decoder time: {decoder_time * 1000:.1f}ms")
    print(f"   Audio shape: {audio.shape}")
    print(f"   Audio duration: {audio.shape[-1] / SAMPLE_RATE:.2f}s")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"LM Prefill: {prefill_time * 1000:.1f}ms")
    print(f"LM Decode: {decode_time:.2f}s for {len(generated_tokens)} tokens")
    print(f"Decoder: {decoder_time * 1000:.1f}ms")
    print("\nNote: Full pipeline requires Code Predictor (not yet converted to CoreML)")
    print("The LM generates first codebook layer; Code Predictor generates remaining 15 layers")


if __name__ == "__main__":
    test_end_to_end()
