# Test Qwen3-TTS CoreML Models
import numpy as np
import coremltools as ct
from pathlib import Path
import time

# Configuration
MAX_TEXT_LENGTH = 128
PREFILL_LEN = MAX_TEXT_LENGTH + 2


def test_lm_models():
    """Test LM prefill and decode models."""
    print("=" * 60)
    print("Testing Qwen3-TTS LM CoreML Models")
    print("=" * 60)

    # Load models
    print("\n1. Loading models...")
    t0 = time.time()
    prefill_model = ct.models.MLModel("qwen3_tts_lm_prefill.mlpackage")
    print(f"   Prefill loaded in {time.time() - t0:.2f}s")

    t0 = time.time()
    decode_model = ct.models.MLModel("qwen3_tts_lm_decode.mlpackage")
    print(f"   Decode loaded in {time.time() - t0:.2f}s")

    # Test prefill
    print("\n2. Testing Prefill...")
    # Pad text to MAX_TEXT_LENGTH
    # "Hello world" tokens: [9707, 1879]
    text_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
    text_ids[0, :2] = [9707, 1879]

    t0 = time.time()
    prefill_result = prefill_model.predict({"text_ids": text_ids})
    prefill_time = time.time() - t0

    logits = prefill_result["logits"]
    kv_cache = prefill_result["kv_cache"]

    print(f"   Time: {prefill_time * 1000:.1f}ms")
    print(f"   Logits shape: {logits.shape}")
    print(f"   KV cache shape: {kv_cache.shape}")
    print(f"   Top 5 tokens: {np.argsort(logits[0])[-5:][::-1].tolist()}")

    # Test decode
    print("\n3. Testing Decode...")
    first_token = np.argmax(logits, axis=-1).reshape(1, 1).astype(np.int32)
    position = np.array([PREFILL_LEN], dtype=np.int32)

    t0 = time.time()
    decode_result = decode_model.predict({
        "token_id": first_token,
        "kv_cache": kv_cache,
        "position": position,
    })
    decode_time = time.time() - t0

    new_logits = decode_result["logits"]
    new_kv_cache = decode_result["new_kv_cache"]

    print(f"   Time: {decode_time * 1000:.1f}ms")
    print(f"   New logits shape: {new_logits.shape}")
    print(f"   New KV cache shape: {new_kv_cache.shape}")
    print(f"   Next top 5: {np.argsort(new_logits[0])[-5:][::-1].tolist()}")

    # Multi-step generation
    print("\n4. Multi-step Generation...")
    generated = [first_token[0, 0]]
    current_kv = kv_cache
    current_pos = PREFILL_LEN

    EOS_TOKEN = 2150
    MAX_STEPS = 50

    t0 = time.time()
    for step in range(MAX_STEPS):
        result = decode_model.predict({
            "token_id": np.array([[generated[-1]]], dtype=np.int32),
            "kv_cache": current_kv,
            "position": np.array([current_pos], dtype=np.int32),
        })
        next_token = np.argmax(result["logits"], axis=-1)[0]
        generated.append(int(next_token))
        current_kv = result["new_kv_cache"]
        current_pos += 1

        if next_token == EOS_TOKEN:
            print(f"   EOS reached at step {step + 1}")
            break

    gen_time = time.time() - t0
    tokens_per_sec = len(generated) / gen_time

    print(f"   Generated {len(generated)} tokens in {gen_time:.2f}s ({tokens_per_sec:.1f} tok/s)")
    print(f"   Tokens: {generated[:20]}{'...' if len(generated) > 20 else ''}")

    return generated, current_kv


def compare_with_pytorch():
    """Compare CoreML output with PyTorch."""
    print("\n" + "=" * 60)
    print("Comparing CoreML vs PyTorch")
    print("=" * 60)

    import torch
    from qwen_tts import Qwen3TTSModel

    # Load PyTorch model
    print("\n1. Loading PyTorch model...")
    model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    talker = model.model.talker
    config = talker.config

    # Load CoreML model
    prefill_model = ct.models.MLModel("qwen3_tts_lm_prefill.mlpackage")

    # Test input
    text_ids_np = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
    text_ids_np[0, :2] = [9707, 1879]

    # CoreML forward
    print("\n2. Running CoreML...")
    coreml_result = prefill_model.predict({"text_ids": text_ids_np})
    coreml_logits = coreml_result["logits"]

    # PyTorch forward
    print("3. Running PyTorch...")
    text_ids = torch.tensor([[9707, 1879]])
    with torch.no_grad():
        text_embed = talker.model.text_embedding(text_ids)
        text_projected = talker.text_projection(text_embed)
        lang_embed = talker.model.codec_embedding(torch.tensor([[config.codec_language_id["english"]]]))
        bos_embed = talker.model.codec_embedding(torch.tensor([[config.codec_bos_id]]))
        combined = torch.cat([lang_embed, text_projected, bos_embed], dim=1)
        outputs = talker.model(inputs_embeds=combined, use_cache=True, return_dict=True)
        pytorch_logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1).numpy()

    # Compare
    print("\n4. Comparison:")
    print(f"   PyTorch top 5: {np.argsort(pytorch_logits[0])[-5:][::-1].tolist()}")
    print(f"   CoreML top 5:  {np.argsort(coreml_logits[0])[-5:][::-1].tolist()}")

    diff = np.abs(pytorch_logits - coreml_logits).max()
    print(f"   Max diff: {diff:.6f}")

    correlation = np.corrcoef(pytorch_logits.flatten(), coreml_logits.flatten())[0, 1]
    print(f"   Correlation: {correlation:.6f}")


if __name__ == "__main__":
    # Test CoreML models
    generated, final_kv = test_lm_models()

    # Compare with PyTorch
    compare_with_pytorch()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
