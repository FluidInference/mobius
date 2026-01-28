# Test PyTorch vs CoreML with SAME padded input
import torch
import numpy as np
import coremltools as ct
from qwen_tts import Qwen3TTSModel

MAX_TEXT_LENGTH = 128

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "./model_0.6b",
    device_map="cpu",
    torch_dtype=torch.float32,
)
talker = model.model.talker
config = talker.config
processor = model.processor
print("Model loaded!")

text = "Hello world"
inputs = processor(text=text, return_tensors="pt")
text_ids = inputs.input_ids
text_len = text_ids.shape[1]
print(f"\nText: '{text}'")
print(f"Text tokens: {text_ids[0].tolist()}")

# Pad to 128 with PAD token (not zeros!)
PAD_TOKEN = 0  # Test with 0 first
print(f"\n=== Test 1: Padding with token 0 ===")

padded_text_ids = torch.full((1, MAX_TEXT_LENGTH), PAD_TOKEN, dtype=torch.long)
padded_text_ids[0, :text_len] = text_ids[0]
print(f"Padded shape: {padded_text_ids.shape}")

# Run PyTorch with PADDED input (same as CoreML)
print("\n=== PyTorch with padded input ===")
with torch.no_grad():
    # Embed ALL 128 tokens
    text_embed = talker.model.text_embedding(padded_text_ids)
    text_projected = talker.text_projection(text_embed)

    # Language and BOS
    lang_embed = talker.model.codec_embedding(torch.tensor([[config.codec_language_id["english"]]]))
    bos_embed = talker.model.codec_embedding(torch.tensor([[config.codec_bos_id]]))

    # Combine: [lang, 128 text tokens, bos] = 130 tokens
    combined = torch.cat([lang_embed, text_projected, bos_embed], dim=1)
    print(f"Combined shape: {combined.shape}")

    # Run transformer
    outputs = talker.model(
        inputs_embeds=combined,
        use_cache=False,
        return_dict=True,
    )

    # Get logits from last position
    logits_pytorch = talker.codec_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1)
    first_token_pytorch = torch.argmax(logits_pytorch, dim=-1).item()
    print(f"PyTorch first token: {first_token_pytorch}")
    print(f"PyTorch top 5: {torch.argsort(logits_pytorch[0])[-5:].flip(0).tolist()}")

# Run CoreML
print("\n=== CoreML with same padded input ===")
lm_prefill = ct.models.MLModel("qwen3_tts_lm_prefill.mlpackage")

text_np = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
text_np[0, :text_len] = text_ids[0].numpy()

result = lm_prefill.predict({"text_ids": text_np})
logits_coreml = result["logits"]
first_token_coreml = int(np.argmax(logits_coreml, axis=-1)[0])
print(f"CoreML first token: {first_token_coreml}")
print(f"CoreML top 5: {np.argsort(logits_coreml[0])[-5:][::-1].tolist()}")

# Compare
print("\n=== Comparison ===")
print(f"Match: {first_token_pytorch == first_token_coreml}")

corr = np.corrcoef(logits_pytorch.numpy().flatten(), logits_coreml.flatten())[0, 1]
print(f"Correlation: {corr:.6f}")

diff = np.abs(logits_pytorch.numpy() - logits_coreml).max()
print(f"Max diff: {diff:.6f}")

# Now test with NO padding (real length only)
print("\n" + "=" * 60)
print("=== Test 2: PyTorch with REAL length (no padding) ===")
print("=" * 60)

with torch.no_grad():
    # Only embed real tokens
    text_embed_real = talker.model.text_embedding(text_ids)
    text_projected_real = talker.text_projection(text_embed_real)

    # Combine: [lang, real text, bos]
    combined_real = torch.cat([lang_embed, text_projected_real, bos_embed], dim=1)
    print(f"Combined shape (real): {combined_real.shape}")

    outputs_real = talker.model(
        inputs_embeds=combined_real,
        use_cache=False,
        return_dict=True,
    )

    logits_real = talker.codec_head(outputs_real.last_hidden_state[:, -1:, :]).squeeze(1)
    first_token_real = torch.argmax(logits_real, dim=-1).item()
    print(f"PyTorch (real) first token: {first_token_real}")
    print(f"PyTorch (real) top 5: {torch.argsort(logits_real[0])[-5:].flip(0).tolist()}")

print(f"\nPadded token: {first_token_pytorch}")
print(f"Real token: {first_token_real}")
print(f"Different due to padding: {first_token_pytorch != first_token_real}")
