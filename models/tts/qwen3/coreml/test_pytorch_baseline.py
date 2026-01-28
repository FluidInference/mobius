# Test PyTorch baseline to verify model works correctly
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "./model_0.6b",
    device_map="cpu",
    torch_dtype=torch.float32,
)
tokenizer = Qwen3TTSTokenizer.from_pretrained("./tokenizer_12hz", device_map="cpu")
print("Model loaded!")

talker = model.model.talker
config = talker.config
processor = model.processor

text = "Hello world, this is a test of the text to speech system."
print(f"\nText: '{text}'")

# Tokenize
inputs = processor(text=text, return_tensors="pt")
text_ids = inputs.input_ids
print(f"Text tokens: {text_ids.shape} = {text_ids[0].tolist()}")

# Check special tokens
print(f"\nSpecial tokens:")
print(f"  Language ID (english): {config.codec_language_id['english']}")
print(f"  BOS ID: {config.codec_bos_id}")
print(f"  EOS ID: {config.codec_eos_token_id}")
print(f"  PAD ID: {config.codec_pad_id}")

# Build proper input for the talker
language_id = config.codec_language_id["english"]
bos_id = config.codec_bos_id
eos_id = config.codec_eos_token_id

print(f"\n=== Testing Talker Generation ===")

with torch.no_grad():
    # The talker uses inputs_embeds, not input_ids
    # Build embeddings manually:
    # - Language token → codec_embedding
    # - Text tokens → text_embedding → text_projection
    # - BOS token → codec_embedding

    # Text embeddings
    text_embed = talker.model.text_embedding(text_ids)
    text_projected = talker.text_projection(text_embed)
    print(f"Text embed: {text_embed.shape} → projected: {text_projected.shape}")

    # Language embedding
    lang_tokens = torch.tensor([[language_id]])
    lang_embed = talker.model.codec_embedding(lang_tokens)
    print(f"Language embed: {lang_embed.shape}")

    # BOS embedding
    bos_tokens = torch.tensor([[bos_id]])
    bos_embed = talker.model.codec_embedding(bos_tokens)
    print(f"BOS embed: {bos_embed.shape}")

    # Combine: [lang, text, bos]
    combined_embeds = torch.cat([lang_embed, text_projected, bos_embed], dim=1)
    print(f"Combined embeds: {combined_embeds.shape}")

    # Run prefill
    outputs = talker.model(
        inputs_embeds=combined_embeds,
        use_cache=True,
        return_dict=True,
    )
    print(f"Hidden states: {outputs.last_hidden_state.shape}")

    # Get first codec token
    logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :])
    first_token = torch.argmax(logits, dim=-1)
    print(f"First token: {first_token.item()}")

    # Generate more tokens
    past_kv = outputs.past_key_values
    generated = [first_token.item()]
    current_token = first_token

    MAX_TOKENS = 50
    for i in range(MAX_TOKENS):
        # Embed the current token using codec_embedding
        token_embed = talker.model.codec_embedding(current_token)

        outputs = talker.model(
            inputs_embeds=token_embed,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        logits = talker.codec_head(outputs.last_hidden_state)
        current_token = torch.argmax(logits, dim=-1)
        generated.append(current_token.item())
        past_kv = outputs.past_key_values

        if current_token.item() == eos_id:
            print(f"EOS at step {i+1}")
            break

    print(f"\nGenerated {len(generated)} tokens:")
    print(f"  {generated[:20]}...")

# Compare with CoreML
print(f"\n=== Comparing with CoreML ===")

import coremltools as ct

lm_prefill = ct.models.MLModel("qwen3_tts_lm_prefill.mlpackage")

# Prepare input for CoreML - need to match the format
MAX_TEXT_LENGTH = 128
text_np = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
text_len = text_ids.shape[1]
text_np[0, :text_len] = text_ids[0].numpy()

prefill_result = lm_prefill.predict({"text_ids": text_np})
coreml_logits = prefill_result["logits"]
coreml_first_token = int(np.argmax(coreml_logits, axis=-1)[0])

print(f"PyTorch first token: {first_token.item()}")
print(f"CoreML first token: {coreml_first_token}")
print(f"Match: {first_token.item() == coreml_first_token}")

# Check logits correlation
pytorch_logits = logits.squeeze().numpy()
print(f"\nLogits comparison:")
print(f"  PyTorch logits shape: {pytorch_logits.shape}")
print(f"  CoreML logits shape: {coreml_logits.shape}")

corr = np.corrcoef(pytorch_logits.flatten(), coreml_logits.flatten())[0, 1]
print(f"  Correlation: {corr:.6f}")

diff = np.abs(pytorch_logits - coreml_logits).max()
print(f"  Max diff: {diff:.6f}")

print(f"\n  PyTorch top 5: {np.argsort(pytorch_logits)[-5:][::-1].tolist()}")
print(f"  CoreML top 5: {np.argsort(coreml_logits[0])[-5:][::-1].tolist()}")
