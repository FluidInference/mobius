# Full TTS Pipeline Test - Pure PyTorch
import torch
import numpy as np
import soundfile as sf
import time

SAMPLE_RATE = 24000

print("=" * 60)
print("Full TTS Pipeline Test - Pure PyTorch Reference")
print("=" * 60)

# Load models
print("\n1. Loading models...")
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

t0 = time.time()
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
tokenizer = Qwen3TTSTokenizer.from_pretrained("./tokenizer_12hz", device_map="cpu")
processor = tts_model.processor
talker = tts_model.model.talker
config = talker.config
print(f"   Loaded in {time.time() - t0:.1f}s")

text = "Hello world, this is a test of the text to speech system."
print(f"\n2. Input text: '{text}'")

inputs = processor(text=text, return_tensors="pt")
text_ids = inputs.input_ids
text_len = text_ids.shape[1]
print(f"   Text tokens: {text_len}")

# === LM Generation (PyTorch) ===
print("\n3. LM Generation (PyTorch)...")

EOS_TOKEN = config.codec_eos_token_id
MAX_CODEC_TOKENS = 125

with torch.no_grad():
    # Prefill
    t0 = time.time()
    text_embed = talker.model.text_embedding(text_ids)
    text_projected = talker.text_projection(text_embed)

    lang_embed = talker.model.codec_embedding(torch.tensor([[config.codec_language_id["english"]]]))
    bos_embed = talker.model.codec_embedding(torch.tensor([[config.codec_bos_id]]))

    combined = torch.cat([lang_embed, text_projected, bos_embed], dim=1)
    outputs = talker.model(inputs_embeds=combined, use_cache=True, return_dict=True)

    logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :])
    first_token = torch.argmax(logits, dim=-1).item()
    past_kv = outputs.past_key_values
    prefill_time = time.time() - t0
    print(f"   Prefill: {prefill_time * 1000:.1f}ms")
    print(f"   First token: {first_token}")

    # Decode
    generated_tokens = [first_token]
    current_token = torch.tensor([[first_token]])

    t0 = time.time()
    while len(generated_tokens) < MAX_CODEC_TOKENS:
        token_embed = talker.model.codec_embedding(current_token)
        outputs = talker.model(
            inputs_embeds=token_embed,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        logits = talker.codec_head(outputs.last_hidden_state)
        next_token = torch.argmax(logits, dim=-1).item()
        generated_tokens.append(next_token)
        current_token = torch.tensor([[next_token]])
        past_kv = outputs.past_key_values

        if next_token == EOS_TOKEN:
            print(f"   EOS at token {len(generated_tokens)}")
            break

    lm_time = time.time() - t0
    num_tokens = len(generated_tokens)
    print(f"   Generated {num_tokens} tokens in {lm_time:.2f}s ({num_tokens/lm_time:.1f} tok/s)")
    print(f"   Codebook 0: {generated_tokens[:10]}...")

# === Decode (PyTorch - Code Predictor + Decoder combined) ===
print("\n4. Decode codes to audio (PyTorch)...")

t0 = time.time()
codebook0_tensor = torch.tensor([generated_tokens], dtype=torch.long)
# Pad to 16 codebooks (only codebook 0 is provided, rest will be predicted)
codes_input = torch.zeros(1, 16, len(generated_tokens), dtype=torch.long)
codes_input[:, 0, :] = codebook0_tensor

with torch.no_grad():
    audio_pytorch = tokenizer.decode(codes_input)
decode_time = time.time() - t0
print(f"   Decode: {decode_time:.2f}s")
print(f"   Audio shape: {audio_pytorch.shape}")

audio_np = audio_pytorch[0, 0, :].numpy()
duration = len(audio_np) / SAMPLE_RATE

output_file = "test_pytorch_full_output.wav"
sf.write(output_file, audio_np, SAMPLE_RATE)
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
