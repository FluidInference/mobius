# Pure PyTorch Reference - Generate audio and compare with V9
import torch
import numpy as np
import soundfile as sf
import time

SAMPLE_RATE = 24000
MAX_CODEC_TOKENS = 125

print("=" * 60)
print("Pure PyTorch Reference - Proper Generation")
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
print(f"   Token IDs: {text_ids[0, :10].tolist()}...")

# === Try simple LM generation ===
print("\n3. Simple LM Generation...")

EOS_TOKEN = config.codec_eos_token_id
print(f"   EOS Token: {EOS_TOKEN}")

with torch.no_grad():
    # Prefill - simple version without speaker
    text_embed = talker.model.text_embedding(text_ids)
    text_projected = talker.text_projection(text_embed)

    lang_id = config.codec_language_id["english"]
    bos_id = config.codec_bos_id
    print(f"   Language ID: {lang_id}, BOS ID: {bos_id}")

    lang_embed = talker.model.codec_embedding(torch.tensor([[lang_id]]))
    bos_embed = talker.model.codec_embedding(torch.tensor([[bos_id]]))

    combined = torch.cat([lang_embed, text_projected, bos_embed], dim=1)
    print(f"   Combined shape: {combined.shape}")

    outputs = talker.model(inputs_embeds=combined, use_cache=True, return_dict=True)

    logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :])
    first_token = torch.argmax(logits, dim=-1).item()
    past_kv = outputs.past_key_values
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

# === Use code predictor to get codebooks 1-14 ===
print("\n4. Code Predictor...")

t0 = time.time()
codebook0_tensor = torch.tensor([generated_tokens], dtype=torch.long)

with torch.no_grad():
    # Use the code_predictor to get codebooks 1-14
    code_predictor = talker.code_predictor

    all_codebooks = [codebook0_tensor]
    for gen_steps in range(1, 15):
        output = code_predictor(input_ids=codebook0_tensor, generation_steps=gen_steps)
        predicted = torch.argmax(output.logits, dim=-1)
        all_codebooks.append(predicted)

    # Stack to [1, seq_len, 16] (transpose needed for tokenizer format)
    codes_stacked = torch.stack(all_codebooks, dim=-1)  # [1, seq_len, 16]
    # Add codebook 15 as zeros
    codes_full = torch.cat([codes_stacked, torch.zeros_like(codes_stacked[:, :, 0:1])], dim=-1)  # [1, seq_len, 16]

cp_time = time.time() - t0
print(f"   Code Predictor: {cp_time:.2f}s")
print(f"   Codes shape: {codes_full.shape}")
print(f"   Codebook 1 (first 5): {codes_full[0, :5, 1].tolist()}")

# === Decode ===
print("\n5. Decode codes to audio...")

# Build the format expected by tokenizer.decode
t0 = time.time()
# Create an object that looks like encode output
class EncoderOutput:
    def __init__(self, audio_codes):
        self.audio_codes = audio_codes
    def keys(self):
        return ['audio_codes']
    def items(self):
        return [('audio_codes', self.audio_codes)]
    def __getitem__(self, key):
        if key == 'audio_codes':
            return self.audio_codes
        raise KeyError(key)

encoded_output = EncoderOutput(audio_codes=[codes_full[0]])  # [seq_len, 16]

with torch.no_grad():
    audio_list, sample_rate = tokenizer.decode(encoded_output)
    audio_np = audio_list[0]
decode_time = time.time() - t0
print(f"   Decode: {decode_time:.2f}s")
print(f"   Audio shape: {audio_np.shape}")

duration = len(audio_np) / sample_rate
print(f"   Duration: {duration:.2f}s")
print(f"   Audio RMS: {np.sqrt(np.mean(audio_np**2)):.4f}")
print(f"   Audio range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")

output_file = "test_pytorch_reference_output.wav"
sf.write(output_file, audio_np, SAMPLE_RATE)
print(f"   Saved: {output_file}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Generated {num_tokens} tokens")
print(f"Audio duration: {duration:.2f}s")
print(f"LM time: {lm_time:.2f}s")
print(f"Decode time: {decode_time:.2f}s")

# Save codebook0 for comparison
np.save("pytorch_reference_codebook0.npy", np.array(generated_tokens))
print(f"\nSaved codebook0 to pytorch_reference_codebook0.npy")
