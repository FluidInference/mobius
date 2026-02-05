# Test Official Generate - Use the model's official generate method
import torch
import numpy as np
import soundfile as sf
import time

print("=" * 60)
print("Test Official Generate")
print("=" * 60)

print("\n1. Loading model...")
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
processor = model.processor

text = "Hello world, this is a test of the text to speech system."
print(f"\nText: '{text}'")

# Build input_ids the way the model expects
# The model uses _build_assistant_text which wraps text with special tokens
tokenizer = processor.tokenizer

# Format: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
im_start_token = tokenizer.encode("<|im_start|>", add_special_tokens=False)
im_end_token = tokenizer.encode("<|im_end|>", add_special_tokens=False)
newline_token = tokenizer.encode("\n", add_special_tokens=False)
assistant_token = tokenizer.encode("assistant", add_special_tokens=False)

# Build: <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
text_tokens = tokenizer.encode(text, add_special_tokens=False)
full_tokens = (
    im_start_token + assistant_token + newline_token +
    text_tokens +
    im_end_token + newline_token +
    im_start_token + assistant_token + newline_token
)
input_ids = torch.tensor([full_tokens], dtype=torch.long)
print(f"Input IDs shape: {input_ids.shape}")
print(f"First 20 tokens: {input_ids[0, :20].tolist()}")

# Generate
print("\n2. Generating with official model.model.generate...")
t0 = time.time()
with torch.no_grad():
    talker_codes_list, _ = model.model.generate(
        input_ids=[input_ids],  # Each element is [1, seq_len]
        languages=["english"],
        non_streaming_mode=True,
        max_new_tokens=125,
        do_sample=False,  # Deterministic
    )
gen_time = time.time() - t0
print(f"Generation time: {gen_time:.2f}s")

codes = talker_codes_list[0]
print(f"Codes shape: {codes.shape}")
print(f"First 10 codes: {codes[:10, 0].tolist()}")

# Decode to audio
print("\n3. Decoding to audio...")
t0 = time.time()
with torch.no_grad():
    wavs, sample_rate = model.model.speech_tokenizer.decode([{"audio_codes": codes}])
decode_time = time.time() - t0
print(f"Decode time: {decode_time:.2f}s")

audio = wavs[0]
duration = len(audio) / sample_rate
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sample_rate}")
print(f"Duration: {duration:.2f}s")
print(f"Audio RMS: {np.sqrt(np.mean(audio**2)):.4f}")

output_file = "test_official_generate_output.wav"
sf.write(output_file, audio, sample_rate)
print(f"Saved: {output_file}")

# Save codes for comparison
np.save("official_generate_codes.npy", codes.numpy())
print(f"\nSaved codes to official_generate_codes.npy")
print(f"Codebook 0: {codes[:10, 0].tolist()}")
print(f"Codebook 1: {codes[:10, 1].tolist()}")
