# Generate audio using official PyTorch pipeline for comparison
import torch
import numpy as np
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import warnings
import time
warnings.filterwarnings('ignore')

text = "Hello world, this is a test of the text to speech system."

print("=" * 60)
print("Audio Generation with Official PyTorch Pipeline")
print("=" * 60)

print("\n1. Loading model...")
t0 = time.time()
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
print(f"   Load time: {time.time() - t0:.1f}s")

print(f"\n2. Input text: '{text}'")

# Load speaker embedding
speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024)
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed_np.squeeze(0))],
    'x_vector_only_mode': [True],
    'icl_mode': [False],
    'ref_code': None,
}

print("\n3. Generating audio...")
t0 = time.time()

input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

with torch.no_grad():
    result = tts_model.model.generate(
        input_ids=[full_input_ids],
        languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True,
        max_new_tokens=125,
        do_sample=False,
        subtalker_dosample=False,
    )

gen_time = time.time() - t0

# result is (codes, audio) tuple
codes = result[0][0]  # [num_tokens, 16]
print(f"   Generated codes shape: {codes.shape}")
print(f"   Generation time: {gen_time:.2f}s")

# Decode to audio
print("\n4. Decoding to audio...")
t0 = time.time()

from qwen_tts import Qwen3TTSTokenizer
tokenizer_model = Qwen3TTSTokenizer.from_pretrained("./tokenizer_12hz", device_map="cpu")

with torch.no_grad():
    # codes: [num_tokens, 16] -> [1, 16, num_tokens]
    codes_for_decoder = codes.T.unsqueeze(0)
    audio = tokenizer_model.model.decoder(codes_for_decoder)

decode_time = time.time() - t0
print(f"   Decode time: {decode_time * 1000:.1f}ms")
print(f"   Audio shape: {audio.shape}")

# Save
audio_np = audio[0, 0].numpy()
duration = len(audio_np) / 24000
output_path = "output_official.wav"
sf.write(output_path, audio_np, 24000)
print(f"\n5. Saved: {output_path}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Text: '{text}'")
print(f"Tokens: {codes.shape[0]}")
print(f"Audio duration: {duration:.2f}s")
print(f"Generation: {gen_time:.2f}s")
print(f"Decode: {decode_time * 1000:.1f}ms")
print(f"\nOutput: {output_path}")

# Show first 10 codebook 0 tokens
print(f"\nFirst 10 codebook 0 tokens: {codes[:10, 0].tolist()}")
