# Debug: Get official tokens using the API that's known to work
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
import soundfile as sf

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
processor = tts_model.processor

text = "Hello world, this is a test of the text to speech system."
print(f"\nText: '{text}'")

# Use the known-working API: generate_voice_clone
speaker_embed = np.load("speaker_embedding_official.npy").reshape(1, 1024)
speaker_embed_t = torch.from_numpy(speaker_embed).to(torch.float32)

# Use the high-level API that works
print("\n" + "="*60)
print("Using generate_voice_clone with x_vector_only_mode=True")
print("="*60)

with torch.no_grad():
    audio, sr = tts_model.generate_voice_clone(
        text=text,
        ref_audio="reference_audio.wav",
        x_vector_only_mode=True,
        non_streaming_mode=True,  # This is what we're converting
    )

if isinstance(audio, list):
    audio = audio[0]
print(f"Generated audio shape: {audio.shape}, sr: {sr}")
if hasattr(audio, 'numpy'):
    audio = audio.numpy()
sf.write("debug_official_output.wav", audio, sr)
print("Saved: debug_official_output.wav")

# Now let's trace the model.generate directly to get the codes
print("\n" + "="*60)
print("Using model.generate to get codes directly")
print("="*60)

# Build the full input with template
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]
print(f"Full input IDs: {full_input_ids.tolist()}")

voice_clone_prompt = {
    'ref_spk_embedding': [speaker_embed_t.squeeze(0)],
    'x_vector_only_mode': [True],
    'icl_mode': [False],
    'ref_code': None,
}

with torch.no_grad():
    result = tts_model.model.generate(
        input_ids=[full_input_ids],
        languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True,
        max_new_tokens=100,
        do_sample=False,
    )

codes_list, hidden_list = result
codes = codes_list[0]  # [16, seq_len] or [seq_len, 16]
print(f"Codes shape: {codes.shape}")
print(f"First 20 codebook 0 tokens: {codes[0, :20].tolist()}")
print(f"Codebook 0 tokens (first row): {codes[0, :10].tolist()}")

# Compare with V8 wrapper
print("\n" + "="*60)
print("Testing V8 wrapper")
print("="*60)

from convert_lm_prefill_v8 import TracablePrefillV8, MAX_TEXT_LENGTH
talker = tts_model.model.talker

wrapper = TracablePrefillV8(talker)
wrapper.eval()

TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]

tokenizer = processor.tokenizer
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)

role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
text_ids = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
text_ids[0, :text_len] = torch.tensor(text_ids_list)
text_length = torch.tensor([text_len], dtype=torch.long)

with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

    logits, kv_cache = wrapper(role_ids, text_ids, text_length,
                                tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed_t)

print(f"Wrapper output shapes: logits={logits.shape}, kv_cache={kv_cache.shape}")

# Apply suppression mask
suppress_mask = np.zeros(talker.config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[talker.config.codec_eos_token_id] = False

logits_np = logits.numpy().copy()
logits_np[0, suppress_mask] = -float('inf')
v8_first_token = int(np.argmax(logits_np))
print(f"V8 wrapper first token: {v8_first_token}")

official_first_token = codes[0, 0].item()
print(f"Official first token: {official_first_token}")

if v8_first_token == official_first_token:
    print("\nMATCH! First tokens are identical.")
else:
    print(f"\nMISMATCH: V8={v8_first_token}, Official={official_first_token}")

# Show top 10 tokens for V8
top10 = np.argsort(logits_np[0])[-10:][::-1]
print(f"\nV8 top 10: {top10.tolist()}")
print(f"V8 top 10 logits: {[round(logits_np[0, t], 2) for t in top10]}")
