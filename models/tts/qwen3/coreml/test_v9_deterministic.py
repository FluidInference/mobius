# Test V9 with deterministic seed - compare with official
import torch
import numpy as np
import coremltools as ct
import soundfile as sf
import time

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

MAX_TEXT_LENGTH = 128
MAX_CODEC_TOKENS = 125
SAMPLE_RATE = 24000

TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]

print("Loading models...")
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from convert_lm_decode_v3 import TracableDecodeV3

model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
tokenizer = Qwen3TTSTokenizer.from_pretrained("./tokenizer_12hz", device_map="cpu")
processor = model.processor
talker = model.model.talker
config = talker.config

# CoreML prefill
lm_prefill = ct.models.MLModel("qwen3_tts_lm_prefill_v9.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)

# PyTorch decode wrapper
decode_wrapper = TracableDecodeV3(talker)
decode_wrapper.eval()

# Pre-compute embeddings
with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024).astype(np.float32)
speaker_embed = torch.from_numpy(speaker_embed_np)

text = "Hello world, this is a test of the text to speech system."
print(f"\nText: '{text}'")

tokenizer_obj = processor.tokenizer
text_ids_list = tokenizer_obj.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)

# === Run official model with seed ===
print("\n=== Official Model (seed=42) ===")
torch.manual_seed(42)

voice_clone_prompt = {
    'ref_spk_embedding': [speaker_embed.squeeze(0)],
    'x_vector_only_mode': [True],
    'icl_mode': [False],
    'ref_code': None,
}

with torch.no_grad():
    result = model.model.generate(
        input_ids=[model._tokenize_texts([model._build_assistant_text(text)])[0]],
        languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True,
        max_new_tokens=MAX_CODEC_TOKENS,
        do_sample=False,
    )

official_codes = result[0][0]
print(f"Official codes shape: {official_codes.shape}")
print(f"Official codebook 0 first 20: {official_codes[:20, 0].tolist()}")

# Decode official to audio
with torch.no_grad():
    wavs, sample_rate = model.model.speech_tokenizer.decode([{"audio_codes": official_codes}])
official_audio = wavs[0]
sf.write("test_official_seeded.wav", official_audio, sample_rate)
print(f"Official audio saved: test_official_seeded.wav, RMS={np.sqrt(np.mean(official_audio**2)):.4f}")

# === Run V9 + V3 with seed ===
print("\n=== V9 + V3 (seed=42) ===")
torch.manual_seed(42)

role_ids = np.array([ROLE_PREFIX], dtype=np.int32)
text_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
text_ids[0, :text_len] = text_ids_list
text_length = np.array([text_len], dtype=np.int32)

actual_len = text_len + 11

# V9 Prefill
prefill_result = lm_prefill.predict({
    "role_ids": role_ids,
    "text_ids": text_ids,
    "text_length": text_length,
    "tts_bos_embed": tts_bos_embed.numpy().astype(np.float32),
    "tts_pad_embed": tts_pad_embed.numpy().astype(np.float32),
    "tts_eos_embed": tts_eos_embed.numpy().astype(np.float32),
    "speaker_embed": speaker_embed_np,
})

logits = prefill_result["logits"]
kv_cache = prefill_result["kv_cache"][:, :, :, :actual_len, :]
past_hidden = prefill_result["past_hidden"]

# Sampling
EOS_TOKEN = config.codec_eos_token_id
suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[EOS_TOKEN] = False

def sample_with_suppress(logits_np):
    logits_np = logits_np.copy()
    logits_np[0, suppress_mask] = -float('inf')
    return int(np.argmax(logits_np, axis=-1)[0])

generated_tokens = []
all_codebook_sequences = []
position = actual_len

first_token = sample_with_suppress(logits)
generated_tokens.append(first_token)

kv_cache_torch = torch.from_numpy(kv_cache).float()
past_hidden_torch = torch.from_numpy(past_hidden).float()

while len(generated_tokens) < MAX_CODEC_TOKENS:
    token_id = torch.tensor([[generated_tokens[-1]]], dtype=torch.long)
    position_tensor = torch.tensor([position], dtype=torch.long)

    with torch.no_grad():
        logits_torch, kv_cache_torch, past_hidden_torch, all_codebooks = decode_wrapper(
            token_id, past_hidden_torch, tts_pad_embed, kv_cache_torch, position_tensor
        )

    all_codebook_sequences.append(all_codebooks[0].tolist())
    next_token = sample_with_suppress(logits_torch.numpy())
    generated_tokens.append(next_token)
    position += 1

    if next_token == EOS_TOKEN:
        print(f"EOS at token {len(generated_tokens)}")
        break

print(f"V9 generated {len(generated_tokens)} tokens")
print(f"V9 codebook 0 first 20: {generated_tokens[:20]}")

# Build codes
codes_np = np.array(all_codebook_sequences, dtype=np.int64)
v9_codes = torch.from_numpy(codes_np)  # [seq_len, 16]

# Compare codebook 0
print(f"\n=== Comparison ===")
min_len = min(len(generated_tokens), official_codes.shape[0])
matches = 0
for i in range(min_len):
    official = official_codes[i, 0].item()
    v9 = generated_tokens[i] if i < len(generated_tokens) else -1
    if official == v9:
        matches += 1
        status = "MATCH"
    else:
        status = "DIFF"
    if i < 20:
        print(f"Position {i}: official={official}, v9={v9} - {status}")

print(f"\nMatches: {matches}/{min_len} ({100*matches/min_len:.1f}%)")

# Decode V9 to audio
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

encoded_output = EncoderOutput(audio_codes=[v9_codes])
with torch.no_grad():
    audio_list, sample_rate = tokenizer.decode(encoded_output)
    v9_audio = audio_list[0]

sf.write("test_v9_seeded.wav", v9_audio, sample_rate)
print(f"\nV9 audio saved: test_v9_seeded.wav, RMS={np.sqrt(np.mean(v9_audio**2)):.4f}")
print(f"Audio duration: official={len(official_audio)/sample_rate:.2f}s, v9={len(v9_audio)/sample_rate:.2f}s")
