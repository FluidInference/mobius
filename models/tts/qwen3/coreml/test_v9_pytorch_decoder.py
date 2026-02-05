# Test V9 pipeline with PyTorch decoder instead of CoreML decoder
import torch
import numpy as np
import coremltools as ct
import soundfile as sf
import time

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

role_ids = np.array([ROLE_PREFIX], dtype=np.int32)
text_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
text_ids[0, :text_len] = text_ids_list
text_length = np.array([text_len], dtype=np.int32)

actual_len = text_len + 11
print(f"Text tokens: {text_len}, Actual prefill length: {actual_len}")

# === LM Generation ===
print("\n=== LM Generation (V9 Prefill + V3 Decode) ===")

# Prefill
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
VOCAB_SIZE = config.vocab_size
suppress_mask = np.zeros(VOCAB_SIZE, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[EOS_TOKEN] = False

def sample_with_suppress(logits_np):
    logits_np = logits_np.copy()
    logits_np[0, suppress_mask] = -float('inf')
    return int(np.argmax(logits_np, axis=-1)[0])

generated_tokens = []
all_codebook_sequences = []  # Will store [seq_len, 16] codebook tokens
position = actual_len

first_token = sample_with_suppress(logits)
generated_tokens.append(first_token)
print(f"First token: {first_token}")

# Convert to PyTorch
kv_cache_torch = torch.from_numpy(kv_cache).float()
past_hidden_torch = torch.from_numpy(past_hidden).float()

t0 = time.time()
while len(generated_tokens) < MAX_CODEC_TOKENS:
    token_id = torch.tensor([[generated_tokens[-1]]], dtype=torch.long)
    position_tensor = torch.tensor([position], dtype=torch.long)

    with torch.no_grad():
        logits_torch, kv_cache_torch, past_hidden_torch, all_codebooks = decode_wrapper(
            token_id, past_hidden_torch, tts_pad_embed, kv_cache_torch, position_tensor
        )

    # Store all 16 codebook tokens for this step
    all_codebook_sequences.append(all_codebooks[0].tolist())  # [16]

    next_token = sample_with_suppress(logits_torch.numpy())
    generated_tokens.append(next_token)
    position += 1

    if next_token == EOS_TOKEN:
        print(f"EOS at token {len(generated_tokens)}")
        break

lm_time = time.time() - t0
num_tokens = len(generated_tokens)
print(f"Generated {num_tokens} tokens in {lm_time:.2f}s")
print(f"Codebook 0: {generated_tokens[:10]}...")

# === Build codes from collected sequences ===
print("\n=== Building Codes from Decode Output ===")

# all_codebook_sequences[i] contains codebooks for generated_tokens[i]
# The decode loop generates codebooks for each token as it's processed
# So all_codebook_sequences has length = num_tokens - 1 (we don't generate codebooks for the last token)
# But we need all codebooks including the last generated token

# Actually, the loop structure is:
# - First token (from prefill) = generated_tokens[0]
# - Decode step uses generated_tokens[-1] and produces codebooks for it, then appends next_token
# - So all_codebook_sequences[0] has codebooks for generated_tokens[0] (first token)
# - all_codebook_sequences[i] has codebooks for generated_tokens[i]

# Convert to tensor [seq_len, 16]
codes_np = np.array(all_codebook_sequences, dtype=np.int64)
codes_full = torch.from_numpy(codes_np).unsqueeze(0)  # [1, seq_len, 16]

print(f"Codes shape: {codes_full.shape}")
print(f"Generated {len(generated_tokens)} tokens, have {len(all_codebook_sequences)} codebook sequences")
print(f"Codebook 0 first 5: {codes_full[0, :5, 0].tolist()}")
print(f"Codebook 1 first 5: {codes_full[0, :5, 1].tolist()}")

# === Use PyTorch decoder (via tokenizer) ===
print("\n=== PyTorch Decoder ===")

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

t0 = time.time()
with torch.no_grad():
    audio_list, sample_rate = tokenizer.decode(encoded_output)
    audio_np = audio_list[0]
decode_time = time.time() - t0

print(f"Decode time: {decode_time:.2f}s")
print(f"Audio shape: {audio_np.shape}")
print(f"Audio RMS: {np.sqrt(np.mean(audio_np**2)):.4f}")

sf.write("test_v9_pytorch_decoder_output.wav", audio_np, sample_rate)
print(f"Saved: test_v9_pytorch_decoder_output.wav")
