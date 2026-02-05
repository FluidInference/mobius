# Debug V9 decode vs official
import torch
import numpy as np
import coremltools as ct

print("Loading models...")
from qwen_tts import Qwen3TTSModel
from convert_lm_decode_v3 import TracableDecodeV3

model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
processor = model.processor
talker = model.model.talker
config = talker.config

# Create V3 decode wrapper
decode_wrapper = TracableDecodeV3(talker)
decode_wrapper.eval()

# Load CoreML prefill V9
lm_prefill = ct.models.MLModel("qwen3_tts_lm_prefill_v9.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)

# Constants
TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]
MAX_TEXT_LENGTH = 128

text = "Hello world, this is a test of the text to speech system."
tokenizer = processor.tokenizer
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)

print(f"Text: '{text}'")
print(f"Text tokens: {text_len}")

# Prepare embeddings
with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024).astype(np.float32)
speaker_embed = torch.from_numpy(speaker_embed_np)

# V9 CoreML Prefill
role_ids = np.array([ROLE_PREFIX], dtype=np.int32)
text_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
text_ids[0, :text_len] = text_ids_list
text_length = np.array([text_len], dtype=np.int32)

print("\n=== V9 CoreML Prefill ===")
prefill_result = lm_prefill.predict({
    "role_ids": role_ids,
    "text_ids": text_ids,
    "text_length": text_length,
    "tts_bos_embed": tts_bos_embed.numpy().astype(np.float32),
    "tts_pad_embed": tts_pad_embed.numpy().astype(np.float32),
    "tts_eos_embed": tts_eos_embed.numpy().astype(np.float32),
    "speaker_embed": speaker_embed_np,
})

v9_logits = prefill_result["logits"]
v9_kv_cache = prefill_result["kv_cache"]
v9_past_hidden = prefill_result["past_hidden"]

actual_len = text_len + 11
v9_kv_cache = v9_kv_cache[:, :, :, :actual_len, :]

print(f"V9 logits shape: {v9_logits.shape}")
print(f"V9 KV cache shape: {v9_kv_cache.shape}")
print(f"V9 past_hidden shape: {v9_past_hidden.shape}")

# Get first token from V9
suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

v9_logits_copy = v9_logits.copy()
v9_logits_copy[0, suppress_mask] = -float('inf')
v9_first_token = int(np.argmax(v9_logits_copy))
print(f"V9 first token: {v9_first_token}")

# Now compare with official PyTorch generation
print("\n=== Official PyTorch Prefill ===")
assistant_text = model._build_assistant_text(text)
input_ids = model._tokenize_texts([assistant_text])[0]

voice_clone_prompt = {
    'ref_spk_embedding': [speaker_embed.squeeze(0)],
    'x_vector_only_mode': [True],
    'icl_mode': [False],
    'ref_code': None,
}

# Run official generate with tracing
with torch.no_grad():
    result = model.model.generate(
        input_ids=[input_ids],
        languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True,
        max_new_tokens=10,
        do_sample=False,
    )

official_codes = result[0][0]
print(f"Official codes shape: {official_codes.shape}")
print(f"Official first 10 tokens: {official_codes[:10, 0].tolist()}")

# V9 decode step by step
print("\n=== V9 Decode Step by Step ===")
v9_tokens = [v9_first_token]
kv_cache_torch = torch.from_numpy(v9_kv_cache).float()
past_hidden_torch = torch.from_numpy(v9_past_hidden).float()
position = actual_len

for step in range(9):  # Generate 9 more tokens
    token_id = torch.tensor([[v9_tokens[-1]]], dtype=torch.long)
    position_tensor = torch.tensor([position], dtype=torch.long)

    with torch.no_grad():
        logits_torch, kv_cache_torch, past_hidden_torch = decode_wrapper(
            token_id, past_hidden_torch, tts_pad_embed, kv_cache_torch, position_tensor
        )

    logits_np = logits_torch.numpy().copy()
    logits_np[0, suppress_mask] = -float('inf')
    next_token = int(np.argmax(logits_np))
    v9_tokens.append(next_token)
    position += 1

print(f"V9 first 10 tokens: {v9_tokens}")

print("\n=== Comparison ===")
for i in range(min(10, len(v9_tokens))):
    official_tok = official_codes[i, 0].item()
    v9_tok = v9_tokens[i]
    match = "MATCH" if official_tok == v9_tok else "DIFF"
    print(f"Position {i}: official={official_tok}, v9={v9_tok} - {match}")
