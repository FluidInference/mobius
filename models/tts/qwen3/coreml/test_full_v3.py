# Full test of V9 prefill + V3 decode (PyTorch only)
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH
from convert_lm_decode_v3 import TracableDecodeV3
import warnings
warnings.filterwarnings('ignore')

TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]

text = "Hello world, this is a test."

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = tts_model.model.talker
config = talker.config
tokenizer = tts_model.processor.tokenizer

prefill_wrapper = TracablePrefillV9(talker)
decode_wrapper = TracableDecodeV3(talker)
prefill_wrapper.eval()
decode_wrapper.eval()

# Prepare inputs
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)
actual_len = text_len + 11

print(f"text_len={text_len}, actual_len={actual_len}")

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

speaker_embed = torch.from_numpy(np.load("speaker_embedding_official.npy").reshape(1, 1024)).float()

# Run prefill
print("\n=== Running Prefill V9 ===")
with torch.no_grad():
    logits, kv_cache, past_hidden = prefill_wrapper(
        role_ids, text_ids, text_length,
        tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed
    )

# Slice KV cache to actual length
kv_cache = kv_cache[:, :, :, :actual_len, :]
print(f"KV cache shape: {kv_cache.shape}")
print(f"past_hidden shape: {past_hidden.shape}")

# Get first token
suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

logits_np = logits.numpy().copy()
logits_np[0, suppress_mask] = -float('inf')
first_token = int(np.argmax(logits_np))
print(f"First token: {first_token}")

# Run decode
print("\n=== Running Decode V3 ===")
tokens = [first_token]
pos = actual_len

for i in range(30):
    token_id = torch.tensor([[tokens[-1]]], dtype=torch.long)
    position = torch.tensor([pos], dtype=torch.long)

    with torch.no_grad():
        logits, kv_cache, past_hidden = decode_wrapper(
            token_id, past_hidden, tts_pad_embed, kv_cache, position
        )

    logits_np = logits.numpy().copy()
    logits_np[0, suppress_mask] = -float('inf')
    next_token = int(np.argmax(logits_np))

    if next_token == config.codec_eos_token_id:
        print(f"EOS at step {i}")
        break

    tokens.append(next_token)
    pos += 1

print(f"\nGenerated: {tokens[:20]}")

# Compare with official
print("\n=== Running Official ===")
speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024)
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed_np.squeeze(0))],
    'x_vector_only_mode': [True], 'icl_mode': [False], 'ref_code': None,
}
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

with torch.no_grad():
    result = tts_model.model.generate(
        input_ids=[full_input_ids], languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True, max_new_tokens=30, do_sample=False,
        subtalker_dosample=False,  # Match V3's greedy code_predictor
    )
codes = result[0][0][:, 0].tolist()
print(f"Official: {codes[:20]}")

# Compare
print(f"\n=== Comparison ===")
n_compare = min(20, len(tokens), len(codes))
match = tokens[:n_compare] == codes[:n_compare]
print(f"First {n_compare} tokens match: {match}")
if not match:
    for i in range(n_compare):
        status = "OK" if tokens[i] == codes[i] else "MISMATCH"
        print(f"  Step {i}: Wrapper={tokens[i]}, Official={codes[i]} - {status}")
else:
    print("All tokens match!")
