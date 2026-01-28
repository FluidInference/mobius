# Test official with subtalker_dosample=False
import torch
import numpy as np
import random
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)
actual_len = text_len + 11

with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

speaker_embed = torch.from_numpy(np.load("speaker_embedding_official.npy").reshape(1, 1024)).float()

# Capture official code_predictor tokens
official_predictor_tokens = []
original_predictor_generate = talker.code_predictor.generate

def capture_predictor_generate(*args, **kwargs):
    result = original_predictor_generate(*args, **kwargs)
    official_predictor_tokens.append(result.sequences[0].tolist())
    return result

talker.code_predictor.generate = capture_predictor_generate

# Run official with subtalker_dosample=False
print("\n=== Running official with subtalker_dosample=False ===")
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
        non_streaming_mode=True, max_new_tokens=5, do_sample=False,
        subtalker_dosample=False,  # <-- Key parameter!
    )
codes = result[0][0][:, 0].tolist()
print(f"Official tokens (subtalker_dosample=False): {codes}")
print(f"Official predictor tokens (step 0): {official_predictor_tokens[0] if official_predictor_tokens else 'N/A'}")

talker.code_predictor.generate = original_predictor_generate

# Run V3 prefill and decode
print("\n=== Running V3 ===")
prefill_wrapper = TracablePrefillV9(talker)
prefill_wrapper.eval()

role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
text_ids = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
text_ids[0, :text_len] = torch.tensor(text_ids_list)
text_length = torch.tensor([text_len], dtype=torch.long)

with torch.no_grad():
    v9_logits, v9_kv_cache, v9_past_hidden = prefill_wrapper(
        role_ids, text_ids, text_length,
        tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed
    )

v9_kv_cache = v9_kv_cache[:, :, :, :actual_len, :]

suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

# V3 decode loop
v3_tokens = []
past_hidden = v9_past_hidden.clone()
kv_cache = v9_kv_cache.clone()
pos = actual_len

# Get first token
v9_logits_np = v9_logits.numpy().copy()
v9_logits_np[0, suppress_mask] = -float('inf')
first_token = int(np.argmax(v9_logits_np))
v3_tokens.append(first_token)

from convert_lm_decode_v3 import TracableDecodeV3
decode_wrapper = TracableDecodeV3(talker)
decode_wrapper.eval()

for i in range(4):  # Match official's 5 tokens
    token_id = torch.tensor([[v3_tokens[-1]]], dtype=torch.long)
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

    v3_tokens.append(next_token)
    pos += 1

print(f"V3 tokens: {v3_tokens}")

# Compare
print("\n=== Comparison ===")
print(f"Official (subtalker_dosample=False): {codes}")
print(f"V3 (do_sample=False):                {v3_tokens}")

if len(codes) > 0 and len(v3_tokens) > 0:
    n_compare = min(len(codes), len(v3_tokens))
    for i in range(n_compare):
        match = "OK" if i < len(codes) and i < len(v3_tokens) and codes[i] == v3_tokens[i] else "MISMATCH"
        print(f"  Token {i}: Official={codes[i] if i < len(codes) else 'N/A'}, V3={v3_tokens[i] if i < len(v3_tokens) else 'N/A'} - {match}")
