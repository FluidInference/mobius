# Test full pipeline with V9 prefill and V2 decode
import torch
import numpy as np
import coremltools as ct
from qwen_tts import Qwen3TTSModel
import warnings
warnings.filterwarnings('ignore')

TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]
text = 'Hello world, this is a test.'

print('Loading models...')
tts_model = Qwen3TTSModel.from_pretrained('./model_0.6b', device_map='cpu', torch_dtype=torch.float32)
talker = tts_model.model.talker
config = talker.config
tokenizer = tts_model.processor.tokenizer

prefill = ct.models.MLModel('qwen3_tts_lm_prefill_v9.mlpackage')
decode = ct.models.MLModel('qwen3_tts_lm_decode_v2.mlpackage')

# Prepare inputs
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)
actual_len = text_len + 11  # IMPORTANT: this is the valid length in KV cache
MAX_TEXT_LENGTH = 128

print(f'text_len={text_len}, actual_len={actual_len}')

role_ids = np.array([ROLE_PREFIX], dtype=np.int32)
text_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
text_ids[0, :text_len] = text_ids_list
text_length = np.array([text_len], dtype=np.int32)

with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :].numpy()
    tts_pad_embed = tts_embed[:, 1:2, :].numpy()
    tts_eos_embed = tts_embed[:, 2:3, :].numpy()

speaker_embed = np.load('speaker_embedding_official.npy').reshape(1, 1024).astype(np.float32)

print('Running prefill...')
out = prefill.predict({
    'role_ids': role_ids, 'text_ids': text_ids, 'text_length': text_length,
    'tts_bos_embed': tts_bos_embed, 'tts_pad_embed': tts_pad_embed,
    'tts_eos_embed': tts_eos_embed, 'speaker_embed': speaker_embed,
})
logits = out['logits']
kv_full = out['kv_cache']

# CRITICAL: Slice KV cache to only valid positions
kv = kv_full[:, :, :, :actual_len, :]
print(f'KV cache shape: {kv_full.shape} -> sliced to {kv.shape}')

suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

logits_m = logits.copy()
logits_m[0, suppress_mask] = -float('inf')
first_token = int(np.argmax(logits_m))
print(f'First token: {first_token}')

# Decode
print('Decoding...')
tokens = [first_token]
pos = actual_len  # Start position for decode

for i in range(30):
    out = decode.predict({
        'token_id': np.array([[tokens[-1]]], dtype=np.int32),
        'position': np.array([pos], dtype=np.int32),
        'kv_cache': kv.astype(np.float32),
        'trailing_text_embed': tts_pad_embed.astype(np.float32),
    })
    logits = out['logits']
    kv = out['new_kv_cache']

    logits_m = logits.copy()
    logits_m[0, suppress_mask] = -float('inf')
    next_tok = int(np.argmax(logits_m))

    if next_tok == config.codec_eos_token_id:
        print(f'EOS at step {i}')
        break
    tokens.append(next_tok)
    pos += 1

print(f'CoreML generated: {tokens[:20]}')

# Compare with official
print()
print('Official...')
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed.squeeze(0))],
    'x_vector_only_mode': [True], 'icl_mode': [False], 'ref_code': None,
}
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

with torch.no_grad():
    result = tts_model.model.generate(
        input_ids=[full_input_ids], languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True, max_new_tokens=30, do_sample=False,
    )
codes = result[0][0][:, 0].tolist()
print(f'Official: {codes[:20]}')

match = tokens[:20] == codes[:20]
print(f'First 20 match: {match}')
if not match:
    for i in range(min(20, len(tokens), len(codes))):
        if tokens[i] != codes[i]:
            print(f'  Pos {i}: CoreML={tokens[i]}, Official={codes[i]}')
