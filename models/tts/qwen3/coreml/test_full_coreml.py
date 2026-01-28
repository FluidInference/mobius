# Full CoreML pipeline test: CoreML prefill + CoreML decode + PyTorch code_predictor
import torch
import numpy as np
import coremltools as ct
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import MAX_TEXT_LENGTH
from convert_lm_decode_v4 import compute_decode_inputs
import warnings
warnings.filterwarnings('ignore')

TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]

text = "Hello world, this is a test."

print("Loading PyTorch model for components...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = tts_model.model.talker
config = talker.config
tokenizer = tts_model.processor.tokenizer

# Load CoreML models
print("Loading CoreML prefill model...")
prefill_coreml = ct.models.MLModel("qwen3_tts_lm_prefill_v9.mlpackage")

print("Loading CoreML decode model...")
decode_coreml = ct.models.MLModel("qwen3_tts_lm_decode_v4.mlpackage")

# Prepare inputs
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)
actual_len = text_len + 11

print(f"\nText: '{text}'")
print(f"text_len={text_len}, actual_len={actual_len}")

# Prepare embeddings (still need PyTorch for these)
with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

speaker_embed = torch.from_numpy(np.load("speaker_embedding_official.npy").reshape(1, 1024)).float()

# Run CoreML prefill
print("\n=== Running Prefill (CoreML) ===")

role_ids = np.array([ROLE_PREFIX], dtype=np.int32)
text_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
text_ids[0, :text_len] = np.array(text_ids_list, dtype=np.int32)
text_length = np.array([text_len], dtype=np.int32)

prefill_out = prefill_coreml.predict({
    'role_ids': role_ids,
    'text_ids': text_ids,
    'text_length': text_length,
    'tts_bos_embed': tts_bos_embed.numpy().astype(np.float32),
    'tts_pad_embed': tts_pad_embed.numpy().astype(np.float32),
    'tts_eos_embed': tts_eos_embed.numpy().astype(np.float32),
    'speaker_embed': speaker_embed.numpy().astype(np.float32),
})

logits = prefill_out['logits']
kv_cache = prefill_out['kv_cache']
past_hidden = prefill_out['past_hidden']

# Slice KV cache to actual length
kv_cache = kv_cache[:, :, :, :actual_len, :]
print(f"KV cache shape: {kv_cache.shape}")
print(f"past_hidden shape: {past_hidden.shape}")

# Convert to torch for further processing
kv_cache = torch.from_numpy(kv_cache)
past_hidden = torch.from_numpy(past_hidden)

# Get first token
suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

logits_m = logits.copy()
logits_m[0, suppress_mask] = -float('inf')
first_token = int(np.argmax(logits_m))
print(f"First token: {first_token}")

# Run decode (CoreML)
print("\n=== Running Decode (CoreML main + PyTorch code_predictor) ===")
tokens = [first_token]
pos = actual_len

for i in range(30):
    token_id = torch.tensor([[tokens[-1]]], dtype=torch.long)

    # Compute inputs_embeds using PyTorch code_predictor
    with torch.no_grad():
        inputs_embeds = compute_decode_inputs(talker, token_id, past_hidden, tts_pad_embed)

    # Run CoreML decode
    out = decode_coreml.predict({
        'inputs_embeds': inputs_embeds.numpy().astype(np.float32),
        'kv_cache': kv_cache.numpy().astype(np.float32),
        'position': np.array([pos], dtype=np.int32),
    })

    logits = out['logits']
    kv_cache = torch.from_numpy(out['new_kv_cache'])
    past_hidden = torch.from_numpy(out['past_hidden'])

    logits_m = logits.copy()
    logits_m[0, suppress_mask] = -float('inf')
    next_token = int(np.argmax(logits_m))

    if next_token == config.codec_eos_token_id:
        print(f"EOS at step {i}")
        break

    tokens.append(next_token)
    pos += 1

print(f"\nGenerated (Full CoreML): {tokens[:20]}")

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
        subtalker_dosample=False,  # Match greedy code_predictor
    )
codes = result[0][0][:, 0].tolist()
print(f"Official: {codes[:20]}")

# Compare
print(f"\n=== Comparison ===")
n_compare = min(20, len(tokens), len(codes))
match_count = sum(1 for i in range(n_compare) if tokens[i] == codes[i])
print(f"Matched: {match_count}/{n_compare} tokens")

if tokens[:n_compare] != codes[:n_compare]:
    for i in range(n_compare):
        status = "OK" if tokens[i] == codes[i] else "MISMATCH"
        print(f"  Step {i}: CoreML={tokens[i]}, Official={codes[i]} - {status}")
else:
    print("All tokens match!")
