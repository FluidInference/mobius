# Test V3 decode with proper past_hidden and code_predictor
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
print("\n=== Running PyTorch Prefill ===")
with torch.no_grad():
    logits, kv_cache = prefill_wrapper(
        role_ids, text_ids, text_length,
        tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed
    )

# Slice KV cache to actual length
kv_cache = kv_cache[:, :, :, :actual_len, :]
print(f"KV cache shape after slice: {kv_cache.shape}")

# Get first token and past_hidden
suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

logits_np = logits.numpy().copy()
logits_np[0, suppress_mask] = -float('inf')
first_token = int(np.argmax(logits_np))
print(f"First token: {first_token}")

# For past_hidden, we need to run a forward on the prefill again to get last hidden state
# Actually, let me extract it from the prefill wrapper directly
# The prefill returns logits which come from codec_head(last_hidden), so we need last_hidden

# Re-run to get last_hidden
print("\nExtracting past_hidden from prefill...")
with torch.no_grad():
    # Build hidden states same as prefill
    from convert_lm_prefill_v9 import FIXED_SEQ_LEN
    batch_size = 1
    device = role_ids.device

    hidden_states = torch.zeros(batch_size, FIXED_SEQ_LEN, config.hidden_size)
    role_embed = prefill_wrapper.text_projection(prefill_wrapper.text_embedding(role_ids))
    hidden_states[:, 0:3, :] = role_embed

    codec_think_ids = torch.tensor([
        [config.codec_think_id, config.codec_think_bos_id,
         config.codec_language_id["english"], config.codec_think_eos_id]
    ], dtype=torch.long)
    codec_think_embeds = prefill_wrapper.codec_embedding(codec_think_ids)
    hidden_states[:, 3:7, :] = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds

    hidden_states[:, 7:8, :] = tts_pad_embed + speaker_embed.unsqueeze(1)

    codec_pad_embed = prefill_wrapper.codec_embedding(torch.tensor([[config.codec_pad_id]], dtype=torch.long))
    hidden_states[:, 8:9, :] = tts_bos_embed + codec_pad_embed

    all_text_embed = prefill_wrapper.text_projection(prefill_wrapper.text_embedding(text_ids))
    codec_pad_for_text = prefill_wrapper.codec_embedding(
        torch.full((batch_size, MAX_TEXT_LENGTH), config.codec_pad_id, dtype=torch.long)
    )
    hidden_states[:, 9:9+MAX_TEXT_LENGTH, :] = all_text_embed + codec_pad_for_text

    eos_embed = tts_eos_embed + codec_pad_embed
    codec_bos_embed = prefill_wrapper.codec_embedding(torch.tensor([[config.codec_bos_id]], dtype=torch.long))
    bos_embed = tts_pad_embed + codec_bos_embed

    eos_idx = (9 + text_length).view(batch_size, 1, 1).expand(-1, -1, config.hidden_size)
    hidden_states.scatter_(1, eos_idx, eos_embed)
    bos_idx = (10 + text_length).view(batch_size, 1, 1).expand(-1, -1, config.hidden_size)
    hidden_states.scatter_(1, bos_idx, bos_embed)

    # Build mask
    pos_1d = torch.arange(FIXED_SEQ_LEN, device=device)
    position_ids = pos_1d.unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    cos, sin = prefill_wrapper.rotary_emb(hidden_states, position_ids)

    causal_mask = torch.triu(torch.ones(FIXED_SEQ_LEN, FIXED_SEQ_LEN), diagonal=1)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    k_pos = torch.arange(FIXED_SEQ_LEN).view(1, 1, 1, FIXED_SEQ_LEN)
    actual_len_expanded = (text_length + 11).view(batch_size, 1, 1, 1)
    padding_mask = torch.where(k_pos >= actual_len_expanded, float("-inf"), 0.0)
    combined_mask = causal_mask + padding_mask
    combined_mask = combined_mask.expand(batch_size, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN)

    # Run layers
    for layer in prefill_wrapper.layers:
        hidden_states, _, _ = prefill_wrapper._run_layer(layer, hidden_states, combined_mask, cos, sin)

    hidden_states = prefill_wrapper.norm(hidden_states)

    # Extract past_hidden at bos position
    bos_position = (text_length + 11 - 1).view(batch_size, 1, 1).expand(-1, -1, config.hidden_size)
    past_hidden = torch.gather(hidden_states, 1, bos_position)

print(f"past_hidden shape: {past_hidden.shape}")
print(f"past_hidden sample: {past_hidden[0, 0, :5].tolist()}")

# Run decode
print("\n=== Running V3 Decode ===")
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
    print(f"Step {i}: token={next_token}, pos={pos-1}, kv_len={kv_cache.shape[3]}")

print(f"\nV3 wrapper: {tokens[:15]}")

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
    )
codes = result[0][0][:, 0].tolist()
print(f"Official: {codes[:15]}")

# Compare
print(f"\n=== Comparison ===")
print(f"Match: {tokens[:20] == codes[:20]}")
for i in range(min(20, len(tokens), len(codes))):
    match = "OK" if tokens[i] == codes[i] else "MISMATCH"
    print(f"  Step {i}: V3={tokens[i]}, Official={codes[i]} - {match}")
