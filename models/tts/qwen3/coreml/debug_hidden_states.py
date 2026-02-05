# Debug hidden states at divergence point
# Compare V9 + V3 decode vs official model internals
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

# Constants
TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]
MAX_TEXT_LENGTH = 128

text = "Hello world, this is a test of the text to speech system."
tokenizer_obj = processor.tokenizer
text_ids_list = tokenizer_obj.encode(text, add_special_tokens=False)
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

# === V9 CoreML Prefill ===
print("\n=== V9 CoreML Prefill ===")
lm_prefill = ct.models.MLModel("qwen3_tts_lm_prefill_v9.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)

role_ids = np.array([ROLE_PREFIX], dtype=np.int32)
text_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
text_ids[0, :text_len] = text_ids_list
text_length = np.array([text_len], dtype=np.int32)

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

actual_len = text_len + 11  # role_prefix(3) + think(3) + speaker_proj(1) + text_len + tts_eos(1) + tts_pad(1) + bos(1) + bos_proj(1) = text_len + 11
v9_kv_cache = v9_kv_cache[:, :, :, :actual_len, :]

print(f"V9 past_hidden shape: {v9_past_hidden.shape}")
print(f"V9 past_hidden stats: mean={v9_past_hidden.mean():.4f}, std={v9_past_hidden.std():.4f}")
print(f"V9 KV cache shape: {v9_kv_cache.shape}")

# Get first token
suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

v9_logits_copy = v9_logits.copy()
v9_logits_copy[0, suppress_mask] = -float('inf')
v9_first_token = int(np.argmax(v9_logits_copy))
print(f"V9 first token: {v9_first_token}")

# === Official Model - Hook into internal generation ===
print("\n=== Official Model Internal Generation ===")

# We need to access the talker's generation to get hidden states
# Let's run a manual forward pass that mimics what official does

# Build input embeds the way official does
assistant_text = model._build_assistant_text(text)
input_ids_official = model._tokenize_texts([assistant_text])[0]
print(f"Official input_ids shape: {input_ids_official.shape}")

# Voice clone prompt setup
voice_clone_prompt = {
    'ref_spk_embedding': [speaker_embed.squeeze(0)],
    'x_vector_only_mode': [True],
    'icl_mode': [False],
    'ref_code': None,
}

# Run official generation with max_new_tokens=3 to capture the first few steps
with torch.no_grad():
    # Access model internals through forward passes
    # First get the text embeddings from the input_ids
    text_embedding = talker.model.text_embedding(input_ids_official)
    text_projected = talker.text_projection(text_embedding)
    print(f"Official text_projected shape: {text_projected.shape}")

    # Build the full input embeds including speaker
    # The official model builds: think_tokens + speaker_proj + text + tts_eos + trailing_text + bos_embed

    # Get think tokens (3 tokens)
    think_ids = torch.tensor([[config.think_token_id] * config.num_think_tokens])
    think_embed = talker.model.codec_embedding(think_ids)

    # Speaker projection
    speaker_proj = talker.speaker_projection(speaker_embed)  # [1, 1024] -> [1, 1, hidden_size]

    # Get language/BOS embeddings
    lang_id = config.codec_language_id["english"]
    bos_id = config.codec_bos_id
    lang_embed = talker.model.codec_embedding(torch.tensor([[lang_id]]))
    bos_embed = talker.model.codec_embedding(torch.tensor([[bos_id]]))

    print(f"Think embed shape: {think_embed.shape}")
    print(f"Speaker proj shape: {speaker_proj.shape}")
    print(f"Text projected shape: {text_projected.shape}")

    # Build prefill input: think + speaker + text_projected + tts_eos_embed + trailing(tts_pad) + bos_embed
    prefill_input = torch.cat([
        think_embed,           # [1, 3, hidden]
        speaker_proj,          # [1, 1, hidden]
        text_projected,        # [1, text_len, hidden]
        tts_eos_embed,         # [1, 1, hidden]
        tts_pad_embed,         # [1, 1, hidden] - trailing text embed
        bos_embed,             # [1, 1, hidden]
    ], dim=1)

    print(f"Official prefill_input shape: {prefill_input.shape}")

    # Run forward pass
    outputs = talker.model(inputs_embeds=prefill_input, use_cache=True, return_dict=True)
    official_hidden = outputs.last_hidden_state[:, -1:, :]
    official_kv = outputs.past_key_values

    print(f"Official hidden shape: {official_hidden.shape}")
    print(f"Official hidden stats: mean={official_hidden.mean().item():.4f}, std={official_hidden.std().item():.4f}")

    # Get first token logits
    official_logits = talker.codec_head(official_hidden)
    official_logits_np = official_logits.squeeze(1).numpy()
    official_logits_np[0, suppress_mask] = -float('inf')
    official_first_token = int(np.argmax(official_logits_np))
    print(f"Official first token: {official_first_token}")

# === Compare V9 and Official hidden states ===
print("\n=== Comparison: V9 vs Official ===")
v9_hidden_torch = torch.from_numpy(v9_past_hidden).float()

# Check if the hidden states match
diff = (v9_hidden_torch - official_hidden).abs()
print(f"Hidden state difference: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

# Compare first few values
print(f"\nV9 hidden (first 10 values): {v9_hidden_torch[0, 0, :10].tolist()}")
print(f"Official hidden (first 10 values): {official_hidden[0, 0, :10].tolist()}")

# Compare logits at the first token
print(f"\nV9 logits (top 5):")
v9_log = v9_logits.flatten()
top5_idx_v9 = np.argsort(v9_log)[-5:][::-1]
for idx in top5_idx_v9:
    print(f"  Token {idx}: {v9_log[idx]:.4f}")

print(f"\nOfficial logits (top 5):")
off_log = official_logits.squeeze().numpy()
top5_idx_off = np.argsort(off_log)[-5:][::-1]
for idx in top5_idx_off:
    print(f"  Token {idx}: {off_log[idx]:.4f}")

# === Check KV cache comparison ===
print("\n=== KV Cache Comparison ===")
# Extract first layer key from official
official_key_0 = official_kv[0][0]  # [batch, num_heads, seq_len, head_dim]
v9_key_0 = torch.from_numpy(v9_kv_cache[0])  # [batch, num_heads, seq_len, head_dim]

print(f"Official key[0] shape: {official_key_0.shape}")
print(f"V9 key[0] shape: {v9_key_0.shape}")

if official_key_0.shape == v9_key_0.shape:
    key_diff = (official_key_0 - v9_key_0).abs()
    print(f"Key[0] difference: max={key_diff.max().item():.6f}, mean={key_diff.mean().item():.6f}")
else:
    print("Shapes don't match - checking seq_len alignment")
    # V9 might have different sequence length due to role_prefix handling
    min_seq = min(official_key_0.shape[2], v9_key_0.shape[2])
    key_diff = (official_key_0[:, :, :min_seq, :] - v9_key_0[:, :, :min_seq, :]).abs()
    print(f"Key[0] difference (first {min_seq} positions): max={key_diff.max().item():.6f}, mean={key_diff.mean().item():.6f}")

# === Now run V3 decode step and compare ===
print("\n=== V3 Decode Step 1 vs Official Step 1 ===")

decode_wrapper = TracableDecodeV3(talker)
decode_wrapper.eval()

# V3 decode step 1
token_id = torch.tensor([[v9_first_token]], dtype=torch.long)
position_tensor = torch.tensor([actual_len], dtype=torch.long)
kv_cache_torch = torch.from_numpy(v9_kv_cache).float()
past_hidden_torch = torch.from_numpy(v9_past_hidden).float()

with torch.no_grad():
    v3_logits, v3_kv, v3_hidden = decode_wrapper(
        token_id, past_hidden_torch, tts_pad_embed, kv_cache_torch, position_tensor
    )

v3_logits_np = v3_logits.numpy().copy()
v3_logits_np[0, suppress_mask] = -float('inf')
v3_next_token = int(np.argmax(v3_logits_np))

print(f"V3 decode token: {v3_next_token}")
print(f"V3 hidden stats: mean={v3_hidden.mean().item():.4f}, std={v3_hidden.std().item():.4f}")

# Official decode step 1
with torch.no_grad():
    # Get codec embedding for first token
    first_token_embed = talker.model.codec_embedding(torch.tensor([[official_first_token]]))

    # For official, we need to also add the trailing text embed (tts_pad)
    # But wait - the official model does something different with code_predictor

    # Let's check what the V3 wrapper does vs what official does
    print("\nV3 inputs_embeds computation:")
    print(f"  token_id: {v9_first_token}")
    print(f"  Using past_hidden + last_id_hidden for code_predictor")

    # Official model's next step after first token
    official_next_input = first_token_embed + tts_pad_embed  # Add trailing text
    outputs_step1 = talker.model(
        inputs_embeds=official_next_input,
        past_key_values=official_kv,
        use_cache=True,
        return_dict=True,
    )
    official_hidden_step1 = outputs_step1.last_hidden_state
    official_logits_step1 = talker.codec_head(official_hidden_step1)

    official_logits_step1_np = official_logits_step1.squeeze().numpy()
    official_logits_step1_np[suppress_mask] = -float('inf')
    official_next_token = int(np.argmax(official_logits_step1_np))

    print(f"\nOfficial decode token (simple): {official_next_token}")

print("\n=== Critical Issue Analysis ===")
print(f"V9 first token: {v9_first_token}")
print(f"Official first token: {official_first_token}")
print(f"V3 step 1 produces: {v3_next_token}")
print(f"Official step 1 produces: {official_next_token}")

if v9_first_token == official_first_token:
    print("\nFirst tokens match - divergence is in decode step")
else:
    print("\nFirst tokens DIFFER - divergence starts at prefill")
