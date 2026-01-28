# Debug: Compare hidden states from V9 prefill vs direct model call
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

# Prepare all components
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

# Build V9-style inputs_embeds manually (same logic as TracablePrefillV9)
with torch.no_grad():
    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
    text_ids_tensor = torch.tensor([text_ids_list], dtype=torch.long)

    # Role embedding
    role_embed = talker.text_projection(talker.model.text_embedding(role_ids))

    # Text embedding
    text_embed = talker.text_projection(talker.model.text_embedding(text_ids_tensor))

    # Codec embeddings
    codec_pad_id = config.codec_pad_id
    codec_bos_id = config.codec_bos_id
    codec_pad_embed = talker.model.codec_embedding(torch.tensor([[codec_pad_id]]))
    codec_bos_embed = talker.model.codec_embedding(torch.tensor([[codec_bos_id]]))
    speaker_codec_embed = speaker_embed.view(1, 1, -1)

    # Build sequence (same as V9):
    # [role (3)] + [pad*4 + speaker + bos] codec part + [text + eos] + [pad + bos]
    # Position breakdown:
    # 0-2: role
    # 3-6: pad + pad_codec (4 positions)
    # 7: speaker + pad_codec
    # 8: bos + bos_codec
    # 9 to 9+text_len-1: text[i] + pad_codec
    # 9+text_len: eos + pad_codec
    # 10+text_len: pad + bos_codec  <- this is the last position (actual_len - 1)

    # Text part embeddings
    text_part_embed = []
    for i in range(text_len):
        text_part_embed.append(text_embed[:, i:i+1, :] + codec_pad_embed)
    text_part_embed.append(tts_eos_embed + codec_pad_embed)  # EOS
    text_part = torch.cat(text_part_embed, dim=1) if text_part_embed else torch.zeros(1, 0, 1024)

    # Build full sequence
    inputs_embeds = torch.cat([
        role_embed,                                      # 0-2: role (3)
        tts_pad_embed.expand(1, 4, -1) + codec_pad_embed.expand(1, 4, -1),  # 3-6: pad*4 + pad_codec
        tts_pad_embed + speaker_codec_embed,             # 7: pad + speaker
        tts_bos_embed + codec_bos_embed,                 # 8: bos + bos_codec
        text_part,                                       # 9 to 9+text_len: text + pad_codec, eos + pad_codec
        tts_pad_embed + codec_bos_embed,                 # last: pad + bos_codec
    ], dim=1)

print(f"Built inputs_embeds shape: {inputs_embeds.shape}")
print(f"Expected actual_len: {actual_len}")
assert inputs_embeds.shape[1] == actual_len, f"Mismatch: {inputs_embeds.shape[1]} vs {actual_len}"

# Run through talker.model (main transformer) directly
print("\n=== Running talker.model directly ===")
with torch.no_grad():
    # Create attention mask (all ones, no padding)
    attention_mask = torch.ones(1, actual_len, dtype=torch.long)

    # Run forward
    outputs = talker.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
    )

    direct_hidden = outputs.last_hidden_state
    direct_last_hidden = direct_hidden[:, -1:, :]

print(f"Direct hidden shape: {direct_hidden.shape}")
print(f"Direct last hidden: mean={direct_last_hidden.mean().item():.6f}, std={direct_last_hidden.std().item():.6f}")

# Run V9 prefill
print("\n=== Running V9 Prefill ===")
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

print(f"V9 past_hidden: mean={v9_past_hidden.mean().item():.6f}, std={v9_past_hidden.std().item():.6f}")

# Compare
print("\n=== Comparison ===")
diff = (direct_last_hidden - v9_past_hidden).abs().max().item()
print(f"Max diff between direct last hidden and V9 past_hidden: {diff:.6f}")

if diff < 0.001:
    print("V9 past_hidden MATCHES direct model output!")
else:
    print("V9 past_hidden DIFFERS from direct model output!")

# Also compare logits
direct_logits = talker.codec_head(direct_hidden[:, -1:, :]).squeeze(1)
print(f"\nDirect logits top-5: {torch.topk(direct_logits, 5).indices[0].tolist()}")

suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

v9_logits_np = v9_logits.detach().numpy().copy()
v9_logits_np[0, suppress_mask] = -float('inf')
direct_logits_np = direct_logits.detach().numpy().copy()
direct_logits_np[0, suppress_mask] = -float('inf')

print(f"V9 first token (masked): {np.argmax(v9_logits_np)}")
print(f"Direct first token (masked): {np.argmax(direct_logits_np)}")

# Now the key question: What inputs_embeds does official use for decode step 0?
# In official non_streaming_mode, after prefill, decode uses:
# - past_hidden from prefill (hidden_states[:, -1:])
# - trailing_text_hidden = tts_pad_embed

# Since direct last_hidden matches V9 past_hidden, the issue must be elsewhere.
# Let's check what inputs_embeds the official decode actually uses

print("\n=== Testing code_predictor with V9 past_hidden ===")
first_token = int(np.argmax(v9_logits_np))
print(f"First token: {first_token}")

with torch.no_grad():
    token_id = torch.tensor([[first_token]], dtype=torch.long)
    last_id_hidden = talker.model.codec_embedding(token_id)

    predictor_input = torch.cat([v9_past_hidden, last_id_hidden], dim=1)
    result = talker.code_predictor.generate(
        inputs_embeds=predictor_input,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=False,
        return_dict_in_generate=True,
    )
    v9_predictor_tokens = result.sequences[0].tolist()

print(f"V9 code_predictor tokens: {v9_predictor_tokens}")

# Now test with direct_last_hidden
with torch.no_grad():
    predictor_input_direct = torch.cat([direct_last_hidden, last_id_hidden], dim=1)
    result_direct = talker.code_predictor.generate(
        inputs_embeds=predictor_input_direct,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=False,
        return_dict_in_generate=True,
    )
    direct_predictor_tokens = result_direct.sequences[0].tolist()

print(f"Direct code_predictor tokens: {direct_predictor_tokens}")

# Compute final inputs_embeds
with torch.no_grad():
    codec_hiddens = [last_id_hidden]
    for i in range(config.num_code_groups - 1):
        cb_embed = talker.code_predictor.get_input_embeddings()[i](
            result.sequences[..., i:i+1]
        )
        codec_hiddens.append(cb_embed)
    codec_hiddens_cat = torch.cat(codec_hiddens, dim=1)
    codec_sum = codec_hiddens_cat.sum(dim=1, keepdim=True)
    final_inputs_embeds = codec_sum + tts_pad_embed

print(f"\nFinal inputs_embeds for decode step 0:")
print(f"  codec_sum: mean={codec_sum.mean().item():.6f}, std={codec_sum.std().item():.6f}")
print(f"  tts_pad_embed: mean={tts_pad_embed.mean().item():.6f}")
print(f"  final: mean={final_inputs_embeds.mean().item():.6f}, std={final_inputs_embeds.std().item():.6f}")
