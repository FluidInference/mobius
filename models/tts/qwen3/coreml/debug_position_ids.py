# Debug position_ids between V9 and direct model call
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)

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

# Build inputs_embeds (same as V9)
with torch.no_grad():
    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
    text_ids_tensor = torch.tensor([text_ids_list], dtype=torch.long)

    role_embed = talker.text_projection(talker.model.text_embedding(role_ids))
    text_embed = talker.text_projection(talker.model.text_embedding(text_ids_tensor))

    codec_pad_id = config.codec_pad_id
    codec_bos_id = config.codec_bos_id
    codec_pad_embed = talker.model.codec_embedding(torch.tensor([[codec_pad_id]]))
    codec_bos_embed = talker.model.codec_embedding(torch.tensor([[codec_bos_id]]))
    speaker_codec_embed = speaker_embed.view(1, 1, -1)

    text_part_embed = []
    for i in range(text_len):
        text_part_embed.append(text_embed[:, i:i+1, :] + codec_pad_embed)
    text_part_embed.append(tts_eos_embed + codec_pad_embed)
    text_part = torch.cat(text_part_embed, dim=1)

    inputs_embeds = torch.cat([
        role_embed,
        tts_pad_embed.expand(1, 4, -1) + codec_pad_embed.expand(1, 4, -1),
        tts_pad_embed + speaker_codec_embed,
        tts_bos_embed + codec_bos_embed,
        text_part,
        tts_pad_embed + codec_bos_embed,
    ], dim=1)

print(f"inputs_embeds shape: {inputs_embeds.shape}")

# What position_ids does talker.model use when called with attention_mask?
print("\n=== Direct call with attention_mask ===")

# Capture position_ids from the direct call
captured_pos_ids = []
original_forward = talker.model.forward

def capture_forward(*args, **kwargs):
    pos_ids = kwargs.get('position_ids')
    if pos_ids is not None:
        captured_pos_ids.append(pos_ids.clone())
    return original_forward(*args, **kwargs)

talker.model.forward = capture_forward

with torch.no_grad():
    attention_mask = torch.ones(1, actual_len, dtype=torch.long)
    outputs = talker.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
    )

talker.model.forward = original_forward

if captured_pos_ids:
    direct_pos_ids = captured_pos_ids[0]
    print(f"Direct position_ids shape: {direct_pos_ids.shape}")
    print(f"Direct position_ids:\n{direct_pos_ids}")
else:
    print("No position_ids captured (computed internally)")

    # The model computes position_ids from attention_mask
    # Let's see how it does it
    # In Qwen2Model.forward:
    # position_ids = attention_mask.long().cumsum(-1) - 1
    # position_ids.masked_fill_(attention_mask == 0, 1)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids = position_ids.masked_fill(attention_mask == 0, 1)
    # Then expands to [3, B, seq_len] for 3D rotary
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    print(f"Computed position_ids shape: {position_ids.shape}")
    print(f"Computed position_ids:\n{position_ids}")

# What position_ids does V9 use?
print("\n=== V9 position_ids ===")

# V9 uses simple 0,1,2,3... position_ids
v9_position_ids = torch.arange(actual_len).unsqueeze(0).expand(3, 1, -1)
print(f"V9 position_ids shape: {v9_position_ids.shape}")
print(f"V9 position_ids:\n{v9_position_ids}")

# Now let's run the model with V9-style position_ids
print("\n=== Direct call with V9-style position_ids ===")
with torch.no_grad():
    v9_pos_3d = torch.arange(actual_len).view(1, -1).expand(1, -1)
    v9_pos_3d = v9_pos_3d.unsqueeze(0).expand(3, -1, -1)

    outputs_v9pos = talker.model(
        inputs_embeds=inputs_embeds,
        position_ids=v9_pos_3d,
        use_cache=True,
    )

    direct_with_v9pos_hidden = outputs_v9pos.last_hidden_state[:, -1:, :]

# Compare to V9
prefill_wrapper = TracablePrefillV9(talker)
prefill_wrapper.eval()

role_ids_input = torch.tensor([ROLE_PREFIX], dtype=torch.long)
text_ids_input = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
text_ids_input[0, :text_len] = torch.tensor(text_ids_list)
text_length = torch.tensor([text_len], dtype=torch.long)

with torch.no_grad():
    v9_logits, v9_kv_cache, v9_past_hidden = prefill_wrapper(
        role_ids_input, text_ids_input, text_length,
        tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed
    )

print(f"Direct with V9 pos_ids - last hidden: mean={direct_with_v9pos_hidden.mean().item():.6f}, std={direct_with_v9pos_hidden.std().item():.6f}")
print(f"V9 past_hidden: mean={v9_past_hidden.mean().item():.6f}, std={v9_past_hidden.std().item():.6f}")

diff = (direct_with_v9pos_hidden - v9_past_hidden).abs().max().item()
print(f"Max diff: {diff:.6f}")

# Also check the inputs_embeds that V9 creates
print("\n=== Comparing inputs_embeds ===")
# Run V9 just to check what it builds

# The issue might be in how V9 builds inputs_embeds differently
# Let me check by extracting the inputs_embeds from V9
