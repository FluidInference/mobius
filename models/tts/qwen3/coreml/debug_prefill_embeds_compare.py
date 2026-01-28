# Compare official vs V9 prefill inputs_embeds position by position
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

print(f"Text length: {text_len}")

# Capture official prefill inputs_embeds
official_prefill_embeds = None
official_prefill_hidden = None

original_model_forward = talker.model.forward

def capture_forward(*args, **kwargs):
    global official_prefill_embeds, official_prefill_hidden
    inputs_embeds = kwargs.get('inputs_embeds')
    if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        official_prefill_embeds = inputs_embeds.clone().detach()
    result = original_model_forward(*args, **kwargs)
    if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        official_prefill_hidden = result.last_hidden_state[:, -1:, :].clone().detach()
    return result

talker.model.forward = capture_forward

# Run official
speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024)
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed_np.squeeze(0))],
    'x_vector_only_mode': [True], 'icl_mode': [False], 'ref_code': None,
}
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

print("Running official generate...")
with torch.no_grad():
    result = tts_model.model.generate(
        input_ids=[full_input_ids], languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True, max_new_tokens=3, do_sample=False,
    )
codes = result[0][0][:, 0].tolist()
print(f"Official tokens: {codes}")

talker.model.forward = original_model_forward

print(f"\nOfficial prefill inputs_embeds shape: {official_prefill_embeds.shape}")
print(f"Official prefill last_hidden: mean={official_prefill_hidden.mean().item():.6f}, std={official_prefill_hidden.std().item():.6f}")

# Build V9 inputs_embeds manually
print("\n=== Building V9 prefill inputs_embeds ===")

with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

speaker_embed = torch.from_numpy(np.load("speaker_embedding_official.npy").reshape(1, 1024)).float()

# Build V9 prefill inputs_embeds (same logic as TracablePrefillV9)
actual_len = text_len + 11  # 19

with torch.no_grad():
    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
    text_ids_tensor = torch.tensor([text_ids_list], dtype=torch.long)

    role_embed = talker.text_projection(talker.model.text_embedding(role_ids))
    text_embed = talker.text_projection(talker.model.text_embedding(text_ids_tensor))

    codec_think_ids = torch.tensor([[config.codec_think_id, config.codec_think_bos_id,
                                     config.codec_language_id["english"], config.codec_think_eos_id]])
    codec_think_embeds = talker.model.codec_embedding(codec_think_ids)
    codec_pad_embed = talker.model.codec_embedding(torch.tensor([[config.codec_pad_id]]))
    codec_bos_embed = talker.model.codec_embedding(torch.tensor([[config.codec_bos_id]]))

    v9_inputs_embeds = torch.zeros(1, actual_len, 1024)
    v9_inputs_embeds[:, 0:3, :] = role_embed
    v9_inputs_embeds[:, 3:7, :] = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds
    v9_inputs_embeds[:, 7:8, :] = tts_pad_embed + speaker_embed.unsqueeze(1)
    v9_inputs_embeds[:, 8:9, :] = tts_bos_embed + codec_pad_embed
    for i in range(text_len):
        v9_inputs_embeds[:, 9+i:10+i, :] = text_embed[:, i:i+1, :] + codec_pad_embed
    v9_inputs_embeds[:, 9+text_len:10+text_len, :] = tts_eos_embed + codec_pad_embed
    v9_inputs_embeds[:, 10+text_len:11+text_len, :] = tts_pad_embed + codec_bos_embed

print(f"V9 inputs_embeds shape: {v9_inputs_embeds.shape}")

# Compare position by position
print("\n=== Position-by-position comparison ===")
for pos in range(actual_len):
    off_pos = official_prefill_embeds[0, pos, :]
    v9_pos = v9_inputs_embeds[0, pos, :]
    diff = (off_pos - v9_pos).abs().max().item()
    if diff > 0.001:
        print(f"Position {pos}: max_diff = {diff:.6f}")
    else:
        print(f"Position {pos}: MATCH")

# Find where they diverge
print("\n=== Detailed diff for diverging positions ===")
for pos in range(actual_len):
    off_pos = official_prefill_embeds[0, pos, :]
    v9_pos = v9_inputs_embeds[0, pos, :]
    diff = (off_pos - v9_pos).abs().max().item()
    if diff > 0.001:
        print(f"\nPosition {pos}:")
        print(f"  Official: mean={off_pos.mean().item():.6f}, std={off_pos.std().item():.6f}")
        print(f"  V9:       mean={v9_pos.mean().item():.6f}, std={v9_pos.std().item():.6f}")
        print(f"  Official[:5]: {off_pos[:5].tolist()}")
        print(f"  V9[:5]:       {v9_pos[:5].tolist()}")

# Run V9 prefill to get its past_hidden
print("\n=== Running V9 Prefill ===")
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

print(f"V9 past_hidden: mean={v9_past_hidden.mean().item():.6f}, std={v9_past_hidden.std().item():.6f}")

# Compare past_hidden
past_hidden_diff = (official_prefill_hidden - v9_past_hidden).abs().max().item()
print(f"\nOfficial past_hidden vs V9 past_hidden max_diff: {past_hidden_diff:.6f}")
