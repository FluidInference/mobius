# Debug by comparing transformer inputs at each decode step
import torch
import numpy as np
import random
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH
import warnings
warnings.filterwarnings('ignore')

# Set seeds
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

# Hook the main transformer model to capture inputs
captured_inputs = []
original_model_forward = talker.model.forward

def hooked_model_forward(*args, **kwargs):
    inputs_embeds = kwargs.get('inputs_embeds')
    position_ids = kwargs.get('position_ids')
    if inputs_embeds is not None:
        captured_inputs.append({
            'inputs_embeds': inputs_embeds.clone().detach() if inputs_embeds is not None else None,
            'position_ids': position_ids.clone().detach() if position_ids is not None else None,
        })
    return original_model_forward(*args, **kwargs)

talker.model.forward = hooked_model_forward

# Run official model
speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024)
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed_np.squeeze(0))],
    'x_vector_only_mode': [True], 'icl_mode': [False], 'ref_code': None,
}
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

print("\nRunning official generate...")
with torch.no_grad():
    result = tts_model.model.generate(
        input_ids=[full_input_ids], languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True, max_new_tokens=5, do_sample=False,
    )
codes = result[0][0][:, 0].tolist()
print(f"Official tokens: {codes}")

official_inputs = captured_inputs.copy()
print(f"Captured {len(official_inputs)} transformer calls")

# Reset and run V3
captured_inputs.clear()
talker.model.forward = original_model_forward  # Restore

# Reset seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Run V3 manually to capture transformer inputs
from convert_lm_decode_v3 import TracableDecodeV3

prefill_wrapper = TracablePrefillV9(talker)
decode_wrapper = TracableDecodeV3(talker)
prefill_wrapper.eval()
decode_wrapper.eval()

text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)
actual_len = text_len + 11

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

suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

# V3 Prefill
with torch.no_grad():
    logits, kv_cache, past_hidden = prefill_wrapper(
        role_ids, text_ids, text_length,
        tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed
    )

kv_cache = kv_cache[:, :, :, :actual_len, :]

logits_np = logits.numpy().copy()
logits_np[0, suppress_mask] = -float('inf')
first_token = int(np.argmax(logits_np))

v3_tokens = [first_token]
v3_transformer_inputs = []  # Inputs to the transformer in V3 decode
pos = actual_len

for i in range(5):
    token_id = torch.tensor([[v3_tokens[-1]]], dtype=torch.long)
    position = torch.tensor([pos], dtype=torch.long)

    with torch.no_grad():
        # Manually compute inputs_embeds like V3 does
        last_id_hidden = talker.model.codec_embedding(token_id)
        predictor_input = torch.cat([past_hidden, last_id_hidden], dim=1)
        predictor_result = talker.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=config.num_code_groups - 1,
            do_sample=False,
            return_dict_in_generate=True,
        )
        codec_hiddens = [last_id_hidden]
        for j in range(config.num_code_groups - 1):
            cb_embed = talker.code_predictor.get_input_embeddings()[j](
                predictor_result.sequences[..., j:j+1]
            )
            codec_hiddens.append(cb_embed)
        codec_hiddens_cat = torch.cat(codec_hiddens, dim=1)
        inputs_embeds = codec_hiddens_cat.sum(dim=1, keepdim=True) + tts_pad_embed

        # Compute position_ids like V3 does
        pos_1d = position.unsqueeze(0).expand(3, -1)
        position_ids = pos_1d.unsqueeze(-1)

        v3_transformer_inputs.append({
            'inputs_embeds': inputs_embeds.clone(),
            'position_ids': position_ids.clone(),
            'step': i,
        })

        # Run full decode
        logits, kv_cache, past_hidden = decode_wrapper(
            token_id, past_hidden, tts_pad_embed, kv_cache, position
        )

    logits_np = logits.numpy().copy()
    logits_np[0, suppress_mask] = -float('inf')
    next_token = int(np.argmax(logits_np))

    if next_token == config.codec_eos_token_id:
        break

    v3_tokens.append(next_token)
    pos += 1

print(f"\nV3 tokens: {v3_tokens}")

# Compare
print(f"\n=== Token Comparison ===")
print(f"Official: {codes}")
print(f"V3:       {v3_tokens}")

print(f"\n=== Comparing Transformer Inputs ===")
print(f"Official captured {len(official_inputs)} calls")
print(f"V3 captured {len(v3_transformer_inputs)} decode calls")

# The first official call is prefill, then decodes
# Let's compare decode calls
for i in range(min(3, len(v3_transformer_inputs))):
    off_idx = i + 1  # Skip prefill for official
    if off_idx < len(official_inputs):
        off = official_inputs[off_idx]
        v3 = v3_transformer_inputs[i]

        print(f"\n--- Decode step {i} ---")

        # Compare inputs_embeds
        off_ie = off['inputs_embeds']
        v3_ie = v3['inputs_embeds']
        ie_diff = (off_ie - v3_ie).abs().max().item()
        print(f"inputs_embeds:")
        print(f"  Official: shape={off_ie.shape}, mean={off_ie.mean().item():.6f}, std={off_ie.std().item():.6f}")
        print(f"  V3:       shape={v3_ie.shape}, mean={v3_ie.mean().item():.6f}, std={v3_ie.std().item():.6f}")
        print(f"  Max diff: {ie_diff:.6f}")

        # Compare position_ids
        off_pos = off['position_ids']
        v3_pos = v3['position_ids']
        print(f"position_ids:")
        print(f"  Official: {off_pos.squeeze().tolist() if off_pos is not None else 'None'}")
        print(f"  V3:       {v3_pos.squeeze().tolist()}")
        if off_pos is not None:
            pos_diff = (off_pos - v3_pos).abs().max().item()
            print(f"  Max diff: {pos_diff}")
