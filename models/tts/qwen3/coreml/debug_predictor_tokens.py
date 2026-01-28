# Capture official code_predictor tokens by hooking code_predictor.generate
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

# Capture code_predictor.generate calls
official_predictor_data = []

original_predictor_generate = talker.code_predictor.generate

def capture_predictor_generate(*args, **kwargs):
    inputs_embeds = kwargs.get('inputs_embeds')
    if inputs_embeds is not None:
        official_predictor_data.append({
            'inputs_embeds': inputs_embeds.clone().detach(),
        })
    result = original_predictor_generate(*args, **kwargs)
    if len(official_predictor_data) > 0:
        official_predictor_data[-1]['sequences'] = result.sequences.clone().detach()
    return result

talker.code_predictor.generate = capture_predictor_generate

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

talker.code_predictor.generate = original_predictor_generate

print(f"\nCaptured {len(official_predictor_data)} code_predictor.generate calls")

# Run V3 prefill
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

v9_logits_np = v9_logits.numpy().copy()
v9_logits_np[0, suppress_mask] = -float('inf')
first_token = int(np.argmax(v9_logits_np))

print(f"\nV3 first token: {first_token}")

# V3 decode step 0 - compute predictor input
with torch.no_grad():
    token_id = torch.tensor([[first_token]], dtype=torch.long)
    v3_last_id_hidden = talker.model.codec_embedding(token_id)
    v3_predictor_input = torch.cat([v9_past_hidden, v3_last_id_hidden], dim=1)

    v3_result = talker.code_predictor.generate(
        inputs_embeds=v3_predictor_input,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=False,
        return_dict_in_generate=True,
    )
    v3_predictor_tokens = v3_result.sequences[0].tolist()

print(f"V3 predictor tokens: {v3_predictor_tokens}")

# Compare
if len(official_predictor_data) > 0:
    off = official_predictor_data[0]
    print("\n=== Code Predictor Input Comparison (Decode Step 0) ===")

    off_ie = off['inputs_embeds']
    print(f"Official input shape: {off_ie.shape}")
    print(f"V3 input shape: {v3_predictor_input.shape}")

    print(f"\nOfficial input: mean={off_ie.mean().item():.6f}, std={off_ie.std().item():.6f}")
    print(f"V3 input:       mean={v3_predictor_input.mean().item():.6f}, std={v3_predictor_input.std().item():.6f}")

    input_diff = (off_ie - v3_predictor_input).abs().max().item()
    print(f"Input max diff: {input_diff:.6f}")

    # Split into past_hidden and last_id_hidden
    off_past_hidden = off_ie[:, 0:1, :]
    off_last_id_hidden = off_ie[:, 1:2, :]

    v3_ph = v3_predictor_input[:, 0:1, :]
    v3_lih = v3_predictor_input[:, 1:2, :]

    print(f"\npast_hidden diff: {(off_past_hidden - v3_ph).abs().max().item():.6f}")
    print(f"last_id_hidden diff: {(off_last_id_hidden - v3_lih).abs().max().item():.6f}")

    if 'sequences' in off:
        off_tokens = off['sequences'][0].tolist()
        print(f"\nOfficial predictor tokens: {off_tokens}")
        print(f"V3 predictor tokens:       {v3_predictor_tokens}")
        print(f"Match: {off_tokens == v3_predictor_tokens}")
