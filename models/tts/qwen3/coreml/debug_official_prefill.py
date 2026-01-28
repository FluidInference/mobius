# Trace official model's prefill sequence structure
import torch
import numpy as np
import random
from qwen_tts import Qwen3TTSModel
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

print(f"Text: '{text}'")
print(f"Text token IDs: {text_ids_list}")
print(f"Text length: {text_len}")

# Capture prefill inputs_embeds from official model
captured_prefill_embeds = []

original_model_forward = talker.model.forward

def capture_forward(*args, **kwargs):
    inputs_embeds = kwargs.get('inputs_embeds')
    if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        captured_prefill_embeds.append({
            'inputs_embeds': inputs_embeds.clone().detach(),
            'position_ids': kwargs.get('position_ids').clone().detach() if kwargs.get('position_ids') is not None else None,
        })
    return original_model_forward(*args, **kwargs)

talker.model.forward = capture_forward

# Run official
speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024)
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed_np.squeeze(0))],
    'x_vector_only_mode': [True], 'icl_mode': [False], 'ref_code': None,
}
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

print(f"\nOfficial full_input_ids: {full_input_ids}")
print(f"Official input length: {len(full_input_ids)}")

with torch.no_grad():
    result = tts_model.model.generate(
        input_ids=[full_input_ids], languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True, max_new_tokens=1, do_sample=False,
    )

talker.model.forward = original_model_forward

print(f"\nCaptured {len(captured_prefill_embeds)} prefill calls")
if len(captured_prefill_embeds) > 0:
    prefill = captured_prefill_embeds[0]
    official_ie = prefill['inputs_embeds']
    print(f"Official prefill inputs_embeds shape: {official_ie.shape}")
    print(f"Official prefill sequence length: {official_ie.shape[1]}")

    if prefill['position_ids'] is not None:
        print(f"Official prefill position_ids shape: {prefill['position_ids'].shape}")
        # Print position_ids
        pos_ids = prefill['position_ids']
        print(f"Official prefill position_ids row 0: {pos_ids[0, 0, :].tolist()}")

# Now compare with V9
print("\n=== V9 Prefill ===")
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH

with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

speaker_embed = torch.from_numpy(np.load("speaker_embedding_official.npy").reshape(1, 1024)).float()

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

v9_actual_len = text_len + 11
print(f"V9 actual sequence length: {v9_actual_len}")
print(f"V9 past_hidden: mean={v9_past_hidden.mean().item():.6f}, std={v9_past_hidden.std().item():.6f}")

# Compare sequence lengths
if len(captured_prefill_embeds) > 0:
    official_len = official_ie.shape[1]
    print(f"\n=== Sequence Length Comparison ===")
    print(f"Official: {official_len}")
    print(f"V9: {v9_actual_len}")

    if official_len != v9_actual_len:
        print("*** MISMATCH: Different sequence lengths! ***")
        print("This could explain why past_hidden differs!")

    # Also compare the last position's embedding
    official_last_hidden = None
    # We'd need to run through the model to get hidden states
    # For now, let's just note the sequence length difference

# Let me also check what the official sequence structure looks like
print("\n=== Official Sequence Structure Analysis ===")
# The official code builds:
# talker_input_embed = torch.cat([
#     _talker_input_embed_role,  # 3 tokens: <|im_start|>assistant\n
#     _talker_input_embed,       # codec + tts stuff
#     text embeddings,
# ], dim=1)

# In non_streaming_mode, it appends ALL text tokens + tts_eos
# Then adds tts_pad + codec_bos at the end

# Let me trace through the exact structure by looking at input_ids
print(f"Official full_input_ids: {full_input_ids}")
# Decode to see what tokens these are
tokens_decoded = tokenizer.decode(full_input_ids.tolist())
print(f"Decoded: '{tokens_decoded}'")

# The structure should be:
# <|im_start|>assistant\n + TEXT + <|im_end|>\n<|im_start|>assistant\n
print("\nExpected structure in official non_streaming_mode:")
print("  [role: <|im_start|>assistant\\n (3 tokens)]")
print("  [codec: think, think_bos, lang, think_eos, speaker, pad (6 tokens)]")
print("  [bos: bos + ? (1 token)]")
print("  [text: all text tokens + eos]")
print("  [final: pad + bos_codec (1 token)]")
