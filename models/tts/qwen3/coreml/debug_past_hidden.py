# Debug past_hidden from prefill - compare official vs V3
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

# Get V3 past_hidden from prefill
prefill_wrapper = TracablePrefillV9(talker)
prefill_wrapper.eval()

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

# V3 Prefill
print("\n=== Running V3 Prefill ===")
with torch.no_grad():
    v3_logits, v3_kv_cache, v3_past_hidden = prefill_wrapper(
        role_ids, text_ids, text_length,
        tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed
    )

v3_kv_cache = v3_kv_cache[:, :, :, :actual_len, :]

suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

v3_logits_np = v3_logits.numpy().copy()
v3_logits_np[0, suppress_mask] = -float('inf')
v3_first_token = int(np.argmax(v3_logits_np))

print(f"V3 first token: {v3_first_token}")
print(f"V3 past_hidden shape: {v3_past_hidden.shape}")
print(f"V3 past_hidden: mean={v3_past_hidden.mean().item():.6f}, std={v3_past_hidden.std().item():.6f}")

# Get official past_hidden by calling talker forward directly with inputs_embeds
print("\n=== Running Official Prefill (manual) ===")

# Build the same inputs_embeds sequence
with torch.no_grad():
    # role tokens
    role_embed = talker.text_projection(talker.model.text_embedding(role_ids))

    # Speaker embedding
    speaker_embed_expanded = speaker_embed.view(1, 1, -1)

    # Text tokens
    text_ids_tensor = torch.tensor([text_ids_list], dtype=torch.long)
    text_embed = talker.text_projection(talker.model.text_embedding(text_ids_tensor))

    # Codec embeddings
    codec_pad_id = config.codec_pad_id
    codec_bos_id = config.codec_bos_id
    codec_prefill_list = [[codec_pad_id] * 5 + [codec_bos_id]]
    codec_embed = talker.model.codec_embedding(torch.tensor(codec_prefill_list))

    # Build sequence following official non_streaming_mode
    # role (3) + [pad*5, speaker, bos] codec (7) + [text + eos] + pad + bos
    # Wait, let me look at the exact official sequence structure

# Actually, let me just build it the same way as V9 and compare

print("\nV3 sequence structure (from prefill_wrapper):")
print(f"  actual_len: {actual_len}")
print(f"  text_len: {text_len}")

# The V9 prefill sequence is:
# Position 0-2: role_embed (3 tokens)
# Position 3: pad + pad_codec_embed
# Position 4: pad + pad_codec_embed
# Position 5: pad + pad_codec_embed
# Position 6: pad + pad_codec_embed
# Position 7: speaker + pad_codec_embed
# Position 8: bos + bos_codec_embed
# Position 9-(9+text_len-1): text[i] + pad_codec_embed
# Position (9+text_len): eos + pad_codec_embed
# Position (10+text_len): pad + bos_codec_embed  <- this is the bos position

# In the official code for non_streaming_mode, it's similar but different...
# Let me trace through what the official generates

# For now, let's compare the code_predictor outputs given the same input token
print("\n=== Comparing code_predictor with same inputs ===")

# Both use first_token = 1995
first_token = v3_first_token
token_id = torch.tensor([[first_token]], dtype=torch.long)

# Compute code_predictor result with V3 past_hidden
with torch.no_grad():
    last_id_hidden = talker.model.codec_embedding(token_id)
    v3_predictor_input = torch.cat([v3_past_hidden, last_id_hidden], dim=1)
    v3_result = talker.code_predictor.generate(
        inputs_embeds=v3_predictor_input,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=False,
        return_dict_in_generate=True,
    )
    v3_predictor_tokens = v3_result.sequences[0].tolist()

print(f"V3 code_predictor tokens: {v3_predictor_tokens}")

# Now let's see what past_hidden the official model would have
# The official model uses hidden_states[:, -1:, :] as past_hidden
# In non_streaming_mode, after prefill, the last position is where we get the first logits

# Check if V3's past_hidden is from the correct position
# V3 uses `last_hidden = torch.gather(hidden_states, 1, bos_position)`
# where bos_position = actual_len - 1

# The key question: is V3's past_hidden the same as what official would have?

# Let me run the official model's talker forward manually with the same inputs_embeds
# as V9 to see what hidden_states[:, -1:] would be

print("\n=== Testing if past_hidden values matter ===")

# Let's generate a random past_hidden and see what code_predictor produces
random_past_hidden = torch.randn_like(v3_past_hidden)
with torch.no_grad():
    random_predictor_input = torch.cat([random_past_hidden, last_id_hidden], dim=1)
    random_result = talker.code_predictor.generate(
        inputs_embeds=random_predictor_input,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=False,
        return_dict_in_generate=True,
    )
    random_predictor_tokens = random_result.sequences[0].tolist()

print(f"Random past_hidden -> code_predictor tokens: {random_predictor_tokens}")

# Try with past_hidden scaled
scaled_past_hidden = v3_past_hidden * 1.1
with torch.no_grad():
    scaled_predictor_input = torch.cat([scaled_past_hidden, last_id_hidden], dim=1)
    scaled_result = talker.code_predictor.generate(
        inputs_embeds=scaled_predictor_input,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=False,
        return_dict_in_generate=True,
    )
    scaled_predictor_tokens = scaled_result.sequences[0].tolist()

print(f"Scaled (1.1x) past_hidden -> code_predictor tokens: {scaled_predictor_tokens}")

# The code_predictor tokens differ significantly based on past_hidden!
# This confirms that different past_hidden values cause different code_predictor outputs

# Now let's check the trailing_text difference
print("\n=== Checking trailing_text_hidden ===")

# In official non_streaming_mode, trailing_text_hidden = tts_pad_embed (shape [1,1,1024])
# In V3, we also use tts_pad_embed

# But wait - let me check generation_step in official
# if generation_step < trailing_text_hidden.shape[1]:
#     inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step]
# else:
#     inputs_embeds = inputs_embeds + tts_pad_embed

# In non_streaming_mode, trailing_text_hidden.shape[1] = 1 (just tts_pad_embed)
# So generation_step 0: 0 < 1, use trailing_text_hidden[:, 0] = tts_pad_embed
# generation_step 1+: >=1, use tts_pad_embed
# Both paths give tts_pad_embed, so V3 should be correct

print(f"tts_pad_embed shape: {tts_pad_embed.shape}")
print(f"tts_pad_embed mean: {tts_pad_embed.mean().item():.6f}")

# The issue must be in past_hidden!
# Let me check if V9's past_hidden is correct by comparing to official prefill

print("\n=== Running official generate with just prefill (max_new_tokens=1) ===")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Capture the hidden_states from the first forward call
captured_hidden = []
original_talker_model_forward = talker.model.forward

def capture_hidden_forward(*args, **kwargs):
    result = original_talker_model_forward(*args, **kwargs)
    if result.last_hidden_state is not None:
        captured_hidden.append(result.last_hidden_state.clone())
    return result

talker.model.forward = capture_hidden_forward

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
        non_streaming_mode=True, max_new_tokens=1, do_sample=False,
    )

talker.model.forward = original_talker_model_forward

print(f"Captured {len(captured_hidden)} hidden states")
if len(captured_hidden) > 0:
    # First one is prefill, get the last position
    prefill_hidden = captured_hidden[0]
    official_past_hidden = prefill_hidden[:, -1:, :]
    print(f"Official prefill hidden shape: {prefill_hidden.shape}")
    print(f"Official past_hidden (from -1): mean={official_past_hidden.mean().item():.6f}, std={official_past_hidden.std().item():.6f}")
    print(f"V3 past_hidden:                 mean={v3_past_hidden.mean().item():.6f}, std={v3_past_hidden.std().item():.6f}")

    diff = (official_past_hidden - v3_past_hidden).abs().max().item()
    print(f"\nMax diff between official and V3 past_hidden: {diff:.6f}")

    if diff > 0.01:
        print("\n*** FOUND THE ISSUE: past_hidden values differ! ***")
        print("V3 prefill returns a different hidden state than official!")

        # Compare a few values
        print(f"\nOfficial past_hidden[:5]: {official_past_hidden[0, 0, :5].tolist()}")
        print(f"V3 past_hidden[:5]:       {v3_past_hidden[0, 0, :5].tolist()}")
