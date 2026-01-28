# Debug code_predictor differences between V3 and official
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

# Get V9 prefill results
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

print(f"First token: {first_token}")
print(f"V9 past_hidden shape: {v9_past_hidden.shape}")
print(f"V9 past_hidden: mean={v9_past_hidden.mean().item():.6f}, std={v9_past_hidden.std().item():.6f}")

# Now test code_predictor with V9's past_hidden
print("\n=== Code predictor test ===")

with torch.no_grad():
    token_id = torch.tensor([[first_token]], dtype=torch.long)
    last_id_hidden = talker.model.codec_embedding(token_id)

    # V3-style: use talker.model.codec_embedding
    predictor_input = torch.cat([v9_past_hidden, last_id_hidden], dim=1)

    result = talker.code_predictor.generate(
        inputs_embeds=predictor_input,
        max_new_tokens=config.num_code_groups - 1,  # 15
        do_sample=False,
        return_dict_in_generate=True,
    )
    v3_predictor_tokens = result.sequences[0].tolist()

print(f"V3-style code_predictor tokens: {v3_predictor_tokens}")

# What does official use? It uses talker.get_input_embeddings() not talker.model.codec_embedding
# Let me check if they're the same
print("\n=== Checking codec embedding sources ===")
print(f"talker.model.codec_embedding: {talker.model.codec_embedding}")
print(f"talker.get_input_embeddings(): {talker.get_input_embeddings()}")

# Are they the same?
with torch.no_grad():
    v3_embed = talker.model.codec_embedding(token_id)
    official_embed = talker.get_input_embeddings()(token_id)

    embed_diff = (v3_embed - official_embed).abs().max().item()
    print(f"Embedding difference: {embed_diff}")

# If they're different, that could be the issue!

# Let me also check the code_predictor.get_input_embeddings
print("\n=== Code predictor input embeddings ===")
predictor_embeds = talker.code_predictor.get_input_embeddings()
print(f"Type: {type(predictor_embeds)}")
print(f"Num embeddings: {len(predictor_embeds) if hasattr(predictor_embeds, '__len__') else 'N/A'}")

# The code predictor has multiple embedding layers for different codebooks
# Let's check if V3 accesses them correctly

# Official code:
# codec_hiddens = torch.cat(
#     [last_id_hidden]
#     + [self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i:i+1])
#        for i in range(self.config.num_code_groups - 1)],
#     dim=1,
# )

# V3 code:
# for i in range(self.num_code_groups - 1):
#     cb_embed = self.code_predictor.get_input_embeddings()[i](
#         predictor_result.sequences[..., i:i+1]
#     )

# These look the same. Let me verify by computing the full codec_sum

print("\n=== Computing full codec_sum ===")

with torch.no_grad():
    codec_hiddens = [last_id_hidden]
    for i in range(config.num_code_groups - 1):
        cb_embed = talker.code_predictor.get_input_embeddings()[i](
            result.sequences[..., i:i+1]
        )
        codec_hiddens.append(cb_embed)

    codec_hiddens_cat = torch.cat(codec_hiddens, dim=1)
    codec_sum = codec_hiddens_cat.sum(dim=1, keepdim=True)

    print(f"codec_sum shape: {codec_sum.shape}")
    print(f"codec_sum: mean={codec_sum.mean().item():.6f}, std={codec_sum.std().item():.6f}")

    # Final inputs_embeds
    final_inputs_embeds = codec_sum + tts_pad_embed
    print(f"final_inputs_embeds: mean={final_inputs_embeds.mean().item():.6f}, std={final_inputs_embeds.std().item():.6f}")

# Now let me check what the official model produces
print("\n=== Running official model to capture decode inputs ===")

# We need to capture what official produces for the same past_hidden
# Since we can't easily hook, let's simulate by calling the official forward

# Actually, the issue might be that official uses different kwargs for code_predictor.generate
# Let me check if there are any differences

print("\n=== Code predictor generate kwargs ===")
# Official uses:
# predictor_result = self.code_predictor.generate(
#     inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
#     max_new_tokens=self.config.num_code_groups - 1,
#     do_sample=subtalker_dosample,
#     top_p=subtalker_top_p,
#     top_k=subtalker_top_k,
#     temperature=subtalker_temperature,
#     output_hidden_states=True,
#     return_dict_in_generate=True,
# )

# V3 uses:
# predictor_result = self.code_predictor.generate(
#     inputs_embeds=predictor_input,
#     max_new_tokens=self.num_code_groups - 1,
#     do_sample=False,
#     output_hidden_states=True,
#     return_dict_in_generate=True,
# )

# The difference: official passes subtalker_* parameters
# When do_sample=False (greedy), the other params shouldn't matter
# But let me verify by checking what official uses by default

# Check what kwargs the official model passes
print("Official uses subtalker_* params (but with do_sample=False, they shouldn't matter)")
print("Let me verify by running with different params...")

with torch.no_grad():
    # With explicit do_sample=False (should be same)
    result_explicit = talker.code_predictor.generate(
        inputs_embeds=predictor_input,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=False,
        top_p=None,
        top_k=None,
        temperature=None,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    tokens_explicit = result_explicit.sequences[0].tolist()

print(f"With explicit params: {tokens_explicit}")
print(f"Original:             {v3_predictor_tokens}")
print(f"Match: {tokens_explicit == v3_predictor_tokens}")

# The code_predictor tokens should be deterministic with do_sample=False
