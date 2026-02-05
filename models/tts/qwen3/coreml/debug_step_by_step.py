# Debug step by step - compare V9+V3 decode with official at each step
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

# === Run official model to get reference tokens and internal states ===
print("\n=== Running Official Model with Hook ===")

# We'll hook into the talker forward to capture hidden states
captured_hidden_states = []
captured_inputs_embeds = []

original_forward = talker.model.forward

def hooked_forward(*args, **kwargs):
    result = original_forward(*args, **kwargs)
    if 'inputs_embeds' in kwargs and kwargs['inputs_embeds'] is not None:
        captured_inputs_embeds.append(kwargs['inputs_embeds'].clone())
    if hasattr(result, 'last_hidden_state'):
        captured_hidden_states.append(result.last_hidden_state.clone())
    return result

talker.model.forward = hooked_forward

voice_clone_prompt = {
    'ref_spk_embedding': [speaker_embed.squeeze(0)],
    'x_vector_only_mode': [True],
    'icl_mode': [False],
    'ref_code': None,
}

with torch.no_grad():
    result = model.model.generate(
        input_ids=[model._tokenize_texts([model._build_assistant_text(text)])[0]],
        languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True,
        max_new_tokens=10,
        do_sample=False,
    )

talker.model.forward = original_forward  # Restore

official_codes = result[0][0]
print(f"Official tokens (first 10): {official_codes[:10, 0].tolist()}")

# === V9 Prefill ===
print("\n=== V9 Prefill ===")
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

actual_len = text_len + 11
v9_kv_cache = v9_kv_cache[:, :, :, :actual_len, :]

# Sampling
suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

v9_logits_copy = v9_logits.copy()
v9_logits_copy[0, suppress_mask] = -float('inf')
v9_first_token = int(np.argmax(v9_logits_copy))
print(f"V9 first token: {v9_first_token}")

# === V3 Decode Step by Step ===
print("\n=== V3 Decode - Detailed Trace ===")

decode_wrapper = TracableDecodeV3(talker)
decode_wrapper.eval()

v9_tokens = [v9_first_token]
kv_cache_torch = torch.from_numpy(v9_kv_cache).float()
past_hidden_torch = torch.from_numpy(v9_past_hidden).float()
position = actual_len

# Trace each step in detail
for step in range(5):
    print(f"\n--- Step {step} ---")
    token_id = torch.tensor([[v9_tokens[-1]]], dtype=torch.long)
    position_tensor = torch.tensor([position], dtype=torch.long)

    # Get last_id_hidden (codec embedding of current token)
    last_id_hidden = talker.model.codec_embedding(token_id)
    print(f"last_id_hidden stats: mean={last_id_hidden.mean().item():.6f}, std={last_id_hidden.std().item():.6f}")
    print(f"last_id_hidden first 5: {last_id_hidden[0, 0, :5].tolist()}")

    # Prepare code_predictor input
    predictor_input = torch.cat([past_hidden_torch, last_id_hidden], dim=1)
    print(f"predictor_input shape: {predictor_input.shape}")
    print(f"predictor_input mean: {predictor_input.mean().item():.6f}")

    # Run code_predictor
    with torch.no_grad():
        predictor_result = talker.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=config.num_code_groups - 1,
            do_sample=True,
            temperature=0.9,
            top_p=1.0,
            top_k=50,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    print(f"code_predictor sequences: {predictor_result.sequences.tolist()}")

    # Get embeddings for all codebooks and sum them
    codec_hiddens = [last_id_hidden]
    for i in range(config.num_code_groups - 1):
        cb_embed = talker.code_predictor.get_input_embeddings()[i](
            predictor_result.sequences[..., i:i+1]
        )
        codec_hiddens.append(cb_embed)

    codec_hiddens_cat = torch.cat(codec_hiddens, dim=1)
    inputs_embeds = codec_hiddens_cat.sum(dim=1, keepdim=True)
    print(f"summed codec_hiddens: mean={inputs_embeds.mean().item():.6f}, std={inputs_embeds.std().item():.6f}")

    # Add trailing text embed
    inputs_embeds_with_text = inputs_embeds + tts_pad_embed
    print(f"after adding tts_pad: mean={inputs_embeds_with_text.mean().item():.6f}")

    # Now run through the full V3 decode
    with torch.no_grad():
        logits_torch, kv_cache_torch, past_hidden_torch = decode_wrapper(
            token_id, past_hidden_torch, tts_pad_embed, kv_cache_torch, position_tensor
        )

    logits_np = logits_torch.numpy().copy()
    logits_np[0, suppress_mask] = -float('inf')
    next_token = int(np.argmax(logits_np))

    # Top 5 tokens
    top5_idx = np.argsort(logits_np[0])[-5:][::-1]
    top5_vals = [logits_np[0, idx] for idx in top5_idx]
    print(f"Top 5 tokens: {list(zip(top5_idx.tolist(), [f'{v:.2f}' for v in top5_vals]))}")

    official_expected = official_codes[step + 1, 0].item() if step + 1 < len(official_codes) else "N/A"
    print(f"V3 token: {next_token}, Official expected: {official_expected}")

    if next_token != official_expected:
        print(f"*** DIVERGENCE at step {step}! ***")
        # Check if official token is in top-5
        if official_expected in top5_idx:
            rank = list(top5_idx).index(official_expected)
            print(f"Official token IS in top-5 at rank {rank}")
        else:
            print(f"Official token NOT in top-5")

    v9_tokens.append(next_token)
    position += 1

    print(f"new past_hidden: mean={past_hidden_torch.mean().item():.6f}, std={past_hidden_torch.std().item():.6f}")

print(f"\n=== Summary ===")
print(f"Official tokens: {official_codes[:6, 0].tolist()}")
print(f"V9+V3 tokens: {v9_tokens}")

# Check if the issue is the code_predictor sampling
print("\n=== Testing code_predictor with fixed seed ===")
torch.manual_seed(42)
with torch.no_grad():
    token_id = torch.tensor([[v9_first_token]], dtype=torch.long)
    last_id_hidden = talker.model.codec_embedding(token_id)
    predictor_input = torch.cat([torch.from_numpy(v9_past_hidden).float(), last_id_hidden], dim=1)
    predictor_result1 = talker.code_predictor.generate(
        inputs_embeds=predictor_input,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=True,
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    print(f"Run 1: {predictor_result1.sequences.tolist()}")

torch.manual_seed(42)
with torch.no_grad():
    predictor_result2 = talker.code_predictor.generate(
        inputs_embeds=predictor_input,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=True,
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    print(f"Run 2: {predictor_result2.sequences.tolist()}")

if predictor_result1.sequences.tolist() == predictor_result2.sequences.tolist():
    print("code_predictor IS deterministic with fixed seed")
else:
    print("code_predictor is NOT deterministic even with fixed seed!")
