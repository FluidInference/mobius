# Debug where decode divergence occurs
# V3/V4 and Official match for first 2 tokens, diverge at token 3
import torch
import numpy as np
import random
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH
from convert_lm_decode_v3 import TracableDecodeV3
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
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

prefill_wrapper = TracablePrefillV9(talker)
decode_wrapper = TracableDecodeV3(talker)
prefill_wrapper.eval()
decode_wrapper.eval()

# Prepare inputs
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)
actual_len = text_len + 11

print(f"text_len={text_len}, actual_len={actual_len}")

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

# Run V3 and capture intermediate states
print("\n=== Running V3 with detailed tracing ===")
with torch.no_grad():
    logits, kv_cache, past_hidden = prefill_wrapper(
        role_ids, text_ids, text_length,
        tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed
    )

kv_cache = kv_cache[:, :, :, :actual_len, :]
print(f"V3 Prefill KV cache shape: {kv_cache.shape}")
print(f"V3 Prefill past_hidden shape: {past_hidden.shape}")

logits_np = logits.numpy().copy()
logits_np[0, suppress_mask] = -float('inf')
first_token = int(np.argmax(logits_np))
print(f"V3 First token: {first_token}")

# Store V3 decode states
v3_tokens = [first_token]
v3_states = []  # Store intermediate states
pos = actual_len

for i in range(5):  # Just first few steps
    token_id = torch.tensor([[v3_tokens[-1]]], dtype=torch.long)
    position = torch.tensor([pos], dtype=torch.long)

    # Get intermediate values from decode wrapper
    with torch.no_grad():
        # Get codec embedding
        last_id_hidden = talker.model.codec_embedding(token_id)

        # Get code_predictor result
        predictor_input = torch.cat([past_hidden, last_id_hidden], dim=1)
        predictor_result = talker.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=config.num_code_groups - 1,
            do_sample=False,
            return_dict_in_generate=True,
        )

        # Get all codebook embeddings
        codec_hiddens = [last_id_hidden]
        predictor_tokens = []
        for j in range(config.num_code_groups - 1):
            pred_token = predictor_result.sequences[..., j:j+1]
            predictor_tokens.append(pred_token.item())
            cb_embed = talker.code_predictor.get_input_embeddings()[j](pred_token)
            codec_hiddens.append(cb_embed)

        codec_hiddens_cat = torch.cat(codec_hiddens, dim=1)
        inputs_embeds = codec_hiddens_cat.sum(dim=1, keepdim=True)
        inputs_embeds_with_text = inputs_embeds + tts_pad_embed

        # Store state
        v3_states.append({
            'step': i,
            'input_token': v3_tokens[-1],
            'last_id_hidden': last_id_hidden.clone(),
            'past_hidden': past_hidden.clone(),
            'predictor_input': predictor_input.clone(),
            'predictor_tokens': predictor_tokens,
            'codec_hiddens_sum': inputs_embeds.clone(),
            'inputs_embeds': inputs_embeds_with_text.clone(),
        })

        # Now run full decode
        logits, kv_cache, past_hidden = decode_wrapper(
            token_id, past_hidden, tts_pad_embed, kv_cache, position
        )

    logits_np = logits.numpy().copy()
    logits_np[0, suppress_mask] = -float('inf')
    next_token = int(np.argmax(logits_np))

    if next_token == config.codec_eos_token_id:
        print(f"EOS at step {i}")
        break

    v3_tokens.append(next_token)
    pos += 1

print(f"\nV3 tokens: {v3_tokens}")

# Now run official with tracing using hooks
print("\n=== Running Official with detailed tracing ===")

# Reset seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024)
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed_np.squeeze(0))],
    'x_vector_only_mode': [True], 'icl_mode': [False], 'ref_code': None,
}
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

# Add hook to capture decode states
official_states = []

original_forward = talker.forward

def hooked_forward(self, *args, **kwargs):
    result = original_forward(*args, **kwargs)

    # Capture some info if we're in decode mode
    if kwargs.get('input_ids') is not None and kwargs['input_ids'].shape[1] == 1:
        # Single token = decode step
        inputs_embeds = kwargs.get('inputs_embeds')
        past_hidden = kwargs.get('past_hidden')
        if inputs_embeds is not None:
            official_states.append({
                'inputs_embeds': inputs_embeds.clone() if inputs_embeds is not None else None,
                'past_hidden': past_hidden.clone() if past_hidden is not None else None,
            })

    return result

# Can't easily hook the official model, so let's trace manually
# by looking at what generate does

# Instead, let's compare the prefill outputs directly
print("\n=== Comparing prefill outputs ===")

# Run official prefill by calling the model directly
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Build the same input as official
full_text = tts_model._build_assistant_text(text)
full_input_ids_list = tts_model._tokenize_texts([full_text])[0]

print(f"Full input IDs length: {len(full_input_ids_list)}")
print(f"First 20 IDs: {full_input_ids_list[:20]}")

# The official model's generate() builds the input differently
# Let's see what the actual prefill sequence looks like
# by examining the full_input_ids

# Run official
with torch.no_grad():
    result = tts_model.model.generate(
        input_ids=[full_input_ids_list], languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True, max_new_tokens=5, do_sample=False,
    )
codes = result[0][0][:, 0].tolist()
print(f"Official tokens: {codes}")

# Compare
print(f"\n=== Token Comparison ===")
print(f"V3:       {v3_tokens}")
print(f"Official: {codes[:len(v3_tokens)]}")
for i in range(min(len(v3_tokens), len(codes))):
    match = "OK" if v3_tokens[i] == codes[i] else "MISMATCH"
    print(f"  Token {i}: V3={v3_tokens[i]}, Official={codes[i]} - {match}")

# Print V3 intermediate states
print(f"\n=== V3 Intermediate States ===")
for state in v3_states:
    print(f"\nStep {state['step']} (input token: {state['input_token']}):")
    print(f"  past_hidden: mean={state['past_hidden'].mean().item():.6f}, std={state['past_hidden'].std().item():.6f}")
    print(f"  last_id_hidden: mean={state['last_id_hidden'].mean().item():.6f}, std={state['last_id_hidden'].std().item():.6f}")
    print(f"  predictor_tokens: {state['predictor_tokens']}")
    print(f"  codec_sum: mean={state['codec_hiddens_sum'].mean().item():.6f}, std={state['codec_hiddens_sum'].std().item():.6f}")
    print(f"  inputs_embeds: mean={state['inputs_embeds'].mean().item():.6f}, std={state['inputs_embeds'].std().item():.6f}")
