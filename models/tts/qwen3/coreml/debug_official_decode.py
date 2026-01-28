# Debug official model's decode to see what inputs_embeds it uses
import torch
import numpy as np
import random
from qwen_tts import Qwen3TTSModel
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

text = "Hello world, this is a test."

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = tts_model.model.talker
config = talker.config

# Add hooks to capture inputs_embeds during decode
decode_calls = []

original_talker_forward = talker.forward.__func__

def hooked_talker_forward(self, input_ids=None, inputs_embeds=None, past_hidden=None, **kwargs):
    if inputs_embeds is not None and inputs_embeds.shape[1] == 1:
        # This is a decode step with single token
        decode_calls.append({
            'inputs_embeds': inputs_embeds.clone().detach(),
            'past_hidden': past_hidden.clone().detach() if past_hidden is not None else None,
        })
    return original_talker_forward(self, input_ids=input_ids, inputs_embeds=inputs_embeds, past_hidden=past_hidden, **kwargs)

# Monkey-patch
talker.forward = lambda *args, **kwargs: hooked_talker_forward(talker, *args, **kwargs)

# Run official generate
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

print(f"\n=== Captured {len(decode_calls)} decode calls ===")
for i, call in enumerate(decode_calls):
    print(f"\nDecode call {i}:")
    if call['inputs_embeds'] is not None:
        ie = call['inputs_embeds']
        print(f"  inputs_embeds: shape={ie.shape}, mean={ie.mean().item():.6f}, std={ie.std().item():.6f}")
        print(f"  inputs_embeds[0,0,:5]: {ie[0, 0, :5].tolist()}")
    if call['past_hidden'] is not None:
        ph = call['past_hidden']
        print(f"  past_hidden: shape={ph.shape}, mean={ph.mean().item():.6f}, std={ph.std().item():.6f}")

# Now run V3 and compare
print("\n=== Running V3 for comparison ===")
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH
from convert_lm_decode_v3 import TracableDecodeV3

TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]

tokenizer = tts_model.processor.tokenizer
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
v3_inputs_embeds = []
pos = actual_len

for i in range(5):
    token_id = torch.tensor([[v3_tokens[-1]]], dtype=torch.long)
    position = torch.tensor([pos], dtype=torch.long)

    # Compute inputs_embeds manually like decode_wrapper does
    with torch.no_grad():
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

        v3_inputs_embeds.append({
            'inputs_embeds': inputs_embeds.clone(),
            'past_hidden': past_hidden.clone(),
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

print("\n=== Comparing inputs_embeds ===")
for i in range(min(len(decode_calls), len(v3_inputs_embeds))):
    off_ie = decode_calls[i]['inputs_embeds']
    v3_ie = v3_inputs_embeds[i]['inputs_embeds']
    diff = (off_ie - v3_ie).abs().max().item()
    print(f"\nDecode step {i}:")
    print(f"  Official inputs_embeds: mean={off_ie.mean().item():.6f}, std={off_ie.std().item():.6f}")
    print(f"  V3 inputs_embeds:       mean={v3_ie.mean().item():.6f}, std={v3_ie.std().item():.6f}")
    print(f"  Max diff: {diff:.6f}")

    if decode_calls[i]['past_hidden'] is not None and v3_inputs_embeds[i]['past_hidden'] is not None:
        off_ph = decode_calls[i]['past_hidden']
        v3_ph = v3_inputs_embeds[i]['past_hidden']
        ph_diff = (off_ph - v3_ph).abs().max().item()
        print(f"  Official past_hidden: mean={off_ph.mean().item():.6f}, std={off_ph.std().item():.6f}")
        print(f"  V3 past_hidden:       mean={v3_ph.mean().item():.6f}, std={v3_ph.std().item():.6f}")
        print(f"  Past hidden max diff: {ph_diff:.6f}")
