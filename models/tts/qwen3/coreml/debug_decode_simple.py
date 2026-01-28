# Simpler approach: capture inputs_embeds at model.forward level
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

# Capture at model.forward level
official_decode_inputs = []

original_model_forward = talker.model.forward

def capture_model_forward(*args, **kwargs):
    inputs_embeds = kwargs.get('inputs_embeds')
    if inputs_embeds is not None and inputs_embeds.shape[1] == 1:
        # Decode call
        official_decode_inputs.append({
            'inputs_embeds': inputs_embeds.clone().detach(),
        })
    return original_model_forward(*args, **kwargs)

talker.model.forward = capture_model_forward

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

print(f"\nCaptured {len(official_decode_inputs)} decode inputs")

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

# V3 decode step 0
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

    v3_codec_hiddens = [v3_last_id_hidden]
    for i in range(config.num_code_groups - 1):
        cb_embed = talker.code_predictor.get_input_embeddings()[i](
            v3_result.sequences[..., i:i+1]
        )
        v3_codec_hiddens.append(cb_embed)
    v3_codec_hiddens_cat = torch.cat(v3_codec_hiddens, dim=1)
    v3_codec_sum = v3_codec_hiddens_cat.sum(dim=1, keepdim=True)

    v3_trailing = tts_pad_embed
    v3_inputs_embeds = v3_codec_sum + v3_trailing

print(f"V3 inputs_embeds: mean={v3_inputs_embeds.mean().item():.6f}, std={v3_inputs_embeds.std().item():.6f}")

# Compare
if len(official_decode_inputs) > 0:
    off_ie = official_decode_inputs[0]['inputs_embeds']
    print(f"\n=== Decode Step 0 inputs_embeds comparison ===")
    print(f"Official: mean={off_ie.mean().item():.6f}, std={off_ie.std().item():.6f}")
    print(f"V3:       mean={v3_inputs_embeds.mean().item():.6f}, std={v3_inputs_embeds.std().item():.6f}")

    diff = (off_ie - v3_inputs_embeds).abs().max().item()
    print(f"Max diff: {diff:.6f}")

    if diff > 0.01:
        print("\n*** MISMATCH ***")
        # Show some values
        print(f"Official[:10]: {off_ie[0, 0, :10].tolist()}")
        print(f"V3[:10]:       {v3_inputs_embeds[0, 0, :10].tolist()}")

        # Check individual components
        print("\n=== Decomposing the difference ===")

        # V3 uses: codec_sum + tts_pad_embed
        # Official uses: codec_sum + trailing_text_hidden[:, generation_step]

        # In non_streaming_mode, trailing_text_hidden = tts_pad_embed
        # So at generation_step=0: trailing = tts_pad_embed[:, 0].unsqueeze(1)

        # The question is: is tts_pad_embed[:, 0].unsqueeze(1) the same as tts_pad_embed?
        tts_pad_0 = tts_pad_embed[:, 0].unsqueeze(1)
        print(f"\ntts_pad_embed shape: {tts_pad_embed.shape}")
        print(f"tts_pad_embed[:, 0].unsqueeze(1) shape: {tts_pad_0.shape}")
        print(f"Are they equal: {torch.equal(tts_pad_embed, tts_pad_0)}")

        # What if the official uses a different trailing_text_hidden?
        # Let me check what trailing_text_hidden the official model actually uses

        # From the earlier debug, official prefill embeds matched V9
        # So the issue must be in how trailing_text_hidden is passed

        # Actually, let me check if codec_sum differs
        v3_cs = v3_codec_sum
        # We can't easily get official codec_sum without hooking

        # Let me try: subtract tts_pad_embed from official inputs_embeds to get codec_sum
        inferred_off_cs = off_ie - tts_pad_embed
        print(f"\nInferred official codec_sum (off_ie - tts_pad):")
        print(f"  mean={inferred_off_cs.mean().item():.6f}, std={inferred_off_cs.std().item():.6f}")
        print(f"V3 codec_sum:")
        print(f"  mean={v3_cs.mean().item():.6f}, std={v3_cs.std().item():.6f}")

        cs_diff = (inferred_off_cs - v3_cs).abs().max().item()
        print(f"Inferred codec_sum diff: {cs_diff:.6f}")

        if cs_diff > 0.01:
            print("\n*** codec_sum DIFFERS! ***")
            print("This means the code_predictor produced different tokens or")
            print("the codec embeddings are computed differently")
        else:
            print("\n*** codec_sum MATCHES, so trailing differs ***")
            # Check if official is using tts_pad_embed or something else
            inferred_trailing = off_ie - v3_cs
            print(f"\nInferred official trailing (off_ie - v3_codec_sum):")
            print(f"  mean={inferred_trailing.mean().item():.6f}")
            print(f"tts_pad_embed:")
            print(f"  mean={tts_pad_embed.mean().item():.6f}")
            tr_diff = (inferred_trailing - tts_pad_embed).abs().max().item()
            print(f"Trailing diff from tts_pad_embed: {tr_diff:.6f}")
