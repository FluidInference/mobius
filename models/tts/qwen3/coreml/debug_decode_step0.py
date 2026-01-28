# Debug decode step 0 in detail - trace official vs V3
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

# Capture official decode step 0 components
official_decode_data = []

original_talker_forward = talker.forward.__func__

def hooked_talker_forward(self, input_ids=None, inputs_embeds=None, past_hidden=None,
                          trailing_text_hidden=None, tts_pad_embed=None, generation_step=None, **kwargs):
    # Capture decode step data
    if input_ids is not None and input_ids.shape[1] == 1:
        with torch.no_grad():
            last_id_hidden = self.get_input_embeddings()(input_ids)

            # Run code_predictor
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=False,
                return_dict_in_generate=True,
            )

            # Compute codec_sum
            codec_hiddens = [last_id_hidden]
            for i in range(self.config.num_code_groups - 1):
                cb_embed = self.code_predictor.get_input_embeddings()[i](
                    predictor_result.sequences[..., i:i+1]
                )
                codec_hiddens.append(cb_embed)
            codec_hiddens_cat = torch.cat(codec_hiddens, dim=1)
            codec_sum = codec_hiddens_cat.sum(1, keepdim=True)

            # Compute trailing
            if generation_step < trailing_text_hidden.shape[1]:
                trailing = trailing_text_hidden[:, generation_step].unsqueeze(1)
            else:
                trailing = tts_pad_embed

            official_decode_data.append({
                'input_id': input_ids[0, 0].item(),
                'past_hidden': past_hidden.clone(),
                'last_id_hidden': last_id_hidden.clone(),
                'predictor_tokens': predictor_result.sequences[0].tolist(),
                'codec_sum': codec_sum.clone(),
                'trailing': trailing.clone(),
                'trailing_text_hidden_shape': trailing_text_hidden.shape,
                'generation_step': generation_step,
            })

    return original_talker_forward(self, input_ids=input_ids, inputs_embeds=inputs_embeds,
                                   past_hidden=past_hidden, trailing_text_hidden=trailing_text_hidden,
                                   tts_pad_embed=tts_pad_embed, generation_step=generation_step, **kwargs)

talker.forward = hooked_talker_forward.__get__(talker, type(talker))

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

talker.forward = original_talker_forward.__get__(talker, type(talker))

print(f"\nCaptured {len(official_decode_data)} decode steps")

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
    v3_predictor_tokens = v3_result.sequences[0].tolist()

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

print(f"V3 predictor tokens: {v3_predictor_tokens}")

# Compare official decode step 0 vs V3
if len(official_decode_data) > 0:
    off = official_decode_data[0]
    print("\n=== Decode Step 0 Comparison ===")
    print(f"\nInput token:")
    print(f"  Official: {off['input_id']}")
    print(f"  V3: {first_token}")

    print(f"\npast_hidden:")
    off_ph = off['past_hidden']
    print(f"  Official: mean={off_ph.mean().item():.6f}, std={off_ph.std().item():.6f}")
    print(f"  V3:       mean={v9_past_hidden.mean().item():.6f}, std={v9_past_hidden.std().item():.6f}")
    ph_diff = (off_ph - v9_past_hidden).abs().max().item()
    print(f"  Max diff: {ph_diff:.6f}")

    print(f"\nlast_id_hidden:")
    off_lih = off['last_id_hidden']
    print(f"  Official: mean={off_lih.mean().item():.6f}, std={off_lih.std().item():.6f}")
    print(f"  V3:       mean={v3_last_id_hidden.mean().item():.6f}, std={v3_last_id_hidden.std().item():.6f}")
    lih_diff = (off_lih - v3_last_id_hidden).abs().max().item()
    print(f"  Max diff: {lih_diff:.6f}")

    print(f"\npredictor_tokens:")
    print(f"  Official: {off['predictor_tokens']}")
    print(f"  V3:       {v3_predictor_tokens}")
    print(f"  Match: {off['predictor_tokens'] == v3_predictor_tokens}")

    print(f"\ncodec_sum:")
    off_cs = off['codec_sum']
    print(f"  Official: mean={off_cs.mean().item():.6f}, std={off_cs.std().item():.6f}")
    print(f"  V3:       mean={v3_codec_sum.mean().item():.6f}, std={v3_codec_sum.std().item():.6f}")
    cs_diff = (off_cs - v3_codec_sum).abs().max().item()
    print(f"  Max diff: {cs_diff:.6f}")

    print(f"\ntrailing:")
    off_tr = off['trailing']
    print(f"  Official: shape={off_tr.shape}, mean={off_tr.mean().item():.6f}")
    print(f"  V3:       shape={v3_trailing.shape}, mean={v3_trailing.mean().item():.6f}")
    print(f"  Official trailing_text_hidden shape: {off['trailing_text_hidden_shape']}")
    print(f"  Official generation_step: {off['generation_step']}")
    tr_diff = (off_tr - v3_trailing).abs().max().item()
    print(f"  Max diff: {tr_diff:.6f}")

    print(f"\nfinal inputs_embeds (codec_sum + trailing):")
    off_final = off_cs + off_tr
    print(f"  Official: mean={off_final.mean().item():.6f}, std={off_final.std().item():.6f}")
    print(f"  V3:       mean={v3_inputs_embeds.mean().item():.6f}, std={v3_inputs_embeds.std().item():.6f}")
    final_diff = (off_final - v3_inputs_embeds).abs().max().item()
    print(f"  Max diff: {final_diff:.6f}")
