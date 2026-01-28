# Debug the components of inputs_embeds: past_hidden, code_predictor, trailing_text
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

# Capture official components
official_past_hiddens = []
official_codec_sums = []
official_trailing = []
official_final_inputs = []

original_talker_forward = talker.forward.__func__

def make_hooked_forward():
    step_counter = [0]

    def hooked_forward(self, input_ids=None, inputs_embeds=None, past_hidden=None,
                       trailing_text_hidden=None, tts_pad_embed=None, generation_step=None, **kwargs):
        # Decode phase detection
        if input_ids is not None and input_ids.shape[1] == 1:
            # This is a decode step
            with torch.no_grad():
                last_id_hidden = self.get_input_embeddings()(input_ids)

                # Capture past_hidden before code_predictor
                official_past_hiddens.append(past_hidden.clone() if past_hidden is not None else None)

                # Run code_predictor
                predictor_result = self.code_predictor.generate(
                    inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                    max_new_tokens=self.config.num_code_groups - 1,
                    do_sample=False,
                    return_dict_in_generate=True,
                )

                # Compute codec sum
                codec_hiddens = torch.cat(
                    [last_id_hidden]
                    + [self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i:i+1])
                       for i in range(self.config.num_code_groups - 1)],
                    dim=1,
                )
                codec_sum = codec_hiddens.sum(1, keepdim=True)
                official_codec_sums.append(codec_sum.clone())

                # Capture trailing
                if generation_step is not None and generation_step < trailing_text_hidden.shape[1]:
                    trailing = trailing_text_hidden[:, generation_step].unsqueeze(1)
                else:
                    trailing = tts_pad_embed
                official_trailing.append(trailing.clone() if trailing is not None else None)

                # Final inputs_embeds
                final = codec_sum + trailing
                official_final_inputs.append(final.clone())

                step_counter[0] += 1

        return original_talker_forward(self, input_ids=input_ids, inputs_embeds=inputs_embeds,
                                       past_hidden=past_hidden, trailing_text_hidden=trailing_text_hidden,
                                       tts_pad_embed=tts_pad_embed, generation_step=generation_step, **kwargs)
    return hooked_forward

talker.forward = make_hooked_forward().__get__(talker, type(talker))

# Run official
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
        non_streaming_mode=True, max_new_tokens=3, do_sample=False,
    )
codes = result[0][0][:, 0].tolist()
print(f"Official tokens: {codes}")

print(f"Captured {len(official_past_hiddens)} decode steps")

# Reset talker.forward
talker.forward = original_talker_forward.__get__(talker, type(talker))

# Now run V3
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
with torch.no_grad():
    logits, kv_cache, v3_past_hidden = prefill_wrapper(
        role_ids, text_ids, text_length,
        tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed
    )

kv_cache = kv_cache[:, :, :, :actual_len, :]

suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

logits_np = logits.numpy().copy()
logits_np[0, suppress_mask] = -float('inf')
first_token = int(np.argmax(logits_np))

print(f"\nV3 first token: {first_token}")

# Now compute V3 decode components for step 0
with torch.no_grad():
    token_id = torch.tensor([[first_token]], dtype=torch.long)

    # Get codec embedding
    v3_last_id_hidden = talker.model.codec_embedding(token_id)

    # Run code_predictor
    v3_predictor_input = torch.cat([v3_past_hidden, v3_last_id_hidden], dim=1)
    v3_predictor_result = talker.code_predictor.generate(
        inputs_embeds=v3_predictor_input,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=False,
        return_dict_in_generate=True,
    )

    # Compute codec sum
    v3_codec_hiddens = [v3_last_id_hidden]
    for i in range(config.num_code_groups - 1):
        cb_embed = talker.code_predictor.get_input_embeddings()[i](
            v3_predictor_result.sequences[..., i:i+1]
        )
        v3_codec_hiddens.append(cb_embed)
    v3_codec_hiddens_cat = torch.cat(v3_codec_hiddens, dim=1)
    v3_codec_sum = v3_codec_hiddens_cat.sum(dim=1, keepdim=True)

    v3_trailing = tts_pad_embed
    v3_final = v3_codec_sum + v3_trailing

# Compare components for decode step 0
print("\n=== Decode Step 0 Component Comparison ===")

if len(official_past_hiddens) > 0:
    off_ph = official_past_hiddens[0]
    print("\n1. past_hidden:")
    print(f"   Official: mean={off_ph.mean().item():.6f}, std={off_ph.std().item():.6f}")
    print(f"   V3:       mean={v3_past_hidden.mean().item():.6f}, std={v3_past_hidden.std().item():.6f}")
    ph_diff = (off_ph - v3_past_hidden).abs().max().item()
    print(f"   Max diff: {ph_diff:.6f}")

if len(official_codec_sums) > 0:
    off_cs = official_codec_sums[0]
    print("\n2. codec_sum (sum of 16 codebook embeddings):")
    print(f"   Official: mean={off_cs.mean().item():.6f}, std={off_cs.std().item():.6f}")
    print(f"   V3:       mean={v3_codec_sum.mean().item():.6f}, std={v3_codec_sum.std().item():.6f}")
    cs_diff = (off_cs - v3_codec_sum).abs().max().item()
    print(f"   Max diff: {cs_diff:.6f}")

if len(official_trailing) > 0:
    off_tr = official_trailing[0]
    print("\n3. trailing_text:")
    print(f"   Official: mean={off_tr.mean().item():.6f}, std={off_tr.std().item():.6f}")
    print(f"   V3:       mean={v3_trailing.mean().item():.6f}, std={v3_trailing.std().item():.6f}")
    tr_diff = (off_tr - v3_trailing).abs().max().item()
    print(f"   Max diff: {tr_diff:.6f}")

if len(official_final_inputs) > 0:
    off_fi = official_final_inputs[0]
    print("\n4. final inputs_embeds (codec_sum + trailing):")
    print(f"   Official: mean={off_fi.mean().item():.6f}, std={off_fi.std().item():.6f}")
    print(f"   V3:       mean={v3_final.mean().item():.6f}, std={v3_final.std().item():.6f}")
    fi_diff = (off_fi - v3_final).abs().max().item()
    print(f"   Max diff: {fi_diff:.6f}")

# Print first few values for detailed comparison
print("\n=== First 5 values ===")
if len(official_past_hiddens) > 0:
    print(f"past_hidden Official[:5]: {official_past_hiddens[0][0, 0, :5].tolist()}")
    print(f"past_hidden V3[:5]:       {v3_past_hidden[0, 0, :5].tolist()}")

if len(official_codec_sums) > 0:
    print(f"codec_sum Official[:5]:   {official_codec_sums[0][0, 0, :5].tolist()}")
    print(f"codec_sum V3[:5]:         {v3_codec_sum[0, 0, :5].tolist()}")

if len(official_trailing) > 0:
    print(f"trailing Official[:5]:    {official_trailing[0][0, 0, :5].tolist()}")
    print(f"trailing V3[:5]:          {v3_trailing[0, 0, :5].tolist()}")
