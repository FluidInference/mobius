# Trace official model's prefill - capture inputs_embeds shape before error
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
print(f"Text length: {text_len}")

# Capture prefill inputs_embeds from official model
captured_prefill_info = []

original_model_forward = talker.model.forward

def capture_forward(*args, **kwargs):
    inputs_embeds = kwargs.get('inputs_embeds')
    if inputs_embeds is not None:
        captured_prefill_info.append({
            'seq_len': inputs_embeds.shape[1],
            'mean': inputs_embeds.mean().item(),
            'std': inputs_embeds.std().item(),
        })
        if inputs_embeds.shape[1] > 1:
            # This is prefill
            print(f"  Prefill captured: seq_len={inputs_embeds.shape[1]}, mean={inputs_embeds.mean().item():.6f}")
    return original_model_forward(*args, **kwargs)

talker.model.forward = capture_forward

# Run official with more tokens to avoid the error
speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024)
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed_np.squeeze(0))],
    'x_vector_only_mode': [True], 'icl_mode': [False], 'ref_code': None,
}
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

print(f"\nOfficial full_input_ids length: {len(full_input_ids)}")
print(f"Running official generate (max_new_tokens=3)...")

try:
    with torch.no_grad():
        result = tts_model.model.generate(
            input_ids=[full_input_ids], languages=['english'],
            voice_clone_prompt=voice_clone_prompt,
            non_streaming_mode=True, max_new_tokens=3, do_sample=False,
        )
    codes = result[0][0][:, 0].tolist()
    print(f"Official tokens: {codes}")
except Exception as e:
    print(f"Error: {e}")

talker.model.forward = original_model_forward

print(f"\nCaptured {len(captured_prefill_info)} forward calls")
for i, info in enumerate(captured_prefill_info):
    print(f"  Call {i}: seq_len={info['seq_len']}")

# Find the prefill call (first one with seq_len > 1)
prefill_info = None
for info in captured_prefill_info:
    if info['seq_len'] > 1:
        prefill_info = info
        break

if prefill_info:
    official_prefill_len = prefill_info['seq_len']
    print(f"\nOfficial prefill sequence length: {official_prefill_len}")

    # V9 prefill length
    v9_prefill_len = text_len + 11
    print(f"V9 prefill sequence length: {v9_prefill_len}")

    if official_prefill_len != v9_prefill_len:
        print(f"\n*** SEQUENCE LENGTH MISMATCH: {official_prefill_len} vs {v9_prefill_len} ***")
        print("This is the root cause of the divergence!")

# Let's calculate what the official length should be
# Based on the code analysis:
# In non_streaming_mode with x_vector_only_mode:
# - role: 3 tokens
# - codec_think + speaker + bos: 6 tokens
# - text tokens: text_len
# - tts_eos: 1 token
# - tts_pad + codec_bos: 1 token
# Total: 3 + 6 + text_len + 1 + 1 = text_len + 11

# But the official builds differently. Let me check the actual code...

# From the official code, the talker_input_embed is built as:
# 1. _talker_input_embed_role: 3 tokens (role prefix)
# 2. _talker_input_embed: tts_pad * (codec_embed.shape[1] - 2) + tts_bos + codec_embed[:-1]
#    - codec_embed includes: think, think_bos, lang, think_eos, [speaker], pad, bos
#    - So codec_embed.shape[1] = 7 (if with speaker) or 6 (without)
#    - With speaker: 7 - 2 = 5 tts_pads, then tts_bos, total 7 positions
# 3. In non_streaming_mode:
#    - Removes last token (text[0])
#    - Adds: text[all] + eos + pad_codec, then pad + bos_codec
#    - So: role(3) + codec(6) + text(N) + eos(1) + pad_bos(1) = 3 + 6 + N + 2 = N + 11

# This matches V9! So why is there a difference?

print("\n=== Debug: official sequence structure ===")
# Let me print what input_ids the official builds
print(f"full_input_ids: {full_input_ids.tolist()}")
print(f"full_input_ids length: {len(full_input_ids)}")

# Decode each token
for i, tid in enumerate(full_input_ids.tolist()):
    try:
        decoded = tokenizer.decode([tid])
    except:
        decoded = f"<special:{tid}>"
    print(f"  {i}: {tid} -> '{decoded}'")
