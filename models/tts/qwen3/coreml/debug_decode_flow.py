# Debug the official decode flow to understand trailing_text usage
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
import warnings
warnings.filterwarnings('ignore')

TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = tts_model.model.talker
config = talker.config

text = "Hello world, this is a test."

# Build the same input as the test
speaker_embed = np.load("speaker_embedding_official.npy").reshape(1, 1024)
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed.squeeze(0))],
    'x_vector_only_mode': [True], 'icl_mode': [False], 'ref_code': None,
}
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

print(f"Input text: {input_text}")
print(f"Full input_ids: {full_input_ids.tolist()}")
print(f"Full input_ids shape: {full_input_ids.shape}")

# Hook into the talker's forward to capture what it's doing
original_forward = talker.forward
captured_data = []

def hooked_forward(*args, **kwargs):
    generation_step = kwargs.get('generation_step', None)
    trailing_text_hidden = kwargs.get('trailing_text_hidden', None)
    tts_pad_embed = kwargs.get('tts_pad_embed', None)
    inputs_embeds = kwargs.get('inputs_embeds', None)
    input_ids = kwargs.get('input_ids', None)

    if inputs_embeds is not None:
        embed_shape = inputs_embeds.shape
    else:
        embed_shape = None

    captured_data.append({
        'generation_step': generation_step,
        'trailing_text_hidden_shape': trailing_text_hidden.shape if trailing_text_hidden is not None else None,
        'tts_pad_embed_shape': tts_pad_embed.shape if tts_pad_embed is not None else None,
        'inputs_embeds_shape': embed_shape,
        'input_ids': input_ids.tolist() if input_ids is not None else None,
    })

    return original_forward(*args, **kwargs)

talker.forward = hooked_forward

print("\nRunning generate...")
with torch.no_grad():
    result = tts_model.model.generate(
        input_ids=[full_input_ids], languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True, max_new_tokens=10, do_sample=False,
    )

codes = result[0][0][:, 0].tolist()
print(f"\nGenerated codes: {codes}")

print(f"\n=== Captured {len(captured_data)} forward calls ===")
for i, data in enumerate(captured_data[:15]):
    print(f"\nCall {i}:")
    print(f"  generation_step: {data['generation_step']}")
    print(f"  trailing_text_hidden_shape: {data['trailing_text_hidden_shape']}")
    print(f"  tts_pad_embed_shape: {data['tts_pad_embed_shape']}")
    print(f"  inputs_embeds_shape: {data['inputs_embeds_shape']}")
    if data['input_ids']:
        print(f"  input_ids: {data['input_ids']}")
