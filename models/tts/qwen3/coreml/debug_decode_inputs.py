# Debug what inputs_embeds the official model uses in decode
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
import warnings
warnings.filterwarnings('ignore')

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = tts_model.model.talker
config = talker.config

text = "Hello world, this is a test."

speaker_embed = np.load("speaker_embedding_official.npy").reshape(1, 1024)
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed.squeeze(0))],
    'x_vector_only_mode': [True], 'icl_mode': [False], 'ref_code': None,
}
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

# Hook into talker.model (the transformer) to see what inputs_embeds it receives
captured_embeds = []
original_model_forward = talker.model.forward

def hooked_model_forward(input_ids=None, attention_mask=None, position_ids=None,
                         past_key_values=None, inputs_embeds=None, use_cache=None,
                         output_attentions=None, output_hidden_states=None,
                         cache_position=None, **kwargs):
    if inputs_embeds is not None:
        captured_embeds.append({
            'shape': inputs_embeds.shape,
            'mean': inputs_embeds.mean().item(),
            'std': inputs_embeds.std().item(),
            'sample': inputs_embeds[0, 0, :5].tolist() if inputs_embeds.shape[1] > 0 else None,
        })
    return original_model_forward(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
        output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        cache_position=cache_position, **kwargs
    )

talker.model.forward = hooked_model_forward

print("\nRunning generate...")
with torch.no_grad():
    result = tts_model.model.generate(
        input_ids=[full_input_ids], languages=['english'],
        voice_clone_prompt=voice_clone_prompt,
        non_streaming_mode=True, max_new_tokens=5, do_sample=False,
    )

codes = result[0][0][:, 0].tolist()
print(f"Generated codes: {codes}")

print(f"\n=== Captured {len(captured_embeds)} transformer forward calls ===")
for i, data in enumerate(captured_embeds):
    print(f"\nCall {i}:")
    print(f"  Shape: {data['shape']}")
    print(f"  Mean: {data['mean']:.6f}")
    print(f"  Std: {data['std']:.6f}")
    print(f"  Sample (first 5 dims): {data['sample']}")

# Now compare with what my wrapper produces
print("\n\n=== Comparing with wrapper inputs ===")
from convert_lm_decode_v2 import TracableDecodeV2

decode_wrapper = TracableDecodeV2(talker)
decode_wrapper.eval()

TTS_PAD_TOKEN_ID = 151671
with torch.no_grad():
    tts_pad_ids = torch.tensor([[TTS_PAD_TOKEN_ID]])
    tts_pad_embed = talker.text_projection(talker.model.text_embedding(tts_pad_ids))

# First decode step
first_token = codes[0]  # 1995
with torch.no_grad():
    token_embed = talker.model.codec_embedding(torch.tensor([[first_token]]))
    wrapper_input = token_embed + tts_pad_embed

print(f"\nWrapper input for first decode step:")
print(f"  Token: {first_token}")
print(f"  Shape: {wrapper_input.shape}")
print(f"  Mean: {wrapper_input.mean().item():.6f}")
print(f"  Std: {wrapper_input.std().item():.6f}")
print(f"  Sample (first 5 dims): {wrapper_input[0, 0, :5].tolist()}")

# Compare with call 1 (first decode step)
if len(captured_embeds) > 1:
    print(f"\nOfficial's first decode inputs:")
    print(f"  Shape: {captured_embeds[1]['shape']}")
    print(f"  Mean: {captured_embeds[1]['mean']:.6f}")
    print(f"  Std: {captured_embeds[1]['std']:.6f}")
    print(f"  Sample (first 5 dims): {captured_embeds[1]['sample']}")
