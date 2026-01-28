# Debug V8 wrapper vs official model output
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
processor = tts_model.processor
talker = tts_model.model.talker

# Special token IDs
TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]

text = "Hello world, this is a test of the text to speech system."
tokenizer = processor.tokenizer
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)

print(f"\nText: '{text}'")
print(f"Text tokens: {text_ids_list} ({text_len} total)")

# Use official high-level API to trace what happens
input_text = tts_model._build_assistant_text(text)
print(f"\nInput text (templated): repr={repr(input_text)}")

full_input_ids = tts_model._tokenize_texts([input_text])[0]
print(f"Full input IDs: {full_input_ids.tolist()}")
print(f"Full input IDs length: {len(full_input_ids)}")

# Decode each token to understand the structure
print("\nToken decoding:")
for i, tok in enumerate(full_input_ids.tolist()):
    decoded = tokenizer.decode(tok)
    print(f"  [{i}] {tok}: {repr(decoded)}")

# Load speaker embedding
speaker_embed = np.load("speaker_embedding_official.npy").reshape(1, 1024)
speaker_embed_t = torch.from_numpy(speaker_embed).to(torch.float32)

print("\n" + "="*60)
print("Testing official talker generate")
print("="*60)

# Call talker generate directly with text IDs
with torch.no_grad():
    text_input_ids = torch.tensor([text_ids_list])
    print(f"\nText input IDs for talker: {text_input_ids[0].tolist()}")

    result = talker.generate(
        input_ids=text_input_ids,
        non_streaming_mode=True,
        speaker_embeddings=speaker_embed_t,
        languages=['english'],
        max_new_tokens=10,
        do_sample=False,
    )

    print(f"\nResult type: {type(result)}")
    if hasattr(result, 'sequences'):
        print(f"Sequences: {result.sequences[0, :10].tolist()}")
    elif isinstance(result, tuple):
        codes, hidden = result
        print(f"Codes shape: {codes.shape}")
        print(f"First 10 codes: {codes[0, :10].tolist()}")

print("\n" + "="*60)
print("Testing V8 wrapper")
print("="*60)

# Import V8 wrapper
from convert_lm_prefill_v8 import TracablePrefillV8, MAX_TEXT_LENGTH

wrapper = TracablePrefillV8(talker)
wrapper.eval()

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

    logits, kv_cache = wrapper(role_ids, text_ids, text_length,
                                tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed_t)

print(f"Wrapper output shapes: logits={logits.shape}, kv_cache={kv_cache.shape}")

# Apply suppression mask
suppress_mask = np.zeros(talker.config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[talker.config.codec_eos_token_id] = False

logits_np = logits.numpy().copy()
logits_np[0, suppress_mask] = -float('inf')
first_token = int(np.argmax(logits_np))
print(f"Wrapper first token: {first_token}")

# Show top 10 tokens
top10_indices = np.argsort(logits_np[0])[-10:][::-1]
print(f"Wrapper top 10 tokens: {top10_indices.tolist()}")
print(f"Wrapper top 10 logits: {logits_np[0, top10_indices].tolist()}")
