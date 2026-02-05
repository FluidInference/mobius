# %% [markdown]
# # Qwen3-TTS LM Exploration
#
# Understand the LM generation flow before converting to CoreML.

# %%
import torch
import torch.nn as nn
from pathlib import Path

# Load model
from qwen_tts import Qwen3TTSModel

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "./model_0.6b",
    device_map="cpu",
    torch_dtype=torch.float32,
)
print("Model loaded!")

talker = model.model.talker
processor = model.processor

# %%
# Test the full TTS pipeline
print("\n=== Testing Full TTS Pipeline ===")

text = "Hello world"
language = "English"

# Generate speech
print(f"Generating speech for: '{text}'")
try:
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=None,  # No reference for basic TTS
        ref_text=None,
    )
    print(f"Generated audio: {wavs[0].shape} @ {sr}Hz")
except Exception as e:
    print(f"Voice clone failed (expected without ref): {e}")

# %%
# Understand the generation process step by step
print("\n=== Step-by-step Generation ===")

# 1. Tokenize text
inputs = processor(text=text, return_tensors="pt")
text_ids = inputs.input_ids
print(f"1. Text tokens: {text_ids.shape} = {text_ids.tolist()}")

# 2. Create the prompt for talker
# The prompt includes: [language_id, BOS, ...text_tokens...]
config = talker.config
language_id = config.codec_language_id.get(language.lower(), config.codec_language_id["english"])
bos_id = config.codec_bos_id
eos_id = config.codec_eos_token_id

print(f"2. Special tokens: language={language_id}, BOS={bos_id}, EOS={eos_id}")

# %%
# Look at the embedding process
print("\n=== Embedding Process ===")

# Text embeddings
text_embed = talker.model.text_embedding(text_ids)
print(f"Text embedding: {text_embed.shape}")  # [B, seq_len, 2048]

# Project to codec space
text_projected = talker.text_projection(text_embed)
print(f"Text projected: {text_projected.shape}")  # [B, seq_len, 1024]

# Codec embeddings (for BOS token)
codec_tokens = torch.tensor([[bos_id]])
codec_embed = talker.model.codec_embedding(codec_tokens)
print(f"Codec embedding: {codec_embed.shape}")  # [B, 1, 1024]

# %%
# Check the forward pass
print("\n=== Forward Pass Test ===")

# The talker model takes inputs_embeds (combined text + codec embeddings)
# Let's see the embed_tokens function
import inspect

if hasattr(talker.model, "embed_tokens"):
    print("embed_tokens source:")
    source = inspect.getsource(talker.model.embed_tokens)
    print(source[:1000])

# %%
# Test a single forward pass
print("\n=== Single Forward Pass ===")

# Create combined input
# For TTS: text tokens come first, then we generate codec tokens
# Input IDs contain both text and codec tokens
# The model distinguishes them by checking if ID >= codec_vocab_start

# Let's trace the actual input creation
combined_ids = torch.cat(
    [
        torch.tensor([[language_id]]),  # Language
        text_ids,  # Text
        torch.tensor([[bos_id]]),  # Codec BOS
    ],
    dim=1,
)
print(f"Combined input IDs: {combined_ids.shape}")

# Run forward
with torch.no_grad():
    outputs = talker.model(
        input_ids=combined_ids,
        use_cache=True,
        return_dict=True,
    )

print(f"Hidden states: {outputs.last_hidden_state.shape}")
print(f"Past KV length: {len(outputs.past_key_values)}")

# Get logits for next token
logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :])
print(f"Logits shape: {logits.shape}")

# Sample next token
next_token = torch.argmax(logits, dim=-1)
print(f"Next token: {next_token}")

# %%
# Now trace multiple steps
print("\n=== Multi-step Generation ===")

past_key_values = outputs.past_key_values
generated_tokens = [next_token.item()]

for step in range(5):
    with torch.no_grad():
        outputs = talker.model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = talker.codec_head(outputs.last_hidden_state)
        next_token = torch.argmax(logits, dim=-1)
        past_key_values = outputs.past_key_values
        generated_tokens.append(next_token.item())

print(f"Generated tokens: {generated_tokens}")
print(f"EOS token: {eos_id}")

# %%
# Check KV cache shape
print("\n=== KV Cache Analysis ===")

kv = past_key_values
print(f"Number of layers: {len(kv)}")
if hasattr(kv, "key_cache"):
    print(f"Key cache shape: {kv.key_cache[0].shape}")
    print(f"Value cache shape: {kv.value_cache[0].shape}")
else:
    # DynamicCache
    print(f"Cache type: {type(kv)}")
    print(f"Cache length: {kv.get_seq_length()}")

# %%
print("\n=== Model Sizes ===")
for name, child in talker.named_children():
    params = sum(p.numel() for p in child.parameters())
    print(f"{name}: {params / 1e6:.1f}M params")
