# Trace the official prefill logic step by step
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
import inspect

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
processor = tts_model.processor
talker = tts_model.model.talker

# Find the _prefill_forward source
print("\n=== Looking for _prefill_forward ===")
if hasattr(talker, '_prefill_forward'):
    print("Found _prefill_forward method")
    # print(inspect.getsource(talker._prefill_forward))
else:
    print("No _prefill_forward method")
    print(f"Available methods: {[m for m in dir(talker) if not m.startswith('_')][:20]}")

# Check the forward method
print("\n=== Talker forward signature ===")
sig = inspect.signature(talker.forward)
print(f"Parameters: {list(sig.parameters.keys())}")

# Examine the generate method to understand the flow
print("\n=== Tracing generate flow ===")

# The key insight is that model.generate uses model._prefill_forward
# Let's look at what arguments it passes

# From the qwen_tts source, the flow is:
# 1. model.generate() calls _non_streaming_generate() for non_streaming_mode=True
# 2. _non_streaming_generate() builds input_embeds and calls the model

# Let's manually build the same input_embeds the model builds

config = talker.config
text = "Hello world, this is a test of the text to speech system."
tokenizer = processor.tokenizer
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)

print(f"\nText: '{text}'")
print(f"Text tokens: {text_ids_list} ({text_len} total)")

# Special token IDs
TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]  # <|im_start|>assistant\n

# Load speaker embedding (the OFFICIAL one)
speaker_embed = np.load("speaker_embedding_official.npy").reshape(1, 1024)
speaker_embed_t = torch.from_numpy(speaker_embed).to(torch.float32)

print("\n=== Understanding the official non-streaming prefill ===")

# From examining the model source, non_streaming_mode builds:
# 1. Role prefix embeddings
# 2. Think token embeddings
# 3. Speaker embedding
# 4. Text embeddings
# 5. EOS token
# 6. BOS token (for generation start)

# The critical difference between streaming and non-streaming:
# - Streaming: text is fed incrementally
# - Non-streaming: all text is embedded at prefill

# Let's build the exact same sequence the model builds

with torch.no_grad():
    # Get TTS token embeddings
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

    print(f"TTS BOS embed shape: {tts_bos_embed.shape}")
    print(f"TTS BOS embed first few: {tts_bos_embed[0, 0, :5].tolist()}")

    # Get role prefix embeddings
    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
    role_text_embed = talker.text_projection(talker.model.text_embedding(role_ids))
    print(f"\nRole text embed shape: {role_text_embed.shape}")
    print(f"Role text embed first few: {role_text_embed[0, 0, :5].tolist()}")

    # Get codec embeddings for special tokens
    codec_think_id = config.codec_think_id
    codec_think_bos_id = config.codec_think_bos_id
    codec_think_eos_id = config.codec_think_eos_id
    codec_pad_id = config.codec_pad_id
    codec_bos_id = config.codec_bos_id
    english_language_id = config.codec_language_id["english"]

    print(f"\nCodec token IDs:")
    print(f"  codec_think_id: {codec_think_id}")
    print(f"  codec_think_bos_id: {codec_think_bos_id}")
    print(f"  codec_think_eos_id: {codec_think_eos_id}")
    print(f"  codec_pad_id: {codec_pad_id}")
    print(f"  codec_bos_id: {codec_bos_id}")
    print(f"  english_language_id: {english_language_id}")

    # Build the OFFICIAL sequence structure
    # Looking at the model source more carefully...

    # The key is in _non_streaming_generate:
    # 1. Build think tokens: [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
    # 2. For each position, combine TTS token embedding + codec embedding

    # For non-streaming mode with voice clone:
    # Position 0-2: role prefix (tts_pad + no_codec) OR (text_embed + 0)?
    # Position 3-6: think tokens (tts_pad + codec_think_embeds)
    # Position 7: speaker position (tts_bos + speaker_embed)
    # Position 8+: text positions (text_embed + codec_pad)
    # Position N: EOS (tts_eos + codec_pad)
    # Position N+1: BOS for generation (tts_pad + codec_bos)

    print("\n=== Building official sequence manually ===")

    # Step 1: Role prefix (positions 0-2)
    # Looking at the code, for role prefix, only text embedding is used, no codec
    # Actually, let me check what the model does...

    # From modeling_qwen3_tts.py, _non_streaming_generate:
    # The role prefix uses:
    # - text_embed = text_projection(text_embedding(role_ids))
    # - NO codec embedding for role prefix (it's set to zeros)

    # Wait, let me check V8 - it uses role_embed + zeros for positions 0-2
    # But the issue is V8 adds codec embedding for positions 0-2

    # Let me check V8's code again
    from convert_lm_prefill_v8 import TracablePrefillV8, MAX_TEXT_LENGTH, FIXED_SEQ_LEN

    print(f"\nV8 FIXED_SEQ_LEN: {FIXED_SEQ_LEN}")
    print(f"V8 MAX_TEXT_LENGTH: {MAX_TEXT_LENGTH}")

    # The issue might be that V8 is NOT adding zeros for role prefix codec
    # Let me trace through V8's forward

    print("\n=== V8 sequence construction trace ===")
    wrapper = TracablePrefillV8(talker)
    wrapper.eval()

    # Trace V8 hidden states construction
    batch_size = 1
    device = role_ids.device

    hidden_states = torch.zeros(batch_size, FIXED_SEQ_LEN, config.hidden_size,
                               device=device, dtype=tts_bos_embed.dtype)

    # V8 Position 0-2: Role prefix (text only, no codec!)
    role_embed = talker.text_projection(talker.model.text_embedding(role_ids))
    hidden_states[:, 0:3, :] = role_embed
    print(f"Position 0-2 (role): text embed only, no codec")
    print(f"  role_embed[0,0,:5]: {role_embed[0,0,:5].tolist()}")

    # V8 Position 3-6: tts_pad + Codec think tokens
    codec_think_ids = torch.tensor([
        [codec_think_id, codec_think_bos_id, english_language_id, codec_think_eos_id]
    ], dtype=torch.long, device=device)
    codec_think_embeds = talker.model.codec_embedding(codec_think_ids)
    hidden_states[:, 3:7, :] = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds
    print(f"Position 3-6 (think): tts_pad + codec_think")
    print(f"  tts_pad_embed[0,0,:5]: {tts_pad_embed[0,0,:5].tolist()}")
    print(f"  codec_think_embeds[0,0,:5]: {codec_think_embeds[0,0,:5].tolist()}")

    # V8 Position 7: tts_bos + Speaker
    hidden_states[:, 7:8, :] = tts_bos_embed + speaker_embed_t.unsqueeze(1)
    print(f"Position 7 (speaker): tts_bos + speaker")
    print(f"  speaker_embed[0,:5]: {speaker_embed_t[0,:5].tolist()}")

    # V8 Position 8+: Text tokens + codec_pad
    text_ids = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
    text_ids[0, :text_len] = torch.tensor(text_ids_list)
    all_text_embed = talker.text_projection(talker.model.text_embedding(text_ids))
    codec_pad_for_text = talker.model.codec_embedding(
        torch.full((batch_size, MAX_TEXT_LENGTH), codec_pad_id, dtype=torch.long, device=device)
    )
    hidden_states[:, 8:8+MAX_TEXT_LENGTH, :] = all_text_embed + codec_pad_for_text
    print(f"Position 8+ (text): text_embed + codec_pad")

    # EOS and BOS positions are scattered based on text_length

    print("\n=== Checking if V8 role prefix should include codec_pad ===")

    # Looking at the model code, for non-streaming mode, the role prefix
    # might need to include some codec embedding

    # Let me check what embedding is used for role prefix in streaming vs non-streaming

    # From the model code comments, the role prefix is typically:
    # - For streaming: just text embedding (streamed character by character)
    # - For non-streaming: text embedding + zeros (no codec for role)

    # But wait - looking at V8, it sets hidden_states[:, 0:3, :] = role_embed
    # This is just the text embedding, no codec

    # Let me check if the official model uses codec_pad for role prefix
    codec_pad_embed_single = talker.model.codec_embedding(
        torch.tensor([[codec_pad_id]], dtype=torch.long, device=device)
    )
    print(f"codec_pad_embed[0,0,:5]: {codec_pad_embed_single[0,0,:5].tolist()}")

    # Compare: role_embed vs role_embed + codec_pad
    role_with_codec = role_embed + codec_pad_embed_single.expand(-1, 3, -1)
    print(f"\nRole WITHOUT codec (V8): {role_embed[0,0,:5].tolist()}")
    print(f"Role WITH codec_pad:     {role_with_codec[0,0,:5].tolist()}")

    print("\n=== THE KEY QUESTION ===")
    print("Does the official model use codec_pad for role prefix positions?")
    print("If yes, V8 is missing it!")

    # Let me try running V8 with codec_pad added to role prefix
    print("\n=== Testing V8 with codec_pad on role prefix ===")

    # Modify hidden_states to include codec_pad on role prefix
    hidden_states_modified = hidden_states.clone()
    hidden_states_modified[:, 0:3, :] = role_embed + codec_pad_embed_single.expand(-1, 3, -1)

    # Now run the rest of V8's forward manually
    text_length = torch.tensor([text_len], dtype=torch.long)
    actual_len = text_length + 10

    # Position embeddings
    pos_1d = torch.arange(FIXED_SEQ_LEN, device=device)
    position_ids = pos_1d.unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    cos, sin = talker.model.rotary_emb(hidden_states_modified, position_ids)

    # Causal mask
    causal_mask = torch.triu(
        torch.ones(FIXED_SEQ_LEN, FIXED_SEQ_LEN, dtype=hidden_states_modified.dtype, device=device),
        diagonal=1
    )
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Padding mask
    q_pos = torch.arange(FIXED_SEQ_LEN, device=device).view(1, 1, FIXED_SEQ_LEN, 1)
    k_pos = torch.arange(FIXED_SEQ_LEN, device=device).view(1, 1, 1, FIXED_SEQ_LEN)
    actual_len_expanded = actual_len.view(batch_size, 1, 1, 1)

    padding_mask = torch.where(
        k_pos >= actual_len_expanded,
        torch.tensor(float("-inf"), dtype=hidden_states_modified.dtype, device=device),
        torch.tensor(0.0, dtype=hidden_states_modified.dtype, device=device),
    )

    combined_mask = causal_mask + padding_mask
    combined_mask = combined_mask.expand(batch_size, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN)

    # Run through layers (using original V8 logic but with modified hidden_states)
    # This is too complex to do inline, let me just test if adding codec_pad to role makes a difference

    print("\nTo test properly, I need to modify V8 and re-run.")
    print("The hypothesis is: V8 is missing codec_pad on role prefix positions 0-2")
