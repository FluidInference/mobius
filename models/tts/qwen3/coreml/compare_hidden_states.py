# Compare hidden states between V9 wrapper and official model
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH, FIXED_SEQ_LEN

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
processor = tts_model.processor
talker = tts_model.model.talker
config = talker.config

# Special token IDs
TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]

text = "Hello world, this is a test of the text to speech system."
tokenizer = processor.tokenizer
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)

# Load speaker embedding
speaker_embed = np.load("speaker_embedding_official.npy").reshape(1, 1024)
speaker_embed_t = torch.from_numpy(speaker_embed).to(torch.float32)

print(f"\nText: '{text}'")
print(f"Text tokens: {text_len}")

# Build official prefill embeddings manually
print("\n=== Building official inputs_embeds ===")
with torch.no_grad():
    # Build full input IDs
    input_text = tts_model._build_assistant_text(text)
    full_input_ids = tts_model._tokenize_texts([input_text])[0]

    # TTS embeddings
    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor([[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]], dtype=torch.long)
        )
    ).chunk(3, dim=1)

    # Role prefix - full_input_ids is [1, seq_len] so we need [0, :3]
    role_embed = talker.text_projection(
        talker.get_text_embeddings()(full_input_ids[0, :3].unsqueeze(0))
    )
    print(f"Role embed shape: {role_embed.shape}")

    # Build codec embeddings
    language_id = config.codec_language_id["english"]
    codec_prefill_list = [[
        config.codec_think_id,
        config.codec_think_bos_id,
        language_id,
        config.codec_think_eos_id,
    ]]
    codec_input_emebdding_0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, dtype=torch.long)
    )
    codec_input_emebdding_1 = talker.get_input_embeddings()(
        torch.tensor([[config.codec_pad_id, config.codec_bos_id]], dtype=torch.long)
    )

    # Full codec embedding with speaker
    codec_input_emebdding = torch.cat([
        codec_input_emebdding_0,  # 4 tokens
        speaker_embed_t.view(1, 1, -1),  # 1 token (speaker)
        codec_input_emebdding_1  # 2 tokens (pad, bos)
    ], dim=1)

    # Build _talker_input_embed (6 positions: think*4 + speaker + pad)
    _talker_input_embed = torch.cat((
        tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),  # 5 tts_pad
        tts_bos_embed,  # 1 tts_bos
    ), dim=1) + codec_input_emebdding[:, :-1]  # 6 codec (excluding final bos)

    # Combine role + _talker_input_embed
    talker_input_embed = torch.cat((role_embed, _talker_input_embed), dim=1)
    print(f"Pre-text embed shape: {talker_input_embed.shape}")

    # Add first text token placeholder (will be removed for non-streaming)
    talker_input_embed = torch.cat([
        talker_input_embed,
        talker.text_projection(talker.get_text_embeddings()(full_input_ids[0, 3:4].unsqueeze(0))) + codec_input_emebdding[:, -1:]
    ], dim=1)

    # Remove for non-streaming
    talker_input_embed = talker_input_embed[:, :-1]

    # Add text + eos + final bos (non-streaming mode)
    text_part = full_input_ids[0, 3:-5]
    text_embed = talker.text_projection(
        talker.get_text_embeddings()(text_part.unsqueeze(0))
    )
    text_with_eos = torch.cat([text_embed, tts_eos_embed], dim=1)

    codec_pad_for_text = talker.get_input_embeddings()(
        torch.tensor([[config.codec_pad_id] * (text_part.shape[0] + 1)], dtype=torch.long)
    )

    final_bos = tts_pad_embed + talker.get_input_embeddings()(
        torch.tensor([[config.codec_bos_id]], dtype=torch.long)
    )

    official_inputs_embeds = torch.cat([
        talker_input_embed,
        text_with_eos + codec_pad_for_text,
        final_bos
    ], dim=1)

    print(f"Official inputs_embeds shape: {official_inputs_embeds.shape}")
    expected_len = 3 + 6 + text_len + 1 + 1
    print(f"Expected length: {expected_len}")

# Build V9 inputs_embeds
print("\n=== Building V9 inputs_embeds ===")
with torch.no_grad():
    wrapper = TracablePrefillV9(talker)
    wrapper.eval()

    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
    text_ids = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
    text_ids[0, :text_len] = torch.tensor(text_ids_list)
    text_length = torch.tensor([text_len], dtype=torch.long)

    # Manually build the hidden_states like V9 does
    batch_size = 1
    device = role_ids.device

    v9_hidden_states = torch.zeros(batch_size, FIXED_SEQ_LEN, config.hidden_size,
                                   device=device, dtype=torch.float32)

    # Position 0-2: Role prefix (text embed ONLY, no codec!)
    role_embed_v9 = wrapper.text_projection(wrapper.text_embedding(role_ids))
    v9_hidden_states[:, 0:3, :] = role_embed_v9

    # Position 3-6: tts_pad + Codec think tokens (4 positions)
    codec_think_ids = torch.tensor([
        [wrapper.codec_think_id, wrapper.codec_think_bos_id,
         wrapper.english_language_id, wrapper.codec_think_eos_id]
    ], dtype=torch.long, device=device)
    codec_think_embeds = wrapper.codec_embedding(codec_think_ids)
    v9_hidden_states[:, 3:7, :] = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds

    # Position 7: tts_pad + Speaker
    v9_hidden_states[:, 7:8, :] = tts_pad_embed + speaker_embed_t.unsqueeze(1)

    # Position 8: tts_bos + codec_pad
    codec_pad_embed = wrapper.codec_embedding(
        torch.tensor([[wrapper.codec_pad_id]], dtype=torch.long, device=device)
    )
    v9_hidden_states[:, 8:9, :] = tts_bos_embed + codec_pad_embed

    # Position 9+: Text tokens + codec_pad
    all_text_embed = wrapper.text_projection(wrapper.text_embedding(text_ids))
    codec_pad_for_text = wrapper.codec_embedding(
        torch.full((batch_size, MAX_TEXT_LENGTH), wrapper.codec_pad_id, dtype=torch.long, device=device)
    )
    v9_hidden_states[:, 9:9+MAX_TEXT_LENGTH, :] = all_text_embed + codec_pad_for_text

    # Scatter eos and bos to correct positions
    eos_embed = tts_eos_embed + codec_pad_embed
    codec_bos_embed = wrapper.codec_embedding(
        torch.tensor([[wrapper.codec_bos_id]], dtype=torch.long, device=device)
    )
    bos_embed = tts_pad_embed + codec_bos_embed

    eos_idx = (9 + text_length).view(batch_size, 1, 1).expand(-1, -1, config.hidden_size)
    v9_hidden_states.scatter_(1, eos_idx, eos_embed)

    bos_idx = (10 + text_length).view(batch_size, 1, 1).expand(-1, -1, config.hidden_size)
    v9_hidden_states.scatter_(1, bos_idx, bos_embed)

    # Extract actual sequence
    actual_len = text_len + 11
    v9_inputs_embeds = v9_hidden_states[:, :actual_len, :]

    print(f"V9 inputs_embeds shape: {v9_inputs_embeds.shape}")

# Compare position by position
print("\n=== Comparing positions ===")
with torch.no_grad():
    for pos in range(min(official_inputs_embeds.shape[1], v9_inputs_embeds.shape[1])):
        diff = (official_inputs_embeds[0, pos, :] - v9_inputs_embeds[0, pos, :]).abs().max().item()
        status = "SAME" if diff < 1e-5 else f"DIFF={diff:.6f}"
        if pos < 3:
            name = f"role[{pos}]"
        elif pos < 9:
            name = f"think/spk/pad[{pos-3}]"
        elif pos < 9 + text_len:
            name = f"text[{pos-9}]"
        elif pos == 9 + text_len:
            name = "eos"
        else:
            name = "final_bos"
        print(f"  Position {pos:2d} ({name:15s}): {status}")

# Run through model
print("\n=== Running through model ===")
with torch.no_grad():
    # Official model forward
    attention_mask = torch.ones(1, official_inputs_embeds.shape[1], dtype=torch.long)

    # Get position_ids
    position_ids = torch.arange(official_inputs_embeds.shape[1]).unsqueeze(0)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs = talker.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=official_inputs_embeds,
        use_cache=False,
    )

    official_hidden = outputs.last_hidden_state
    official_logits = talker.codec_head(official_hidden[:, -1:, :]).squeeze(1)

    print(f"Official final hidden shape: {official_hidden.shape}")
    print(f"Official logits shape: {official_logits.shape}")

    # Apply suppression and get token
    suppress_mask = np.zeros(config.vocab_size, dtype=bool)
    suppress_mask[2048:] = True
    suppress_mask[config.codec_eos_token_id] = False

    official_logits_np = official_logits.numpy().copy()
    official_logits_np[0, suppress_mask] = -float('inf')
    official_token = int(np.argmax(official_logits_np))
    print(f"Official first token: {official_token}")

# Run V9
print("\n=== Running V9 ===")
with torch.no_grad():
    logits, kv_cache = wrapper(role_ids, text_ids, text_length,
                                tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed_t)
    v9_logits_np = logits.numpy().copy()
    v9_logits_np[0, suppress_mask] = -float('inf')
    v9_token = int(np.argmax(v9_logits_np))
    print(f"V9 first token: {v9_token}")

# Compare logits
print("\n=== Comparing logits ===")
with torch.no_grad():
    logits_diff = (official_logits - logits).abs()
    print(f"Max logits diff: {logits_diff.max().item():.6f}")
    print(f"Mean logits diff: {logits_diff.mean().item():.6f}")

    # Top tokens from each
    official_top5 = np.argsort(official_logits_np[0])[-5:][::-1]
    v9_top5 = np.argsort(v9_logits_np[0])[-5:][::-1]
    print(f"Official top 5: {official_top5.tolist()}")
    print(f"V9 top 5: {v9_top5.tolist()}")
