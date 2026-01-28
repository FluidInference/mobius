# Debug which position has the layer 0 difference
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH, FIXED_SEQ_LEN, apply_rotary_pos_emb_simple

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
processor = tts_model.processor
talker = tts_model.model.talker
config = talker.config

# Use same setup as before
TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]

text = "Hello world, this is a test of the text to speech system."
tokenizer = processor.tokenizer
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)

speaker_embed = np.load("speaker_embedding_official.npy").reshape(1, 1024)
speaker_embed_t = torch.from_numpy(speaker_embed).to(torch.float32)

# Build official inputs_embeds
with torch.no_grad():
    input_text = tts_model._build_assistant_text(text)
    full_input_ids = tts_model._tokenize_texts([input_text])[0]

    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor([[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]], dtype=torch.long)
        )
    ).chunk(3, dim=1)

    role_embed = talker.text_projection(
        talker.get_text_embeddings()(full_input_ids[0, :3].unsqueeze(0))
    )

    language_id = config.codec_language_id["english"]
    codec_prefill_list = [[config.codec_think_id, config.codec_think_bos_id, language_id, config.codec_think_eos_id]]
    codec_input_emebdding_0 = talker.get_input_embeddings()(torch.tensor(codec_prefill_list, dtype=torch.long))
    codec_input_emebdding_1 = talker.get_input_embeddings()(torch.tensor([[config.codec_pad_id, config.codec_bos_id]], dtype=torch.long))
    codec_input_emebdding = torch.cat([codec_input_emebdding_0, speaker_embed_t.view(1, 1, -1), codec_input_emebdding_1], dim=1)

    _talker_input_embed = torch.cat((
        tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
        tts_bos_embed,
    ), dim=1) + codec_input_emebdding[:, :-1]

    talker_input_embed = torch.cat((role_embed, _talker_input_embed), dim=1)
    talker_input_embed = torch.cat([
        talker_input_embed,
        talker.text_projection(talker.get_text_embeddings()(full_input_ids[0, 3:4].unsqueeze(0))) + codec_input_emebdding[:, -1:]
    ], dim=1)
    talker_input_embed = talker_input_embed[:, :-1]

    text_part = full_input_ids[0, 3:-5]
    text_embed = talker.text_projection(talker.get_text_embeddings()(text_part.unsqueeze(0)))
    text_with_eos = torch.cat([text_embed, tts_eos_embed], dim=1)
    codec_pad_for_text = talker.get_input_embeddings()(torch.tensor([[config.codec_pad_id] * (text_part.shape[0] + 1)], dtype=torch.long))
    final_bos = tts_pad_embed + talker.get_input_embeddings()(torch.tensor([[config.codec_bos_id]], dtype=torch.long))

    inputs_embeds = torch.cat([talker_input_embed, text_with_eos + codec_pad_for_text, final_bos], dim=1)
    seq_len = inputs_embeds.shape[1]

print(f"inputs_embeds shape: {inputs_embeds.shape}")

# Run official model
print("\n=== Running official model ===")
with torch.no_grad():
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    outputs_official = talker.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=False,
    )
    official_hidden = outputs_official.last_hidden_state
    print(f"Official hidden shape: {official_hidden.shape}")

# Run V9 wrapper
print("\n=== Running V9 wrapper ===")
with torch.no_grad():
    wrapper = TracablePrefillV9(talker)
    wrapper.eval()

    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
    text_ids = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
    text_ids[0, :text_len] = torch.tensor(text_ids_list)
    text_length = torch.tensor([text_len], dtype=torch.long)

    logits_v9, kv_cache_v9 = wrapper(role_ids, text_ids, text_length,
                                      tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed_t)

# To compare hidden states, I need to get V9's hidden states before the final norm+head
# Let me run V9 manually up to the point of getting hidden states

print("\n=== Comparing hidden states position by position ===")
with torch.no_grad():
    # Build V9 hidden states manually
    batch_size = 1
    device = role_ids.device

    v9_hidden = torch.zeros(batch_size, FIXED_SEQ_LEN, config.hidden_size, device=device, dtype=torch.float32)

    # Position 0-2: Role prefix
    role_embed_v9 = wrapper.text_projection(wrapper.text_embedding(role_ids))
    v9_hidden[:, 0:3, :] = role_embed_v9

    # Position 3-6: think tokens
    codec_think_ids = torch.tensor([[wrapper.codec_think_id, wrapper.codec_think_bos_id, wrapper.english_language_id, wrapper.codec_think_eos_id]], dtype=torch.long)
    codec_think_embeds = wrapper.codec_embedding(codec_think_ids)
    v9_hidden[:, 3:7, :] = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds

    # Position 7: speaker
    v9_hidden[:, 7:8, :] = tts_pad_embed + speaker_embed_t.unsqueeze(1)

    # Position 8: tts_bos + codec_pad
    codec_pad_embed = wrapper.codec_embedding(torch.tensor([[wrapper.codec_pad_id]], dtype=torch.long))
    v9_hidden[:, 8:9, :] = tts_bos_embed + codec_pad_embed

    # Position 9+: Text
    all_text_embed = wrapper.text_projection(wrapper.text_embedding(text_ids))
    codec_pad_for_text_v9 = wrapper.codec_embedding(torch.full((batch_size, MAX_TEXT_LENGTH), wrapper.codec_pad_id, dtype=torch.long))
    v9_hidden[:, 9:9+MAX_TEXT_LENGTH, :] = all_text_embed + codec_pad_for_text_v9

    # Scatter eos and bos
    eos_embed = tts_eos_embed + codec_pad_embed
    codec_bos_embed = wrapper.codec_embedding(torch.tensor([[wrapper.codec_bos_id]], dtype=torch.long))
    bos_embed = tts_pad_embed + codec_bos_embed
    eos_idx = (9 + text_length).view(batch_size, 1, 1).expand(-1, -1, config.hidden_size)
    v9_hidden.scatter_(1, eos_idx, eos_embed)
    bos_idx = (10 + text_length).view(batch_size, 1, 1).expand(-1, -1, config.hidden_size)
    v9_hidden.scatter_(1, bos_idx, bos_embed)

    # Extract actual sequence for comparison
    actual_len = text_len + 11
    v9_inputs = v9_hidden[:, :actual_len, :]

    print(f"V9 inputs shape: {v9_inputs.shape}")

    # Compare input embeddings
    input_diff = (inputs_embeds - v9_inputs).abs()
    print(f"\nInput embeddings diff:")
    for pos in range(actual_len):
        max_diff = input_diff[0, pos, :].max().item()
        if max_diff > 1e-5:
            print(f"  Position {pos}: max_diff = {max_diff:.6f}")

    # Now run through layers using official model but V9's inputs
    print("\n=== Running official model with V9 inputs ===")
    outputs_v9_inputs = talker.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids,
        inputs_embeds=v9_inputs,
        use_cache=False,
    )
    v9_inputs_hidden = outputs_v9_inputs.last_hidden_state
    print(f"V9 inputs hidden shape: {v9_inputs_hidden.shape}")

    # Compare hidden states
    hidden_diff = (official_hidden - v9_inputs_hidden).abs()
    print(f"\nHidden states diff (using V9 inputs through official model):")
    for pos in range(actual_len):
        max_diff = hidden_diff[0, pos, :].max().item()
        if max_diff > 1e-5:
            print(f"  Position {pos}: max_diff = {max_diff:.6f}")

    # Compare logits
    official_logits = talker.codec_head(official_hidden[:, -1:, :]).squeeze(1)
    v9_inputs_logits = talker.codec_head(v9_inputs_hidden[:, -1:, :]).squeeze(1)

    suppress_mask = np.zeros(config.vocab_size, dtype=bool)
    suppress_mask[2048:] = True
    suppress_mask[config.codec_eos_token_id] = False

    official_logits_np = official_logits.numpy().copy()
    official_logits_np[0, suppress_mask] = -float('inf')
    official_token = int(np.argmax(official_logits_np))

    v9_inputs_logits_np = v9_inputs_logits.numpy().copy()
    v9_inputs_logits_np[0, suppress_mask] = -float('inf')
    v9_inputs_token = int(np.argmax(v9_inputs_logits_np))

    print(f"\nOfficial token: {official_token}")
    print(f"V9 inputs token (through official model): {v9_inputs_token}")

    logits_v9_np = logits_v9.numpy().copy()
    logits_v9_np[0, suppress_mask] = -float('inf')
    v9_wrapper_token = int(np.argmax(logits_v9_np))
    print(f"V9 wrapper token: {v9_wrapper_token}")
