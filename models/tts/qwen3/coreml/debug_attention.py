# Debug attention differences between V9 and official model
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH, FIXED_SEQ_LEN

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
processor = tts_model.processor
talker = tts_model.model.talker
config = talker.config

# Use a simple test
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

# Build the same inputs_embeds
print("\n=== Building inputs_embeds ===")
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

    codec_input_emebdding = torch.cat([
        codec_input_emebdding_0,
        speaker_embed_t.view(1, 1, -1),
        codec_input_emebdding_1
    ], dim=1)

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

    inputs_embeds = torch.cat([
        talker_input_embed,
        text_with_eos + codec_pad_for_text,
        final_bos
    ], dim=1)

    print(f"inputs_embeds shape: {inputs_embeds.shape}")
    seq_len = inputs_embeds.shape[1]

# Compare position_ids
print("\n=== Comparing position_ids ===")
with torch.no_grad():
    # Official position_ids (from model code)
    official_position_ids = torch.arange(seq_len).unsqueeze(0)
    official_position_ids = official_position_ids.unsqueeze(0).expand(3, -1, -1)
    print(f"Official position_ids shape: {official_position_ids.shape}")
    print(f"Official position_ids[0,0,:5]: {official_position_ids[0,0,:5].tolist()}")

    # V9 position_ids
    pos_1d = torch.arange(FIXED_SEQ_LEN)
    v9_position_ids = pos_1d.unsqueeze(0)
    v9_position_ids = v9_position_ids.unsqueeze(0).expand(3, -1, -1)
    print(f"V9 position_ids shape: {v9_position_ids.shape}")
    print(f"V9 position_ids[0,0,:5]: {v9_position_ids[0,0,:5].tolist()}")

# Compare rotary embeddings
print("\n=== Comparing rotary embeddings ===")
with torch.no_grad():
    # Official rotary
    cos_off, sin_off = talker.model.rotary_emb(inputs_embeds, official_position_ids)
    print(f"Official cos shape: {cos_off.shape}")
    print(f"Official cos[0,0,:5]: {cos_off[0,0,:5].tolist()}")

    # V9 uses a fixed-length sequence for rotary
    v9_hidden = torch.zeros(1, FIXED_SEQ_LEN, config.hidden_size, dtype=torch.float32)
    cos_v9, sin_v9 = talker.model.rotary_emb(v9_hidden, v9_position_ids)
    print(f"V9 cos shape: {cos_v9.shape}")
    print(f"V9 cos[0,0,:5]: {cos_v9[0,0,:5].tolist()}")

    # Compare first seq_len positions (dimension order is [3, batch, seq_len, head_dim])
    cos_diff = (cos_off - cos_v9[:, :, :seq_len, :]).abs().max().item()
    sin_diff = (sin_off - sin_v9[:, :, :seq_len, :]).abs().max().item()
    print(f"Cos diff (first {seq_len} positions): {cos_diff}")
    print(f"Sin diff (first {seq_len} positions): {sin_diff}")

# Compare attention mask
print("\n=== Comparing attention mask ===")
with torch.no_grad():
    # Official model uses attention_mask (all 1s for our case)
    official_attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    print(f"Official attention_mask shape: {official_attention_mask.shape}")

    # V9 builds a causal + padding mask
    # Causal mask
    causal_mask = torch.triu(
        torch.ones(FIXED_SEQ_LEN, FIXED_SEQ_LEN, dtype=torch.float32),
        diagonal=1
    )
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    # Padding mask (for V9's fixed length)
    actual_len = torch.tensor([seq_len], dtype=torch.long)
    q_pos = torch.arange(FIXED_SEQ_LEN).view(1, 1, FIXED_SEQ_LEN, 1)
    k_pos = torch.arange(FIXED_SEQ_LEN).view(1, 1, 1, FIXED_SEQ_LEN)
    actual_len_expanded = actual_len.view(1, 1, 1, 1)

    padding_mask = torch.where(
        k_pos >= actual_len_expanded,
        torch.tensor(float("-inf"), dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
    )

    v9_combined_mask = causal_mask + padding_mask
    print(f"V9 combined_mask shape: {v9_combined_mask.shape}")
    print(f"V9 mask[0,0,0,:10]: {v9_combined_mask[0,0,0,:10].tolist()}")
    print(f"V9 mask[0,0,5,:10]: {v9_combined_mask[0,0,5,:10].tolist()}")

# Run through first layer only to compare
print("\n=== Comparing first layer output ===")
with torch.no_grad():
    layer = talker.model.layers[0]

    # Official forward through first layer
    # The model's internal forward builds attention mask differently
    # Let me run the official model layer by layer

    # First, run layer norm
    residual = inputs_embeds
    hidden_states_off = layer.input_layernorm(inputs_embeds)

    # Get Q, K, V
    q_off = layer.self_attn.q_proj(hidden_states_off)
    k_off = layer.self_attn.k_proj(hidden_states_off)
    v_off = layer.self_attn.v_proj(hidden_states_off)

    print(f"Q shape: {q_off.shape}")
    print(f"K shape: {k_off.shape}")

    # Reshape for attention
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    q_off = q_off.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    k_off = k_off.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v_off = v_off.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    print(f"Q reshaped: {q_off.shape}")
    print(f"K reshaped: {k_off.shape}")

    # Apply rotary
    from convert_lm_prefill_v9 import apply_rotary_pos_emb_simple
    q_rotated, k_rotated = apply_rotary_pos_emb_simple(q_off, k_off, cos_off, sin_off)

    print(f"Q rotated[0,0,0,:5]: {q_rotated[0,0,0,:5].tolist()}")

    # GQA expansion
    n_rep = num_heads // num_kv_heads
    k_expanded = k_rotated.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
    k_expanded = k_expanded.reshape(1, num_heads, seq_len, head_dim)
    v_expanded = v_off.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
    v_expanded = v_expanded.reshape(1, num_heads, seq_len, head_dim)

    # Attention weights
    attn_weights = torch.matmul(q_rotated, k_expanded.transpose(-2, -1)) / (head_dim ** 0.5)
    print(f"Attn weights before mask: {attn_weights[0,0,0,:5].tolist()}")

    # Apply causal mask only (official model builds it internally)
    causal_only = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.float32),
        diagonal=1
    )
    causal_only = causal_only.masked_fill(causal_only == 1, float("-inf"))
    causal_only = causal_only.unsqueeze(0).unsqueeze(0)

    attn_weights_masked = attn_weights + causal_only
    attn_weights_softmax = torch.softmax(attn_weights_masked, dim=-1)
    print(f"Attn weights after softmax: {attn_weights_softmax[0,0,0,:5].tolist()}")

    attn_output = torch.matmul(attn_weights_softmax, v_expanded)
    attn_output = attn_output.transpose(1, 2).reshape(1, seq_len, -1)
    attn_output = layer.self_attn.o_proj(attn_output)

    hidden_after_attn = residual + attn_output

    # MLP
    residual = hidden_after_attn
    hidden_states_mlp = layer.post_attention_layernorm(hidden_after_attn)
    hidden_states_mlp = layer.mlp(hidden_states_mlp)
    hidden_after_layer = residual + hidden_states_mlp

    print(f"Hidden after layer 0: {hidden_after_layer[0,0,:5].tolist()}")

    # Now compare with official model's layer 0 output
    # We need to hook into the model to get intermediate outputs
    layer0_output = [None]
    def hook_fn(module, input, output):
        layer0_output[0] = output[0].clone()

    handle = layer.register_forward_hook(hook_fn)

    # Run official forward
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    outputs = talker.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=official_position_ids,
        inputs_embeds=inputs_embeds,
        use_cache=False,
    )
    handle.remove()

    print(f"Official layer 0 output: {layer0_output[0][0,0,:5].tolist()}")

    diff = (hidden_after_layer - layer0_output[0]).abs().max().item()
    print(f"Diff: {diff}")
