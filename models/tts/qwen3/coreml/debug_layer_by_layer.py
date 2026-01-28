# Debug layer by layer divergence
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH, FIXED_SEQ_LEN, apply_rotary_pos_emb_simple

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
processor = tts_model.processor
talker = tts_model.model.talker
config = talker.config

wrapper = TracablePrefillV9(talker)
wrapper.eval()

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

with torch.no_grad():
    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor([[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]], dtype=torch.long)
        )
    ).chunk(3, dim=1)

# Build V9 hidden states (exactly matching official)
batch_size = 1
device = torch.device('cpu')

with torch.no_grad():
    v9_hidden = torch.zeros(batch_size, FIXED_SEQ_LEN, config.hidden_size, device=device, dtype=torch.float32)

    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
    role_embed = wrapper.text_projection(wrapper.text_embedding(role_ids))
    v9_hidden[:, 0:3, :] = role_embed

    codec_think_ids = torch.tensor([[wrapper.codec_think_id, wrapper.codec_think_bos_id, wrapper.english_language_id, wrapper.codec_think_eos_id]], dtype=torch.long)
    codec_think_embeds = wrapper.codec_embedding(codec_think_ids)
    v9_hidden[:, 3:7, :] = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds

    v9_hidden[:, 7:8, :] = tts_pad_embed + speaker_embed_t.unsqueeze(1)

    codec_pad_embed = wrapper.codec_embedding(torch.tensor([[wrapper.codec_pad_id]], dtype=torch.long))
    v9_hidden[:, 8:9, :] = tts_bos_embed + codec_pad_embed

    text_ids = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
    text_ids[0, :text_len] = torch.tensor(text_ids_list)
    all_text_embed = wrapper.text_projection(wrapper.text_embedding(text_ids))
    codec_pad_for_text = wrapper.codec_embedding(torch.full((batch_size, MAX_TEXT_LENGTH), wrapper.codec_pad_id, dtype=torch.long))
    v9_hidden[:, 9:9+MAX_TEXT_LENGTH, :] = all_text_embed + codec_pad_for_text

    eos_embed = tts_eos_embed + codec_pad_embed
    codec_bos_embed = wrapper.codec_embedding(torch.tensor([[wrapper.codec_bos_id]], dtype=torch.long))
    bos_embed = tts_pad_embed + codec_bos_embed
    text_length = torch.tensor([text_len], dtype=torch.long)
    eos_idx = (9 + text_length).view(batch_size, 1, 1).expand(-1, -1, config.hidden_size)
    v9_hidden.scatter_(1, eos_idx, eos_embed)
    bos_idx = (10 + text_length).view(batch_size, 1, 1).expand(-1, -1, config.hidden_size)
    v9_hidden.scatter_(1, bos_idx, bos_embed)

    actual_len = text_length + 11  # 25

# Build mask
with torch.no_grad():
    pos_1d = torch.arange(FIXED_SEQ_LEN, device=device)
    position_ids = pos_1d.unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    cos, sin = wrapper.rotary_emb(v9_hidden, position_ids)

    causal_mask = torch.triu(
        torch.ones(FIXED_SEQ_LEN, FIXED_SEQ_LEN, dtype=torch.float32, device=device),
        diagonal=1
    )
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    q_pos = torch.arange(FIXED_SEQ_LEN, device=device).view(1, 1, FIXED_SEQ_LEN, 1)
    k_pos = torch.arange(FIXED_SEQ_LEN, device=device).view(1, 1, 1, FIXED_SEQ_LEN)
    actual_len_expanded = actual_len.view(batch_size, 1, 1, 1)

    padding_mask = torch.where(
        k_pos >= actual_len_expanded,
        torch.tensor(float("-inf"), dtype=torch.float32, device=device),
        torch.tensor(0.0, dtype=torch.float32, device=device),
    )

    combined_mask = causal_mask + padding_mask
    combined_mask = combined_mask.expand(batch_size, 1, FIXED_SEQ_LEN, FIXED_SEQ_LEN)

# Prepare official model hooks
layer_outputs_official = []
def make_hook(idx):
    def hook_fn(module, input, output):
        layer_outputs_official.append(output[0].clone())
    return hook_fn

handles = []
for idx, layer in enumerate(talker.model.layers):
    handles.append(layer.register_forward_hook(make_hook(idx)))

# Run official model
with torch.no_grad():
    v9_inputs = v9_hidden[:, :actual_len.item(), :]
    seq_len_actual = actual_len.item()

    attention_mask = torch.ones(1, seq_len_actual, dtype=torch.long)
    position_ids_official = torch.arange(seq_len_actual).unsqueeze(0)
    position_ids_official = position_ids_official.unsqueeze(0).expand(3, -1, -1)

    outputs_official = talker.model(
        input_ids=None,
        attention_mask=attention_mask,
        position_ids=position_ids_official,
        inputs_embeds=v9_inputs,
        use_cache=False,
    )

for handle in handles:
    handle.remove()

print(f"Captured {len(layer_outputs_official)} layer outputs from official model")

# Run V9 layer by layer and compare
print("\n=== Layer by layer comparison ===")
with torch.no_grad():
    hidden_states = v9_hidden.clone()
    num_heads = wrapper.num_heads
    num_kv_heads = wrapper.num_kv_heads
    head_dim = wrapper.head_dim

    for layer_idx, layer in enumerate(wrapper.layers):
        residual = hidden_states
        hidden_states_normed = layer.input_layernorm(hidden_states)

        q = layer.self_attn.q_proj(hidden_states_normed)
        k = layer.self_attn.k_proj(hidden_states_normed)
        v = layer.self_attn.v_proj(hidden_states_normed)

        q = q.view(batch_size, FIXED_SEQ_LEN, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, FIXED_SEQ_LEN, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, FIXED_SEQ_LEN, num_kv_heads, head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb_simple(q, k, cos, sin)

        n_rep = num_heads // num_kv_heads
        k_expanded = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
        k_expanded = k_expanded.reshape(batch_size, num_heads, FIXED_SEQ_LEN, head_dim)
        v_expanded = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
        v_expanded = v_expanded.reshape(batch_size, num_heads, FIXED_SEQ_LEN, head_dim)

        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = attn_weights + combined_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_expanded)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, FIXED_SEQ_LEN, -1)
        attn_output = layer.self_attn.o_proj(attn_output)

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Compare with official
        v9_pos24 = hidden_states[0, 24, :]
        official_pos24 = layer_outputs_official[layer_idx][0, 24, :]
        diff = (v9_pos24 - official_pos24).abs().max().item()

        if layer_idx < 5 or diff > 0.1:
            print(f"Layer {layer_idx:2d}: max diff at pos 24 = {diff:.6f}")

print("\n=== Checking if first divergence is in attention vs MLP ===")
# Re-run layer 0 with more detailed comparison
with torch.no_grad():
    hidden_states = v9_hidden.clone()
    layer = wrapper.layers[0]

    residual = hidden_states
    hidden_states_normed = layer.input_layernorm(hidden_states)

    q = layer.self_attn.q_proj(hidden_states_normed)
    k = layer.self_attn.k_proj(hidden_states_normed)
    v = layer.self_attn.v_proj(hidden_states_normed)

    q = q.view(batch_size, FIXED_SEQ_LEN, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, FIXED_SEQ_LEN, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, FIXED_SEQ_LEN, num_kv_heads, head_dim).transpose(1, 2)

    q, k = apply_rotary_pos_emb_simple(q, k, cos, sin)

    n_rep = num_heads // num_kv_heads
    k_expanded = k.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
    k_expanded = k_expanded.reshape(batch_size, num_heads, FIXED_SEQ_LEN, head_dim)
    v_expanded = v.unsqueeze(2).expand(-1, -1, n_rep, -1, -1)
    v_expanded = v_expanded.reshape(batch_size, num_heads, FIXED_SEQ_LEN, head_dim)

    attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights_raw = attn_weights.clone()
    attn_weights = attn_weights + combined_mask
    attn_weights = torch.softmax(attn_weights, dim=-1)

    # Check softmax at position 24
    attn_24 = attn_weights[0, 0, 24, :]  # Head 0, position 24
    sum_first_25 = attn_24[:25].sum().item()
    sum_rest = attn_24[25:].sum().item()
    print(f"\nSoftmax at position 24, head 0:")
    print(f"  Sum of positions 0-24: {sum_first_25:.10f}")
    print(f"  Sum of positions 25+: {sum_rest:.10f}")

    # Check if attention to padded positions is truly zero
    max_attn_to_padding = attn_24[25:].max().item()
    print(f"  Max attention to padding positions: {max_attn_to_padding}")

    attn_output = torch.matmul(attn_weights, v_expanded)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, FIXED_SEQ_LEN, -1)
    attn_output = layer.self_attn.o_proj(attn_output)

    hidden_after_attn = residual + attn_output

    print(f"\nV9 hidden after attn at position 24[:5]: {hidden_after_attn[0, 24, :5].tolist()}")

    # Now compare with official layer 0 output
    official_after_layer0 = layer_outputs_official[0][0, 24, :]
    v9_after_layer0 = hidden_states[0, 24, :]  # Wait, I need to complete the MLP

    residual = hidden_after_attn
    hidden_mlp = layer.post_attention_layernorm(hidden_after_attn)
    hidden_mlp = layer.mlp(hidden_mlp)
    hidden_after_layer0_v9 = residual + hidden_mlp

    print(f"V9 hidden after layer 0 at position 24[:5]: {hidden_after_layer0_v9[0, 24, :5].tolist()}")
    print(f"Official after layer 0 at position 24[:5]: {official_after_layer0[:5].tolist()}")
