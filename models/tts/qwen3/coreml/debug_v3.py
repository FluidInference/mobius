# Debug V3 prefill
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
from convert_lm_v3 import TracablePrefillV3, MAX_TEXT_LENGTH

print("Loading model...")
model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = model.model.talker
processor = model.processor

wrapper = TracablePrefillV3(talker)
wrapper.eval()

text = "Hello world"
inputs = processor(text=text, return_tensors="pt")
text_ids = inputs.input_ids
text_len = text_ids.shape[1]
print(f"Text: '{text}', tokens: {text_ids[0].tolist()}, length: {text_len}")

# Pad
padded = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
padded[0, :text_len] = text_ids[0]
text_length = torch.tensor([text_len])

print(f"\nPadded shape: {padded.shape}")
print(f"Text length: {text_length}")

with torch.no_grad():
    # Check intermediate values
    print("\n=== Debugging intermediate values ===")

    # Embeddings
    text_embed = wrapper.text_embedding(padded)
    text_projected = wrapper.text_projection(text_embed)
    print(f"Text projected: {text_projected.shape}, has nan: {torch.isnan(text_projected).any()}")

    lang_ids = torch.full((1, 1), wrapper.language_id, dtype=torch.long)
    bos_ids = torch.full((1, 1), wrapper.bos_id, dtype=torch.long)
    lang_embed = wrapper.codec_embedding(lang_ids)
    bos_embed = wrapper.codec_embedding(bos_ids)

    hidden_states = torch.cat([lang_embed, text_projected, bos_embed], dim=1)
    seq_len = hidden_states.shape[1]
    print(f"Hidden states: {hidden_states.shape}")

    # Position embeddings
    pos_1d = torch.arange(seq_len)
    position_ids = pos_1d.unsqueeze(0).expand(1, -1).unsqueeze(0).expand(3, -1, -1)
    cos, sin = wrapper.rotary_emb(hidden_states, position_ids)
    print(f"Cos: {cos.shape}, Sin: {sin.shape}")

    # Masks
    actual_len = text_length + 2  # [B]
    print(f"Actual len: {actual_len}")

    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    print(f"Causal mask: {causal_mask.shape}")

    # Padding mask
    q_pos = torch.arange(seq_len).view(1, 1, seq_len, 1)
    k_pos = torch.arange(seq_len).view(1, 1, 1, seq_len)
    actual_len_expanded = actual_len.view(1, 1, 1, 1)

    padding_mask = (k_pos >= actual_len_expanded).float() * float("-inf")
    print(f"Padding mask shape: {padding_mask.shape}")
    print(f"Padding mask (first row of last position): {padding_mask[0, 0, -1, :10]}")
    print(f"Padding mask (last row valid positions): {padding_mask[0, 0, -1, actual_len[0]-3:actual_len[0]+2]}")

    combined_mask = causal_mask + padding_mask
    print(f"Combined mask shape: {combined_mask.shape}")
    print(f"Combined mask has inf: {torch.isinf(combined_mask).any()}")

    # Check a few positions
    bos_pos = actual_len[0].item() - 1
    print(f"\nBOS position: {bos_pos}")
    print(f"Mask at BOS position (valid keys): {combined_mask[0, 0, bos_pos, :5]}")
    print(f"Mask at BOS position (around actual_len): {combined_mask[0, 0, bos_pos, actual_len[0]-2:actual_len[0]+2]}")

    # The BOS position should only see positions 0 to actual_len-1
    # Let's check if the mask is correct
    print(f"\nExpected: BOS at {bos_pos} should see positions 0 to {bos_pos}")
    for k in range(min(seq_len, 10)):
        val = combined_mask[0, 0, bos_pos, k].item()
        expected = 0.0 if k <= bos_pos else float("-inf")
        print(f"  k={k}: mask={val:.1f}, expected={expected:.1f}, match={val == expected}")

    # Run forward
    logits, kv = wrapper(padded, text_length)
    print(f"\nLogits: {logits.shape}")
    print(f"Logits has nan: {torch.isnan(logits).any()}")
    print(f"Logits has inf: {torch.isinf(logits).any()}")
    print(f"Logits min/max: {logits.min():.2f} / {logits.max():.2f}")

    if not torch.isnan(logits).any():
        print(f"First token: {torch.argmax(logits).item()}")
