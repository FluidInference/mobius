# Debug transformer implementation
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "./model_0.6b",
    device_map="cpu",
    torch_dtype=torch.float32,
)

code_predictor = model.model.talker.code_predictor
config = code_predictor.config

print("\n=== Comparing single layer forward ===")

# Create test input
batch_size = 1
seq_len = 5
hidden_dim = config.hidden_size

test_input = torch.randn(batch_size, seq_len, hidden_dim)

# Run through original model's layers
with torch.no_grad():
    # Get first layer
    layer = code_predictor.model.layers[0]

    # Get rotary embeddings from the model
    rotary_emb = code_predictor.model.rotary_emb
    position_ids = torch.arange(seq_len).unsqueeze(0)
    position_embeddings = rotary_emb(test_input, position_ids)

    # Run the layer
    original_output = layer(
        test_input,
        position_embeddings=position_embeddings,
    )
    print(f"Original layer output shape: {original_output[0].shape}")

    # Now test with our wrapper's implementation
    from convert_code_predictor import TracableCodePredictor

    wrapper = TracableCodePredictor(code_predictor, config)
    wrapper.eval()

    # Create causal mask
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    # Get RoPE embeddings
    cos = wrapper.cos_cached[:, :, :seq_len, :]
    sin = wrapper.sin_cached[:, :, :seq_len, :]

    # Run wrapper's layer
    wrapper_output = wrapper._run_layer(layer, test_input, causal_mask, cos, sin)
    print(f"Wrapper layer output shape: {wrapper_output.shape}")

    # Compare
    diff = (original_output[0] - wrapper_output).abs().max().item()
    print(f"Max diff: {diff:.6f}")

    corr = np.corrcoef(
        original_output[0].numpy().flatten(),
        wrapper_output.numpy().flatten()
    )[0, 1]
    print(f"Correlation: {corr:.6f}")

# If layer output matches, the issue is elsewhere
# Let's also check if the embeddings are correct
print("\n=== Checking embedding output ===")

test_tokens = torch.randint(0, 2048, (1, 5))
with torch.no_grad():
    # Original embedding
    orig_embed = code_predictor.model.codec_embedding[0](test_tokens)
    print(f"Original embed shape: {orig_embed.shape}")

    # Wrapper embedding
    wrap_embed = wrapper.embeddings[0](test_tokens)
    print(f"Wrapper embed shape: {wrap_embed.shape}")

    diff = (orig_embed - wrap_embed).abs().max().item()
    print(f"Embed diff: {diff:.6f}")

# Check lm_head
print("\n=== Checking lm_head output ===")
test_hidden = torch.randn(1, 5, hidden_dim)
with torch.no_grad():
    orig_logits = code_predictor.lm_head[0](test_hidden)
    wrap_logits = wrapper.lm_heads[0](test_hidden)

    diff = (orig_logits - wrap_logits).abs().max().item()
    print(f"LM head diff: {diff:.6f}")

# The issue might be in how the original model processes things
# Let's trace through the full forward
print("\n=== Tracing full forward ===")

with torch.no_grad():
    # Use gen_steps=0 which uses embed[-1] (wrap around to embed[14])
    test_tokens = torch.randint(0, 2048, (1, 5))

    # Original forward
    output = code_predictor(input_ids=test_tokens, generation_steps=0)
    print(f"Original logits shape: {output.logits.shape}")

    # What embedding does gen_steps=0 use?
    # From code: inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)
    # gen_steps=0 → embed[-1] = embed[14]!
    print(f"gen_steps=0 uses embed[{0-1}] = embed[14]!")

    # So the original is embedding with the LAST embedding layer, not the first!
    # Let me run wrapper with embed[14] to compare
    embed14_input = wrapper.embeddings[14](test_tokens)
    wrapper_hidden = wrapper._run_transformer(embed14_input)
    wrapper_logits = wrapper.lm_heads[0](wrapper_hidden)

    diff = (output.logits - wrapper_logits).abs().max().item()
    print(f"Logits diff (using embed[14]): {diff:.6f}")

    corr = np.corrcoef(
        output.logits.numpy().flatten(),
        wrapper_logits.numpy().flatten()
    )[0, 1]
    print(f"Correlation (using embed[14]): {corr:.6f}")

print("\n=== Understanding gen_steps ===")
print("gen_steps=N uses:")
print("  - embed[N-1] (wraps for N=0 → embed[-1] = embed[14])")
print("  - lm_head[N]")
print("")
print("For proper generation flow:")
print("  - gen_steps=1: embed[0] + lm_head[1] (predicts codebook 2)")
print("  - gen_steps=2: embed[1] + lm_head[2] (predicts codebook 3)")
print("  - ...")
print("  - gen_steps=14: embed[13] + lm_head[14] (predicts codebook 15)")
print("")
print("So what predicts codebook 1?")
print("  - gen_steps=0 is broken (uses embed[14])")
print("  - Must use prefill mode with inputs_embeds!")
