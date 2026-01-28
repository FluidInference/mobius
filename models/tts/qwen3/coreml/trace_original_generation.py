# Trace the original Code Predictor generation to understand exact flow
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

print("\n=== Tracing Original Generation ===")

# Create test codebook 0 tokens
codebook0 = torch.randint(0, 2048, (1, 10))
print(f"Codebook 0 input: {codebook0[0, :5].tolist()}...")

# The original model uses HuggingFace generate() which does autoregressive generation
# Let me trace step by step what happens

with torch.no_grad():
    # For prefill, we need inputs_embeds
    # The format is: inputs_embeds includes [hidden_states, codebook0_embed, ...]

    # Looking at the talker source, the prefill is built as:
    # - LM hidden states (optional, for conditioning)
    # - Codebook 0 embeddings

    # For the code predictor, get_input_embeddings() returns codec_embedding
    # codec_embedding has 15 layers (indices 0-14)

    # For prefill with just codebook 0:
    # We need to use some embedding for codebook 0
    # But the code predictor's embeddings are for codebooks 1-15 (in decode mode)

    # Let me check what the talker actually passes to code_predictor

    # From talker source:
    # sub_talker_inputs_embeds = [hidden_states_for_subtaker]  # LM hidden
    # for i in range(1, self.config.num_code_groups):  # i = 1 to 15
    #     sub_talker_inputs_embeds.append(
    #         self.code_predictor.get_input_embeddings()[i-1](codec_ids[:, i:i+1])
    #     )

    # So for codebook i (1-15), it uses embed[i-1]
    # This means embed[0] is for codebook 1, embed[14] is for codebook 15

    # For codebook 0, the talker uses hidden_states_for_subtaker (LM hidden states)
    # NOT an embedding lookup!

    print("\n=== Understanding the prefill format ===")
    print("The code predictor prefill uses:")
    print("  - inputs_embeds[:, 0, :] = LM hidden states (projected)")
    print("  - inputs_embeds[:, 1, :] = embed[0](codebook 1 tokens)")
    print("  - inputs_embeds[:, 2, :] = embed[1](codebook 2 tokens)")
    print("  - ...")
    print("")
    print("For generation from codebook 0 only:")
    print("  - inputs_embeds[:, 0, :] = LM hidden states")
    print("  - inputs_embeds[:, 1, :] = codec_embedding_for_codebook_0(codebook0)")
    print("")
    print("But wait - there's no codec_embedding_for_codebook_0 in the code predictor!")

    # Let me check the main talker's embedding
    talker = model.model.talker
    print(f"\nTalker has codec_embedding: {hasattr(talker.model, 'codec_embedding')}")

    # The talker has its own codec_embedding for the main LM
    # The code predictor has separate codec_embedding for autoregressive prediction

    # For the TTS pipeline:
    # 1. Main LM generates codebook 0 tokens + hidden states
    # 2. Code predictor takes hidden states + codebook 0 to generate codebooks 1-15

    # The hidden states from main LM are projected and used as the first "embedding"
    # Then codebook 0 needs to be embedded... but with which embedding?

    # Looking at the code_predictor forward more carefully:
    # In prefill mode, inputs_embeds is provided externally
    # The generation_steps = inputs_embeds.shape[1] - 2

    # If inputs_embeds has shape [B, 2, hidden]:
    # - inputs_embeds[:, 0, :] = hidden states
    # - inputs_embeds[:, 1, :] = codebook 0 embedding (from talker's embedding)
    # - generation_steps = 2 - 2 = 0
    # - Uses lm_head[0] to predict codebook 1

    # So for prefill, the codebook 0 embedding comes from the TALKER's codec_embedding,
    # not the code_predictor's codec_embedding!

    print("\n=== The key insight ===")
    print("For prefill:")
    print("  - Codebook 0 is embedded using TALKER's codec_embedding")
    print("  - Not the code_predictor's codec_embedding!")
    print("")
    print("Let me check the talker's codec_embedding:")

    talker_codec_emb = talker.model.codec_embedding
    print(f"Talker codec_embedding: {talker_codec_emb}")

    # The talker's codec_embedding is a single Embedding, not a ModuleList
    # It embeds ALL codebook tokens (0-15) into the main LM's hidden space

    # The code_predictor's codec_embedding is a ModuleList of 15 Embeddings
    # Each one embeds a specific codebook's tokens

    print(f"\nCode predictor codec_embedding: {code_predictor.model.codec_embedding}")
    print(f"Number of embeddings: {len(code_predictor.model.codec_embedding)}")

    # So the architecture is:
    # - Talker's codec_embedding: [vocab_size, talker_hidden_dim] - embeds any codebook token
    # - Code predictor's codec_embedding[N]: [vocab_size, cp_hidden_dim] - embeds codebook N+1 tokens

    # For prefill, the talker embeds codebook 0 with its own embedding
    # Then this is projected to the code predictor's hidden dimension

    # Let me check the projection
    print(f"\nCode predictor small_to_mtp_projection: {code_predictor.small_to_mtp_projection}")

    # It's an Identity! So there's no projection, the dimensions must match

    print(f"\nTalker hidden size: {talker.config.hidden_size}")
    print(f"Code predictor hidden size: {config.hidden_size}")

    # Both are 1024! So no projection needed.

    # For my CoreML wrapper, I need to:
    # 1. Use talker's codec_embedding for codebook 0
    # 2. Use code_predictor's codec_embedding[N-1] for codebook N (N >= 1)

    # But I'm trying to convert just the code_predictor...
    # Let me check if I can use the code_predictor's embeddings for all codebooks

    print("\n=== Simplified approach ===")
    print("For CoreML, I'll use a simpler approach:")
    print("  - Step 0: Use code_predictor.codec_embedding[0](codebook0) → lm_head[0]")
    print("    This uses embed[0] for codebook 0, which is trained for codebook 1")
    print("    It's not perfect but should approximate the behavior")
    print("")
    print("  - Step N (1-14): Use embed[N-1](codebook_N) → lm_head[N]")
    print("    This matches the original decode mode exactly")

    # Let me verify this by comparing outputs

    print("\n=== Verification ===")

    # Step 0: My approach vs original prefill
    # My approach: embed[0](codebook0) → transformer → lm_head[0]
    step0_embed = code_predictor.model.codec_embedding[0](codebook0)
    print(f"Step 0 embed shape: {step0_embed.shape}")

    # Run through transformer
    # Need to use the code_predictor's forward path
    # But we can't easily extract intermediate results

    # Let's compare with original gen_steps=1 which uses embed[0]
    # Original gen_steps=1: embed[0](codebook1) → lm_head[1]
    # If we use the same input (codebook0 as if it were codebook1):
    output_gs1 = code_predictor(input_ids=codebook0, generation_steps=1)
    print(f"Original gen_steps=1 logits shape: {output_gs1.logits.shape}")

    # This uses embed[0] on codebook0 and lm_head[1]
    # My step 1 would do the same: embed[0](codebook1) → lm_head[1]
    # So for step 1+, my wrapper should match!

    # The only difference is step 0:
    # - My step 0: embed[0](codebook0) → lm_head[0]
    # - Original prefill: different embedding + lm_head[0]

    # Let me check if using the talker's codec_embedding for codebook 0 makes a difference
    codebook0_talker_embed = talker.model.codec_embedding(codebook0)
    print(f"Talker codebook0 embed shape: {codebook0_talker_embed.shape}")

    # Compare with code_predictor's embed[0]
    diff = (codebook0_talker_embed - step0_embed).abs().max().item()
    print(f"Diff between talker embed and CP embed[0]: {diff:.6f}")

    # They're different! The talker has its own embedding space.

    print("\n=== Solution ===")
    print("For step 0, I need to use the talker's codec_embedding, not the code_predictor's")
    print("Let me add the talker's codec_embedding to my wrapper")
