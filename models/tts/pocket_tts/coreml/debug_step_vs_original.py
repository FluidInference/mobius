"""Debug script to compare step model vs original model frame by frame."""
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_flowlm_step import TraceableFlowLMStep


def debug_step_vs_original():
    """Compare step model to original model frame by frame."""
    print("Loading models...")
    from pocket_tts import TTSModel
    from pocket_tts.models.tts_model import prepare_text_prompt

    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    step_model = TraceableFlowLMStep.from_flowlm(model.flow_lm, max_seq_len=200)
    step_model.eval()

    # Initialize voice state
    print("\nInitializing voice state...")
    voice_state = model.get_state_for_audio_prompt("alba")
    model._expand_kv_cache(voice_state, sequence_length=200)

    # Process text
    print("Processing text...")
    prepared_text, frames_after_eos = prepare_text_prompt("Hello world")
    tokenized = model.flow_lm.conditioner.prepare(prepared_text)
    text_tokens = tokenized.tokens

    # First call: process text to fill KV cache
    model._run_flow_lm_and_increment_step(model_state=voice_state, text_tokens=text_tokens)

    # Get position after text processing
    layer0_state = voice_state['transformer.layers.0.self_attn']
    position_after_text = len(layer0_state['current_end'])
    print(f"Position after text processing: {position_after_text}")

    # Extract initial state for step model
    print("\nExtracting state for step model...")
    bos_emb = model.flow_lm.bos_emb.data

    caches = []
    positions = []
    for i in range(6):
        key = f'transformer.layers.{i}.self_attn'
        layer_state = voice_state[key]
        cache = layer_state['cache'].clone()
        pos = torch.tensor([float(len(layer_state['current_end']))])
        caches.append(cache)
        positions.append(pos)

    # Now generate a few frames and compare
    print("\n=== Generating frames and comparing ===")
    backbone_input = torch.full((1, 1, 32), float('nan'))

    for frame_idx in range(3):
        print(f"\n--- Frame {frame_idx} ---")

        # Original model forward pass
        with torch.no_grad():
            # Save state before running original model
            orig_state_snapshot = {}
            for i in range(6):
                key = f'transformer.layers.{i}.self_attn'
                layer_state = voice_state[key]
                orig_state_snapshot[key] = {
                    'cache': layer_state['cache'].clone(),
                    'current_end_len': len(layer_state['current_end'])
                }

            # Run original model
            next_latent_orig, is_eos_orig = model._run_flow_lm_and_increment_step(
                model_state=voice_state,
                backbone_input_latents=backbone_input
            )

        # Get transformer output from original (we need to hack this)
        # The flow decoding uses random noise, so we need to compare transformer outputs
        # Let's run the original model's backbone directly

        # For fair comparison, we need to access the transformer output before flow decoding
        # Let's create a version that just returns transformer output

        print(f"  Original model: latent range [{next_latent_orig.min():.4f}, {next_latent_orig.max():.4f}]")
        print(f"  Original EOS: {is_eos_orig.item()}")

        # Step model forward pass (using snapshot state)
        with torch.no_grad():
            step_outputs = step_model(
                backbone_input, bos_emb,
                caches[0], positions[0],
                caches[1], positions[1],
                caches[2], positions[2],
                caches[3], positions[3],
                caches[4], positions[4],
                caches[5], positions[5],
            )

        transformer_out_step = step_outputs[0]
        eos_step = step_outputs[1]

        # Update step model state
        caches[0] = step_outputs[2]
        positions[0] = step_outputs[3]
        caches[1] = step_outputs[4]
        positions[1] = step_outputs[5]
        caches[2] = step_outputs[6]
        positions[2] = step_outputs[7]
        caches[3] = step_outputs[8]
        positions[3] = step_outputs[9]
        caches[4] = step_outputs[10]
        positions[4] = step_outputs[11]
        caches[5] = step_outputs[12]
        positions[5] = step_outputs[13]

        print(f"  Step model: transformer_out range [{transformer_out_step.min():.4f}, {transformer_out_step.max():.4f}]")
        print(f"  Step EOS logit: {eos_step.item():.4f}")
        print(f"  New position: {positions[0].item()}")

        # Update backbone input for next frame
        backbone_input = next_latent_orig

    # Now let's do a more direct comparison
    print("\n\n=== Direct backbone comparison ===")

    # Reset voice state
    voice_state = model.get_state_for_audio_prompt("alba")
    model._expand_kv_cache(voice_state, sequence_length=200)
    model._run_flow_lm_and_increment_step(model_state=voice_state, text_tokens=text_tokens)

    # Reset step model state
    caches = []
    positions = []
    for i in range(6):
        key = f'transformer.layers.{i}.self_attn'
        layer_state = voice_state[key]
        cache = layer_state['cache'].clone()
        pos = torch.tensor([float(len(layer_state['current_end']))])
        caches.append(cache)
        positions.append(pos)

    # Test with BOS input
    backbone_input = torch.full((1, 1, 32), float('nan'))

    # Run step model
    with torch.no_grad():
        step_outputs = step_model(
            backbone_input, bos_emb,
            caches[0], positions[0],
            caches[1], positions[1],
            caches[2], positions[2],
            caches[3], positions[3],
            caches[4], positions[4],
            caches[5], positions[5],
        )

    transformer_out_step = step_outputs[0]
    print(f"Step model transformer output shape: {transformer_out_step.shape}")
    print(f"Step model transformer output range: [{transformer_out_step.min():.4f}, {transformer_out_step.max():.4f}]")

    # Run original model backbone directly (need to access internal state)
    # Let's compare by running the original FlowLM forward with same input
    with torch.no_grad():
        # Prepare input same as step model
        sequence = backbone_input.clone()
        sequence = torch.where(torch.isnan(sequence), model.flow_lm.bos_emb, sequence)
        input_ = model.flow_lm.input_linear(sequence)

        # In original, text_embeddings would be empty for generation step
        text_embeddings = torch.empty((1, 0, 1024), device=input_.device, dtype=input_.dtype)

        # Concatenate (with empty text, this is just input_)
        backbone_input_cat = torch.cat([text_embeddings, input_], dim=1)

        # Run transformer
        transformer_out_orig = model.flow_lm.transformer(backbone_input_cat, voice_state)
        transformer_out_orig = model.flow_lm.out_norm(transformer_out_orig)

        # Take last position (audio portion)
        transformer_out_orig = transformer_out_orig[:, -sequence.shape[1]:]

    print(f"Original model transformer output shape: {transformer_out_orig.shape}")
    print(f"Original model transformer output range: [{transformer_out_orig.min():.4f}, {transformer_out_orig.max():.4f}]")

    # Compare
    diff = (transformer_out_step - transformer_out_orig).abs()
    print(f"\nDifference: max={diff.max():.6f}, mean={diff.mean():.6f}")

    # Correlation
    step_flat = transformer_out_step.flatten().numpy()
    orig_flat = transformer_out_orig.flatten().numpy()
    corr = np.corrcoef(step_flat, orig_flat)[0, 1]
    print(f"Correlation: {corr:.6f}")

    if corr < 0.99:
        print("\n*** MISMATCH DETECTED ***")
        print("Step model transformer output doesn't match original!")

        # Debug: check layer by layer
        print("\n=== Layer-by-layer debugging ===")
        debug_layer_by_layer(model, step_model, backbone_input, bos_emb, caches, positions, voice_state)


def debug_layer_by_layer(model, step_model, backbone_input, bos_emb, caches, positions, voice_state):
    """Debug each layer to find where divergence starts."""
    sequence = backbone_input.clone()
    sequence = torch.where(torch.isnan(sequence), model.flow_lm.bos_emb, sequence)

    # Original model input projection
    input_orig = model.flow_lm.input_linear(sequence)

    # Step model input projection
    sequence_step = backbone_input.clone()
    sequence_step = torch.where(torch.isnan(sequence_step), bos_emb, sequence_step)
    input_step = step_model.input_linear(sequence_step)

    diff = (input_orig - input_step).abs()
    print(f"After input_linear: max_diff={diff.max():.6f}")

    # Now compare attention layer by layer
    x_orig = input_orig
    x_step = input_step

    for i in range(6):
        print(f"\n--- Layer {i} ---")

        # Get original layer
        orig_layer = model.flow_lm.transformer.layers[i]

        # Original: norm1 -> attention
        x_orig_norm = orig_layer.norm1(x_orig)

        # Step: norm1 -> attention
        x_step_norm = getattr(step_model, f'norm{i}_1')(x_step)

        diff = (x_orig_norm - x_step_norm).abs()
        print(f"  After norm1: max_diff={diff.max():.6f}")

        # Run attention
        key = f'transformer.layers.{i}.self_attn'
        orig_attn_state = voice_state[key]

        # Original attention
        with torch.no_grad():
            attn_out_orig = orig_layer.self_attn(x_orig_norm, voice_state)

        # Step attention
        with torch.no_grad():
            attn_out_step, new_cache, new_pos = step_model._streaming_attention(
                x_step_norm,
                getattr(step_model, f'attn{i}_in_proj'),
                getattr(step_model, f'attn{i}_out_proj'),
                caches[i],
                positions[i]
            )

        diff = (attn_out_orig - attn_out_step).abs()
        print(f"  After attention: max_diff={diff.max():.6f}")

        if diff.max() > 0.01:
            print(f"  *** Significant difference at layer {i} attention! ***")

            # Debug the attention internals
            debug_attention(orig_layer.self_attn, step_model, i, x_orig_norm, x_step_norm,
                          caches[i], positions[i], orig_attn_state)
            break

        # Continue through FFN
        x_orig = x_orig + attn_out_orig
        x_step = x_step + attn_out_step

        # FFN
        x_orig_ffn = orig_layer.norm2(x_orig)
        x_step_ffn = getattr(step_model, f'norm{i}_2')(x_step)

        x_orig_ffn = orig_layer.linear2(torch.nn.functional.gelu(orig_layer.linear1(x_orig_ffn)))
        x_step_ffn = getattr(step_model, f'linear{i}_2')(torch.nn.functional.gelu(getattr(step_model, f'linear{i}_1')(x_step_ffn)))

        x_orig = x_orig + x_orig_ffn
        x_step = x_step + x_step_ffn


def debug_attention(orig_attn, step_model, layer_idx, x_orig, x_step, cache, position, orig_state):
    """Debug attention differences in detail."""
    print("\n  === Debugging attention ===")

    # Get Q, K, V projections
    in_proj_orig = orig_attn.in_proj
    in_proj_step = getattr(step_model, f'attn{layer_idx}_in_proj')

    # Check weight equality
    weight_diff = (in_proj_orig.weight - in_proj_step.weight).abs().max()
    print(f"  in_proj weight diff: {weight_diff:.6f}")

    # Project
    qkv_orig = in_proj_orig(x_orig)
    qkv_step = in_proj_step(x_step)

    diff = (qkv_orig - qkv_step).abs()
    print(f"  After in_proj: max_diff={diff.max():.6f}")

    # Reshape to Q, K, V
    B, T = x_orig.shape[:2]
    H = 16
    D = 64

    qkv_orig = qkv_orig.reshape(B, T, 3, H, D)
    q_orig, k_orig, v_orig = qkv_orig[:, :, 0], qkv_orig[:, :, 1], qkv_orig[:, :, 2]

    qkv_step = qkv_step.reshape(B, T, 3, H, D)
    q_step, k_step, v_step = qkv_step[:, :, 0], qkv_step[:, :, 1], qkv_step[:, :, 2]

    print(f"  Q diff: {(q_orig - q_step).abs().max():.6f}")
    print(f"  K diff: {(k_orig - k_step).abs().max():.6f}")
    print(f"  V diff: {(v_orig - v_step).abs().max():.6f}")

    # Check RoPE application
    # Original uses its rope module
    offset_orig = len(orig_state['current_end'])
    print(f"  Original offset: {offset_orig}")
    print(f"  Step position: {position.item()}")

    q_rot_orig, k_rot_orig = orig_attn.rope(q_orig, k_orig, offset=offset_orig)
    q_rot_step, k_rot_step = step_model._apply_rope_tensor(q_step, k_step, position)

    print(f"  Q after RoPE diff: {(q_rot_orig - q_rot_step).abs().max():.6f}")
    print(f"  K after RoPE diff: {(k_rot_orig - k_rot_step).abs().max():.6f}")

    # Check cache state
    orig_cache = orig_state['cache']
    print(f"\n  Original cache shape: {orig_cache.shape}")
    print(f"  Step cache shape: {cache.shape}")

    # Check valid portion of cache
    valid_len = offset_orig
    orig_cache_valid = orig_cache[:, :, :valid_len]
    step_cache_valid = cache[:, :, :valid_len]

    cache_diff = (orig_cache_valid - step_cache_valid).abs()
    print(f"  Cache valid portion diff: max={cache_diff.max():.6f}, mean={cache_diff.mean():.6f}")

    if cache_diff.max() > 0.01:
        print("  *** Cache mismatch! ***")
        # Find first position with significant difference
        for pos in range(valid_len):
            pos_diff = (orig_cache_valid[:, :, pos] - step_cache_valid[:, :, pos]).abs().max()
            if pos_diff > 0.01:
                print(f"  First mismatch at position {pos}: diff={pos_diff:.6f}")
                break


if __name__ == "__main__":
    debug_step_vs_original()
