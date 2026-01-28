"""Generate audio using pure CoreML models.

Pipeline:
- PyTorch: Text preparation only (tokenization + embedding) - done once
- CoreML FlowLM backbone: text_emb + sequence -> transformer_out + eos (per frame)
- CoreML flow decoder: transformer_out + noise -> latent (per frame)
- CoreML Mimi decoder: latent -> audio (per frame)
"""
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_pure_coreml(text: str, voice: str = "alba", output_path: str = "pure_coreml_v2.wav"):
    """Generate audio using pure CoreML models."""
    print(f"Text: '{text}'")
    print(f"Voice: {voice}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load PyTorch model just for text preparation
    print("\nLoading PyTorch model (for text prep only)...")
    from pocket_tts import TTSModel
    from pocket_tts.models.tts_model import prepare_text_prompt
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    # Load CoreML models
    print("Loading CoreML models...")
    coreml_flowlm = ct.models.MLModel(
        os.path.join(script_dir, 'flowlm_backbone_v2.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print("  FlowLM backbone loaded")

    coreml_flow_decoder = ct.models.MLModel(
        os.path.join(script_dir, 'flow_decoder_v2.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print("  Flow decoder v2 loaded (with variable time step)")

    coreml_mimi = ct.models.MLModel(
        os.path.join(script_dir, 'mimi_decoder_v2.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print("  Mimi decoder loaded")

    # === TEXT PREPARATION (PyTorch - done once) ===
    print("\nPreparing text (PyTorch)...")
    prepared_text, frames_after_eos = prepare_text_prompt(text)
    frames_after_eos += 2
    tokenized = model.flow_lm.conditioner.prepare(prepared_text)

    # Get text embeddings
    with torch.no_grad():
        text_emb = model.flow_lm.conditioner.embed(tokenized.tokens)  # [1, T_text, 1024]

    # Pad to 100 tokens (fixed buffer size for CoreML)
    T_text_max = 100
    actual_text_len = text_emb.shape[1]
    if actual_text_len > T_text_max:
        print(f"  Warning: Text too long ({actual_text_len} > {T_text_max}), truncating")
        text_emb = text_emb[:, :T_text_max, :]
    elif actual_text_len < T_text_max:
        pad_len = T_text_max - actual_text_len
        text_emb = torch.cat([
            text_emb,
            torch.zeros(1, pad_len, 1024)
        ], dim=1)
    # NOTE: Text embeddings are processed by PyTorch when initializing voice state
    # For CoreML backbone, we pass zeros since conditioning is already in the KV cache
    text_emb_np = np.zeros((1, T_text_max, 1024), dtype=np.float32)
    print(f"  Text embeddings: zeros (conditioning will be in KV cache)")

    # Get BOS embedding and normalization constants
    bos_emb = model.flow_lm.bos_emb.data.numpy().astype(np.float32)
    emb_mean = model.flow_lm.emb_mean.numpy().reshape(1, -1, 1)  # [1, 512, 1]
    emb_std = model.flow_lm.emb_std.numpy().reshape(1, -1, 1)

    # Get quantizer projection weights (32 -> 512)
    quantizer_weight = model.mimi.quantizer.output_proj.weight.data.numpy().squeeze(-1)  # [512, 32]

    # === INITIALIZE COREML STATES WITH VOICE CONDITIONING ===
    print("Initializing states with voice conditioning...")

    # Get voice state from PyTorch (this includes the audio prompt KV cache)
    voice_state = model.get_state_for_audio_prompt(voice)
    model._expand_kv_cache(voice_state, sequence_length=200)

    # Process text prompt through PyTorch to update the voice state
    # (This prepends the text conditioning to the KV cache)
    tokenized = model.flow_lm.conditioner.prepare(prepared_text)
    model._run_flow_lm_and_increment_step(model_state=voice_state, text_tokens=tokenized.tokens)

    # Now extract the KV cache from the voice state for CoreML
    # The voice state has keys like 'transformer.layers.0.self_attn'
    max_seq_len = 200
    flowlm_state = {}

    # Find the actual position by looking at cache content
    layer0_cache = voice_state['transformer.layers.0.self_attn']['cache']
    # Position is the max index with meaningful content
    non_zero_positions = (layer0_cache.abs().sum(dim=(0, 1, 3, 4)) > 1e-6).nonzero()
    if len(non_zero_positions) > 0:
        actual_position = non_zero_positions.max().item() + 1
    else:
        actual_position = 0

    for i in range(6):
        layer_key = f'transformer.layers.{i}.self_attn'
        layer_state = voice_state[layer_key]
        cache = layer_state['cache']  # [2, B, L, H, D]

        # Pad/truncate cache to fixed size
        _, B, L, H, D = cache.shape
        if L < max_seq_len:
            # Pad with zeros
            padded_cache = torch.zeros(2, B, max_seq_len, H, D)
            padded_cache[:, :, :L, :, :] = cache
            cache = padded_cache
        elif L > max_seq_len:
            cache = cache[:, :, :max_seq_len, :, :]

        flowlm_state[f'cache{i}'] = cache.detach().numpy().astype(np.float32)
        flowlm_state[f'position{i}'] = np.array([float(actual_position)], dtype=np.float32)

    print(f"  Voice state loaded, position: {flowlm_state['position0'][0]:.0f}")

    # Mimi decoder state
    from traceable_decoder import TraceableMimiDecoder
    pytorch_decoder = TraceableMimiDecoder.from_mimi(model.mimi)
    mimi_state = pytorch_decoder.init_state(batch_size=1)

    coreml_mimi_state = {
        'upsample_partial': mimi_state['upsample_partial'].numpy().astype(np.float32),
        'attn0_cache': mimi_state['attn0_cache'].numpy().astype(np.float32),
        'attn0_offset': np.array([0.0], dtype=np.float32),
        'attn0_end_offset': np.array([0.0], dtype=np.float32),
        'attn1_cache': mimi_state['attn1_cache'].numpy().astype(np.float32),
        'attn1_offset': np.array([0.0], dtype=np.float32),
        'attn1_end_offset': np.array([0.0], dtype=np.float32),
        'conv0_prev': mimi_state['conv0_prev'].numpy().astype(np.float32),
        'conv0_first': mimi_state['conv0_first'].numpy().astype(np.float32),
        'convtr0_partial': mimi_state['convtr0_partial'].numpy().astype(np.float32),
        'res0_conv0_prev': mimi_state['res0_conv0_prev'].numpy().astype(np.float32),
        'res0_conv0_first': mimi_state['res0_conv0_first'].numpy().astype(np.float32),
        'res0_conv1_prev': mimi_state['res0_conv1_prev'].numpy().astype(np.float32),
        'res0_conv1_first': mimi_state['res0_conv1_first'].numpy().astype(np.float32),
        'convtr1_partial': mimi_state['convtr1_partial'].numpy().astype(np.float32),
        'res1_conv0_prev': mimi_state['res1_conv0_prev'].numpy().astype(np.float32),
        'res1_conv0_first': mimi_state['res1_conv0_first'].numpy().astype(np.float32),
        'res1_conv1_prev': mimi_state['res1_conv1_prev'].numpy().astype(np.float32),
        'res1_conv1_first': mimi_state['res1_conv1_first'].numpy().astype(np.float32),
        'convtr2_partial': mimi_state['convtr2_partial'].numpy().astype(np.float32),
        'res2_conv0_prev': mimi_state['res2_conv0_prev'].numpy().astype(np.float32),
        'res2_conv0_first': mimi_state['res2_conv0_first'].numpy().astype(np.float32),
        'res2_conv1_prev': mimi_state['res2_conv1_prev'].numpy().astype(np.float32),
        'res2_conv1_first': mimi_state['res2_conv1_first'].numpy().astype(np.float32),
        'conv_final_prev': mimi_state['conv_final_prev'].numpy().astype(np.float32),
        'conv_final_first': mimi_state['conv_final_first'].numpy().astype(np.float32),
    }

    # === GENERATION LOOP (Pure CoreML) ===
    gen_len_sec = len(text.split()) * 1 + 2.0
    max_gen_len = int(gen_len_sec * 12.5)
    print(f"\nGenerating (max {max_gen_len} frames) - Pure CoreML...")

    audio_chunks = []
    sequence = np.full((1, 1, 32), np.nan, dtype=np.float32)  # BOS marker

    eos_step = None
    for step in range(max_gen_len):
        # 1. FlowLM backbone (CoreML)
        flowlm_inputs = {
            'sequence': sequence,
            'text_embeddings': text_emb_np,
            'bos_emb': bos_emb,
            **{f'cache{i}': flowlm_state[f'cache{i}'] for i in range(6)},
            **{f'position{i}': flowlm_state[f'position{i}'] for i in range(6)},
        }
        flowlm_outputs = coreml_flowlm.predict(flowlm_inputs)

        transformer_out = flowlm_outputs['input']  # [1, 1, 1024]
        is_eos_logit = flowlm_outputs['var_2414']  # [1, 1, 1]

        # Update FlowLM state
        flowlm_state['cache0'] = flowlm_outputs['new_cache_1_internal_tensor_assign_2']
        flowlm_state['position0'] = flowlm_outputs['var_431']
        flowlm_state['cache1'] = flowlm_outputs['new_cache_3_internal_tensor_assign_2']
        flowlm_state['position1'] = flowlm_outputs['var_819']
        flowlm_state['cache2'] = flowlm_outputs['new_cache_5_internal_tensor_assign_2']
        flowlm_state['position2'] = flowlm_outputs['var_1207']
        flowlm_state['cache3'] = flowlm_outputs['new_cache_7_internal_tensor_assign_2']
        flowlm_state['position3'] = flowlm_outputs['var_1595']
        flowlm_state['cache4'] = flowlm_outputs['new_cache_9_internal_tensor_assign_2']
        flowlm_state['position4'] = flowlm_outputs['var_1983']
        flowlm_state['cache5'] = flowlm_outputs['new_cache_internal_tensor_assign_2']
        flowlm_state['position5'] = flowlm_outputs['var_2371']

        # Check EOS
        is_eos = is_eos_logit.flatten()[0] > 0
        if is_eos and eos_step is None:
            eos_step = step
            print(f"  EOS at step {step}")

        if eos_step is not None and step >= eos_step + frames_after_eos:
            break

        # 2. Flow decoder (CoreML) - Proper LSD iterative decoding
        transformer_out_squeezed = transformer_out.reshape(1, 1024).astype(np.float32)

        # LSD decoding: start with noise, iteratively refine
        num_lsd_steps = 8
        dt = 1.0 / num_lsd_steps
        latent = np.random.randn(1, 32).astype(np.float32)  # Initial noise

        for lsd_step in range(num_lsd_steps):
            t = np.array([[lsd_step * dt]], dtype=np.float32)  # Time step [0, 1/8, 2/8, ...]
            velocity = coreml_flow_decoder.predict({
                'transformer_out': transformer_out_squeezed,
                'latent': latent,
                't': t,
            })['var_368']  # Velocity output

            latent = latent + velocity * dt  # Euler step

        # 3. Mimi decoder (CoreML)
        # Convert latent to mimi input format:
        # - Denormalize the 32-dim latent
        # - Project 32 -> 512 via quantizer weights (pure numpy)
        latent_expanded = latent.reshape(1, 32, 1)  # [1, 32] -> [1, 32, 1]
        latent_denorm = latent_expanded * emb_std[:, :32, :] + emb_mean[:, :32, :]

        # Quantizer projection: [512, 32] @ [32, 1] -> [512, 1]
        quantized = np.dot(quantizer_weight, latent_denorm.squeeze(0)).reshape(1, 512, 1).astype(np.float32)

        # CoreML decode
        coreml_inputs = {'latent': quantized, **coreml_mimi_state}
        mimi_outputs = coreml_mimi.predict(coreml_inputs)

        audio_frame = mimi_outputs['var_1445']
        audio_chunks.append(audio_frame)

        # Update Mimi state
        coreml_mimi_state['upsample_partial'] = mimi_outputs['y_end_1']
        coreml_mimi_state['attn0_cache'] = mimi_outputs['new_cache_1_internal_tensor_assign_2']
        coreml_mimi_state['attn0_offset'] = mimi_outputs['var_402']
        coreml_mimi_state['attn0_end_offset'] = mimi_outputs['new_end_offset_1']
        coreml_mimi_state['attn1_cache'] = mimi_outputs['new_cache_internal_tensor_assign_2']
        coreml_mimi_state['attn1_offset'] = mimi_outputs['var_825']
        coreml_mimi_state['attn1_end_offset'] = mimi_outputs['new_end_offset']
        coreml_mimi_state['conv0_prev'] = mimi_outputs['var_998']
        coreml_mimi_state['conv0_first'] = mimi_outputs['var_1006']
        coreml_mimi_state['convtr0_partial'] = mimi_outputs['var_1048']
        coreml_mimi_state['res0_conv0_prev'] = mimi_outputs['var_1105']
        coreml_mimi_state['res0_conv0_first'] = mimi_outputs['var_1113']
        coreml_mimi_state['res0_conv1_prev'] = mimi_outputs['cast_13']
        coreml_mimi_state['res0_conv1_first'] = mimi_outputs['var_1134']
        coreml_mimi_state['convtr1_partial'] = mimi_outputs['var_1178']
        coreml_mimi_state['res1_conv0_prev'] = mimi_outputs['var_1235']
        coreml_mimi_state['res1_conv0_first'] = mimi_outputs['var_1243']
        coreml_mimi_state['res1_conv1_prev'] = mimi_outputs['cast_18']
        coreml_mimi_state['res1_conv1_first'] = mimi_outputs['var_1264']
        coreml_mimi_state['convtr2_partial'] = mimi_outputs['var_1308']
        coreml_mimi_state['res2_conv0_prev'] = mimi_outputs['var_1365']
        coreml_mimi_state['res2_conv0_first'] = mimi_outputs['var_1373']
        coreml_mimi_state['res2_conv1_prev'] = mimi_outputs['cast_23']
        coreml_mimi_state['res2_conv1_first'] = mimi_outputs['var_1394']
        coreml_mimi_state['conv_final_prev'] = mimi_outputs['var_1450']
        coreml_mimi_state['conv_final_first'] = mimi_outputs['var_1458']

        # Update sequence for next step
        sequence = latent.reshape(1, 1, 32)  # [1, 32] -> [1, 1, 32]

        if step % 20 == 0:
            print(f"  Step {step}...")

    print(f"Generated {len(audio_chunks)} frames")

    # Concatenate and save
    audio = np.concatenate(audio_chunks, axis=-1)
    audio = audio[0, 0]  # Remove batch and channel dims
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.9

    sample_rate = 24000
    wavfile.write(output_path, sample_rate, (audio * 32767).astype(np.int16))

    print(f"\nSaved to {output_path}")
    print(f"Duration: {len(audio) / sample_rate:.2f}s")


if __name__ == "__main__":
    generate_pure_coreml(
        "Hello, this is generated using pure CoreML models.",
        voice="alba",
        output_path="pure_coreml_v2.wav"
    )
