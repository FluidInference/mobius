"""Generate audio using pure CoreML with step model.

This approach:
1. PyTorch: Text prep and voice conditioning (fills KV cache)
2. CoreML FlowLM step: Frame-by-frame generation
3. CoreML flow decoder: LSD decoding (8 steps)
4. CoreML Mimi decoder: Audio synthesis
"""
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_pure_coreml_v3(text: str, voice: str = "alba", output_path: str = "pure_coreml_v3.wav"):
    """Generate audio using pure CoreML."""
    print(f"Text: '{text}'")
    print(f"Voice: {voice}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # PyTorch for setup and conditioning
    print("\nLoading PyTorch model for setup...")
    from pocket_tts import TTSModel
    from pocket_tts.models.tts_model import prepare_text_prompt
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    # Load CoreML models
    print("Loading CoreML models...")
    coreml_step = ct.models.MLModel(
        os.path.join(script_dir, 'flowlm_step.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    coreml_flow_decoder = ct.models.MLModel(
        os.path.join(script_dir, 'flow_decoder_v2.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    coreml_mimi = ct.models.MLModel(
        os.path.join(script_dir, 'mimi_decoder_v2.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )

    # Initialize voice state and process text (PyTorch)
    print("\nPreparing voice and text conditioning...")
    voice_state = model.get_state_for_audio_prompt(voice)
    model._expand_kv_cache(voice_state, sequence_length=200)

    prepared_text, frames_after_eos = prepare_text_prompt(text)
    frames_after_eos += 2
    tokenized = model.flow_lm.conditioner.prepare(prepared_text)
    text_tokens = tokenized.tokens

    # Process text tokens (fills KV cache)
    model._run_flow_lm_and_increment_step(model_state=voice_state, text_tokens=text_tokens)

    # Extract state for CoreML
    print("Extracting state for CoreML...")
    bos_emb = model.flow_lm.bos_emb.data.numpy().astype(np.float32)
    emb_mean = model.flow_lm.emb_mean.numpy()
    emb_std = model.flow_lm.emb_std.numpy()
    quantizer_weight = model.mimi.quantizer.output_proj.weight.detach().numpy().astype(np.float32)

    # Extract caches and positions
    coreml_caches = {}
    coreml_positions = {}
    for i in range(6):
        key = f'transformer.layers.{i}.self_attn'
        layer_state = voice_state[key]
        cache = layer_state['cache'].detach().numpy().astype(np.float32)
        # Replace NaN with 0
        cache = np.where(np.isnan(cache), 0.0, cache)
        coreml_caches[f'cache{i}'] = cache
        coreml_positions[f'position{i}'] = np.array([float(len(layer_state['current_end']))], dtype=np.float32)

    print(f"Starting position: {coreml_positions['position0'][0]}")

    # Initialize Mimi decoder state
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

    # Generate
    gen_len_sec = len(text.split()) * 1 + 2.0
    max_gen_len = int(gen_len_sec * 12.5)
    print(f"\nGenerating (max {max_gen_len} frames)...")

    # Set seed for reproducible generation
    np.random.seed(42)

    audio_chunks = []
    eos_step = None
    sequence = np.full((1, 1, 32), np.nan, dtype=np.float32)
    num_lsd_steps = 8
    dt = 1.0 / num_lsd_steps

    for step in range(max_gen_len):
        # 1. Run FlowLM step (CoreML)
        step_inputs = {
            'sequence': sequence,
            'bos_emb': bos_emb,
            **coreml_caches,
            **coreml_positions,
        }
        step_outputs = coreml_step.predict(step_inputs)

        # Get transformer output and EOS
        transformer_out = step_outputs['input']  # [1, 1, 1024]
        eos_logit = step_outputs['var_2492']  # [1, 1, 1]

        # Update caches and positions for next step
        coreml_caches['cache0'] = step_outputs['new_cache_1_internal_tensor_assign_2']
        coreml_positions['position0'] = step_outputs['var_443']
        coreml_caches['cache1'] = step_outputs['new_cache_3_internal_tensor_assign_2']
        coreml_positions['position1'] = step_outputs['var_847']
        coreml_caches['cache2'] = step_outputs['new_cache_5_internal_tensor_assign_2']
        coreml_positions['position2'] = step_outputs['var_1251']
        coreml_caches['cache3'] = step_outputs['new_cache_7_internal_tensor_assign_2']
        coreml_positions['position3'] = step_outputs['var_1655']
        coreml_caches['cache4'] = step_outputs['new_cache_9_internal_tensor_assign_2']
        coreml_positions['position4'] = step_outputs['var_2059']
        coreml_caches['cache5'] = step_outputs['new_cache_internal_tensor_assign_2']
        coreml_positions['position5'] = step_outputs['var_2463']

        # Check EOS (threshold is -4.0 from model config)
        eos_threshold = -4.0
        is_eos = eos_logit.flatten()[0] > eos_threshold
        if is_eos and eos_step is None:
            eos_step = step
            print(f"  EOS at step {step}")

        if eos_step is not None and step >= eos_step + frames_after_eos:
            break

        # 2. Flow decode with LSD (CoreML)
        transformer_out_flat = transformer_out.reshape(1, 1024)  # [1, 1024]
        temp = 0.7
        latent = np.random.randn(1, 32).astype(np.float32) * (temp ** 0.5)

        for lsd_step in range(num_lsd_steps):
            s_np = np.array([[lsd_step * dt]], dtype=np.float32)
            t_np = np.array([[(lsd_step + 1) * dt]], dtype=np.float32)
            flow_inputs = {
                'transformer_out': transformer_out_flat,
                'latent': latent,
                's': s_np,
                't': t_np,
            }
            flow_outputs = coreml_flow_decoder.predict(flow_inputs)
            velocity = list(flow_outputs.values())[0]  # [1, 32]
            latent = latent + velocity * dt

        # 3. Denormalize and project
        latent_denorm = latent * emb_std.reshape(1, -1) + emb_mean.reshape(1, -1)
        quantized = np.dot(latent_denorm, quantizer_weight.T).reshape(1, 512, 1)

        # 4. Decode audio (CoreML)
        mimi_inputs = {'latent': quantized.astype(np.float32), **coreml_mimi_state}
        mimi_outputs = coreml_mimi.predict(mimi_inputs)

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
        sequence = latent.reshape(1, 1, 32)

        if step % 20 == 0:
            print(f"  Step {step}...")

    print(f"Generated {len(audio_chunks)} frames")

    # Concatenate and save
    audio = np.concatenate(audio_chunks, axis=-1)
    audio = audio[0, 0]
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.9

    sample_rate = 24000
    wavfile.write(output_path, sample_rate, (audio * 32767).astype(np.int16))

    print(f"\nSaved to {output_path}")
    print(f"Duration: {len(audio) / sample_rate:.2f}s")

    return output_path


if __name__ == "__main__":
    generate_pure_coreml_v3(
        "Hello, this is pure CoreML text to speech generation.",
        voice="alba",
        output_path="pure_coreml_v3.wav"
    )
