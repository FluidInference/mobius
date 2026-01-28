"""Generate audio using pure CoreML (FlowLM + Mimi decoder)."""
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_pure_coreml(text: str, voice: str = "alba", output_path: str = "pure_coreml_output.wav"):
    """Generate audio using pure CoreML models."""
    print(f"Text: '{text}'")
    print(f"Voice: {voice}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # We still need PyTorch for:
    # 1. Text tokenization/preparation
    # 2. Voice prompt/conditioning
    # 3. The flow decoder (LSD steps)
    # These could be converted to CoreML later, but for now we focus on FlowLM + Mimi
    print("\nLoading PyTorch model for setup...")
    from pocket_tts import TTSModel
    from pocket_tts.models.tts_model import prepare_text_prompt
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    # Load CoreML models
    print("Loading CoreML FlowLM backbone...")
    coreml_flowlm = ct.models.MLModel(
        os.path.join(script_dir, 'flowlm_backbone_v2.mlpackage'),
        compute_units=ct.ComputeUnit.ALL  # CPU_ONLY causes segfault
    )

    print("Loading CoreML Mimi decoder...")
    coreml_decoder = ct.models.MLModel(
        os.path.join(script_dir, 'mimi_decoder_v2.mlpackage'),
        compute_units=ct.ComputeUnit.ALL  # CPU_ONLY causes segfault
    )

    # Prepare text and voice
    print("\nPreparing generation...")
    voice_state = model.get_state_for_audio_prompt(voice)
    prepared_text, frames_after_eos = prepare_text_prompt(text)
    frames_after_eos += 2
    tokenized = model.flow_lm.conditioner.prepare(prepared_text)
    text_tokens = tokenized.tokens

    # Get text embeddings from conditioner (still PyTorch)
    # This prepares the text conditioning that goes into FlowLM
    with torch.no_grad():
        # The conditioner uses an embedding layer on the tokens
        text_emb = model.flow_lm.conditioner.embed(tokenized.tokens)  # [1, T_text, 1024]

    print(f"Text embeddings shape: {text_emb.shape}")

    # Pad text embeddings to fixed buffer size (100 tokens)
    T_text_max = 100
    actual_text_len = text_emb.shape[1]
    if actual_text_len > T_text_max:
        print(f"Warning: Text too long ({actual_text_len} > {T_text_max}), truncating")
        text_emb = text_emb[:, :T_text_max, :]
    elif actual_text_len < T_text_max:
        # Pad with zeros
        pad_len = T_text_max - actual_text_len
        text_emb = torch.cat([
            text_emb,
            torch.zeros(1, pad_len, 1024, device=text_emb.device)
        ], dim=1)
    print(f"Padded text embeddings shape: {text_emb.shape}")

    # Initialize FlowLM state (all zeros, not NaN for CoreML compatibility)
    max_seq_len = 200
    flowlm_state = {}
    for i in range(6):
        flowlm_state[f'cache{i}'] = np.zeros((2, 1, max_seq_len, 16, 64), dtype=np.float32)
        flowlm_state[f'position{i}'] = np.array([0.0], dtype=np.float32)

    # Initialize Mimi decoder state
    from traceable_decoder import TraceableMimiDecoder
    pytorch_decoder = TraceableMimiDecoder.from_mimi(model.mimi)
    mimi_state = pytorch_decoder.init_state(batch_size=1)

    # Convert mimi state to numpy
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

    # Get constants
    bos_emb = model.flow_lm.bos_emb.data.numpy().astype(np.float32)
    emb_mean = model.flow_lm.emb_mean.numpy()
    emb_std = model.flow_lm.emb_std.numpy()

    gen_len_sec = len(text.split()) * 1 + 2.0
    max_gen_len = int(gen_len_sec * 12.5)

    print(f"Generating (max {max_gen_len} frames)...")

    # First pass: process text prompt
    # The FlowLM backbone takes text embeddings as conditioning
    # For the first step, we feed BOS (represented as NaN in sequence)
    text_emb_np = text_emb.numpy().astype(np.float32)

    audio_chunks = []
    eos_step = None

    # Create BOS sequence (NaN triggers replacement with bos_emb)
    sequence = np.full((1, 1, 32), np.nan, dtype=np.float32)

    for step in range(max_gen_len):
        # Run FlowLM backbone (CoreML)
        flowlm_inputs = {
            'sequence': sequence,
            'text_embeddings': text_emb_np,
            'bos_emb': bos_emb,
            **{f'cache{i}': flowlm_state[f'cache{i}'] for i in range(6)},
            **{f'position{i}': flowlm_state[f'position{i}'] for i in range(6)},
        }

        flowlm_outputs = coreml_flowlm.predict(flowlm_inputs)

        # Get transformer output and EOS
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

        # Decode latent using flow network (PyTorch - LSD multi-step decoding)
        # The flow_net converts transformer_out [1024] -> latent [32] via iterative refinement
        transformer_out_torch = torch.from_numpy(transformer_out).squeeze(1)  # [1, 1024]
        with torch.no_grad():
            # Start with noise
            latent = torch.randn(1, model.flow_lm.ldim)  # [1, 32]
            num_steps = 8  # LSD decode steps (same as model loading)

            # LSD iterative decoding
            for step_idx in range(num_steps):
                t = torch.tensor([step_idx / num_steps])  # Time conditioning
                # Flow network predicts velocity
                velocity = model.flow_lm.flow_net(latent, t, transformer_out_torch)
                # Update latent
                dt = 1.0 / num_steps
                latent = latent + velocity * dt

            next_latent = latent.unsqueeze(2)  # [1, 32, 1]

        # Decode audio using CoreML Mimi decoder
        mimi_input = next_latent.numpy() * emb_std.reshape(1, -1, 1) + emb_mean.reshape(1, -1, 1)
        mimi_input = mimi_input.astype(np.float32)

        # Quantize (passthrough in Mimi)
        quantized = model.mimi.quantizer(torch.from_numpy(mimi_input)).detach().numpy().astype(np.float32)

        # CoreML decode
        coreml_inputs = {'latent': quantized, **coreml_mimi_state}
        coreml_outputs = coreml_decoder.predict(coreml_inputs)

        audio_frame = coreml_outputs['var_1445']
        audio_chunks.append(audio_frame)

        # Update Mimi decoder state
        coreml_mimi_state['upsample_partial'] = coreml_outputs['y_end_1']
        coreml_mimi_state['attn0_cache'] = coreml_outputs['new_cache_1_internal_tensor_assign_2']
        coreml_mimi_state['attn0_offset'] = coreml_outputs['var_402']
        coreml_mimi_state['attn0_end_offset'] = coreml_outputs['new_end_offset_1']
        coreml_mimi_state['attn1_cache'] = coreml_outputs['new_cache_internal_tensor_assign_2']
        coreml_mimi_state['attn1_offset'] = coreml_outputs['var_825']
        coreml_mimi_state['attn1_end_offset'] = coreml_outputs['new_end_offset']
        coreml_mimi_state['conv0_prev'] = coreml_outputs['var_998']
        coreml_mimi_state['conv0_first'] = coreml_outputs['var_1006']
        coreml_mimi_state['convtr0_partial'] = coreml_outputs['var_1048']
        coreml_mimi_state['res0_conv0_prev'] = coreml_outputs['var_1105']
        coreml_mimi_state['res0_conv0_first'] = coreml_outputs['var_1113']
        coreml_mimi_state['res0_conv1_prev'] = coreml_outputs['cast_13']
        coreml_mimi_state['res0_conv1_first'] = coreml_outputs['var_1134']
        coreml_mimi_state['convtr1_partial'] = coreml_outputs['var_1178']
        coreml_mimi_state['res1_conv0_prev'] = coreml_outputs['var_1235']
        coreml_mimi_state['res1_conv0_first'] = coreml_outputs['var_1243']
        coreml_mimi_state['res1_conv1_prev'] = coreml_outputs['cast_18']
        coreml_mimi_state['res1_conv1_first'] = coreml_outputs['var_1264']
        coreml_mimi_state['convtr2_partial'] = coreml_outputs['var_1308']
        coreml_mimi_state['res2_conv0_prev'] = coreml_outputs['var_1365']
        coreml_mimi_state['res2_conv0_first'] = coreml_outputs['var_1373']
        coreml_mimi_state['res2_conv1_prev'] = coreml_outputs['cast_23']
        coreml_mimi_state['res2_conv1_first'] = coreml_outputs['var_1394']
        coreml_mimi_state['conv_final_prev'] = coreml_outputs['var_1450']
        coreml_mimi_state['conv_final_first'] = coreml_outputs['var_1458']

        # Update sequence for next step (feed back the latent)
        sequence = next_latent.numpy().transpose(0, 2, 1).astype(np.float32)  # [B, 32, T] -> [B, T, 32]

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
        "Hello, this is generated with almost pure CoreML. The FlowLM backbone and Mimi decoder are both running on CoreML.",
        voice="alba",
        output_path="pure_coreml_generated.wav"
    )
