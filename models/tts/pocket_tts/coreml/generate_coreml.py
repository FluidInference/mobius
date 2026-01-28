"""Generate audio using CoreML decoder."""
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import coremltools as ct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_decoder import TraceableMimiDecoder


def generate_with_coreml(text: str, voice: str = "alba", output_path: str = "coreml_output.wav"):
    """Generate audio using CoreML decoder."""
    print(f"Text: '{text}'")
    print(f"Voice: {voice}")
    
    # Load PyTorch model for FlowLM (latent generation)
    print("\nLoading models...")
    from pocket_tts import TTSModel
    from pocket_tts.models.tts_model import prepare_text_prompt
    from pocket_tts.modules.stateful_module import init_states
    
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()
    
    # Load CoreML decoder
    print("Loading CoreML decoder...")
    coreml_decoder = ct.models.MLModel(
        'mimi_decoder_v2.mlpackage',
        compute_units=ct.ComputeUnit.CPU_ONLY
    )
    
    # Initialize decoder state (using PyTorch wrapper for init)
    pytorch_decoder = TraceableMimiDecoder.from_mimi(model.mimi)
    state = pytorch_decoder.init_state(batch_size=1)
    
    # Convert state to numpy for CoreML
    coreml_state = {
        'upsample_partial': state['upsample_partial'].detach().numpy().astype(np.float32),
        'attn0_cache': state['attn0_cache'].detach().numpy().astype(np.float32),
        'attn0_offset': np.array([0.0], dtype=np.float32),
        'attn0_end_offset': np.array([0.0], dtype=np.float32),
        'attn1_cache': state['attn1_cache'].detach().numpy().astype(np.float32),
        'attn1_offset': np.array([0.0], dtype=np.float32),
        'attn1_end_offset': np.array([0.0], dtype=np.float32),
        'conv0_prev': state['conv0_prev'].detach().numpy().astype(np.float32),
        'conv0_first': state['conv0_first'].detach().numpy().astype(np.float32),
        'convtr0_partial': state['convtr0_partial'].detach().numpy().astype(np.float32),
        'res0_conv0_prev': state['res0_conv0_prev'].detach().numpy().astype(np.float32),
        'res0_conv0_first': state['res0_conv0_first'].detach().numpy().astype(np.float32),
        'res0_conv1_prev': state['res0_conv1_prev'].detach().numpy().astype(np.float32),
        'res0_conv1_first': state['res0_conv1_first'].detach().numpy().astype(np.float32),
        'convtr1_partial': state['convtr1_partial'].detach().numpy().astype(np.float32),
        'res1_conv0_prev': state['res1_conv0_prev'].detach().numpy().astype(np.float32),
        'res1_conv0_first': state['res1_conv0_first'].detach().numpy().astype(np.float32),
        'res1_conv1_prev': state['res1_conv1_prev'].detach().numpy().astype(np.float32),
        'res1_conv1_first': state['res1_conv1_first'].detach().numpy().astype(np.float32),
        'convtr2_partial': state['convtr2_partial'].detach().numpy().astype(np.float32),
        'res2_conv0_prev': state['res2_conv0_prev'].detach().numpy().astype(np.float32),
        'res2_conv0_first': state['res2_conv0_first'].detach().numpy().astype(np.float32),
        'res2_conv1_prev': state['res2_conv1_prev'].detach().numpy().astype(np.float32),
        'res2_conv1_first': state['res2_conv1_first'].detach().numpy().astype(np.float32),
        'conv_final_prev': state['conv_final_prev'].detach().numpy().astype(np.float32),
        'conv_final_first': state['conv_final_first'].detach().numpy().astype(np.float32),
    }
    
    # Prepare text
    print("Preparing generation...")
    voice_state = model.get_state_for_audio_prompt(voice)
    prepared_text, frames_after_eos = prepare_text_prompt(text)
    frames_after_eos += 2
    tokenized = model.flow_lm.conditioner.prepare(prepared_text)
    text_tokens = tokenized.tokens
    model._expand_kv_cache(voice_state, sequence_length=1000)
    
    gen_len_sec = len(text.split()) * 1 + 2.0
    max_gen_len = int(gen_len_sec * 12.5)
    
    # Process text prompt
    model._run_flow_lm_and_increment_step(model_state=voice_state, text_tokens=text_tokens)
    
    # Get constants
    emb_mean = model.flow_lm.emb_mean.numpy()
    emb_std = model.flow_lm.emb_std.numpy()
    
    print(f"Generating (max {max_gen_len} frames)...")
    
    audio_chunks = []
    backbone_input = torch.full((1, 1, model.flow_lm.ldim), float('nan'))
    
    eos_step = None
    for step in range(max_gen_len):
        # Generate latent using PyTorch FlowLM
        with torch.no_grad():
            next_latent, is_eos = model._run_flow_lm_and_increment_step(
                model_state=voice_state,
                backbone_input_latents=backbone_input
            )
        
        if is_eos.item() and eos_step is None:
            eos_step = step
            print(f"  EOS at step {step}")
        
        if eos_step is not None and step >= eos_step + frames_after_eos:
            break
        
        # Decode using CoreML
        mimi_input = next_latent.numpy() * emb_std + emb_mean
        mimi_input = mimi_input.transpose(0, 2, 1).astype(np.float32)  # [B, T, C] -> [B, C, T]
        
        # Quantize (passthrough)
        quantized = model.mimi.quantizer(torch.from_numpy(mimi_input)).detach().numpy().astype(np.float32)
        
        # CoreML decode
        coreml_inputs = {'latent': quantized, **coreml_state}
        coreml_outputs = coreml_decoder.predict(coreml_inputs)
        
        audio_frame = coreml_outputs['var_1445']
        audio_chunks.append(audio_frame)
        
        # Update state from outputs
        coreml_state['upsample_partial'] = coreml_outputs['y_end_1']
        coreml_state['attn0_cache'] = coreml_outputs['new_cache_1_internal_tensor_assign_2']
        coreml_state['attn0_offset'] = coreml_outputs['var_402']
        coreml_state['attn0_end_offset'] = coreml_outputs['new_end_offset_1']
        coreml_state['attn1_cache'] = coreml_outputs['new_cache_internal_tensor_assign_2']
        coreml_state['attn1_offset'] = coreml_outputs['var_825']
        coreml_state['attn1_end_offset'] = coreml_outputs['new_end_offset']
        coreml_state['conv0_prev'] = coreml_outputs['var_998']
        coreml_state['conv0_first'] = coreml_outputs['var_1006']
        coreml_state['convtr0_partial'] = coreml_outputs['var_1048']
        coreml_state['res0_conv0_prev'] = coreml_outputs['var_1105']
        coreml_state['res0_conv0_first'] = coreml_outputs['var_1113']
        coreml_state['res0_conv1_prev'] = coreml_outputs['cast_13']
        coreml_state['res0_conv1_first'] = coreml_outputs['var_1134']
        coreml_state['convtr1_partial'] = coreml_outputs['var_1178']
        coreml_state['res1_conv0_prev'] = coreml_outputs['var_1235']
        coreml_state['res1_conv0_first'] = coreml_outputs['var_1243']
        coreml_state['res1_conv1_prev'] = coreml_outputs['cast_18']
        coreml_state['res1_conv1_first'] = coreml_outputs['var_1264']
        coreml_state['convtr2_partial'] = coreml_outputs['var_1308']
        coreml_state['res2_conv0_prev'] = coreml_outputs['var_1365']
        coreml_state['res2_conv0_first'] = coreml_outputs['var_1373']
        coreml_state['res2_conv1_prev'] = coreml_outputs['cast_23']
        coreml_state['res2_conv1_first'] = coreml_outputs['var_1394']
        coreml_state['conv_final_prev'] = coreml_outputs['var_1450']
        coreml_state['conv_final_first'] = coreml_outputs['var_1458']
        
        backbone_input = next_latent
        
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
    generate_with_coreml(
        "Hello, this is generated using CoreML decoder.",
        voice="alba",
        output_path="coreml_generated.wav"
    )
