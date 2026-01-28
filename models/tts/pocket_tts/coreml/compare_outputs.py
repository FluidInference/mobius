"""Compare traceable decoder output vs original PyTorch output."""
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_decoder import TraceableMimiDecoder


def generate_with_original(model, text, voice, lsd_steps=8):
    """Generate audio using fully original PyTorch pipeline."""
    from pocket_tts.models.tts_model import prepare_text_prompt
    from pocket_tts.modules.stateful_module import init_states
    
    voice_state = model.get_state_for_audio_prompt(voice)
    prepared_text, frames_after_eos = prepare_text_prompt(text)
    frames_after_eos += 2
    tokenized = model.flow_lm.conditioner.prepare(prepared_text)
    text_tokens = tokenized.tokens
    model._expand_kv_cache(voice_state, sequence_length=1000)
    
    gen_len_sec = len(text.split()) * 1 + 2.0
    max_gen_len = int(gen_len_sec * 12.5)
    
    # Process text prompt
    model._run_flow_lm_and_increment_step(
        model_state=voice_state,
        text_tokens=text_tokens
    )
    
    # Get constants
    emb_mean = model.flow_lm.emb_mean
    emb_std = model.flow_lm.emb_std
    
    # Initialize ORIGINAL mimi decoder state
    mimi_state = init_states(model.mimi, batch_size=1, sequence_length=1000)

    audio_chunks = []
    latents = []
    backbone_input = torch.full((1, 1, model.flow_lm.ldim), float('nan'))
    
    eos_step = None
    for step in range(max_gen_len):
        with torch.no_grad():
            next_latent, is_eos = model._run_flow_lm_and_increment_step(
                model_state=voice_state,
                backbone_input_latents=backbone_input
            )
        
        if is_eos.item() and eos_step is None:
            eos_step = step
        
        if eos_step is not None and step >= eos_step + frames_after_eos:
            break
        
        latents.append(next_latent.clone())
        
        # Decode using ORIGINAL mimi decoder
        mimi_input = next_latent * emb_std + emb_mean
        mimi_input = mimi_input.transpose(-1, -2)  # [B, T, C] -> [B, C, T]

        # Quantizer converts 32-dim to 512-dim
        quantized = model.mimi.quantizer(mimi_input)

        with torch.no_grad():
            audio_frame = model.mimi.decode_from_latent(quantized, mimi_state)

        # Increment mimi state
        from pocket_tts.modules.stateful_module import increment_steps
        increment_steps(model.mimi, mimi_state, increment=16)
        
        audio_chunks.append(audio_frame)
        backbone_input = next_latent
    
    audio = torch.cat(audio_chunks, dim=-1)
    return audio[0, 0].numpy(), latents


def generate_with_traceable(model, latents):
    """Generate audio using traceable decoder from pre-computed latents."""
    emb_mean = model.flow_lm.emb_mean
    emb_std = model.flow_lm.emb_std
    
    traceable_decoder = TraceableMimiDecoder.from_mimi(model.mimi)
    traceable_decoder.eval()
    decoder_state = traceable_decoder.init_state(batch_size=1)
    
    audio_chunks = []
    
    for next_latent in latents:
        mimi_input = next_latent * emb_std + emb_mean
        mimi_input = mimi_input.transpose(-1, -2)
        quantized = model.mimi.quantizer(mimi_input)
        
        with torch.no_grad():
            decoder_outputs = traceable_decoder(
                quantized,
                decoder_state['upsample_partial'],
                decoder_state['attn0_cache'], decoder_state['attn0_offset'], decoder_state['attn0_end_offset'],
                decoder_state['attn1_cache'], decoder_state['attn1_offset'], decoder_state['attn1_end_offset'],
                decoder_state['conv0_prev'], decoder_state['conv0_first'],
                decoder_state['convtr0_partial'],
                decoder_state['res0_conv0_prev'], decoder_state['res0_conv0_first'],
                decoder_state['res0_conv1_prev'], decoder_state['res0_conv1_first'],
                decoder_state['convtr1_partial'],
                decoder_state['res1_conv0_prev'], decoder_state['res1_conv0_first'],
                decoder_state['res1_conv1_prev'], decoder_state['res1_conv1_first'],
                decoder_state['convtr2_partial'],
                decoder_state['res2_conv0_prev'], decoder_state['res2_conv0_first'],
                decoder_state['res2_conv1_prev'], decoder_state['res2_conv1_first'],
                decoder_state['conv_final_prev'], decoder_state['conv_final_first'],
            )
        
        audio_frame = decoder_outputs[0]
        # Update state
        decoder_state['upsample_partial'] = decoder_outputs[1]
        decoder_state['attn0_cache'] = decoder_outputs[2]
        decoder_state['attn0_offset'] = decoder_outputs[3]
        decoder_state['attn0_end_offset'] = decoder_outputs[4]
        decoder_state['attn1_cache'] = decoder_outputs[5]
        decoder_state['attn1_offset'] = decoder_outputs[6]
        decoder_state['attn1_end_offset'] = decoder_outputs[7]
        decoder_state['conv0_prev'] = decoder_outputs[8]
        decoder_state['conv0_first'] = decoder_outputs[9]
        decoder_state['convtr0_partial'] = decoder_outputs[10]
        decoder_state['res0_conv0_prev'] = decoder_outputs[11]
        decoder_state['res0_conv0_first'] = decoder_outputs[12]
        decoder_state['res0_conv1_prev'] = decoder_outputs[13]
        decoder_state['res0_conv1_first'] = decoder_outputs[14]
        decoder_state['convtr1_partial'] = decoder_outputs[15]
        decoder_state['res1_conv0_prev'] = decoder_outputs[16]
        decoder_state['res1_conv0_first'] = decoder_outputs[17]
        decoder_state['res1_conv1_prev'] = decoder_outputs[18]
        decoder_state['res1_conv1_first'] = decoder_outputs[19]
        decoder_state['convtr2_partial'] = decoder_outputs[20]
        decoder_state['res2_conv0_prev'] = decoder_outputs[21]
        decoder_state['res2_conv0_first'] = decoder_outputs[22]
        decoder_state['res2_conv1_prev'] = decoder_outputs[23]
        decoder_state['res2_conv1_first'] = decoder_outputs[24]
        decoder_state['conv_final_prev'] = decoder_outputs[25]
        decoder_state['conv_final_first'] = decoder_outputs[26]
        
        audio_chunks.append(audio_frame)
    
    audio = torch.cat(audio_chunks, dim=-1)
    return audio[0, 0].numpy()


def compute_mfcc(audio, sr=24000, n_mfcc=13):
    """Compute MFCC features for audio comparison."""
    import scipy.signal as signal
    
    # Pre-emphasis
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    
    # Frame parameters
    frame_len = int(0.025 * sr)  # 25ms
    frame_step = int(0.010 * sr)  # 10ms
    
    # Simple STFT-based features
    f, t, Zxx = signal.stft(audio, sr, nperseg=frame_len, noverlap=frame_len-frame_step)
    power = np.abs(Zxx) ** 2
    
    # Mel filterbank (simplified)
    n_mels = 40
    mel_points = np.linspace(0, 2595 * np.log10(1 + sr/2 / 700), n_mels + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    
    # Return mean power spectrum as simple embedding
    return np.mean(power, axis=1)


def main():
    print("Loading PocketTTS model...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()
    
    text = "Hello, this is a test of the traceable PocketTTS decoder."
    voice = "alba"
    
    print("\n" + "="*60)
    print("Generating with ORIGINAL PyTorch decoder...")
    print("="*60)
    original_audio, latents = generate_with_original(model, text, voice)
    print(f"Original audio shape: {original_audio.shape}")
    print(f"Generated {len(latents)} latent frames")
    
    print("\n" + "="*60)
    print("Generating with TRACEABLE decoder (same latents)...")
    print("="*60)
    traceable_audio = generate_with_traceable(model, latents)
    print(f"Traceable audio shape: {traceable_audio.shape}")
    
    # Align lengths
    min_len = min(len(original_audio), len(traceable_audio))
    original_audio = original_audio[:min_len]
    traceable_audio = traceable_audio[:min_len]
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    # Raw waveform comparison
    diff = original_audio - traceable_audio
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))
    
    print(f"Waveform max difference: {max_diff:.6f}")
    print(f"Waveform mean difference: {mean_diff:.6f}")
    print(f"Waveform RMS difference: {rms_diff:.6f}")
    
    # Correlation
    corr = np.corrcoef(original_audio, traceable_audio)[0, 1]
    print(f"Waveform correlation: {corr:.6f}")
    
    # Spectral comparison
    orig_spec = compute_mfcc(original_audio)
    trace_spec = compute_mfcc(traceable_audio)
    spec_corr = np.corrcoef(orig_spec, trace_spec)[0, 1]
    print(f"Spectral correlation: {spec_corr:.6f}")
    
    # Save both for listening
    sample_rate = 24000
    
    original_norm = original_audio / (np.abs(original_audio).max() + 1e-8) * 0.9
    traceable_norm = traceable_audio / (np.abs(traceable_audio).max() + 1e-8) * 0.9
    
    wavfile.write("original_pytorch.wav", sample_rate, (original_norm * 32767).astype(np.int16))
    wavfile.write("traceable_decoder.wav", sample_rate, (traceable_norm * 32767).astype(np.int16))
    
    print(f"\nSaved original_pytorch.wav and traceable_decoder.wav for comparison")
    print(f"Duration: {min_len / sample_rate:.2f}s")
    
    if corr > 0.99:
        print("\n✓ Outputs are nearly identical!")
    elif corr > 0.95:
        print("\n⚠ Outputs are similar but have some differences")
    else:
        print(f"\n✗ Significant differences detected (corr={corr:.4f})")


if __name__ == "__main__":
    main()
