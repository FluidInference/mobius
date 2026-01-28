"""Generate audio using traceable PocketTTS components.

This demonstrates the full pipeline using PyTorch traceable wrappers
(same code that was converted to CoreML).
"""
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import sys
import os
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_decoder import TraceableMimiDecoder
from traceable_flowlm import TraceableFlowLMBackbone


def lsd_decode(v_t, x_0: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
    """Lagrangian Self Distillation decoding."""
    current = x_0
    for i in range(num_steps):
        s = i / num_steps
        t = (i + 1) / num_steps
        flow_dir = v_t(
            s * torch.ones_like(x_0[..., :1]),
            t * torch.ones_like(x_0[..., :1]),
            current
        )
        current = current + flow_dir / num_steps
    return current


def generate_audio(text: str, voice: str = "alba", output_path: str = "output.wav", lsd_steps: int = 8):
    """Generate audio from text using traceable decoder component.

    Args:
        text: Text to synthesize
        voice: Voice name (e.g., "alba")
        output_path: Output WAV file path
        lsd_steps: LSD decode steps (more = better prosody, default 8)
    """

    print(f"Generating audio for: '{text}'")
    print(f"Voice: {voice}")
    print(f"LSD decode steps: {lsd_steps}")
    print()

    # Load original model
    print("Loading PocketTTS model...")
    from pocket_tts import TTSModel
    from pocket_tts.modules.stateful_module import init_states, increment_steps

    # Use more LSD steps for better prosody (default is 1, which sounds robotic)
    model = TTSModel.load_model(lsd_decode_steps=lsd_steps)
    model.eval()

    # Create traceable decoder
    print("Creating traceable decoder...")
    traceable_decoder = TraceableMimiDecoder.from_mimi(model.mimi)
    traceable_decoder.eval()

    # Initialize decoder state
    decoder_state = traceable_decoder.init_state(batch_size=1)

    # Get constants
    emb_mean = model.flow_lm.emb_mean
    emb_std = model.flow_lm.emb_std

    # Use original model's generate_audio but intercept the mimi decoding
    # For simplicity, let's generate latents first using original model
    print(f"Loading voice '{voice}'...")
    voice_state = model.get_state_for_audio_prompt(voice)

    # Prepare text
    from pocket_tts.models.tts_model import prepare_text_prompt
    prepared_text, frames_after_eos = prepare_text_prompt(text)
    frames_after_eos += 2

    # Tokenize text
    print("Tokenizing text...")
    tokenized = model.flow_lm.conditioner.prepare(prepared_text)
    text_tokens = tokenized.tokens

    # Expand KV cache back to full size
    model._expand_kv_cache(voice_state, sequence_length=1000)

    # Estimate generation length
    gen_len_sec = len(text.split()) * 1 + 2.0
    max_gen_len = int(gen_len_sec * 12.5)

    print(f"Max generation length: {max_gen_len} frames")
    print()

    # Run text prompt through original model
    print("Processing text prompt...")
    model._run_flow_lm_and_increment_step(
        model_state=voice_state,
        text_tokens=text_tokens
    )

    print("Generating latents...")

    # Autoregressive generation
    audio_chunks = []
    backbone_input = torch.full((1, 1, model.flow_lm.ldim), float('nan'))

    eos_step = None
    for step in range(max_gen_len):
        # Generate next latent using original model
        with torch.no_grad():
            next_latent, is_eos = model._run_flow_lm_and_increment_step(
                model_state=voice_state,
                backbone_input_latents=backbone_input
            )

        if is_eos.item() and eos_step is None:
            eos_step = step
            print(f"EOS detected at step {step}")

        if eos_step is not None and step >= eos_step + frames_after_eos:
            break

        # Decode latent to audio using TRACEABLE decoder
        mimi_input = next_latent * emb_std + emb_mean
        mimi_input = mimi_input.transpose(-1, -2)  # [B, T, C] -> [B, C, T]

        # Quantize (passthrough in this model)
        quantized = model.mimi.quantizer(mimi_input)

        # Run traceable decoder
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

        # Extract audio and update state
        audio_frame = decoder_outputs[0]
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
        backbone_input = next_latent

        if step % 10 == 0:
            print(f"  Step {step}/{max_gen_len}...")

    print(f"Generated {len(audio_chunks)} frames")

    # Concatenate audio
    audio = torch.cat(audio_chunks, dim=-1)
    audio = audio[0, 0].numpy()  # Remove batch and channel dims

    # Normalize
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.9

    # Save WAV
    sample_rate = 24000
    wavfile.write(output_path, sample_rate, (audio * 32767).astype(np.int16))

    print(f"\nSaved to {output_path}")
    print(f"Duration: {len(audio) / sample_rate:.2f}s")

    return output_path


if __name__ == "__main__":
    text = "Hello, this is a test of the traceable PocketTTS decoder."
    # Use 8 LSD steps for natural prosody (vs 1 which sounds robotic)
    output = generate_audio(text, voice="alba", output_path="coreml_test_output.wav", lsd_steps=8)
