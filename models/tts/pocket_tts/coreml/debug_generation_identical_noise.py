"""Generate with identical noise to compare step model vs original."""
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traceable_flowlm_step import TraceableFlowLMStep
from traceable_flow_decoder import TraceableFlowDecoder


def generate_with_identical_noise():
    """Generate audio with identical noise using both approaches."""
    print("Loading models...")
    from pocket_tts import TTSModel
    from pocket_tts.models.tts_model import prepare_text_prompt

    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    step_model = TraceableFlowLMStep.from_flowlm(model.flow_lm, max_seq_len=200)
    step_model.eval()

    flow_decoder = TraceableFlowDecoder.from_flowlm(model.flow_lm)
    flow_decoder.eval()

    # Initialize voice state for both
    print("\nInitializing voice states...")
    voice_state_orig = model.get_state_for_audio_prompt("alba")
    model._expand_kv_cache(voice_state_orig, sequence_length=200)

    voice_state_step = model.get_state_for_audio_prompt("alba")
    model._expand_kv_cache(voice_state_step, sequence_length=200)

    # Process text
    text = "Hello world"
    prepared_text, frames_after_eos = prepare_text_prompt(text)
    frames_after_eos += 2
    tokenized = model.flow_lm.conditioner.prepare(prepared_text)
    text_tokens = tokenized.tokens

    # Process text for both (fills KV cache)
    model._run_flow_lm_and_increment_step(model_state=voice_state_orig, text_tokens=text_tokens)
    model._run_flow_lm_and_increment_step(model_state=voice_state_step, text_tokens=text_tokens)

    # Extract state for step model
    bos_emb = model.flow_lm.bos_emb.data
    caches = []
    positions = []
    for i in range(6):
        key = f'transformer.layers.{i}.self_attn'
        layer_state = voice_state_step[key]
        cache = layer_state['cache'].clone()
        pos = torch.tensor([float(len(layer_state['current_end']))])
        caches.append(cache)
        positions.append(pos)

    # Get constants
    emb_mean = model.flow_lm.emb_mean
    emb_std = model.flow_lm.emb_std

    # Fixed seed for identical noise
    torch.manual_seed(42)
    np.random.seed(42)

    # Pre-generate all noise for both approaches (to ensure identity)
    max_gen_len = 20
    num_lsd_steps = 8
    all_noise = [torch.randn(1, 32) for _ in range(max_gen_len)]

    print(f"\n=== Generating with ORIGINAL model ===")
    torch.manual_seed(42)  # Reset for same noise

    backbone_input_orig = torch.full((1, 1, 32), float('nan'))
    latents_orig = []

    for step in range(max_gen_len):
        with torch.no_grad():
            # Manually run FlowLM to control noise
            sequence = backbone_input_orig.clone()
            sequence = torch.where(torch.isnan(sequence), model.flow_lm.bos_emb, sequence)
            input_ = model.flow_lm.input_linear(sequence)

            # Empty text embeddings (text already in cache)
            text_embeddings = torch.empty((1, 0, 1024), device=input_.device, dtype=input_.dtype)
            backbone_input_cat = torch.cat([text_embeddings, input_], dim=1)

            transformer_out = model.flow_lm.transformer(backbone_input_cat, voice_state_orig)
            transformer_out = model.flow_lm.out_norm(transformer_out)
            transformer_out = transformer_out[:, -sequence.shape[1]:]

            # Flow decode with controlled noise
            transformer_out_flat = transformer_out[:, -1]  # [1, 1024]
            noise = all_noise[step].to(transformer_out.device)

            dt = 1.0 / num_lsd_steps
            latent = noise.clone()
            for lsd_step in range(num_lsd_steps):
                t = torch.tensor([[lsd_step * dt]])
                velocity = flow_decoder(transformer_out_flat, latent, t)
                latent = latent + velocity * dt

            latents_orig.append(latent.clone())

            # Update state
            from pocket_tts.modules.stateful_module import increment_steps
            increment_steps(model.flow_lm, voice_state_orig, increment=1)

            backbone_input_orig = latent.unsqueeze(1)

        if step % 5 == 0:
            print(f"  Step {step}: latent range [{latent.min():.4f}, {latent.max():.4f}]")

    print(f"\n=== Generating with STEP model ===")

    backbone_input_step = torch.full((1, 1, 32), float('nan'))
    latents_step = []

    for step in range(max_gen_len):
        with torch.no_grad():
            # Run step model
            step_outputs = step_model(
                backbone_input_step, bos_emb,
                caches[0], positions[0],
                caches[1], positions[1],
                caches[2], positions[2],
                caches[3], positions[3],
                caches[4], positions[4],
                caches[5], positions[5],
            )

            transformer_out = step_outputs[0]

            # Update caches and positions
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

            # Flow decode with SAME noise
            transformer_out_flat = transformer_out[:, -1]  # [1, 1024]
            noise = all_noise[step].to(transformer_out.device)

            dt = 1.0 / num_lsd_steps
            latent = noise.clone()
            for lsd_step in range(num_lsd_steps):
                t = torch.tensor([[lsd_step * dt]])
                velocity = flow_decoder(transformer_out_flat, latent, t)
                latent = latent + velocity * dt

            latents_step.append(latent.clone())

            backbone_input_step = latent.unsqueeze(1)

        if step % 5 == 0:
            print(f"  Step {step}: latent range [{latent.min():.4f}, {latent.max():.4f}]")

    print(f"\n=== Comparing latents ===")
    for i in range(min(5, max_gen_len)):
        diff = (latents_orig[i] - latents_step[i]).abs()
        corr = np.corrcoef(latents_orig[i].flatten().numpy(), latents_step[i].flatten().numpy())[0, 1]
        print(f"Frame {i}: max_diff={diff.max():.6f}, correlation={corr:.6f}")

    # Check if they diverge over time
    print(f"\n=== Checking divergence over time ===")
    for i in range(0, max_gen_len, 5):
        diff = (latents_orig[i] - latents_step[i]).abs()
        print(f"Frame {i}: max_diff={diff.max():.6f}")

    # Decode both to audio and save
    print("\n=== Decoding to audio ===")

    # Concatenate all latents and decode at once
    all_latents_orig = torch.stack(latents_orig, dim=2)  # [1, 32, T]
    all_latents_step = torch.stack(latents_step, dim=2)  # [1, 32, T]

    # Denormalize
    all_latents_orig = all_latents_orig * emb_std.view(1, -1, 1) + emb_mean.view(1, -1, 1)
    all_latents_step = all_latents_step * emb_std.view(1, -1, 1) + emb_mean.view(1, -1, 1)

    # Quantize and decode
    with torch.no_grad():
        quantized_orig = model.mimi.quantizer(all_latents_orig)
        quantized_step = model.mimi.quantizer(all_latents_step)
        audio_orig = model.mimi.decoder(quantized_orig)
        audio_step = model.mimi.decoder(quantized_step)

    # Save audio
    audio_orig = audio_orig[0, 0].numpy()
    audio_step = audio_step[0, 0].numpy()

    audio_orig = audio_orig / (np.abs(audio_orig).max() + 1e-8) * 0.9
    audio_step = audio_step / (np.abs(audio_step).max() + 1e-8) * 0.9

    wavfile.write("debug_orig.wav", 24000, (audio_orig * 32767).astype(np.int16))
    wavfile.write("debug_step.wav", 24000, (audio_step * 32767).astype(np.int16))

    print("Saved debug_orig.wav and debug_step.wav")

    # Compare audio
    if len(audio_orig) == len(audio_step):
        audio_diff = np.abs(audio_orig - audio_step)
        audio_corr = np.corrcoef(audio_orig, audio_step)[0, 1]
        print(f"Audio correlation: {audio_corr:.6f}")
        print(f"Audio max diff: {audio_diff.max():.6f}")


if __name__ == "__main__":
    generate_with_identical_noise()
