"""Verify VoxCPM 1.5 loads and runs inference via PyTorch.

Downloads the model from HuggingFace on first run, then generates a short
audio clip to confirm the pipeline works end-to-end.
"""

import sys
import time

import numpy as np


def main():
    print("=== VoxCPM 1.5 PyTorch Verification ===\n")

    # Step 1: Load model
    print("[1/4] Loading VoxCPM 1.5 from HuggingFace...")
    t0 = time.time()
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5", load_denoiser=False, optimize=False)
    print(f"  Loaded in {time.time() - t0:.1f}s\n")

    # Step 2: Inspect model components
    print("[2/4] Model components:")
    tts = model.tts_model
    for name, param in [
        ("base_lm", tts.base_lm),
        ("residual_lm", tts.residual_lm),
        ("feat_encoder", tts.feat_encoder),
        ("feat_decoder", tts.feat_decoder),
        ("audio_vae", tts.audio_vae),
    ]:
        n_params = sum(p.numel() for p in param.parameters())
        print(f"  {name}: {n_params / 1e6:.1f}M params")

    # Print projection layers
    for name in ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj",
                 "stop_proj", "stop_actn", "stop_head"]:
        layer = getattr(tts, name, None)
        if layer is not None:
            n_params = sum(p.numel() for p in layer.parameters())
            print(f"  {name}: {n_params / 1e3:.1f}K params")

    # Print config
    print(f"\n  Config:")
    print(f"    patch_size: {tts.patch_size}")
    print(f"    hidden_size: {tts.config.lm_config.hidden_size}")
    print(f"    lm layers: {tts.config.lm_config.num_hidden_layers}")
    print(f"    encoder_config: {tts.config.encoder_config}")
    print(f"    dit_config: {tts.config.dit_config}")
    print(f"    vocab_size: {tts.config.lm_config.vocab_size}")
    # AudioVAE details
    vae = tts.audio_vae
    for attr in ["sample_rate", "hop_length", "latent_dim"]:
        print(f"    audio_vae.{attr}: {getattr(vae, attr, 'N/A')}")
    print(f"    audio_vae_config: {tts.config.audio_vae_config}")

    # Step 3: Generate audio
    print("\n[3/4] Generating test audio...")
    t0 = time.time()
    audio = model.generate(
        text="Hello, this is a test of VoxCPM one point five.",
        inference_timesteps=5,  # fewer steps for speed
        cfg_value=2.0,
        normalize=False,
        denoise=False,
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")
    print(f"  Audio shape: {audio.shape}")
    print(f"  Duration: {len(audio) / 44100:.2f}s")
    print(f"  RTF: {gen_time / (len(audio) / 44100):.2f}")

    # Step 4: Save audio
    from scipy.io import wavfile
    out_path = "test_output.wav"
    audio_int16 = (audio * 32767).astype(np.int16)
    wavfile.write(out_path, 44100, audio_int16)
    print(f"\n[4/4] Saved to {out_path}")
    print("\n=== Verification complete ===")


if __name__ == "__main__":
    main()
