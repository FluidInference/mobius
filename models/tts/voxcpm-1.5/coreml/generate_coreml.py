"""Zero-PyTorch VoxCPM 1.5 generation using CoreML models.

Runs the full TTS pipeline using only CoreML models and numpy,
validating that the conversion is end-to-end functional.

Usage:
    python generate_coreml.py --text "Hello world" --prompt prompt.wav
"""

import argparse
import json
import os
import subprocess
import time

import coremltools as ct
import numpy as np


# VoxCPM special tokens
AUDIO_START_TOKEN = 101


def load_models():
    """Load all CoreML models."""
    print("Loading CoreML models...")
    models = {
        "audio_vae_encoder": ct.models.MLModel("audio_vae_encoder.mlpackage"),
        "audio_vae_decoder": ct.models.MLModel("audio_vae_decoder.mlpackage"),
        "feat_encoder": ct.models.MLModel("feat_encoder.mlpackage"),
        "base_lm_step": ct.models.MLModel("base_lm_step.mlpackage"),
        "residual_lm_step": ct.models.MLModel("residual_lm_step.mlpackage"),
        "locdit_step": ct.models.MLModel("locdit_step.mlpackage"),
    }
    print(f"  Loaded {len(models)} models")
    return models


def load_constants():
    """Load numpy constants and config."""
    print("Loading constants...")
    constants = {
        "embed_tokens": np.load("constants/embed_tokens.npy"),
        "enc_to_lm_proj_w": np.load("constants/enc_to_lm_proj_w.npy"),
        "enc_to_lm_proj_b": np.load("constants/enc_to_lm_proj_b.npy"),
        "lm_to_dit_proj_w": np.load("constants/lm_to_dit_proj_w.npy"),
        "lm_to_dit_proj_b": np.load("constants/lm_to_dit_proj_b.npy"),
        "res_to_dit_proj_w": np.load("constants/res_to_dit_proj_w.npy"),
        "res_to_dit_proj_b": np.load("constants/res_to_dit_proj_b.npy"),
    }

    with open("constants/config.json") as f:
        config = json.load(f)

    print(f"  embed_tokens: {constants['embed_tokens'].shape}")
    print(f"  scale_emb: {config['scale_emb']}")
    return constants, config


def linear(x, w, b):
    """Apply linear projection: y = x @ w.T + b"""
    return x @ w.T + b


def encode_prompt_audio(models, audio_np, config):
    """Encode prompt audio to latent features.

    The encoder CoreML model uses a fixed input size of 5s (220500 samples).
    Input audio is truncated or padded to fit. Valid latent frames are trimmed.
    """
    sr = config["sample_rate"]
    hop = config["hop_length"]
    n = audio_np.shape[-1]

    # Fixed encoder size: 5 seconds (must match convert_audio_vae_encoder.py)
    encoder_samples = 5 * sr  # 220500

    if n > encoder_samples:
        # Truncate to last 5s (keep the most recent audio for voice cloning)
        audio_np = audio_np[:, :, -encoder_samples:]
        n = encoder_samples

    # Pad to encoder size
    if n < encoder_samples:
        pad = encoder_samples - n
        audio_np = np.pad(audio_np, ((0, 0), (0, 0), (0, pad)))

    # Compute how many valid latent frames we have (before padding)
    n_valid_frames = n // hop

    pred = models["audio_vae_encoder"].predict({"audio": audio_np.astype(np.float32)})
    latent = pred["latent"]  # [1, 64, T]

    # Trim to valid frames (remove latent frames from padding)
    if n_valid_frames > 0 and n_valid_frames < latent.shape[2]:
        latent = latent[:, :, :n_valid_frames]

    return latent


def encode_features(models, latent_patch):
    """Encode a latent patch through feat_encoder.

    Args:
        latent_patch: [1, 1, patch_size, 64] - single patch
    Returns:
        embedding: [1, 1, 1024]
    """
    pred = models["feat_encoder"].predict({"feat": latent_patch.astype(np.float32)})
    return pred["embedding"]  # [1, 1, 1024]


def run_locdit(models, mu, cond, n_timesteps, cfg_value, config):
    """Run the LocDiT diffusion to generate a latent patch.

    Args:
        mu: [1, 1024] - combined dit_hidden
        cond: [1, 64, patch_size] - prefix conditioning features (last patch)
        n_timesteps: number of Euler steps
        cfg_value: classifier-free guidance scale
    Returns:
        pred_feat: [1, 64, patch_size] - generated latent patch
    """
    patch_size = config["patch_size"]
    feat_dim = config["feat_dim"]

    # Initialize noise
    noise = np.random.randn(1, feat_dim, patch_size).astype(np.float32)

    # Build batch=2 for CFG (conditioned + unconditioned)
    if cfg_value > 0:
        mu_batch = np.concatenate([mu, np.zeros_like(mu)], axis=0)  # [2, 1024]
        noise_batch = np.concatenate([noise, noise], axis=0)  # [2, 64, patch_size]
        cond_batch = np.concatenate([cond, cond], axis=0)  # [2, 64, patch_size]
    else:
        mu_batch = mu
        noise_batch = noise
        cond_batch = cond

    batch = mu_batch.shape[0]

    # Euler solver (backward: time goes from 1 -> ~0, matching VoxCPM solve_euler)
    # VoxCPM: t_span = linspace(1, 1e-3, n_timesteps + 1)
    #         dt = t_span[i] - t_span[i+1]  (positive)
    #         x = x - dt * velocity
    t_span = np.linspace(1.0, 1e-3, n_timesteps + 1, dtype=np.float32)
    x = noise_batch.copy()

    for step in range(n_timesteps):
        t_val = t_span[step]
        step_dt = t_span[step] - t_span[step + 1]
        t = np.full((batch,), t_val, dtype=np.float32)
        dt = np.zeros((batch,), dtype=np.float32)

        pred = models["locdit_step"].predict({
            "x": x.astype(np.float32),
            "mu": mu_batch.astype(np.float32),
            "t": t,
            "cond": cond_batch.astype(np.float32),
            "dt": dt,
        })
        velocity = pred["velocity"]  # [2, 64, patch_size]

        if cfg_value > 0 and batch == 2:
            # CFG: v = v_uncond + cfg * (v_cond - v_uncond)
            v_cond = velocity[0:1]
            v_uncond = velocity[1:2]
            v = v_uncond + cfg_value * (v_cond - v_uncond)
            # Apply to single sample (backward Euler)
            x_single = x[0:1] - step_dt * v
            x = np.concatenate([x_single, x_single], axis=0)
        else:
            x = x - step_dt * velocity

    return x[0:1]  # [1, 64, patch_size]


def run_base_lm_step(models, embed, position, base_caches):
    """Run one base LM step and update caches."""
    input_dict = {
        "embed": embed.astype(np.float32),
        "position": np.array([position], dtype=np.int32),
    }
    for i in range(24):
        input_dict[f"k{i}"] = base_caches[i * 2]
        input_dict[f"v{i}"] = base_caches[i * 2 + 1]

    pred = models["base_lm_step"].predict(input_dict)

    for i in range(24):
        base_caches[i * 2] = pred[f"out_k{i}"]
        base_caches[i * 2 + 1] = pred[f"out_v{i}"]

    return pred["lm_hidden"], pred["lm_hidden_fsq"], pred["stop_logit"]


def run_residual_lm_step(models, embed, position, res_caches):
    """Run one residual LM step and update caches."""
    res_dict = {
        "embed": embed.astype(np.float32),
        "position": np.array([position], dtype=np.int32),
    }
    for i in range(8):
        res_dict[f"k{i}"] = res_caches[i * 2]
        res_dict[f"v{i}"] = res_caches[i * 2 + 1]

    res_pred = models["residual_lm_step"].predict(res_dict)

    for i in range(8):
        res_caches[i * 2] = res_pred[f"out_k{i}"]
        res_caches[i * 2 + 1] = res_pred[f"out_v{i}"]

    return res_pred["res_hidden"]


def generate(models, constants, config, text_ids, prompt_audio, max_len=200, min_len=5, disable_stop=False, prompt_text_ids=None):
    """Run the full VoxCPM generation pipeline.

    Following VoxCPM._inference():
    1. Encode prompt audio -> latent patches -> feat_encoder -> enc_to_lm_proj
    2. Embed text tokens (+ audio_start_token) with scale_emb
    3. Combine: text_mask * text_emb + feat_mask * audio_emb
    4. Prefill both LMs with combined embeddings
    5. Generate loop: dit_hidden -> LocDiT diffusion -> encode -> next step
    6. Decode accumulated latents
    """
    scale_emb = config["scale_emb"]
    patch_size = config["patch_size"]
    feat_dim = config["feat_dim"]
    max_seq_len = config["max_seq_len"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    n_timesteps = config["default_inference_timesteps"]
    cfg_value = config["default_cfg_value"]

    print(f"\n--- Generation ---")
    print(f"  Text tokens: {len(text_ids)}")
    if prompt_audio is not None:
        print(f"  Prompt audio: {prompt_audio.shape[-1] / config['sample_rate']:.1f}s")
    else:
        print(f"  Prompt audio: none (unconditioned)")

    # Step 1-3: Encode prompt audio (if provided)
    if prompt_audio is not None:
        print("\n  [1] Encoding prompt audio...")
        prompt_latent = encode_prompt_audio(models, prompt_audio, config)
        T = prompt_latent.shape[2]
        print(f"    Latent shape: {prompt_latent.shape} ({T} frames)")

        n_patches = T // patch_size
        prompt_patches = prompt_latent[:, :, :n_patches * patch_size]
        prompt_patches = prompt_patches.reshape(1, feat_dim, n_patches, patch_size)
        prompt_patches = prompt_patches.transpose(0, 2, 3, 1)

        print("  [2] Encoding prompt features...")
        prompt_embeddings = []
        for p in range(n_patches):
            patch = prompt_patches[:, p:p+1, :, :]
            emb = encode_features(models, patch)
            prompt_embeddings.append(emb[:, 0, :])
        feat_emb = np.stack(prompt_embeddings, axis=1)  # [1, n_patches, 1024]
        print(f"    Prompt embeddings: {feat_emb.shape}")

        feat_lm_emb = linear(feat_emb, constants["enc_to_lm_proj_w"], constants["enc_to_lm_proj_b"])
        prefix_cond = prompt_latent[:, :, -patch_size:]  # [1, 64, patch_size]
    else:
        # No prompt: VoxCPM uses text-only with zero conditioning
        print("\n  [1-2] No prompt audio (unconditioned generation)")
        n_patches = 0
        feat_lm_emb = np.zeros((1, 0, 1024), dtype=np.float32)
        prefix_cond = np.zeros((1, feat_dim, patch_size), dtype=np.float32)

    # Step 4: Text embeddings
    # VoxCPM: text = prompt_text + target_text, then appends audio_start_token
    if prompt_text_ids is not None:
        all_text_ids = list(prompt_text_ids) + list(text_ids) + [AUDIO_START_TOKEN]
    else:
        all_text_ids = list(text_ids) + [AUDIO_START_TOKEN]
    text_len = len(all_text_ids)
    text_emb = constants["embed_tokens"][all_text_ids] * scale_emb
    text_emb = text_emb[np.newaxis, :]  # [1, text_len, 1024]

    # Step 5: Build combined embedding sequence
    if n_patches > 0:
        combined = np.concatenate([text_emb, feat_lm_emb], axis=1)
    else:
        combined = text_emb
    seq_len = combined.shape[1]
    print(f"    Combined sequence: {seq_len} tokens (text={text_len}, audio={n_patches})")

    if seq_len >= max_seq_len:
        print(f"    WARNING: Sequence length {seq_len} exceeds max_seq_len {max_seq_len}!")
        print(f"    Truncating prompt audio to fit.")
        max_audio_patches = max_seq_len - text_len - 50
        if max_audio_patches < 1:
            raise ValueError("Text too long for max_seq_len")
        n_patches = max(0, max_audio_patches)
        feat_lm_emb = feat_lm_emb[:, :n_patches, :]
        if n_patches > 0:
            combined = np.concatenate([text_emb, feat_lm_emb], axis=1)
        else:
            combined = text_emb
        seq_len = combined.shape[1]

    # Step 6: Prefill
    print("  [3] Prefilling...")
    cache_shape = (1, num_kv_heads, max_seq_len, head_dim)
    base_caches = [np.zeros(cache_shape, dtype=np.float32) for _ in range(48)]
    res_caches = [np.zeros(cache_shape, dtype=np.float32) for _ in range(16)]

    t0 = time.time()
    for pos in range(seq_len):
        token_emb = combined[:, pos, :]

        lm_hidden, lm_hidden_fsq, stop_logit = run_base_lm_step(
            models, token_emb, pos, base_caches
        )

        is_audio_pos = pos >= text_len
        if is_audio_pos:
            audio_idx = pos - text_len
            res_input = lm_hidden_fsq + feat_lm_emb[:, audio_idx, :]
        else:
            res_input = lm_hidden

        res_hidden = run_residual_lm_step(models, res_input, pos, res_caches)

    prefill_time = time.time() - t0
    print(f"    Prefill: {seq_len} tokens in {prefill_time:.1f}s ({seq_len / prefill_time:.1f} tok/s)")

    # Step 7: Generate
    print("  [4] Generating...")

    generated_latents = []
    pos = seq_len  # Current position in KV cache

    t0 = time.time()
    for step in range(max_len):
        if pos >= max_seq_len:
            print(f"    Max KV cache position reached at step {step}")
            break

        # Combine projections for LocDiT
        # VoxCPM: dit_hidden = lm_to_dit_proj(lm_hidden) + res_to_dit_proj(residual_hidden)
        # lm_hidden is FSQ'd in VoxCPM (quantized output), not raw
        dit_hidden = (
            linear(lm_hidden_fsq, constants["lm_to_dit_proj_w"], constants["lm_to_dit_proj_b"]) +
            linear(res_hidden, constants["res_to_dit_proj_w"], constants["res_to_dit_proj_b"])
        )

        # Run LocDiT diffusion
        pred_feat = run_locdit(models, dit_hidden, prefix_cond, n_timesteps, cfg_value, config)
        # pred_feat: [1, 64, patch_size]
        generated_latents.append(pred_feat)

        # Check stop using base LM stop logit
        if not disable_stop and step >= min_len:
            stop_flag = np.argmax(stop_logit[0])
            if stop_flag == 1:
                print(f"    Stop at step {step + 1}")
                break

        # Update prefix conditioning with newly generated patch
        # VoxCPM: prefix_feat_cond = pred_feat
        prefix_cond = pred_feat  # [1, 64, patch_size]

        # Encode predicted feature for next step input
        # pred_feat: [1, 64, patch_size] -> [1, patch_size, 64] -> [1, 1, patch_size, 64]
        pred_patch_4d = pred_feat.transpose(0, 2, 1)[:, np.newaxis, :, :]  # [1, 1, patch_size, 64]

        curr_emb = encode_features(models, pred_patch_4d)  # [1, 1, 1024]
        curr_emb = curr_emb[:, 0, :]  # [1, 1024]
        curr_lm_emb = linear(curr_emb, constants["enc_to_lm_proj_w"], constants["enc_to_lm_proj_b"])

        # Run base LM step
        lm_hidden, lm_hidden_fsq, stop_logit = run_base_lm_step(
            models, curr_lm_emb, pos, base_caches
        )

        # Run residual LM step
        # VoxCPM: residual input = fsq(lm_hidden) + feat_embed (audio position)
        res_input = lm_hidden_fsq + curr_lm_emb
        res_hidden = run_residual_lm_step(models, res_input, pos, res_caches)

        pos += 1

        if (step + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Step {step + 1}: {elapsed:.1f}s ({(step + 1) / elapsed:.2f} steps/s)")

    gen_time = time.time() - t0
    n_steps = len(generated_latents)
    print(f"    Generated {n_steps} steps in {gen_time:.1f}s ({n_steps / gen_time:.2f} steps/s)")

    # Step 8: Decode
    print("  [5] Decoding audio...")
    all_latents = np.concatenate(generated_latents, axis=2)  # [1, 64, n_steps * patch_size]
    print(f"    Latent shape: {all_latents.shape}")

    pred = models["audio_vae_decoder"].predict({"latent": all_latents.astype(np.float32)})
    audio = pred["audio"]  # [1, 1, M]
    duration = audio.shape[-1] / config["sample_rate"]
    print(f"    Audio: {audio.shape} ({duration:.2f}s)")

    return audio


def transcribe(wav_path, fluidaudio_root=None):
    """Transcribe a WAV file using FluidAudio CLI.

    Resamples to 16kHz mono first (required by ASR), then runs
    `swift run fluidaudiocli transcribe`.

    Returns the transcription text, or None on failure.
    """
    if fluidaudio_root is None:
        # Walk up from this script to find FluidAudio root
        here = os.path.dirname(os.path.abspath(__file__))
        fluidaudio_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))

    # Resample to 16kHz mono WAV for ASR
    wav_path = os.path.abspath(wav_path)
    wav_16k = wav_path.replace(".wav", "_16k.wav")
    try:
        import scipy.io.wavfile as wavmod
        sr_in, data = wavmod.read(wav_path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        if len(data.shape) > 1:
            data = data[:, 0]
        # Resample to 16kHz
        ratio = 16000 / sr_in
        new_len = int(len(data) * ratio)
        indices = np.linspace(0, len(data) - 1, new_len)
        data_16k = np.interp(indices, np.arange(len(data)), data)
        out_16k = (data_16k * 32768).clip(-32768, 32767).astype(np.int16)
        wavmod.write(wav_16k, 16000, out_16k)
    except Exception as e:
        print(f"  Resample failed: {e}")
        return None

    print(f"\n  [Transcribe] Transcribing {wav_16k}...")
    try:
        result = subprocess.run(
            ["swift", "run", "fluidaudiocli", "transcribe", wav_16k],
            capture_output=True, text=True, timeout=120,
            cwd=fluidaudio_root,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            print(f"  Transcription failed (rc={result.returncode}): {result.stderr[:200]}")
            return None
        # Extract transcript from CLI output
        # The CLI prints log lines with [INFO] tags, then the raw transcript on the last line
        lines = [l.strip() for l in output.split("\n") if l.strip()]
        # Look for labeled transcript line first
        for line in lines:
            if "final transcription:" in line.lower():
                text = line.split(":", 2)[-1].strip()
                if text:
                    print(f"  Transcript: {text}")
                    return text
        # Fall back to last line (raw transcript output)
        if lines:
            # Skip log lines (contain [INFO], [DEBUG], etc.)
            text_lines = [l for l in lines if not l.startswith("[") and "PM]" not in l and "AM]" not in l]
            if text_lines:
                print(f"  Transcript: {text_lines[-1]}")
                return text_lines[-1]
        print(f"  No transcript found in output")
        return None
    except subprocess.TimeoutExpired:
        print(f"  Transcription timed out")
        return None
    except Exception as e:
        print(f"  Transcription error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="VoxCPM 1.5 CoreML generation")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the voice cloning system.")
    parser.add_argument("--prompt", type=str, help="Path to prompt audio WAV file")
    parser.add_argument("--prompt-text", type=str, help="Transcript of prompt audio (for voice cloning)")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--no-stop", action="store_true", help="Disable stop head, generate max-len steps")
    parser.add_argument("--transcribe", action="store_true", help="Transcribe output via FluidAudio CLI")
    args = parser.parse_args()

    models = load_models()
    constants, config = load_constants()

    # Tokenize text
    # VoxCPM uses mask_multichar_chinese_tokens which splits multi-character
    # Chinese tokens into individual characters. This is required for Chinese.
    print("\nTokenizing text...")
    from transformers import AutoTokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_name", "openbmb/MiniCPM-1B-sft-bf16"),
        trust_remote_code=True,
    )
    from voxcpm.model.utils import mask_multichar_chinese_tokens
    tokenizer = mask_multichar_chinese_tokens(base_tokenizer)

    # VoxCPM concatenates: prompt_text + target_text, then tokenizes
    if args.prompt_text:
        full_text = args.prompt_text + args.text
    else:
        full_text = args.text
    all_ids = tokenizer(full_text)
    # Split back into prompt_text_ids and text_ids based on the prompt portion
    if args.prompt_text:
        prompt_text_ids = tokenizer(args.prompt_text)
        text_ids = all_ids[len(prompt_text_ids):]
        print(f"  Prompt text: '{args.prompt_text}' -> {len(prompt_text_ids)} tokens")
    else:
        prompt_text_ids = None
        text_ids = all_ids
    print(f"  Text: '{args.text}' -> {len(text_ids)} tokens (total: {len(all_ids)})")

    # Load prompt audio
    if args.prompt:
        import scipy.io.wavfile as wav
        sr, audio_data = wav.read(args.prompt)
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]  # mono
        if sr != 44100:
            # Simple resampling (for proper use, use librosa)
            ratio = 44100 / sr
            new_len = int(len(audio_data) * ratio)
            indices = np.linspace(0, len(audio_data) - 1, new_len)
            audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
        prompt_audio = audio_data[np.newaxis, np.newaxis, :].astype(np.float32)
    else:
        # No prompt: VoxCPM uses empty audio features (all zeros)
        print("  No prompt audio provided, using zeros (unconditioned)")
        prompt_audio = None

    # Generate
    audio = generate(models, constants, config, text_ids, prompt_audio,
                     max_len=args.max_len, min_len=args.min_len, disable_stop=args.no_stop,
                     prompt_text_ids=prompt_text_ids)

    # Save
    import scipy.io.wavfile as wav
    audio_out = (audio[0, 0] * 32768).clip(-32768, 32767).astype(np.int16)
    wav.write(args.output, 44100, audio_out)
    print(f"\nSaved to {args.output} ({len(audio_out) / 44100:.2f}s)")

    # Transcribe
    if args.transcribe:
        transcript = transcribe(args.output)
        if transcript:
            print(f"\n--- Transcription ---")
            print(f"  Input text:  {args.text}")
            print(f"  Output text: {transcript}")
        else:
            print(f"\n--- Transcription failed ---")


if __name__ == "__main__":
    main()
