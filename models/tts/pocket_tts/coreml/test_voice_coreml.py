#!/usr/bin/env python3
"""Test voice cloning with CoreML inference.

End-to-end workflow:
1. Load exported voice (.bin or .safetensors)
2. Run CoreML TTS inference
3. Optionally evaluate against reference audio

Usage:
    # Test with exported voice
    python test_voice_coreml.py --voice custom_audio_prompt.bin --text "Hello world"

    # Full workflow: export voice, generate TTS, evaluate
    python test_voice_coreml.py --reference speaker.wav --text "Hello world" --evaluate

Dependencies: numpy, sentencepiece, coremltools, scipy
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent

# Voice cloning constants
VOICE_PROMPT_LENGTH = 125
EMBEDDING_DIM = 1024
SAMPLE_RATE = 24000

# Model output keys (from generate_coreml_v4.py)
COND_CACHE_KEYS = [
    "new_cache_1_internal_tensor_assign_2",
    "new_cache_3_internal_tensor_assign_2",
    "new_cache_5_internal_tensor_assign_2",
    "new_cache_7_internal_tensor_assign_2",
    "new_cache_9_internal_tensor_assign_2",
    "new_cache_internal_tensor_assign_2",
]
COND_POS_KEYS = [
    "var_445", "var_864", "var_1283", "var_1702", "var_2121", "var_2365",
]
STEP_CACHE_KEYS = [
    "new_cache_1_internal_tensor_assign_2",
    "new_cache_3_internal_tensor_assign_2",
    "new_cache_5_internal_tensor_assign_2",
    "new_cache_7_internal_tensor_assign_2",
    "new_cache_9_internal_tensor_assign_2",
    "new_cache_internal_tensor_assign_2",
]
STEP_POS_KEYS = [
    "var_458", "var_877", "var_1296", "var_1715", "var_2134", "var_2553",
]


def load_voice_bin(path: Path) -> np.ndarray:
    """Load voice conditioning from .bin file (Swift format)."""
    data = np.fromfile(path, dtype=np.float32)
    num_floats = len(data)

    # Expected: VOICE_PROMPT_LENGTH * EMBEDDING_DIM
    expected = VOICE_PROMPT_LENGTH * EMBEDDING_DIM
    if num_floats != expected:
        print(f"Warning: Expected {expected} floats, got {num_floats}")
        # Try to infer shape
        if num_floats % EMBEDDING_DIM == 0:
            frames = num_floats // EMBEDDING_DIM
            print(f"  Inferred shape: [{frames}, {EMBEDDING_DIM}]")
        else:
            raise ValueError(f"Cannot reshape {num_floats} floats to [?, {EMBEDDING_DIM}]")

    frames = num_floats // EMBEDDING_DIM
    voice_emb = data.reshape(1, frames, EMBEDDING_DIM)
    return voice_emb


def load_voice_safetensors(path: Path) -> np.ndarray:
    """Load voice conditioning from .safetensors file."""
    from safetensors.numpy import load_file
    data = load_file(str(path))
    voice_emb = data['audio_prompt']
    if voice_emb.ndim == 2:
        voice_emb = voice_emb[np.newaxis, :, :]
    return voice_emb


def load_voice(path: Path) -> np.ndarray:
    """Load voice conditioning from .bin or .safetensors file."""
    if path.suffix == '.bin':
        return load_voice_bin(path)
    elif path.suffix == '.safetensors':
        return load_voice_safetensors(path)
    else:
        raise ValueError(f"Unknown voice format: {path.suffix}")


def prepare_text_prompt(text: str):
    """Normalize text for TTS."""
    text = text.strip()
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'  +', ' ', text)

    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]

    if text and text[-1] not in '.!?':
        text += '.'

    word_count = len(text.split())
    if word_count < 5:
        text = ' ' * 8 + text
        frames_after_eos = 3
    else:
        frames_after_eos = 1

    return text, frames_after_eos


def find_constants_dir() -> Path:
    """Find the constants directory."""
    # Try multiple locations
    candidates = [
        SCRIPT_DIR / "constants",
        SCRIPT_DIR / "constants_bin",
        PROJECT_DIR / "coreml" / "constants",
    ]
    for path in candidates:
        if path.exists() and (path / "tokenizer.model").exists():
            return path
    raise FileNotFoundError(f"Constants directory not found. Tried: {candidates}")


def find_models_dir() -> Path:
    """Find the CoreML models directory."""
    # Models should be in same directory as constants
    candidates = [
        SCRIPT_DIR,
        SCRIPT_DIR / "models",
        PROJECT_DIR / "coreml",
    ]
    for path in candidates:
        if (path / "cond_step.mlpackage").exists():
            return path
    raise FileNotFoundError("CoreML models not found (cond_step.mlpackage)")


def generate_with_voice(
    text: str,
    voice_emb: np.ndarray,
    output_path: Path,
    seed: int = 42,
    temperature: float = 0.7,
) -> Path:
    """Generate audio using CoreML with custom voice embedding."""
    import sentencepiece as sp
    import coremltools as ct

    print(f"Text: '{text}'")
    print(f"Voice shape: {voice_emb.shape}")
    print(f"Seed: {seed}")

    # Find directories
    const_dir = find_constants_dir()
    models_dir = find_models_dir()
    print(f"Constants: {const_dir}")
    print(f"Models: {models_dir}")

    # 1. Text preparation
    prepared_text, frames_after_eos = prepare_text_prompt(text)
    frames_after_eos += 2
    print(f"Prepared: '{prepared_text}'")

    # 2. Tokenize
    tokenizer = sp.SentencePieceProcessor()
    tokenizer.load(str(const_dir / "tokenizer.model"))
    token_ids = tokenizer.encode(prepared_text)
    print(f"Tokens: {len(token_ids)}")

    # 3. Embed text
    embed_table_path = const_dir / "text_embed_table.npy"
    if not embed_table_path.exists():
        # Try .bin format
        embed_table_path = const_dir / "text_embed_table.bin"
        embed_table = np.fromfile(embed_table_path, dtype=np.float32)
        embed_table = embed_table.reshape(-1, EMBEDDING_DIM)
    else:
        embed_table = np.load(str(embed_table_path))

    text_emb = embed_table[token_ids]
    text_emb = text_emb[np.newaxis, :, :]
    print(f"Text embeddings: {text_emb.shape}")

    # 4. Load constants
    bos_path = const_dir / "bos_emb.npy"
    if not bos_path.exists():
        bos_path = const_dir / "bos_emb.bin"
        bos_emb = np.fromfile(bos_path, dtype=np.float32)
    else:
        bos_emb = np.load(str(bos_path))

    mimi_state_path = const_dir / "mimi_init_state.npz"
    if mimi_state_path.exists():
        mimi_state_npz = dict(np.load(str(mimi_state_path)))
    else:
        # Load from individual .bin files
        mimi_state_dir = const_dir / "mimi_init_state"
        mimi_state_npz = {}
        if mimi_state_dir.exists():
            for f in mimi_state_dir.glob("*.bin"):
                name = f.stem
                mimi_state_npz[name] = np.fromfile(f, dtype=np.float32)

    # 5. Load CoreML models
    print("\nLoading CoreML models...")
    t0 = time.time()

    coreml_cond = ct.models.MLModel(
        str(models_dir / 'cond_step.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    coreml_step = ct.models.MLModel(
        str(models_dir / 'flowlm_step.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    coreml_flow = ct.models.MLModel(
        str(models_dir / 'flow_decoder.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    coreml_mimi = ct.models.MLModel(
        str(models_dir / 'mimi_decoder.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    print(f"Models loaded in {time.time() - t0:.2f}s")

    # 6. Combine conditioning
    combined = np.concatenate([voice_emb, text_emb], axis=1)
    cond_len = combined.shape[1]
    print(f"Conditioning: {cond_len} tokens (voice={voice_emb.shape[1]}, text={text_emb.shape[1]})")

    # 7. Initialize KV caches
    caches = {}
    positions = {}
    for i in range(6):
        caches[f'cache{i}'] = np.zeros((2, 1, 512, 16, 64), dtype=np.float32)
        positions[f'position{i}'] = np.array([0.0], dtype=np.float32)

    # 8. Prefill
    print(f"Prefilling KV cache ({cond_len} tokens)...")
    t0 = time.time()
    for tok_idx in range(cond_len):
        cond_token = combined[:, tok_idx:tok_idx + 1, :]
        cond_inputs = {
            'conditioning': cond_token.astype(np.float32),
            **caches,
            **positions,
        }
        cond_out = coreml_cond.predict(cond_inputs)

        for i in range(6):
            caches[f'cache{i}'] = cond_out[COND_CACHE_KEYS[i]]
            positions[f'position{i}'] = cond_out[COND_POS_KEYS[i]]

    print(f"Prefill done in {time.time() - t0:.2f}s")

    # 9. Generation
    gen_len_sec = len(text.split()) + 2.0
    max_gen_len = int(gen_len_sec * 12.5)
    print(f"\nGenerating (max {max_gen_len} frames)...")

    np.random.seed(seed)

    audio_chunks = []
    eos_step = None
    sequence = np.full((1, 1, 32), float('nan'), dtype=np.float32)
    num_lsd_steps = 8
    dt = 1.0 / num_lsd_steps

    # Initialize Mimi state
    coreml_mimi_state = {}
    for k, v in mimi_state_npz.items():
        if isinstance(v, np.ndarray):
            coreml_mimi_state[k] = v.astype(np.float32)
    coreml_mimi_state.setdefault('attn0_offset', np.array([0.0], dtype=np.float32))
    coreml_mimi_state.setdefault('attn0_end_offset', np.array([0.0], dtype=np.float32))
    coreml_mimi_state.setdefault('attn1_offset', np.array([0.0], dtype=np.float32))
    coreml_mimi_state.setdefault('attn1_end_offset', np.array([0.0], dtype=np.float32))

    t0 = time.time()
    for step in range(max_gen_len):
        # Step model
        step_inputs = {
            'sequence': sequence,
            'bos_emb': bos_emb,
            **caches,
            **positions,
        }
        step_out = coreml_step.predict(step_inputs)

        transformer_out = step_out['input']
        eos_logit = step_out['var_2582']

        for i in range(6):
            caches[f'cache{i}'] = step_out[STEP_CACHE_KEYS[i]]
            positions[f'position{i}'] = step_out[STEP_POS_KEYS[i]]

        # EOS check
        is_eos = eos_logit.flatten()[0] > -4.0
        if is_eos and eos_step is None:
            eos_step = step
            print(f"  EOS at step {step}")
        if eos_step is not None and step >= eos_step + frames_after_eos:
            break

        # Flow decode
        transformer_out_flat = transformer_out.reshape(1, 1024)
        latent = np.random.randn(1, 32).astype(np.float32) * (temperature ** 0.5)

        for lsd_step in range(num_lsd_steps):
            s_np = np.array([[lsd_step * dt]], dtype=np.float32)
            t_np = np.array([[(lsd_step + 1) * dt]], dtype=np.float32)
            flow_out = coreml_flow.predict({
                'transformer_out': transformer_out_flat,
                'latent': latent,
                's': s_np,
                't': t_np,
            })
            velocity = list(flow_out.values())[0]
            latent = latent + velocity * dt

        # Mimi decode
        mimi_inputs = {'latent': latent.astype(np.float32), **coreml_mimi_state}
        mimi_out = coreml_mimi.predict(mimi_inputs)

        audio_frame = mimi_out['var_1445']
        audio_chunks.append(audio_frame)

        # Update Mimi state
        coreml_mimi_state['upsample_partial'] = mimi_out['y_end_1']
        coreml_mimi_state['attn0_cache'] = mimi_out['new_cache_1_internal_tensor_assign_2']
        coreml_mimi_state['attn0_offset'] = mimi_out['var_402']
        coreml_mimi_state['attn0_end_offset'] = mimi_out['new_end_offset_1']
        coreml_mimi_state['attn1_cache'] = mimi_out['new_cache_internal_tensor_assign_2']
        coreml_mimi_state['attn1_offset'] = mimi_out['var_825']
        coreml_mimi_state['attn1_end_offset'] = mimi_out['new_end_offset']
        coreml_mimi_state['conv0_prev'] = mimi_out['var_998']
        coreml_mimi_state['conv0_first'] = mimi_out['var_1006']
        coreml_mimi_state['convtr0_partial'] = mimi_out['var_1048']
        coreml_mimi_state['res0_conv0_prev'] = mimi_out['var_1105']
        coreml_mimi_state['res0_conv0_first'] = mimi_out['var_1113']
        coreml_mimi_state['res0_conv1_prev'] = mimi_out['cast_13']
        coreml_mimi_state['res0_conv1_first'] = mimi_out['var_1134']
        coreml_mimi_state['convtr1_partial'] = mimi_out['var_1178']
        coreml_mimi_state['res1_conv0_prev'] = mimi_out['var_1235']
        coreml_mimi_state['res1_conv0_first'] = mimi_out['var_1243']
        coreml_mimi_state['res1_conv1_prev'] = mimi_out['cast_18']
        coreml_mimi_state['res1_conv1_first'] = mimi_out['var_1264']
        coreml_mimi_state['convtr2_partial'] = mimi_out['var_1308']
        coreml_mimi_state['res2_conv0_prev'] = mimi_out['var_1365']
        coreml_mimi_state['res2_conv0_first'] = mimi_out['var_1373']
        coreml_mimi_state['res2_conv1_prev'] = mimi_out['cast_23']
        coreml_mimi_state['res2_conv1_first'] = mimi_out['var_1394']
        coreml_mimi_state['conv_final_prev'] = mimi_out['var_1450']
        coreml_mimi_state['conv_final_first'] = mimi_out['var_1458']

        sequence = latent.reshape(1, 1, 32)

        if step % 20 == 0:
            print(f"  Step {step}...")

    gen_time = time.time() - t0
    print(f"Generated {len(audio_chunks)} frames in {gen_time:.2f}s")

    # Concatenate and save
    audio = np.concatenate(audio_chunks, axis=-1)
    audio = audio[0, 0]
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.9

    duration = len(audio) / SAMPLE_RATE
    rtfx = duration / gen_time

    wavfile.write(str(output_path), SAMPLE_RATE, (audio * 32767).astype(np.int16))

    print(f"\nSaved to: {output_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"RTFx: {rtfx:.1f}x real-time")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Test voice cloning with CoreML inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with pre-exported voice
  python test_voice_coreml.py --voice custom_audio_prompt.bin --text "Hello world"

  # Full workflow: export + generate + evaluate
  python test_voice_coreml.py --reference speaker.wav --text "Hello" --evaluate

  # Use built-in voice (if available as .safetensors)
  python test_voice_coreml.py --voice alba --text "Testing alba voice"
"""
    )
    parser.add_argument(
        "--voice",
        type=str,
        help="Voice file (.bin or .safetensors) or built-in voice name"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference audio to clone (runs export_voice_coreml.py first)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of voice cloning with CoreML.",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_output.wav"),
        help="Output WAV file path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate output against reference using spectral similarity"
    )

    args = parser.parse_args()

    # Determine voice source
    if args.reference:
        # Export voice first using CoreML encoder
        print("=" * 60)
        print("Step 1: Exporting voice from reference audio (CoreML)")
        print("=" * 60)

        if not args.reference.exists():
            print(f"Error: Reference file not found: {args.reference}")
            sys.exit(1)

        # Import and run voice export
        voice_bin_path = Path(f"{args.reference.stem}_audio_prompt.bin")

        try:
            # Use the new CoreML-based export
            sys.path.insert(0, str(SCRIPT_DIR))
            from export_voice_coreml import export_voice, find_encoder_model

            encoder_path = find_encoder_model()
            export_voice(
                audio_path=args.reference,
                output_path=voice_bin_path,
                encoder_path=encoder_path,
            )
            voice_path = voice_bin_path
        except Exception as e:
            print(f"Error exporting voice: {e}")
            print("Make sure mimi_encoder.mlpackage exists (run convert_mimi_encoder.py first)")
            sys.exit(1)

        print()
    elif args.voice:
        voice_path = Path(args.voice)
        if not voice_path.exists():
            # Try as built-in voice name
            const_dir = find_constants_dir()
            voice_path = const_dir / f"{args.voice}.safetensors"
            if not voice_path.exists():
                voice_path = const_dir / f"{args.voice}_audio_prompt.bin"
            if not voice_path.exists():
                print(f"Error: Voice not found: {args.voice}")
                sys.exit(1)
    else:
        print("Error: Must specify --voice or --reference")
        sys.exit(1)

    # Load voice and generate
    print("=" * 60)
    print("Step 2: Generating audio with CoreML")
    print("=" * 60)

    voice_emb = load_voice(voice_path)
    output_path = generate_with_voice(
        text=args.text,
        voice_emb=voice_emb,
        output_path=args.output,
        seed=args.seed,
        temperature=args.temperature,
    )

    # Evaluate if requested
    if args.evaluate and args.reference:
        print()
        print("=" * 60)
        print("Step 3: Evaluating voice similarity")
        print("=" * 60)

        try:
            from evaluate_voice import evaluate_voice_cloning
            metrics = evaluate_voice_cloning(args.reference, output_path)
        except ImportError:
            print("evaluate_voice.py not found, skipping evaluation")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
