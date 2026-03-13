"""Pure CoreML TTS generation for Magpie TTS 357M.

Pipeline:
1. Tokenize text
2. Encode text (text_encoder.mlpackage)
3. Prepare speaker context embedding
4. Autoregressive generation loop (decoder_step.mlpackage)
5. Local transformer codebook-by-codebook sampling (numpy)
6. Decode codec tokens to audio (nanocodec_decoder.mlpackage)
7. Save WAV

Dependencies: numpy, coremltools, soundfile
NeMo is required only for tokenization (could be replaced with standalone tokenizer).
"""
import json
import math
import os
import time

import numpy as np
import coremltools as ct
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONST_DIR = os.path.join(SCRIPT_DIR, "constants")
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")

# Decoder step output key names (from CoreML model spec)
DECODER_LOGITS_KEY = "var_2201"
DECODER_HIDDEN_KEY = "input"
# Output cache keys (input keys are cache0..cache11)
DECODER_CACHE_OUT_KEYS = [
    "new_cache_1", "new_cache_3", "new_cache_5", "new_cache_7",
    "new_cache_9", "new_cache_11", "new_cache_13", "new_cache_15",
    "new_cache_17", "new_cache_19", "new_cache_21", "new_cache",
]
DECODER_POSITION_KEYS = [
    "var_169", "var_346", "var_523", "var_700", "var_877", "var_1054",
    "var_1231", "var_1408", "var_1585", "var_1762", "var_1939", "var_2116",
]

# Forbidden token IDs (special tokens that should never be sampled)
# Allow EOS: forbid BOS(2016), CTX_BOS(2018), CTX_EOS(2019), MASK(2020), 2021-2023
FORBIDDEN_TOKENS_ALLOW_EOS = [2016, 2018, 2019, 2020, 2021, 2022, 2023]
# Forbid EOS too: also forbid EOS(2017)
FORBIDDEN_TOKENS_FORBID_EOS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]


def load_constants():
    with open(os.path.join(CONST_DIR, "constants.json")) as f:
        return json.load(f)


def load_speaker_embedding(speaker_idx):
    return np.load(os.path.join(CONST_DIR, f"speaker_{speaker_idx}.npy")).astype(np.float32)


def load_audio_embeddings(constants):
    tables = []
    for i in range(constants["num_audio_codebooks"]):
        tables.append(np.load(os.path.join(CONST_DIR, f"audio_embedding_{i}.npy")).astype(np.float32))
    return tables


def load_local_transformer():
    """Load local transformer weights from exported numpy files."""
    lt_dir = os.path.join(CONST_DIR, "local_transformer")
    lt = {
        "in_proj_weight": np.load(os.path.join(lt_dir, "in_proj_weight.npy")),
        "in_proj_bias": np.load(os.path.join(lt_dir, "in_proj_bias.npy")),
        "pos_emb": np.load(os.path.join(lt_dir, "pos_emb.npy")),
        "norm1_weight": np.load(os.path.join(lt_dir, "norm1_weight.npy")),
        "sa_qkv_weight": np.load(os.path.join(lt_dir, "sa_qkv_weight.npy")),
        "sa_o_weight": np.load(os.path.join(lt_dir, "sa_o_weight.npy")),
        "norm2_weight": np.load(os.path.join(lt_dir, "norm2_weight.npy")),
        "ffn_conv1_weight": np.load(os.path.join(lt_dir, "ffn_conv1_weight.npy")),
        "ffn_conv2_weight": np.load(os.path.join(lt_dir, "ffn_conv2_weight.npy")),
    }
    lt["out_proj_weights"] = []
    lt["out_proj_biases"] = []
    for i in range(8):
        lt["out_proj_weights"].append(np.load(os.path.join(lt_dir, f"out_proj_{i}_weight.npy")))
        lt["out_proj_biases"].append(np.load(os.path.join(lt_dir, f"out_proj_{i}_bias.npy")))
    return lt


def embed_audio_codes(codes, audio_embedding_tables, num_codebooks):
    """Embed audio codes for one frame. codes: (num_codebooks,) int."""
    embedding = None
    for c in range(num_codebooks):
        emb = audio_embedding_tables[c][codes[c]]
        embedding = emb if embedding is None else embedding + emb
    return (embedding / num_codebooks)[np.newaxis, np.newaxis, :]  # (1, 1, d_model)


def numpy_layer_norm(x, weight, eps=1e-5):
    """LayerNorm with no bias."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight


def numpy_gelu(x):
    """GELU (tanh approximation)."""
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def local_transformer_forward(sequence, lt_weights):
    """Run 1-layer causal transformer on sequence.

    Args:
        sequence: (T, 256) float32 — input sequence
        lt_weights: dict of weight arrays

    Returns:
        output: (T, 256) float32
    """
    T, D = sequence.shape

    # Add positional embeddings
    x = sequence + lt_weights["pos_emb"][:T]

    # Layer: pre-norm causal self-attention
    residual = x
    x_norm = numpy_layer_norm(x, lt_weights["norm1_weight"])

    # QKV projection (single head, d_model=256)
    qkv = x_norm @ lt_weights["sa_qkv_weight"].T  # (T, 768)
    q, k, v = np.split(qkv, 3, axis=-1)  # each (T, 256)

    scale = 1.0 / math.sqrt(D)
    attn = (q @ k.T) * scale  # (T, T)

    # Causal mask
    causal = np.tril(np.ones((T, T), dtype=np.float32))
    attn = attn * causal + (1 - causal) * (-1e9)
    attn = np.exp(attn - attn.max(axis=-1, keepdims=True))
    attn = attn / attn.sum(axis=-1, keepdims=True)

    sa_out = attn @ v  # (T, 256)
    sa_out = sa_out @ lt_weights["sa_o_weight"].T  # (T, 256)
    x = residual + sa_out

    # Layer: pre-norm FFN (kernel_size=1, so just matmuls)
    residual = x
    x_norm = numpy_layer_norm(x, lt_weights["norm2_weight"])

    # Conv1d with ks=1 is equivalent to linear: (T, 256) @ (256, 1024) → (T, 1024)
    ffn_w1 = lt_weights["ffn_conv1_weight"].squeeze(-1)  # (1024, 256)
    ffn_w2 = lt_weights["ffn_conv2_weight"].squeeze(-1)  # (256, 1024)
    h = numpy_gelu(x_norm @ ffn_w1.T)
    h = h @ ffn_w2.T
    x = residual + h

    return x


def sample_topk(logits, temperature=0.6, topk=80):
    """Top-k sampling with temperature."""
    if topk > 0:
        topk_vals = np.partition(logits, -topk)[-topk:]
        threshold = topk_vals.min()
        logits = np.where(logits >= threshold, logits, -np.inf)

    logits = logits / max(temperature, 1e-8)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)


def local_transformer_sample(decoder_hidden, lt_weights, audio_emb_tables,
                             num_codebooks, temperature, topk, forbid_eos,
                             uncond_decoder_hidden=None, cfg_scale=1.0):
    """Sample codebook tokens autoregressively using local transformer.

    When CFG is enabled, runs both conditional and unconditional sequences through
    the local transformer and applies CFG at the codebook logit level (matching NeMo).

    Args:
        decoder_hidden: (768,) float32 — conditional decoder hidden state
        lt_weights: dict of LT weight arrays
        audio_emb_tables: list of (2024, 768) embedding tables
        num_codebooks: 8
        temperature: sampling temperature
        topk: top-k value
        forbid_eos: whether to forbid EOS token
        uncond_decoder_hidden: (768,) float32 — unconditional decoder hidden (for CFG)
        cfg_scale: CFG scale factor

    Returns:
        codes: (num_codebooks,) int32 — sampled codebook tokens
    """
    forbidden = FORBIDDEN_TOKENS_FORBID_EOS if forbid_eos else FORBIDDEN_TOKENS_ALLOW_EOS
    use_cfg = uncond_decoder_hidden is not None and cfg_scale != 1.0

    in_proj_w = lt_weights["in_proj_weight"]  # (256, 768)
    in_proj_b = lt_weights["in_proj_bias"]  # (256,)

    # Project decoder hidden → LT dim
    cond_input = decoder_hidden @ in_proj_w.T + in_proj_b  # (256,)
    cond_seq = cond_input[np.newaxis, :]  # (1, 256)

    if use_cfg:
        uncond_input = uncond_decoder_hidden @ in_proj_w.T + in_proj_b
        uncond_seq = uncond_input[np.newaxis, :]

    codes = np.zeros(num_codebooks, dtype=np.int32)

    for cb_idx in range(num_codebooks):
        # Run conditional LT
        cond_out = local_transformer_forward(cond_seq, lt_weights)
        out_w = lt_weights["out_proj_weights"][cb_idx]
        out_b = lt_weights["out_proj_biases"][cb_idx]
        cond_logits = cond_out[-1] @ out_w.T + out_b  # (2024,)

        if use_cfg:
            # Run unconditional LT
            uncond_out = local_transformer_forward(uncond_seq, lt_weights)
            uncond_logits = uncond_out[-1] @ out_w.T + out_b
            # Apply CFG at logit level
            cb_logits = cfg_scale * cond_logits + (1.0 - cfg_scale) * uncond_logits
        else:
            cb_logits = cond_logits

        # Clear forbidden tokens
        for tok_id in forbidden:
            if tok_id < len(cb_logits):
                cb_logits[tok_id] = -np.inf

        # Sample
        codes[cb_idx] = sample_topk(cb_logits, temperature, topk)

        # Embed sampled token and project to LT dim for next step
        token_emb = audio_emb_tables[cb_idx][codes[cb_idx]]  # (768,)
        next_input = token_emb @ in_proj_w.T + in_proj_b  # (256,)
        cond_seq = np.concatenate([cond_seq, next_input[np.newaxis, :]], axis=0)
        if use_cfg:
            # Unconditional path uses same sampled token (matching NeMo line 94)
            uncond_seq = np.concatenate([uncond_seq, next_input[np.newaxis, :]], axis=0)

    return codes


def generate(
    text: str,
    speaker: int = 0,
    language: str = "en",
    output_path: str = "magpie_output.wav",
    temperature: float = 0.6,
    topk: int = 80,
    max_steps: int = 500,
    seed: int = 42,
    use_cfg: bool = True,
    cfg_scale: float = 2.5,
):
    np.random.seed(seed)
    constants = load_constants()

    num_codebooks = constants["num_audio_codebooks"]
    audio_bos_id = constants["special_tokens"]["audio_bos_id"]
    audio_eos_id = constants["special_tokens"]["audio_eos_id"]
    sample_rate = constants["output_sample_rate"]
    d_model = constants["decoder"]["d_model"]
    n_layers = constants["decoder"]["n_layers"]
    sa_n_heads = constants["decoder"]["sa_n_heads"]
    d_head = d_model // sa_n_heads
    max_text_len = 256
    max_seq_len = 512
    min_frames = constants["inference"].get("min_generated_frames", 4)

    print(f"Text: '{text}'")
    print(f"Speaker: {speaker}, Language: {language}")
    if use_cfg:
        print(f"CFG scale: {cfg_scale}")

    # 1. Tokenize text
    print("Tokenizing text...")
    text_tokens = _tokenize_text(text, language, constants)
    T_text = len(text_tokens)
    print(f"  Tokens: {T_text}")

    text_tokens_padded = np.zeros(max_text_len, dtype=np.int32)
    text_tokens_padded[:T_text] = text_tokens
    text_mask = np.zeros(max_text_len, dtype=np.float32)
    text_mask[:T_text] = 1.0

    # 2. Load CoreML models
    print("Loading CoreML models...")
    text_encoder = ct.models.MLModel(
        os.path.join(BUILD_DIR, "text_encoder.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    decoder_step = ct.models.MLModel(
        os.path.join(BUILD_DIR, "decoder_step.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    nanocodec = ct.models.MLModel(
        os.path.join(BUILD_DIR, "nanocodec_decoder.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    # 3. Encode text
    print("Encoding text...")
    enc_out = text_encoder.predict({
        "text_tokens": text_tokens_padded[np.newaxis, :],
        "text_mask": text_mask[np.newaxis, :],
    })
    encoder_output = enc_out["encoder_output"]  # (1, max_text_len, d_model)

    # For CFG: create unconditional encoder output and mask (matching NeMo's prepare_dummy_cond_for_cfg)
    if use_cfg:
        uncond_encoder_output = np.zeros_like(encoder_output)
        # NeMo keeps first position unmasked (value is zero) for numerical stability
        uncond_text_mask = np.zeros_like(text_mask)
        uncond_text_mask[0] = 1.0

    # 4. Load speaker context + audio embeddings + local transformer
    print(f"Loading speaker {speaker} embedding...")
    speaker_emb = load_speaker_embedding(speaker)
    T_ctx = speaker_emb.shape[0]
    audio_emb_tables = load_audio_embeddings(constants)
    lt_weights = load_local_transformer()

    # 5. Initialize KV caches (conditional)
    def make_caches():
        c, p = {}, {}
        for i in range(n_layers):
            c[f"cache{i}"] = np.zeros((2, 1, max_seq_len, sa_n_heads, d_head), dtype=np.float32)
            p[f"position{i}"] = np.array([0.0], dtype=np.float32)
        return c, p

    caches, positions = make_caches()
    if use_cfg:
        uncond_caches, uncond_positions = make_caches()

    def run_decoder_step(audio_embed_np, enc_out_np, mask_np, cache_dict, pos_dict):
        step_inputs = {
            "audio_embed": audio_embed_np.astype(np.float32),
            "encoder_output": enc_out_np.astype(np.float32),
            "encoder_mask": mask_np[np.newaxis, :].astype(np.float32),
        }
        step_inputs.update(cache_dict)
        step_inputs.update(pos_dict)
        step_out = decoder_step.predict(step_inputs)
        for i in range(n_layers):
            # Output cache keys differ from input keys after scatter-based cache rewrite
            cache_dict[f"cache{i}"] = step_out[DECODER_CACHE_OUT_KEYS[i]]
            pos_dict[f"position{i}"] = step_out[DECODER_POSITION_KEYS[i]]
        return step_out[DECODER_HIDDEN_KEY]  # (1, 1, d_model) — decoder hidden

    # 6. Prefill context
    # Conditional path: real speaker context + real encoder output
    # Unconditional path (CFG): ZERO context + zero encoder output (matching NeMo's dummy_additional_decoder_input)
    print(f"Prefilling {T_ctx} context tokens into decoder...")
    uncond_ctx_token = np.zeros((1, 1, d_model), dtype=np.float32)
    for t in range(T_ctx):
        ctx_token = speaker_emb[np.newaxis, np.newaxis, t, :]
        run_decoder_step(ctx_token, encoder_output, text_mask, caches, positions)
        if use_cfg:
            run_decoder_step(uncond_ctx_token, uncond_encoder_output, uncond_text_mask, uncond_caches, uncond_positions)
        if (t + 1) % 50 == 0:
            print(f"  Prefilled {t + 1}/{T_ctx}")
    print(f"  Prefill done. Position: {positions['position0'][0]:.0f}")

    # 7. Autoregressive generation with local transformer
    print(f"\nGenerating (max {max_steps} steps)...")
    start_time = time.time()

    current_codes = np.full(num_codebooks, audio_bos_id, dtype=np.int32)
    all_predictions = []

    for step in range(max_steps):
        # Embed current audio codes
        audio_embed = embed_audio_codes(current_codes, audio_emb_tables, num_codebooks)

        # Run conditional decoder step → get hidden state
        cond_hidden = run_decoder_step(audio_embed, encoder_output, text_mask, caches, positions)

        if use_cfg:
            # Run unconditional decoder step
            uncond_hidden = run_decoder_step(
                audio_embed, uncond_encoder_output, uncond_text_mask, uncond_caches, uncond_positions
            )
            uncond_decoder_hidden = uncond_hidden[0, 0]  # (d_model,)

        decoder_hidden = cond_hidden[0, 0]  # (d_model,)

        # Local transformer: sample codebooks autoregressively
        # CFG is applied at the codebook logit level inside the LT (matching NeMo)
        forbid_eos = (step < min_frames)
        next_codes = local_transformer_sample(
            decoder_hidden, lt_weights, audio_emb_tables,
            num_codebooks, temperature, topk, forbid_eos,
            uncond_decoder_hidden=uncond_decoder_hidden if use_cfg else None,
            cfg_scale=cfg_scale if use_cfg else 1.0,
        )

        # Check EOS
        is_eos = np.any(next_codes == audio_eos_id)
        if is_eos and step >= min_frames:
            print(f"  EOS at step {step}")
            break

        all_predictions.append(next_codes.copy())
        current_codes = next_codes

        if step % 20 == 0:
            print(f"  Step {step}...")

    gen_time = time.time() - start_time
    num_frames = len(all_predictions)
    print(f"Generated {num_frames} frames in {gen_time:.2f}s")

    if num_frames == 0:
        print("No audio generated!")
        return

    # 8. Decode with NanoCodec
    print("Decoding audio with NanoCodec...")
    predicted_codes = np.stack(all_predictions, axis=1)  # (num_cb, T_total)
    T_total = predicted_codes.shape[1]

    max_frames = 256
    if T_total > max_frames:
        print(f"  Warning: {T_total} frames > max {max_frames}, truncating")
        predicted_codes = predicted_codes[:, :max_frames]
        T_total = max_frames

    padded = np.zeros((num_codebooks, max_frames), dtype=np.int32)
    padded[:, :T_total] = predicted_codes

    codec_out = nanocodec.predict({
        "tokens": padded[np.newaxis, :, :].astype(np.int32),
    })

    audio = codec_out["audio"]
    if audio.ndim > 1:
        audio = audio.flatten()

    expected_samples = T_total * constants["codec_samples_per_frame"]
    audio = audio[:expected_samples]

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9

    sf.write(output_path, audio, sample_rate)
    duration = len(audio) / sample_rate
    rtf = gen_time / duration if duration > 0 else float("inf")
    print(f"\nSaved to {output_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"RTF: {rtf:.2f}x (generation time / audio duration)")


def _tokenize_text(text, language, constants):
    """Tokenize text using NeMo tokenizer."""
    try:
        from nemo.collections.tts.models import MagpieTTSModel
        from nemo.collections.tts.parts.utils.tts_dataset_utils import (
            chunk_and_tokenize_text_by_sentence,
        )

        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")

        language_tokenizer_map = {
            "en": ["english_phoneme", "english"],
            "es": ["spanish_phoneme", "spanish"],
            "de": ["german_phoneme", "german"],
            "fr": ["french_chartokenizer", "french"],
            "zh": ["mandarin_phoneme", "mandarin"],
            "ja": ["japanese_phoneme", "japanese"],
            "hi": ["hindi_chartokenizer", "hindi"],
            "it": ["italian_phoneme", "italian"],
            "vi": ["vietnamese_phoneme", "vietnamese"],
        }
        available_tokenizers = list(model.tokenizer.tokenizers.keys())
        tokenizer_name = available_tokenizers[0]
        if language in language_tokenizer_map:
            for candidate in language_tokenizer_map[language]:
                if candidate in available_tokenizers:
                    tokenizer_name = candidate
                    break

        # Force phoneme_probability=1.0 for deterministic phoneme lookup.
        # Default 0.8 randomly falls back to graphemes for ~20% of words,
        # which degrades synthesis quality since the model was trained on phonemes.
        tok = model.tokenizer.tokenizers.get(tokenizer_name)
        if tok is not None and hasattr(tok, "g2p") and hasattr(tok.g2p, "phoneme_probability"):
            tok.g2p.phoneme_probability = 1.0

        tokens, tokens_len, _ = chunk_and_tokenize_text_by_sentence(
            text=text,
            tokenizer_name=tokenizer_name,
            text_tokenizer=model.tokenizer,
            eos_token_id=model.eos_id,
        )
        return tokens[0].numpy().astype(np.int32)
    except ImportError:
        raise RuntimeError(
            "NeMo toolkit required for tokenization. "
            "Install with: uv sync --extra nemo"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate speech with Magpie TTS CoreML")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker index (0-4)")
    parser.add_argument("--language", type=str, default="en", help="Language code")
    parser.add_argument("--output", type=str, default="magpie_output.wav", help="Output WAV path")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=80, help="Top-k for sampling")
    parser.add_argument("--max-steps", type=int, default=500, help="Max decoder steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-cfg", action="store_true", help="Disable classifier-free guidance")
    parser.add_argument("--cfg-scale", type=float, default=2.5, help="CFG scale")
    args = parser.parse_args()

    generate(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        output_path=args.output,
        temperature=args.temperature,
        topk=args.topk,
        max_steps=args.max_steps,
        seed=args.seed,
        use_cfg=not args.no_cfg,
        cfg_scale=args.cfg_scale,
    )
