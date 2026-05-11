"""Pure CoreML TTS generation — zero PyTorch dependency.

Dependencies: numpy, sentencepiece, safetensors, coremltools, scipy
NO torch import anywhere.

Pipeline:
1. Text prep (string ops)
2. Tokenize (sentencepiece)
3. Embed text (numpy lookup)
4. Load voice (safetensors)
5. KV cache prefill (CoreML backbone)
6. Autoregressive generation (CoreML step + flow_decoder + mimi)

Multi-language support: pass `--language <id>` to read converted assets
from `build/<id>/` instead of the legacy `<script_dir>/constants/` +
`<script_dir>/*.mlpackage` layout. When `--language` is omitted, falls
back to the legacy layout for backward compatibility with existing
setups.
"""
import argparse
import os
import sys
import time
import numpy as np
import sentencepiece as sp
import coremltools as ct
import re
import scipy.io.wavfile as wavfile
from safetensors.numpy import load_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_CONST_DIR = os.path.join(SCRIPT_DIR, "constants")
LEGACY_MODEL_DIR = SCRIPT_DIR

# Make the _language_arg helper importable even when running as a script.
sys.path.insert(0, os.path.join(SCRIPT_DIR, "convert_models", "convert"))
from _language_arg import SUPPORTED_LANGUAGES, build_output_dir  # noqa: E402


def _load_voice_embedding(const_dir: str, model_dir: str, voice: str) -> np.ndarray:
    """Load a voice's `[1, prompt_len, 1024]` conditioning latent (v1 format).

    Used by `generate_v4` only when the voice file carries the legacy
    `audio_prompt` tensor (pre-v2.0.0 upstream). Returns `None` when the
    voice file exists but is in the new v2 KV-cache format; callers should
    fall back to `_load_voice_state_v2` in that case.

    Tries, in order:
    1. `<const_dir>/<voice>.safetensors` (legacy English layout).
    2. `<model_dir>/constants_bin/<voice>_audio_prompt.bin` (v1 packer output).
    """
    st_path = os.path.join(const_dir, f"{voice}.safetensors")
    if os.path.isfile(st_path):
        tensors = load_file(st_path)
        if 'audio_prompt' in tensors:
            emb = tensors['audio_prompt']
            if emb.ndim == 2:
                emb = emb[np.newaxis, :, :]
            return emb.astype(np.float32)
        # v2 KV-cache safetensors — caller should use _load_voice_state_v2.
        return None

    bin_path = os.path.join(model_dir, "constants_bin", f"{voice}_audio_prompt.bin")
    if os.path.isfile(bin_path):
        raw = np.fromfile(bin_path, dtype=np.float32)
        if raw.size % 1024 != 0:
            raise ValueError(
                f"{bin_path} size {raw.size} floats is not a multiple of 1024"
            )
        prompt_len = raw.size // 1024
        return raw.reshape(1, prompt_len, 1024)

    return None


def _locate_voice_v2(const_dir: str, model_dir: str, voice: str) -> str | None:
    """Return the path to the v2 voice safetensors, or None if not present."""
    for candidate in (
        os.path.join(model_dir, "constants_bin", f"{voice}.safetensors"),
        os.path.join(const_dir, f"{voice}.safetensors"),
    ):
        if os.path.isfile(candidate):
            try:
                tensors = load_file(candidate)
            except Exception:
                continue
            if any(k.endswith("/cache") for k in tensors.keys()):
                return candidate
    return None


def _load_voice_state_v2(path: str, cache_slots: int, num_layers: int) -> dict:
    """Load a v2 voice safetensors into a CoreML-ready initial KV cache.

    Input tensor layout (per layer N in 0..num_layers-1):
      transformer.layers.N.self_attn/offset  int64 [1]
      transformer.layers.N.self_attn/cache   float32 [2, 1, prompt_len, 16, 64]

    Output dict matches `cond_step`/`flowlm_step` initial-state inputs:
      cache0..cache{num_layers-1}      float32 [2, 1, cache_slots, 16, 64]
      position0..position{num_layers-1} float32 [1]   (= prompt_len per layer)

    Any prompt_len < cache_slots is zero-padded along axis 2 (time).
    """
    tensors = load_file(path)

    caches: dict[str, np.ndarray] = {}
    positions: dict[str, np.ndarray] = {}
    prompt_lens = []

    for layer in range(num_layers):
        cache_key = f"transformer.layers.{layer}.self_attn/cache"
        offset_key = f"transformer.layers.{layer}.self_attn/offset"
        if cache_key not in tensors or offset_key not in tensors:
            raise KeyError(
                f"{path}: missing layer {layer} entries "
                f"({cache_key!r} / {offset_key!r}). Have: {sorted(tensors.keys())}"
            )
        raw_cache = tensors[cache_key].astype(np.float32)
        if raw_cache.ndim != 5 or raw_cache.shape[0] != 2 or raw_cache.shape[1] != 1:
            raise ValueError(
                f"{path}: layer {layer} cache has shape {raw_cache.shape}; "
                f"expected [2, 1, prompt_len, 16, 64]"
            )
        prompt_len = raw_cache.shape[2]
        if prompt_len > cache_slots:
            raise ValueError(
                f"{path}: layer {layer} prompt_len={prompt_len} exceeds "
                f"CoreML cache_slots={cache_slots}"
            )
        padded = np.zeros(
            (2, 1, cache_slots, raw_cache.shape[3], raw_cache.shape[4]),
            dtype=np.float32,
        )
        padded[:, :, :prompt_len, :, :] = raw_cache
        caches[f"cache{layer}"] = padded

        raw_offset = tensors[offset_key]
        offset_val = float(np.asarray(raw_offset).reshape(-1)[0])
        positions[f"position{layer}"] = np.array([offset_val], dtype=np.float32)
        prompt_lens.append(int(offset_val))

    print(
        f"  v2 voice state: {num_layers} layers, prompt_len(s)={prompt_lens}, "
        f"padded to cache_slots={cache_slots}"
    )
    return {"caches": caches, "positions": positions, "prompt_len": max(prompt_lens)}


def _introspect_cache_shape(coreml_model) -> tuple[int, int]:
    """Return `(num_layers, cache_slots)` from a CoreML model's input spec.

    Inspects inputs named `cache0`, `cache1`, ... to count layers and read
    the cache length (axis 2 of shape [2, 1, cache_slots, 16, 64]).
    """
    spec = coreml_model.get_spec()
    num_layers = 0
    cache_slots = None
    for inp in spec.description.input:
        if inp.name.startswith("cache") and inp.name[len("cache"):].isdigit():
            num_layers = max(num_layers, int(inp.name[len("cache"):]) + 1)
            if cache_slots is None:
                shape = list(inp.type.multiArrayType.shape)
                # Expected [2, 1, cache_slots, 16, 64]; guard against unknown layouts.
                if len(shape) == 5:
                    cache_slots = int(shape[2])
    if num_layers == 0 or cache_slots is None:
        raise RuntimeError(
            "Could not introspect cache shape from CoreML model "
            f"(num_layers={num_layers}, cache_slots={cache_slots})"
        )
    return num_layers, cache_slots


def _introspect_cond_output_keys(coreml_cond, num_layers: int) -> tuple[list[str], list[str]]:
    """Return `(cache_output_names, pos_output_names)` for cond_step outputs.

    The traceable wrapper's `forward` returns outputs interleaved as
    (new_cache0, new_pos0, new_cache1, new_pos1, ..., new_cacheN-1, new_posN-1).
    CoreML preserves that positional order in `spec.description.output`, but
    may rename entries to `var_NNN` / `new_cache_NN_internal_tensor_assign_2`.

    We classify by rank: the cache outputs are 5-D ([2, 1, cache_slots, 16, 64]),
    while the position outputs are 1-D ([1]).
    """
    spec = coreml_cond.get_spec()
    cache_names = []
    pos_names = []
    for out in spec.description.output:
        if not out.type.HasField("multiArrayType"):
            continue
        rank = len(out.type.multiArrayType.shape)
        if rank == 5:
            cache_names.append(out.name)
        elif rank == 1:
            pos_names.append(out.name)
    if len(cache_names) != num_layers or len(pos_names) != num_layers:
        raise RuntimeError(
            f"cond_step outputs don't match num_layers={num_layers}: "
            f"found {len(cache_names)} caches, {len(pos_names)} positions"
        )
    return cache_names, pos_names


def _introspect_step_output_keys(
    coreml_step, num_layers: int
) -> tuple[str, str, list[str], list[str]]:
    """Return `(transformer_name, eos_name, cache_names, pos_names)` for flowlm_step.

    Traceable forward returns (x, is_eos, new_cache0, new_pos0, ...). We
    classify by rank:
      - `x` is 3-D `[1, 1, 1024]` → transformer_out
      - `is_eos` is 3-D `[1, 1, 1]` → EOS logit
      - caches are 5-D
      - positions are 1-D
    When two 3-D outputs are present, the one whose last dim is 1 is EOS.
    """
    spec = coreml_step.get_spec()
    rank3, cache_names, pos_names = [], [], []
    for out in spec.description.output:
        if not out.type.HasField("multiArrayType"):
            continue
        shape = list(out.type.multiArrayType.shape)
        rank = len(shape)
        if rank == 3:
            rank3.append((out.name, shape))
        elif rank == 5:
            cache_names.append(out.name)
        elif rank == 1:
            pos_names.append(out.name)
    if len(rank3) != 2:
        raise RuntimeError(f"flowlm_step: expected 2 rank-3 outputs, got {len(rank3)}: {rank3}")
    if len(cache_names) != num_layers or len(pos_names) != num_layers:
        raise RuntimeError(
            f"flowlm_step outputs don't match num_layers={num_layers}: "
            f"found {len(cache_names)} caches, {len(pos_names)} positions"
        )
    # Disambiguate: EOS has last dim == 1; transformer_out has last dim == 1024.
    eos_candidates = [n for n, s in rank3 if s[-1] == 1]
    xfmr_candidates = [n for n, s in rank3 if s[-1] != 1]
    if len(eos_candidates) != 1 or len(xfmr_candidates) != 1:
        raise RuntimeError(f"flowlm_step: cannot disambiguate rank-3 outputs: {rank3}")
    return xfmr_candidates[0], eos_candidates[0], cache_names, pos_names


def resolve_asset_dirs(language: str | None) -> tuple[str, str]:
    """Return `(constants_dir, model_dir)` for the selected language.

    - When `language` is None: use the legacy layout (backward compat).
    - When `language` is set: read from `build/<language>/constants/` and
      `build/<language>/*.mlpackage` produced by the convert scripts.
    """
    if language is None:
        return LEGACY_CONST_DIR, LEGACY_MODEL_DIR
    base = build_output_dir(SCRIPT_DIR, language)
    return os.path.join(base, "constants"), base

def prepare_text_prompt(text: str):
    """Normalize text for TTS (pure string ops, no PyTorch)."""
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


def generate_v4(
    text: str,
    voice: str = "alba",
    output_path: str = "coreml_v4.wav",
    seed: int = 42,
    language: str | None = None,
    compute_units: str | None = None,
):
    """Generate audio using pure CoreML — no PyTorch.

    When `language` is `None`, reads models + constants from the legacy
    `<script_dir>/` layout. Otherwise reads from `build/<language>/`
    (produced by `convert_all_languages.sh`).
    """
    const_dir, model_dir = resolve_asset_dirs(language)
    print(f"Text: '{text}'")
    print(f"Voice: {voice}")
    print(f"Seed: {seed}")
    print(f"Language: {language or '<legacy>'}")
    print(f"Constants dir: {const_dir}")
    print(f"Model dir: {model_dir}")

    # 1. Text preparation
    prepared_text, frames_after_eos = prepare_text_prompt(text)
    frames_after_eos += 2
    print(f"Prepared: '{prepared_text}' (frames_after_eos={frames_after_eos})")

    # 2. Tokenize — prefer constants_bin/tokenizer.model (produced by
    # pack_constants_bin.py), fall back to legacy constants/ layout.
    tokenizer_candidates = [
        os.path.join(model_dir, "constants_bin", "tokenizer.model"),
        os.path.join(const_dir, "tokenizer.model"),
    ]
    tokenizer_path = next((p for p in tokenizer_candidates if os.path.isfile(p)), None)
    if tokenizer_path is None:
        raise FileNotFoundError(
            f"tokenizer.model not found in any of: {tokenizer_candidates}"
        )
    tokenizer = sp.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    token_ids = tokenizer.encode(prepared_text)
    print(f"Tokens: {len(token_ids)} ids  (tokenizer: {tokenizer_path})")

    # 3. Embed text (numpy lookup)
    embed_table = np.load(os.path.join(const_dir, "text_embed_table.npy"))
    text_emb = embed_table[token_ids]  # [T_text, 1024]
    text_emb = text_emb[np.newaxis, :, :]  # [1, T_text, 1024]
    print(f"Text embeddings: {text_emb.shape}")

    # 4. Try loading voice conditioning in both formats.
    voice_emb_v1 = _load_voice_embedding(const_dir, model_dir, voice)  # legacy audio_prompt
    voice_v2_path = _locate_voice_v2(const_dir, model_dir, voice)     # v2 KV cache
    if voice_emb_v1 is None and voice_v2_path is None:
        raise FileNotFoundError(
            f"No voice data for '{voice}' found under {const_dir} or "
            f"{os.path.join(model_dir, 'constants_bin')}."
        )
    if voice_emb_v1 is not None:
        print(f"Voice embeddings (v1 audio_prompt): {voice_emb_v1.shape}")
    else:
        print(f"Voice KV cache (v2) from: {voice_v2_path}")

    # 5. Load constants. The v2 traced mimi_decoder bakes denormalize +
    # quantizer projection internally and accepts a raw [1, 32] latent, so
    # the legacy emb_mean / emb_std / quantizer_weight / mimi_init_state
    # files are no longer consumed.
    bos_emb = np.load(os.path.join(const_dir, "bos_emb.npy"))
    # bos_before_voice is prepended to the v1 audio_prompt during cond_step
    # prefill (matches pocket-tts 2.0.0's `flow_lm.bos_before_voice`). Without
    # it the cond_step output diverges from the deployed v2 cache state and
    # the flow LM emits EOS within a few steps (silent / garbled audio).
    # v2 voices have this token already baked into their KV state.
    bos_before_voice_path = os.path.join(const_dir, "bos_before_voice.npy")
    bos_before_voice = (
        np.load(bos_before_voice_path) if os.path.isfile(bos_before_voice_path) else None
    )

    # 6. Load CoreML models — per-model compute-unit strategy (profiled M-series, FP16):
    #
    #   model         units            median_predict_ms   rationale
    #   ────────────  ───────────────  ──────────────────  ──────────────────────────
    #   cond_step     CPU_AND_GPU      20.42               ANE ≈ GPU (indifferent),
    #                                                      prefer GPU to AVOID a rare
    #                                                      MPSGraph rank-5/zero-shape
    #                                                      assert during prefill on ANE
    #   flowlm_step   ALL              19.91               1.97× faster than GPU;
    #                                                      autoregressive bottleneck
    #   flow_decoder  CPU_AND_NE       0.39                tiny model, pure ANE wins
    #                                                      (and 8 calls/frame)
    #   mimi_decoder  CPU_ONLY          9.90               1.74× faster than CPU+GPU;
    #                                                      cannot use ANE (segfaults
    #                                                      on 64-byte stride misalign)
    #
    # When `--compute-units` is ALL or omitted we apply the per-model overrides
    # above. When the caller explicitly picks CPU_ONLY / CPU_AND_GPU / CPU_AND_NE
    # we honor it for all 4 models (still pin mimi off ANE to avoid segfault).
    units_name = compute_units or "CPU_AND_GPU"
    if units_name == "ALL":
        cond_units_name = "CPU_AND_GPU"
        step_units_name = "ALL"
        flow_units_name = "CPU_AND_NE"
        mimi_units_name = "CPU_ONLY"
    elif units_name == "CPU_AND_NE":
        # Pure ANE: cond_step is slower here so leave it on CPU+GPU; mimi can't
        # do ANE so pin to CPU+GPU.
        cond_units_name = "CPU_AND_GPU"
        step_units_name = "CPU_AND_NE"
        flow_units_name = "CPU_AND_NE"
        mimi_units_name = "CPU_AND_GPU"
    else:
        cond_units_name = step_units_name = flow_units_name = units_name
        mimi_units_name = "CPU_ONLY" if units_name == "CPU_AND_GPU" else units_name

    cond_units = getattr(ct.ComputeUnit, cond_units_name)
    step_units = getattr(ct.ComputeUnit, step_units_name)
    flow_units = getattr(ct.ComputeUnit, flow_units_name)
    mimi_units = getattr(ct.ComputeUnit, mimi_units_name)
    # Maintained for log-summary compatibility.
    units = step_units
    print(
        f"\nLoading CoreML models (units: cond={cond_units_name}, "
        f"flowlm={step_units_name}, flow={flow_units_name}, mimi={mimi_units_name})..."
    )
    t_load0 = time.time()
    coreml_cond = ct.models.MLModel(
        os.path.join(model_dir, 'cond_step.mlpackage'),
        compute_units=cond_units,
    )
    coreml_step = ct.models.MLModel(
        os.path.join(model_dir, 'flowlm_step.mlpackage'),
        compute_units=step_units,
    )
    coreml_flow = ct.models.MLModel(
        os.path.join(model_dir, 'flow_decoder.mlpackage'),
        compute_units=flow_units,
    )
    coreml_mimi = ct.models.MLModel(
        os.path.join(model_dir, 'mimi_decoder.mlpackage'),
        compute_units=mimi_units,
    )
    load_time = time.time() - t_load0
    print(f"Loaded 4 mlpackages in {load_time:.2f}s")

    # 7. Introspect cache shape + output names from CoreML models so we
    # handle both 6-layer and 24-layer packs without hard-coding.
    num_layers, cache_slots = _introspect_cache_shape(coreml_step)
    print(f"flowlm_step cache shape: num_layers={num_layers}, cache_slots={cache_slots}")
    cond_cache_keys, cond_pos_keys = _introspect_cond_output_keys(coreml_cond, num_layers)
    step_xfmr_key, step_eos_key, step_cache_keys, step_pos_keys = (
        _introspect_step_output_keys(coreml_step, num_layers)
    )

    # 8. Initialize KV caches.
    #   - v2: seed from precomputed voice KV cache (skip voice prefill via cond_step)
    #   - v1: start from zeros and prefill with [voice, text] through cond_step
    if voice_v2_path is not None:
        voice_state = _load_voice_state_v2(voice_v2_path, cache_slots, num_layers)
        caches = voice_state["caches"]
        positions = voice_state["positions"]
        prefill_tokens = text_emb  # voice already baked into the KV state
        print(f"Voice-seeded KV state; prefilling text only: {prefill_tokens.shape[1]} tokens")
    else:
        caches = {}
        positions = {}
        for i in range(num_layers):
            caches[f'cache{i}'] = np.zeros(
                (2, 1, cache_slots, 16, 64), dtype=np.float32
            )
            positions[f'position{i}'] = np.array([0.0], dtype=np.float32)
        if bos_before_voice is None:
            raise FileNotFoundError(
                f"Missing {bos_before_voice_path}. The v1 cond_step prefill path "
                "requires the bos_before_voice constant (pocket-tts 2.0.0's "
                "`flow_lm.bos_before_voice`). Regenerate it with "
                "coreml/voice_cloning/export_constants.py or switch the voice to v2."
            )
        bos_voice_text = [bos_before_voice, voice_emb_v1, text_emb]
        prefill_tokens = np.concatenate(bos_voice_text, axis=1).astype(np.float32)
        print(
            f"Conditioning: {prefill_tokens.shape[1]} tokens "
            f"(bos=1, voice={voice_emb_v1.shape[1]}, text={text_emb.shape[1]})"
        )

    # 9. Prefill remaining conditioning tokens (v2: text only; v1: voice+text).
    prefill_len = prefill_tokens.shape[1]
    print(f"Prefilling KV cache ({prefill_len} tokens)...")
    t_prefill0 = time.time()
    for tok_idx in range(prefill_len):
        cond_token = prefill_tokens[:, tok_idx:tok_idx + 1, :]  # [1, 1, 1024]
        cond_inputs = {
            'conditioning': cond_token.astype(np.float32),
            **caches,
            **positions,
        }
        cond_out = coreml_cond.predict(cond_inputs)

        for i in range(num_layers):
            caches[f'cache{i}'] = cond_out[cond_cache_keys[i]]
            positions[f'position{i}'] = cond_out[cond_pos_keys[i]]

    start_pos = positions['position0'][0]
    prefill_time = time.time() - t_prefill0
    print(f"KV cache filled to position: {start_pos} (prefill {prefill_time:.2f}s, "
          f"{prefill_len / max(prefill_time, 1e-6):.1f} tok/s)")

    # 10. Autoregressive generation loop (pure CoreML)
    gen_len_sec = len(text.split()) * 1 + 2.0
    max_gen_len = int(gen_len_sec * 12.5)
    print(f"\nGenerating (max {max_gen_len} frames)...")

    np.random.seed(seed)

    t_gen0 = time.time()
    t_step_total = 0.0
    t_flow_total = 0.0
    t_mimi_total = 0.0

    audio_chunks = []
    eos_step = None
    sequence = np.full((1, 1, 32), float('nan'), dtype=np.float32)
    num_lsd_steps = 8
    dt = 1.0 / num_lsd_steps
    temp = 0.7

    # Initialize Mimi state. v2.0.0 traced mimi_decoder schema:
    #   - input `latent` is raw [1, 32] (denorm + quantize are baked in)
    #   - 24 state inputs named after MIMI_STATE_SPEC (attn_end_offset dropped)
    #   - outputs are [audio, *24 updated states] in that positional order;
    #     names may be `var_NNN` due to CoreML's output preprocess renames.
    coreml_mimi_state = {}
    mimi_spec = coreml_mimi.get_spec()
    mimi_input_order = []  # SPEC order minus 'latent' — needed to pair new state → input name
    for inp in mimi_spec.description.input:
        if inp.name == "latent":
            continue
        shape = tuple(int(d) for d in inp.type.multiArrayType.shape)
        # `*_first` boolean flags must start at 1.0 so the streaming convs take
        # the cold-start replicate-padding path (see traceable_mimi_decoder.py
        # `_functional_streaming_conv1d_forward`: `previous = where(first, init,
        # previous)`). All other state tensors are zero-initialized.
        if inp.name.endswith("_first"):
            coreml_mimi_state[inp.name] = np.ones(shape, dtype=np.float32)
        else:
            coreml_mimi_state[inp.name] = np.zeros(shape, dtype=np.float32)
        mimi_input_order.append(inp.name)
    # Output positional order: [audio, state0, state1, ..., state23].
    mimi_output_names = [out.name for out in mimi_spec.description.output]
    assert len(mimi_output_names) == 1 + len(mimi_input_order), (
        f"mimi output count {len(mimi_output_names)} != 1 + state inputs {len(mimi_input_order)}"
    )

    for step in range(max_gen_len):
        # Step model
        step_inputs = {
            'sequence': sequence,
            'bos_emb': bos_emb,
            **caches,
            **positions,
        }
        _t0 = time.time()
        step_out = coreml_step.predict(step_inputs)
        t_step_total += time.time() - _t0

        transformer_out = step_out[step_xfmr_key]  # [1, 1, 1024]
        eos_logit = step_out[step_eos_key]  # [1, 1, 1]

        # Update caches/positions
        for i in range(num_layers):
            caches[f'cache{i}'] = step_out[step_cache_keys[i]]
            positions[f'position{i}'] = step_out[step_pos_keys[i]]

        # EOS check. Threshold pulled from PocketTTS upstream reference; may
        # need re-calibration per language when we swap to v2.0.0-native
        # EOS logits. Use POCKET_TTS_EOS_THRESHOLD=inf to disable for debug.
        eos_threshold = float(os.environ.get("POCKET_TTS_EOS_THRESHOLD", "-4.0"))
        eos_val = float(eos_logit.flatten()[0])
        is_eos = eos_val > eos_threshold
        if os.environ.get("POCKET_TTS_DEBUG_EOS"):
            print(f"  step={step:3d} eos_logit={eos_val:+.3f}")
        if is_eos and eos_step is None:
            eos_step = step
            print(f"  EOS at step {step} (logit={eos_val:+.3f})")
        if eos_step is not None and step >= eos_step + frames_after_eos:
            break

        # Flow decode (LSD 8 steps)
        transformer_out_flat = transformer_out.reshape(1, 1024)
        latent = np.random.randn(1, 32).astype(np.float32) * (temp ** 0.5)

        for lsd_step in range(num_lsd_steps):
            s_np = np.array([[lsd_step * dt]], dtype=np.float32)
            t_np = np.array([[(lsd_step + 1) * dt]], dtype=np.float32)
            _t0 = time.time()
            flow_out = coreml_flow.predict({
                'transformer_out': transformer_out_flat,
                'latent': latent,
                's': s_np,
                't': t_np,
            })
            t_flow_total += time.time() - _t0
            velocity = list(flow_out.values())[0]
            latent = latent + velocity * dt

        # Mimi decode. v2 traced decoder takes a raw [1, 32] latent and bakes
        # in denormalize + quantize projection internally, so feed `latent`
        # straight in (no numpy-side standardize/quantizer).
        mimi_inputs = {'latent': latent.astype(np.float32), **coreml_mimi_state}
        _t0 = time.time()
        mimi_out = coreml_mimi.predict(mimi_inputs)
        t_mimi_total += time.time() - _t0

        # Outputs are positional: [audio, state_0, state_1, ..., state_23]
        # where state_i matches the i-th non-`latent` input in spec order.
        audio_frame = mimi_out[mimi_output_names[0]]
        audio_chunks.append(audio_frame)
        for state_name, out_name in zip(mimi_input_order, mimi_output_names[1:]):
            coreml_mimi_state[state_name] = mimi_out[out_name]

        # Update sequence for next step
        sequence = latent.reshape(1, 1, 32)

        if step % 20 == 0:
            print(f"  Step {step}...")

    gen_time = time.time() - t_gen0
    n_steps = len(audio_chunks)
    audio_secs = n_steps * 0.08  # 80 ms / frame
    rtfx = audio_secs / max(gen_time, 1e-6)
    print(f"Generated {n_steps} frames")
    print(
        f"\n=== Timing (compute_units={units_name}) ===\n"
        f"  load        : {load_time:.2f}s\n"
        f"  prefill     : {prefill_time:.2f}s ({prefill_len} tokens)\n"
        f"  generation  : {gen_time:.2f}s ({n_steps} frames → {audio_secs:.2f}s audio)\n"
        f"  RTFx        : {rtfx:.3f}x  (>1 means faster than realtime)\n"
        f"  per-step (avg): step={t_step_total/max(n_steps,1)*1000:.1f}ms "
        f"flow×8={t_flow_total/max(n_steps,1)*1000:.1f}ms "
        f"mimi={t_mimi_total/max(n_steps,1)*1000:.1f}ms"
    )

    # Concatenate and save
    audio = np.concatenate(audio_chunks, axis=-1)
    audio = audio[0, 0]
    audio = audio / (np.abs(audio).max() + 1e-8) * 0.9

    sample_rate = 24000
    wavfile.write(output_path, sample_rate, (audio * 32767).astype(np.int16))

    print(f"\nSaved to {output_path}")
    print(f"Duration: {len(audio) / sample_rate:.2f}s")
    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pure-CoreML PocketTTS reference generator. "
            "Runs the full 4-model pipeline end-to-end and writes a 24 kHz wav."
        )
    )
    parser.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default=None,
        help=(
            "Language pack to use. When omitted, falls back to the legacy "
            "root-level English layout (models + constants next to this script)."
        ),
    )
    parser.add_argument(
        "--text",
        default="Hello, this is pure CoreML text to speech generation.",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--voice",
        default="alba",
        help="Voice name. Must exist as a .safetensors (legacy) or .bin (per-language).",
    )
    parser.add_argument(
        "--output",
        default="coreml_v4.wav",
        help="Output wav path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--compute-units",
        choices=("ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"),
        default="CPU_AND_GPU",
        help=(
            "CoreML compute units. ALL lets CoreML route to ANE when supported. "
            "Default CPU_AND_GPU preserves the original release behavior."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    generate_v4(
        text=args.text,
        voice=args.voice,
        output_path=args.output,
        seed=args.seed,
        language=args.language,
        compute_units=args.compute_units,
    )
