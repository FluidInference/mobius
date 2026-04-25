"""Numerical parity: upstream flow_lm prefill vs CoreML cond_step prefill.

Starting from our packed voice KV cache safetensors (identical bytes on both
sides), run:

  A) Upstream PyTorch:  _run_flow_lm_and_increment_step(model_state, text_tokens)
  B) Our CoreML:        for each text embedding, coreml_cond.predict(...)

Both should produce the same post-prefill KV cache at positions
`[prompt_len .. prompt_len + text_len)`. Report max/mean absolute difference
per layer; large diff = our CoreML cond_step is drifting from upstream.

Usage:
  uv run python parity_prefill.py --language german \
      --text "Hallo, das ist ein Sprachsynthesesystem."
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import sentencepiece as sp
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)


def _upstream_prefill(
    language: str, voice_path: str, token_ids: list[int]
) -> tuple[list[np.ndarray], list[int], int]:
    """Run upstream flow_lm prefill with text tokens on top of the voice state.

    Returns `(post_caches_per_layer, offsets_per_layer, start_offset)` where
    `post_caches_per_layer[L]` has shape `[2, 1, prompt_len + T, 16, 64]`
    (actual cache size after the text write).
    """
    from pocket_tts.models.tts_model import TTSModel

    print(f"[upstream] load language={language}")
    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    print(f"[upstream] load voice state from {voice_path}")
    voice_state = model.get_state_for_audio_prompt(voice_path)
    voice_state = copy.deepcopy(voice_state)

    start_offset = int(
        voice_state["transformer.layers.0.self_attn"]["offset"].item()
    )

    # Upstream pulls cache out sliced-down to actual prompt_len; we need to
    # grow it so there's room for the text tokens we're about to append.
    tokens = torch.tensor([token_ids], dtype=torch.int64)
    required_len = start_offset + tokens.shape[1] + 64  # +headroom for gen
    model._expand_kv_cache(voice_state, sequence_length=required_len)

    print(f"[upstream] prefill text_tokens shape={tuple(tokens.shape)}")
    model._run_flow_lm_and_increment_step(
        model_state=voice_state, text_tokens=tokens
    )

    post_caches = []
    offsets = []
    num_layers = sum(
        1 for k in voice_state if k.startswith("transformer.layers.")
    )
    for layer in range(num_layers):
        key = f"transformer.layers.{layer}.self_attn"
        cache = voice_state[key]["cache"].detach().cpu().numpy()
        offset = int(voice_state[key]["offset"].item())
        post_caches.append(cache)
        offsets.append(offset)
    print(
        f"[upstream] num_layers={num_layers} post_offset={offsets[0]} "
        f"(started at {start_offset})"
    )
    return post_caches, offsets, start_offset


def _coreml_prefill(
    language: str, voice_path: str, token_ids: list[int]
) -> tuple[list[np.ndarray], list[int], int]:
    """Replay our CoreML cond_step prefill on the same voice state + text."""
    import coremltools as ct
    from safetensors.numpy import load_file

    model_dir = os.path.join(_SCRIPT_DIR, "build", language)
    const_dir = os.path.join(model_dir, "constants")

    coreml_cond = ct.models.MLModel(
        os.path.join(model_dir, "cond_step.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    # Introspect shapes/output names (same logic as generate_coreml_v4).
    spec = coreml_cond.get_spec()
    num_layers = 0
    cache_slots = None
    for inp in spec.description.input:
        if inp.name.startswith("cache") and inp.name[len("cache"):].isdigit():
            num_layers = max(num_layers, int(inp.name[len("cache"):]) + 1)
            if cache_slots is None:
                cache_slots = int(inp.type.multiArrayType.shape[2])
    cache_keys = []
    pos_keys = []
    for out in spec.description.output:
        if not out.type.HasField("multiArrayType"):
            continue
        rank = len(out.type.multiArrayType.shape)
        if rank == 5:
            cache_keys.append(out.name)
        elif rank == 1:
            pos_keys.append(out.name)
    print(
        f"[coreml] num_layers={num_layers} cache_slots={cache_slots} "
        f"cache_outs={len(cache_keys)} pos_outs={len(pos_keys)}"
    )

    # Seed voice state from the safetensors.
    tensors = load_file(voice_path)
    caches = {}
    positions = {}
    start_offset = None
    for layer in range(num_layers):
        ck = f"transformer.layers.{layer}.self_attn/cache"
        ok = f"transformer.layers.{layer}.self_attn/offset"
        raw = tensors[ck].astype(np.float32)
        padded = np.zeros(
            (2, 1, cache_slots, raw.shape[3], raw.shape[4]), dtype=np.float32
        )
        padded[:, :, : raw.shape[2], :, :] = raw
        caches[f"cache{layer}"] = padded
        offset_val = float(np.asarray(tensors[ok]).reshape(-1)[0])
        positions[f"position{layer}"] = np.array([offset_val], dtype=np.float32)
        if start_offset is None:
            start_offset = int(offset_val)

    # Text embedding via per-language embed table (what generate_coreml_v4 does).
    embed_table = np.load(os.path.join(const_dir, "text_embed_table.npy"))
    text_emb = embed_table[np.array(token_ids, dtype=np.int64)][np.newaxis, :, :]

    print(f"[coreml] prefill {text_emb.shape[1]} tokens from offset {start_offset}")
    for t in range(text_emb.shape[1]):
        inp = {
            "conditioning": text_emb[:, t:t + 1, :].astype(np.float32),
            **caches,
            **positions,
        }
        out = coreml_cond.predict(inp)
        for i in range(num_layers):
            caches[f"cache{i}"] = out[cache_keys[i]]
            positions[f"position{i}"] = out[pos_keys[i]]

    post_caches = [caches[f"cache{i}"] for i in range(num_layers)]
    offsets = [int(positions[f"position{i}"][0]) for i in range(num_layers)]
    print(f"[coreml] final offset={offsets[0]}")
    return post_caches, offsets, start_offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    lang_dir = os.path.join(_SCRIPT_DIR, "build", args.language)
    voice_path = os.path.join(lang_dir, "constants_bin", f"{args.voice}.safetensors")
    tokenizer_path = os.path.join(lang_dir, "constants_bin", "tokenizer.model")

    tok = sp.SentencePieceProcessor()
    tok.load(tokenizer_path)
    # Use the same `prepare_text_prompt` behaviour as generate_coreml_v4:
    # lightly normalise, ensure trailing punctuation, pad short inputs.
    text = args.text.strip()
    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    if len(text.split()) < 5:
        text = " " * 8 + text
    token_ids = tok.encode(text)
    print(f"Text: {text!r}")
    print(f"Tokens ({len(token_ids)}): {token_ids}")

    up_caches, up_offsets, up_start = _upstream_prefill(
        args.language, voice_path, token_ids
    )
    cm_caches, cm_offsets, cm_start = _coreml_prefill(
        args.language, voice_path, token_ids
    )

    if up_start != cm_start:
        print(f"!!! start offsets differ: upstream={up_start} coreml={cm_start}")
    if up_offsets[0] != cm_offsets[0]:
        print(
            f"!!! post-prefill offsets differ: "
            f"upstream={up_offsets[0]} coreml={cm_offsets[0]}"
        )

    print("\nPer-layer diff of text-conditioning region "
          "[prompt_len .. prompt_len + T):")
    prompt_len = up_start
    T = len(token_ids)
    end = prompt_len + T
    print(f"  prompt_len={prompt_len}, text_len={T}, slice=[{prompt_len}..{end})")
    for i, (u, c) in enumerate(zip(up_caches, cm_caches)):
        u_slice = u[:, :, prompt_len:end, :, :]
        c_slice = c[:, :, prompt_len:end, :, :]
        diff = np.abs(u_slice - c_slice)
        rel = diff.mean() / (np.abs(u_slice).mean() + 1e-9)
        print(
            f"  layer {i:2d}: upstream|mean={np.abs(u_slice).mean():.5f} "
            f"coreml|mean={np.abs(c_slice).mean():.5f} "
            f"abs_max={diff.max():.5f} abs_mean={diff.mean():.5f} "
            f"rel={rel:.5f}"
        )


if __name__ == "__main__":
    main()
