"""Pack per-language constants into a Swift-friendly `constants_bin/` layout.

The output layout under `build/<language>/constants_bin/`:

  constants_bin/
  ├── bos_emb.bin                    [32]               — Float32
  ├── text_embed_table.bin           [vocab_size, 1024] — Float32
  ├── tokenizer.model                                   — SentencePiece binary
  └── <voice>.safetensors            per-layer flow_lm KV cache  (v2 format)

**v2.0.0 voice format change.** Upstream kyutai/pocket-tts 2.0.0 switched
voice embeddings from a single `audio_prompt [prompt_len, 1024]` latent to
precomputed flow_lm KV caches with keys:

  transformer.layers.{0..N-1}.self_attn/offset   [1] int64
  transformer.layers.{0..N-1}.self_attn/cache    [2, 1, prompt_len, 16, 64] float32

We keep the safetensors as-is — the runtime loader (Python generator, and
eventually the Swift reader) is responsible for zero-padding the cache to
the CoreML model's expected length and mapping into its state inputs.

Run AFTER `export_constants.py --language <lang>` for the same language.

Usage:
  uv run python convert_assets/pack_constants_bin.py --language <lang>
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COREML_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_COREML_DIR)
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...
sys.path.insert(0, os.path.join(_COREML_DIR, "convert_models", "convert"))  # for _language_arg

from _language_arg import add_language_arg, build_output_dir


# `kyutai/pocket-tts` is gated (requires HF auth + license click-through); the
# `kyutai/pocket-tts-without-voice-cloning` mirror is ungated and carries the
# exact same tokenizer + per-language voice embeddings, so default to it.
# Override with `POCKET_TTS_HF_REPO=kyutai/pocket-tts` if you've signed in.
HF_REPO = os.environ.get(
    "POCKET_TTS_HF_REPO", "kyutai/pocket-tts-without-voice-cloning"
)


def _write_float32_bin(array: np.ndarray, output_path: str) -> None:
    """Write a numpy array as raw little-endian Float32 bytes."""
    flat = np.ascontiguousarray(array, dtype=np.float32)
    with open(output_path, "wb") as f:
        f.write(flat.tobytes())


def _pack_core_constants(constants_npy_dir: str, out_dir: str) -> None:
    """Convert `bos_emb.npy` + `text_embed_table.npy` → `.bin`."""
    for npy_name, bin_name in (
        ("bos_emb.npy", "bos_emb.bin"),
        ("text_embed_table.npy", "text_embed_table.bin"),
    ):
        npy_path = os.path.join(constants_npy_dir, npy_name)
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(
                f"Missing {npy_path}. Run export_constants.py first."
            )
        arr = np.load(npy_path)
        bin_path = os.path.join(out_dir, bin_name)
        _write_float32_bin(arr, bin_path)
        print(f"  wrote {bin_name}: shape={arr.shape}, dtype=float32")


def _copy_tokenizer(language: str, out_dir: str) -> None:
    """Download upstream `tokenizer.model` for the given language pack."""
    from huggingface_hub import hf_hub_download

    remote = f"languages/{language}/tokenizer.model"
    print(f"  downloading {HF_REPO}:{remote} ...")
    src = hf_hub_download(repo_id=HF_REPO, filename=remote)
    dst = os.path.join(out_dir, "tokenizer.model")
    shutil.copyfile(src, dst)
    print(f"  wrote tokenizer.model ({os.path.getsize(dst) / 1024:.1f} KB)")


def _iter_voice_names(language: str) -> list[str]:
    """List voice embedding files upstream ships for this language pack."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo_id=HF_REPO)
    prefix = f"languages/{language}/embeddings/"
    voices = []
    for f in files:
        if f.startswith(prefix) and f.endswith(".safetensors"):
            voices.append(os.path.basename(f)[: -len(".safetensors")])
    voices.sort()
    return voices


def _pack_voice(voice: str, language: str, out_dir: str) -> None:
    """Download one voice's safetensors and stash it under `constants_bin/`.

    Validates it's the v2 KV-cache format (keys look like
    `transformer.layers.<N>.self_attn/{offset,cache}`) before copying.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    remote = f"languages/{language}/embeddings/{voice}.safetensors"
    src = hf_hub_download(repo_id=HF_REPO, filename=remote)
    tensors = load_file(src)

    # Expected: 2 tensors per layer — `.../offset` + `.../cache`
    offset_keys = [k for k in tensors.keys() if k.endswith("/offset")]
    cache_keys = [k for k in tensors.keys() if k.endswith("/cache")]
    if not offset_keys or len(offset_keys) != len(cache_keys):
        raise KeyError(
            f"{remote}: expected v2 KV-cache format (matched offset/cache pairs), "
            f"got {sorted(tensors.keys())}"
        )

    num_layers = len(cache_keys)
    sample_cache = tensors[sorted(cache_keys)[0]]
    prompt_len = sample_cache.shape[2] if sample_cache.ndim == 5 else None

    dst = os.path.join(out_dir, f"{voice}.safetensors")
    shutil.copyfile(src, dst)
    size_kb = os.path.getsize(dst) / 1024
    print(
        f"  wrote {voice}.safetensors: {num_layers} layers, prompt_len={prompt_len}, "
        f"{size_kb:.1f} KB"
    )


def pack(language: str) -> str:
    """Produce `build/<language>/constants_bin/` ready for HF upload."""
    lang_root = build_output_dir(_COREML_DIR, language)
    constants_npy = os.path.join(lang_root, "constants")
    out_dir = os.path.join(lang_root, "constants_bin")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Packing constants for language={language}")
    print(f"  source: {constants_npy}")
    print(f"  dest:   {out_dir}")

    _pack_core_constants(constants_npy, out_dir)
    _copy_tokenizer(language, out_dir)

    voices = _iter_voice_names(language)
    if not voices:
        raise RuntimeError(
            f"No voice embeddings found under languages/{language}/embeddings/"
        )
    print(f"  packing {len(voices)} voice embeddings...")
    for voice in voices:
        _pack_voice(voice, language, out_dir)

    # Summary
    total = 0
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        total += size
    print(f"\nDone. {out_dir} total: {total / (1024 * 1024):.1f} MB")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    args = parser.parse_args()
    pack(args.language)
