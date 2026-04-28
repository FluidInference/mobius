"""Build manifest.json for the Magpie TTS hf-upload directory.

The manifest is a machine-readable index of every artifact in the upload
(models in both .mlmodelc + .mlpackage form, constants, per-language
tokenizer files), along with shapes, sizes, and SHA-256 digests. The Swift
port's MagpieResourceDownloader consumes it to know what to fetch and how
to verify integrity.
"""

from __future__ import annotations

import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent / "hf-upload"
SCHEMA_VERSION = "1.0"
REPO_ID = "FluidInference/magpie-tts-multilingual-357m-coreml"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def file_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


def parse_npy_header(path: Path) -> dict[str, Any]:
    """Read the v1/v2 .npy header and return shape + dtype."""
    with path.open("rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"not an npy file: {path}")
        major = f.read(1)[0]
        f.read(1)  # minor
        if major == 1:
            (header_len,) = struct.unpack("<H", f.read(2))
        else:
            (header_len,) = struct.unpack("<I", f.read(4))
        header = f.read(header_len).decode("latin1")
    # crude eval of the header dict (it is a plain Python literal)
    import ast

    meta = ast.literal_eval(header)
    return {
        "dtype": meta["descr"],
        "shape": list(meta["shape"]),
        "fortran_order": meta["fortran_order"],
    }


def npy_entry(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    info = parse_npy_header(p)
    return {
        "path": rel,
        "bytes": p.stat().st_size,
        "sha256": sha256_file(p),
        "dtype": info["dtype"],
        "shape": info["shape"],
    }


def json_entry(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    return {
        "path": rel,
        "bytes": p.stat().st_size,
        "sha256": sha256_file(p),
    }


def model_pair_entry(name: str, io: dict[str, Any]) -> dict[str, Any]:
    mlmodelc = ROOT / f"{name}.mlmodelc"
    mlpackage = ROOT / f"{name}.mlpackage"
    return {
        "name": name,
        "compiled": {
            "path": f"{name}.mlmodelc",
            "bytes": dir_size(mlmodelc),
            "files": file_count(mlmodelc),
        },
        "package": {
            "path": f"{name}.mlpackage",
            "bytes": dir_size(mlpackage),
            "files": file_count(mlpackage),
        },
        "io": io,
    }


# ---------- model io specs ----------------------------------------------------

# These specs were captured by inspecting the converted .mlpackage descriptions
# during convert_*.py runs (see generate_coreml.py for runtime keys).

MODEL_IO: dict[str, dict[str, Any]] = {
    "text_encoder": {
        "inputs": [
            {"name": "text_tokens", "dtype": "int32", "shape": [1, 256]},
            {"name": "text_mask", "dtype": "fp16", "shape": [1, 256]},
        ],
        "outputs": [
            {"name": "encoder_output", "dtype": "fp16", "shape": [1, 256, 768]},
            {"name": "encoder_mask", "dtype": "fp16", "shape": [1, 256]},
        ],
    },
    "decoder_prefill": {
        "inputs": [
            {"name": "input", "dtype": "fp16", "shape": [1, 110, 768]},
            {"name": "encoder_output", "dtype": "fp16", "shape": [1, 256, 768]},
            {"name": "encoder_mask", "dtype": "fp16", "shape": [1, 256]},
        ],
        "outputs": [
            {"name": "hidden_states", "dtype": "fp16", "shape": [1, 110, 768]},
            {
                "name": "cache_k{i} / cache_v{i}",
                "dtype": "fp16",
                "shape": [1, 512, 12, 64],
                "count": 24,
                "note": (
                    "12 K + 12 V cache outputs for the 12 decoder layers "
                    "(rank-4 split-K/V layout)"
                ),
            },
            {
                "name": "position{i}",
                "dtype": "fp32",
                "shape": [1],
                "count": 12,
                "note": "scalar position counter per layer",
            },
        ],
    },
    "decoder_step": {
        "inputs": [
            {"name": "audio_embed", "dtype": "fp16", "shape": [1, 1, 768]},
            {"name": "encoder_output", "dtype": "fp16", "shape": [1, 256, 768]},
            {"name": "encoder_mask", "dtype": "fp16", "shape": [1, 256]},
            {
                "name": "cache_k{i}",
                "dtype": "fp16",
                "shape": [1, 512, 12, 64],
                "count": 12,
                "note": "rank-4 split-K cache per layer (i=0..11)",
            },
            {
                "name": "cache_v{i}",
                "dtype": "fp16",
                "shape": [1, 512, 12, 64],
                "count": 12,
                "note": "rank-4 split-V cache per layer (i=0..11)",
            },
            {"name": "position{i}", "dtype": "fp32", "shape": [1], "count": 12},
        ],
        "outputs": [
            {
                "name": "input",
                "dtype": "fp16",
                "shape": [1, 1, 768],
                "note": "decoder hidden state (consumed by LocalTransformer)",
            },
            {
                "name": "var_2129",
                "dtype": "fp16",
                "shape": [1, 1, 16192],
                "note": (
                    "logits, reshape to (1, 1, 8, 2024) for 8 codebooks "
                    "(stateful variant uses var_2124 instead — see "
                    "DECODER_LOGITS_KEY_STATEFUL in generate_coreml.py)"
                ),
            },
            {
                "name": "new_k* / new_v*",
                "dtype": "fp16",
                "shape": [1, 512, 12, 64],
                "count": 24,
                "note": (
                    "12 K + 12 V outputs per step with non-uniform names "
                    "(new_k_1..new_k_21, new_k; new_v_1..new_v_21, new_v) — "
                    "see DECODER_CACHE_K_OUT_KEYS / DECODER_CACHE_V_OUT_KEYS in "
                    "generate_coreml.py for the canonical mapping per layer"
                ),
            },
            {
                "name": "var_169 / var_339 / ... / var_2039",
                "dtype": "fp32",
                "shape": [1],
                "count": 12,
                "note": (
                    "advanced position counters per layer — see "
                    "DECODER_POSITION_KEYS in generate_coreml.py"
                ),
            },
        ],
    },
    "nanocodec_decoder": {
        "inputs": [
            {"name": "tokens", "dtype": "int32", "shape": [1, 8, 256]},
        ],
        "outputs": [
            {"name": "audio", "dtype": "fp32", "shape": [1, 262144], "note": "256 frames * 1024 samples = 11.89s @ 22050 Hz"},
        ],
        "limits": {"max_frames": 256, "max_audio_seconds": 11.89},
    },
}


# ---------- constants files ---------------------------------------------------

CONSTANTS_NPY = [
    "constants/audio_embedding_0.npy",
    "constants/audio_embedding_1.npy",
    "constants/audio_embedding_2.npy",
    "constants/audio_embedding_3.npy",
    "constants/audio_embedding_4.npy",
    "constants/audio_embedding_5.npy",
    "constants/audio_embedding_6.npy",
    "constants/audio_embedding_7.npy",
    "constants/speaker_0.npy",
    "constants/speaker_1.npy",
    "constants/speaker_2.npy",
    "constants/speaker_3.npy",
    "constants/speaker_4.npy",
    "constants/speaker_embeddings_raw.npy",
    "constants/text_embedding.npy",
]

CONSTANTS_JSON = [
    "constants/constants.json",
    "constants/speaker_info.json",
    "constants/tokenizer_info.json",
    "constants/tokenizer_metadata.json",
    "constants/tokenizer_references.json",
]

LOCAL_TRANSFORMER_NPY = [
    "constants/local_transformer/in_proj_weight.npy",
    "constants/local_transformer/in_proj_bias.npy",
    "constants/local_transformer/pos_emb.npy",
    "constants/local_transformer/norm1_weight.npy",
    "constants/local_transformer/norm2_weight.npy",
    "constants/local_transformer/sa_qkv_weight.npy",
    "constants/local_transformer/sa_o_weight.npy",
    "constants/local_transformer/ffn_conv1_weight.npy",
    "constants/local_transformer/ffn_conv2_weight.npy",
] + [
    f"constants/local_transformer/out_proj_{i}_{kind}.npy"
    for i in range(8)
    for kind in ("weight", "bias")
]


# ---------- per-language tokenizer files --------------------------------------

# Mirrors MagpieLanguage in the Swift port. Must agree with
# PER_LANGUAGE_TOKENIZER_FILES in prepare_hf_upload.py — every entry here
# is sha256'd against the staged hf-upload/ tree.

LANGUAGE_FILES: dict[str, dict[str, Any]] = {
    "english": {
        "tokenizer_kind": "phoneme",
        "files": [
            "tokenizer/english_phoneme_token2id.json",
            "tokenizer/english_phoneme_phoneme_dict.json",
            "tokenizer/english_phoneme_heteronyms.json",
        ],
    },
    "spanish": {
        "tokenizer_kind": "phoneme",
        "files": [
            "tokenizer/spanish_phoneme_token2id.json",
            "tokenizer/spanish_phoneme_phoneme_dict.json",
        ],
    },
    "german": {
        "tokenizer_kind": "phoneme",
        "files": [
            "tokenizer/german_phoneme_token2id.json",
            "tokenizer/german_phoneme_phoneme_dict.json",
            "tokenizer/german_phoneme_heteronyms.json",
        ],
    },
    "hindi": {
        "tokenizer_kind": "char",
        "files": [
            "tokenizer/hindi_chartokenizer_token2id.json",
        ],
    },
    "mandarin": {
        "tokenizer_kind": "phoneme+jieba+pypinyin",
        "files": [
            "tokenizer/mandarin_phoneme_token2id.json",
            "tokenizer/mandarin_phoneme_phoneme_dict.json",
            "tokenizer/mandarin_phoneme_pinyin_dict.json",
            "tokenizer/mandarin_phoneme_tone_dict.json",
            "tokenizer/mandarin_phoneme_ascii_letter_dict.json",
            "tokenizer/mandarin_pypinyin_char_dict.json",
            "tokenizer/mandarin_pypinyin_phrase_dict.json",
            "tokenizer/mandarin_jieba_dict.json",
        ],
    },
    "french": {
        "tokenizer_kind": "char",
        "files": [
            "tokenizer/french_chartokenizer_token2id.json",
        ],
    },
    "italian": {
        "tokenizer_kind": "phoneme",
        "files": [
            "tokenizer/italian_phoneme_token2id.json",
            "tokenizer/italian_phoneme_phoneme_dict.json",
        ],
    },
    "vietnamese": {
        "tokenizer_kind": "phoneme",
        "files": [
            "tokenizer/vietnamese_phoneme_token2id.json",
            "tokenizer/vietnamese_phoneme_phoneme_dict.json",
        ],
    },
}


def build_manifest() -> dict[str, Any]:
    models = {
        "text_encoder": model_pair_entry("text_encoder", MODEL_IO["text_encoder"]),
        "decoder_prefill": model_pair_entry("decoder_prefill", MODEL_IO["decoder_prefill"]),
        "decoder_step": model_pair_entry("decoder_step", MODEL_IO["decoder_step"]),
        "nanocodec_decoder": model_pair_entry("nanocodec_decoder", MODEL_IO["nanocodec_decoder"]),
    }

    constants = {
        "json": [json_entry(p) for p in CONSTANTS_JSON],
        "npy": [npy_entry(p) for p in CONSTANTS_NPY],
        "local_transformer": [npy_entry(p) for p in LOCAL_TRANSFORMER_NPY],
    }

    languages = {}
    for lang, spec in LANGUAGE_FILES.items():
        entries = [json_entry(p) for p in spec["files"]]
        languages[lang] = {
            "tokenizer_kind": spec["tokenizer_kind"],
            "files": entries,
            "bytes": sum(e["bytes"] for e in entries),
        }

    # Top-level summary
    total_bytes = (
        sum(m["compiled"]["bytes"] + m["package"]["bytes"] for m in models.values())
        + sum(e["bytes"] for e in constants["json"])
        + sum(e["bytes"] for e in constants["npy"])
        + sum(e["bytes"] for e in constants["local_transformer"])
        + sum(lang["bytes"] for lang in languages.values())
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_id": REPO_ID,
        "model": {
            "name": "Magpie TTS Multilingual",
            "params_million": 357,
            "sample_rate": 22050,
            "codec_samples_per_frame": 1024,
            "frames_per_second": 22050.0 / 1024.0,
            "max_decoder_steps": 500,
            "max_decoder_seconds": 500 * 1024 / 22050.0,
            "max_nanocodec_frames": 256,
            "max_nanocodec_seconds": 256 * 1024 / 22050.0,
            "embedding_dim": 768,
            "num_audio_codebooks": 8,
            "codebook_size": 2024,
            "audio_bos_id": 2016,
            "audio_eos_id": 2017,
            "forbidden_token_ids": [2016, 2018, 2019, 2020, 2021, 2022, 2023],
            "num_speakers": 5,
            "speaker_names": ["John", "Sofia", "Aria", "Jason", "Leo"],
            "speaker_context_length": 110,
            "max_text_tokens": 256,
            "supported_languages": list(LANGUAGE_FILES.keys()),
            "supported_features": [
                "ipa_override",
                "deterministic_g2p",
                "classifier_free_guidance",
            ],
            "japanese": {
                "supported": False,
                "note": "Japanese deferred — needs OpenJTalk + MeCab dict (separate follow-up).",
            },
            "streaming_nanocodec": {
                "supported": False,
                "note": (
                    "NanoCodec is exported as a fixed-window batch decoder (max_frames=256). "
                    "True streaming requires MLState conv-cache integration; tested overlap "
                    "warmup yields <15 dB SNR and is unviable as a fallback."
                ),
            },
        },
        "models": models,
        "constants": constants,
        "languages": languages,
        "totals": {
            "bytes": total_bytes,
            "human": f"{total_bytes / 1_000_000_000:.2f} GB",
        },
        "notes": [
            "Both .mlmodelc (compiled, ready-to-run) and .mlpackage (portable source) are shipped.",
            "Swift consumers should prefer .mlmodelc; .mlpackage is provided for inspection / re-targeting.",
            "Per-language tokenizer files under tokenizer/ are lazy: download only the languages you need.",
            "constants/local_transformer/*.npy are loaded once into a Swift fp32 cache — see MagpieLocalTransformerWeights.swift.",
        ],
    }
    return manifest


def main() -> None:
    manifest = build_manifest()
    out = ROOT / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")
    print(f"total assets: {manifest['totals']['human']}")


if __name__ == "__main__":
    main()
