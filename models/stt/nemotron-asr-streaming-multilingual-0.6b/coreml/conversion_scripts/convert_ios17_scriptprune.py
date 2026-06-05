#!/usr/bin/env python3
"""Export Nemotron-3.5-ASR-Multilingual 0.6B to CoreML — UNIQUE-BIAS variant.

Same as `convert_nemotron_multilingual.py` but applies `patch_uniquebias`
to the encoder before tracing. This perturbs each conformer layer's FFN
biases by ~1e-3 in a per-(layer,ffn,linear)-unique pattern, breaking the
`linear_1_bias_0_to_fp16` shared-constant dedup that coremltools applies
to identical FP16 values across layers.

Why this exists: layer-position mixed-precision quantization
(`mixed_layerpos.py`) was blocked at coremltools 8.3 because shared
biases prevented op_name_configs from assigning different compression
configs to different linear ops within a single pass. With unique-valued
biases, that block is removed and layer-position mixed becomes possible.

Expected WER impact of the bias perturbation: << 0.1pp. The perturbation
is well below model noise; the goal is just to make FP16 const bytes
differ per location.

Run:
    uv sync
    uv run python convert_nemotron_multilingual_uniquebias.py \\
        --nemo-path /path/to/multilingual.nemo \\
        --output-dir ./build_fp16_tmp_1120ms_ios18_uniquebias \\
        --precision FLOAT16 \\
        --att-context "56,0"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import coremltools as ct
import numpy as np
import torch
import typer

# Reach into the sibling English package for the shared wrappers.
_ENGLISH_PKG = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "nemotron-speech-streaming-0.6b"
    / "coreml"
    / "conversion_scripts"
)
sys.path.insert(0, str(_ENGLISH_PKG))

from individual_components import (  # type: ignore  # noqa: E402
    DecoderWrapper,
    ExportSettings,
    JointWrapper,
    PreprocessorWrapper,
    _coreml_convert,
)
from multilingual_components import (  # noqa: E402
    EncoderStreamingWithPostPrompt,
    NUM_PROMPTS,
)


def _tensor_shape(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(d) for d in t.shape)


def _parse_cu(name: str) -> ct.ComputeUnit:
    mapping = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    return mapping.get(name.upper(), ct.ComputeUnit.CPU_ONLY)


def _parse_att_context(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise typer.BadParameter("--att-context must be 'left,right' e.g. '56,0'")
    return [int(parts[0]), int(parts[1])]


_LANG_TAG_RE = __import__("re").compile(r"^<[A-Za-z]{2,4}-[A-Za-z]{2,4}>$")


def _lang_tag_token_ids(model) -> List[int]:
    """Token IDs whose text form is a language tag like '<en-US>'.

    These are emitted by greedy decoding but should be stripped from
    the final transcript (per README: strip_lang_tags=true). The 39
    tags in this model are scattered across the full 13,087 vocab
    (not bunched at the start), so we scan everything.
    """
    ids: List[int] = []
    vocab_size = int(model.tokenizer.vocab_size)
    for i in range(vocab_size):
        tok = model.tokenizer.ids_to_tokens([i])[0]
        if _LANG_TAG_RE.match(tok):
            ids.append(i)
    return ids


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    nemo_path: Path = typer.Option(
        ...,
        "--nemo-path",
        help="Path to nemotron-asr-streaming-multilingual-0.6b.nemo",
    ),
    output_dir: Path = typer.Option(
        Path("nemotron_multi_coreml"), "--output-dir"
    ),
    encoder_cu: str = typer.Option("CPU_AND_NE", "--encoder-cu"),
    precision: str = typer.Option("FLOAT16", "--precision"),
    att_context: str = typer.Option(
        "56,0",
        "--att-context",
        help="Attention context as 'left,right'. Supported by this model: "
        "56,0 | 56,3 | 56,6 | 56,13. 56,0 = lowest latency.",
    ),
    chunk_mel_frames: int = typer.Option(
        112, "--chunk-mel-frames", help="Mel frames per chunk (subsamples by 8)."
    ),
    pre_encode_cache: int = typer.Option(9, "--pre-encode-cache"),
    prune_corpus_jsonl: Path = typer.Option(
        ...,
        "--prune-corpus-jsonl",
        help="JSONL file with hyp_raw/ref_raw fields; used to build the "
        "English keep-set for vocab pruning.",
    ),
) -> None:
    """Export the multilingual streaming model to CoreML — VOCAB-PRUNED + unique-bias + [56,*] fix."""

    import nemo.collections.asr as nemo_asr

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading model from {nemo_path}...")
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo_path), map_location="cpu"
    )
    model.eval()
    model_class = f"{type(model).__module__}.{type(model).__name__}"
    typer.echo(f"  class: {model_class}")

    if not hasattr(model, "prompt_kernel"):
        raise RuntimeError(
            "Loaded model has no `prompt_kernel`; expected "
            "EncDecRNNTBPEModelWithPrompt. Wrong checkpoint?"
        )

    sample_rate = int(model.cfg.preprocessor.sample_rate)
    mel_features = int(model.cfg.preprocessor.features)

    encoder = model.encoder
    # === UNIQUE-BIAS PATCH ===
    from patch_uniquebias import patch_and_log
    patch_and_log(encoder)
    # === END PATCH ===

    parsed_att_ctx = _parse_att_context(att_context)
    # CRITICAL: set att_context_size attribute directly (setup_streaming_params
    # is insufficient — see comment in convert_nemotron_multilingual.py).
    encoder.att_context_size = list(parsed_att_ctx)
    encoder.setup_streaming_params(att_context_size=parsed_att_ctx)

    # === ENGLISH VOCAB PRUNE ===
    # Slice decoder.embed and joint.output_proj to keep only tokens needed
    # for English transcription. Pre-tracing so the converter sees a
    # smaller model.
    from patch_vocab_prune import (
        build_english_keep_set,
        prune_vocab_english,
    )
    # SCRIPT-BASED keep-set (domain-independent, no corpus, no overfitting):
    # keep every token whose characters are Latin/Greek or shared
    # (digits/punct/symbols/special) and drop all other scripts (CJK,
    # Hangul, Cyrillic, Arabic, ...). prune_corpus_jsonl is ignored here.
    import unicodedata as _ud
    _OTHER = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC",
              "HEBREW", "DEVANAGARI", "THAI", "BENGALI", "TAMIL", "TELUGU",
              "GEORGIAN", "ARMENIAN")

    def _latin_or_shared(piece: str) -> bool:
        for ch in piece.replace("▁", ""):
            cat = _ud.category(ch)
            if ch.isdigit() or ch.isspace() or cat[0] in ("P", "S", "Z", "C"):
                continue
            try:
                nm = _ud.name(ch)
            except ValueError:
                continue
            if any(s in nm for s in _OTHER):
                return False
        return True

    old_blank = int(model.decoder.blank_idx)
    old_lang_tag_ids = _lang_tag_token_ids(model)
    _vsz = int(model.tokenizer.vocab_size)
    _keep = set()
    for _i in range(_vsz):
        if _i == old_blank:
            continue
        try:
            _p = model.tokenizer.ids_to_tokens([_i])[0]
        except Exception:
            continue
        if _latin_or_shared(_p):
            _keep.add(_i)
    _keep.update(old_lang_tag_ids)
    _keep.discard(old_blank)
    keep_ids = sorted(_keep) + [old_blank]
    typer.echo(
        f"  [vocab-prune] keeping {len(keep_ids)} of 13088 tokens "
        f"({100 * (1 - len(keep_ids)/13088):.1f}% pruned)"
    )
    id_map = prune_vocab_english(model, keep_ids)
    # Cache mapping data for the post-conversion metadata writes
    _pruned_vocab_size = len(keep_ids) - 1  # exclude blank
    _pruned_blank_idx = len(keep_ids) - 1
    _pruned_lang_tag_ids = sorted({id_map[i] for i in old_lang_tag_ids if i in id_map})
    _pruned_id_map = id_map
    # === END PRUNE ===

    # Initial cache state from NeMo (shape [L, B, ...])
    cache_channel, cache_time, cache_len = encoder.get_initial_cache_state(
        batch_size=1, device="cpu"
    )
    cache_len = cache_len.to(torch.int32)
    cache_channel_b = cache_channel.transpose(0, 1)
    cache_time_b = cache_time.transpose(0, 1)
    typer.echo(
        f"  caches: channel={tuple(cache_channel_b.shape)} "
        f"time={tuple(cache_time_b.shape)} len={tuple(cache_len.shape)}"
    )

    # === Prompt-aware streaming encoder wrapper ===
    # The NeMo class applies `prompt_kernel` AFTER the full encoder runs
    # (`conformer_stream_step` → `_apply_prompt_to_encoded`). The wrapper
    # mirrors that exactly so the CoreML graph stays faithful to the
    # PyTorch reference.
    encoder_streaming = EncoderStreamingWithPostPrompt(
        encoder.eval(),
        model.prompt_kernel.eval(),
        num_prompts=NUM_PROMPTS,
    )

    # Shared wrappers (preprocessor/decoder/joint are language-agnostic)
    preprocessor = PreprocessorWrapper(model.preprocessor.eval())
    decoder = DecoderWrapper(model.decoder.eval())
    joint = JointWrapper(model.joint.eval())
    model.decoder._rnnt_export = True

    total_mel_frames = chunk_mel_frames + pre_encode_cache
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS17,
        compute_precision=(
            ct.precision.FLOAT16
            if precision.upper() == "FLOAT16"
            else ct.precision.FLOAT32
        ),
        max_audio_seconds=80.0,
        max_symbol_steps=1,
        chunk_size_frames=chunk_mel_frames // 8,
        cache_size=cache_channel.shape[2],
    )

    # === 1. Preprocessor ===
    typer.echo("Exporting preprocessor...")
    max_samples = int(settings.max_audio_seconds * sample_rate)
    audio = torch.randn(1, max_samples)

    traced = torch.jit.trace(
        preprocessor,
        (audio, torch.tensor([max_samples], dtype=torch.int32)),
        strict=False,
    )
    mlmodel = _coreml_convert(
        traced,
        inputs=[
            ct.TensorType(
                name="audio",
                shape=(1, ct.RangeDim(1, max_samples)),
                dtype=np.float32,
            ),
            ct.TensorType(name="audio_length", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="mel", dtype=np.float32),
            ct.TensorType(name="mel_length", dtype=np.int32),
        ],
        settings=settings,
        compute_units_override=ct.ComputeUnit.CPU_ONLY,
    )
    mlmodel.save(str(output_dir / "preprocessor.mlpackage"))

    # === 2. Encoder (prompt-aware streaming) ===
    typer.echo("Exporting encoder...")
    mel = torch.randn(1, mel_features, total_mel_frames)
    mel_len = torch.tensor([total_mel_frames], dtype=torch.int32)
    prompt_id = torch.tensor([0], dtype=torch.int32)  # any valid id

    traced = torch.jit.trace(
        encoder_streaming,
        (mel, mel_len, cache_channel_b, cache_time_b, cache_len, prompt_id),
        strict=False,
    )
    mlmodel = _coreml_convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=_tensor_shape(mel), dtype=np.float32),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
            ct.TensorType(
                name="cache_channel",
                shape=_tensor_shape(cache_channel_b),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="cache_time",
                shape=_tensor_shape(cache_time_b),
                dtype=np.float32,
            ),
            ct.TensorType(name="cache_len", shape=(1,), dtype=np.int32),
            ct.TensorType(name="prompt_id", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="encoded", dtype=np.float32),
            ct.TensorType(name="encoded_length", dtype=np.int32),
            ct.TensorType(name="cache_channel_out", dtype=np.float32),
            ct.TensorType(name="cache_time_out", dtype=np.float32),
            ct.TensorType(name="cache_len_out", dtype=np.int32),
        ],
        settings=settings,
        compute_units_override=_parse_cu(encoder_cu),
    )
    mlmodel.save(str(output_dir / "encoder.mlpackage"))

    # === 3. Decoder ===
    typer.echo("Exporting decoder...")
    decoder_hidden = int(model.decoder.pred_hidden)
    decoder_layers = int(model.decoder.pred_rnn_layers)
    targets = torch.tensor([[model.decoder.blank_idx]], dtype=torch.int32)
    target_len = torch.tensor([1], dtype=torch.int32)
    h = torch.zeros(decoder_layers, 1, decoder_hidden)
    c = torch.zeros(decoder_layers, 1, decoder_hidden)

    traced = torch.jit.trace(decoder, (targets, target_len, h, c), strict=False)
    mlmodel = _coreml_convert(
        traced,
        inputs=[
            ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="token_length", shape=(1,), dtype=np.int32),
            ct.TensorType(name="h_in", shape=_tensor_shape(h), dtype=np.float32),
            ct.TensorType(name="c_in", shape=_tensor_shape(c), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="decoder_out", dtype=np.float32),
            ct.TensorType(name="h_out", dtype=np.float32),
            ct.TensorType(name="c_out", dtype=np.float32),
        ],
        settings=settings,
        compute_units_override=ct.ComputeUnit.CPU_ONLY,
    )
    mlmodel.save(str(output_dir / "decoder.mlpackage"))

    # === 4. Joint (single step) ===
    typer.echo("Exporting joint...")
    with torch.no_grad():
        mel_test, _ = preprocessor(
            audio[:, :sample_rate],
            torch.tensor([sample_rate], dtype=torch.int32),
        )
        # Reset cache state for the test forward
        cc, ct_, cl = encoder.get_initial_cache_state(batch_size=1, device="cpu")
        enc_out, _, _, _, _ = encoder_streaming(
            mel_test,
            torch.tensor([mel_test.shape[2]], dtype=torch.int32),
            cc.transpose(0, 1),
            ct_.transpose(0, 1),
            cl.to(torch.int32),
            prompt_id,
        )
        dec_out, _, _ = decoder(targets, target_len, h, c)

    enc_step = enc_out[:, :, :1].contiguous()
    dec_step = dec_out[:, :, :1].contiguous()

    traced = torch.jit.trace(joint, (enc_step, dec_step), strict=False)
    mlmodel = _coreml_convert(
        traced,
        inputs=[
            ct.TensorType(
                name="encoder", shape=_tensor_shape(enc_step), dtype=np.float32
            ),
            ct.TensorType(
                name="decoder", shape=_tensor_shape(dec_step), dtype=np.float32
            ),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        settings=settings,
        compute_units_override=ct.ComputeUnit.CPU_ONLY,
    )
    mlmodel.save(str(output_dir / "joint.mlpackage"))

    # === 5. Metadata + tokenizer (VOCAB-PRUNED) ===
    # All vocab/blank/lang_tag fields use the REMAPPED IDs from prune.
    vocab_size = _pruned_vocab_size  # was model.tokenizer.vocab_size (13087)
    prompt_dict = dict(model.cfg.model_defaults.prompt_dictionary)
    lang_tag_ids = _pruned_lang_tag_ids  # already remapped to new IDs

    metadata = {
        "model": "nvidia/nemotron-asr-streaming-multilingual-0.6b",
        "model_class": model_class,
        "sample_rate": sample_rate,
        "mel_features": mel_features,
        "chunk_mel_frames": chunk_mel_frames,
        "pre_encode_cache": pre_encode_cache,
        "total_mel_frames": total_mel_frames,
        "att_context_size": _parse_att_context(att_context),
        "vocab_size": vocab_size,
        "blank_idx": _pruned_blank_idx,
        "vocab_pruned": True,
        "vocab_pruned_original_size": 13087,
        "cache_channel_shape": list(cache_channel_b.shape),
        "cache_time_shape": list(cache_time_b.shape),
        "decoder_hidden": decoder_hidden,
        "decoder_layers": decoder_layers,
        "encoder_dim": int(enc_out.shape[1]),
        "num_prompts": NUM_PROMPTS,
        "prompt_dictionary": prompt_dict,
        "default_prompt_id": prompt_dict.get("auto", 101),
        "lang_tag_token_ids": lang_tag_ids,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Tokenizer: rewrite with NEW IDs mapping to original BPE pieces.
    # _pruned_id_map: old_id -> new_id. We invert and build new-id -> piece.
    new_id_to_piece = {}
    for old_id, new_id in _pruned_id_map.items():
        # blank (last kept entry) gets a sentinel piece; downstream Swift
        # filters blank by id anyway, so the string is informational
        piece = (
            "<blank>"
            if new_id == _pruned_blank_idx
            else model.tokenizer.ids_to_tokens([old_id])[0]
        )
        new_id_to_piece[str(new_id)] = piece
    (output_dir / "tokenizer.json").write_text(
        json.dumps(new_id_to_piece, indent=2, ensure_ascii=False)
    )

    typer.echo(f"Done. Exported to {output_dir}")


if __name__ == "__main__":
    app()
