#!/usr/bin/env python3
"""Export Nemotron-3.5-ASR-Streaming-Multilingual 0.6B to OpenVINO IR.

This is the OpenVINO analogue of mobius's CoreML conversion
(`convert_nemotron_multilingual.py`). It reuses the *exact same* traceable
TorchScript wrappers from the mobius CoreML pipeline — the wrappers are
backend-agnostic — and only swaps the backend call:

    CoreML:   ct.convert(traced, inputs=..., outputs=...)
    OpenVINO: ov.convert_model(traced, example_input=...) + ov.save_model

Produces, in --output-dir, the four IR pairs eddy-audio consumes plus
tokenizer/metadata:

    nemotron_preprocessor.xml/.bin   audio -> mel
    nemotron_encoder.xml/.bin        mel + caches + prompt_id -> encoded + caches
    nemotron_decoder.xml/.bin        token + lstm state -> dec_out + state
    nemotron_joint.xml/.bin          enc_step + dec_step -> logits
    nemotron_vocab.json              id -> token piece
    metadata.json                    shapes, blank_idx, prompt_dictionary, ...

Run:
    python export_openvino.py --nemo-path ./nemotron-3.5-asr-streaming-0.6b.nemo \
        --output-dir ./build_ov --precision FP32 --att-context 56,0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import openvino as ov
import torch
import typer

# Reuse the mobius CoreML wrappers verbatim (backend-agnostic TorchScript).
_MOBIUS = Path.home() / "mobius" / "models" / "stt"
_EN_PKG = _MOBIUS / "nemotron-speech-streaming-0.6b" / "coreml" / "conversion_scripts"
_ML_PKG = _MOBIUS / "nemotron-asr-streaming-multilingual-0.6b" / "coreml" / "conversion_scripts"
sys.path.insert(0, str(_EN_PKG))
sys.path.insert(0, str(_ML_PKG))

from individual_components import (  # type: ignore  # noqa: E402
    DecoderWrapper,
    JointWrapper,
    PreprocessorWrapper,
)
from multilingual_components import (  # type: ignore  # noqa: E402
    EncoderStreamingWithPostPrompt,
    NUM_PROMPTS,
)

_LANG_TAG_RE = __import__("re").compile(r"^<[A-Za-z]{2,4}-[A-Za-z]{2,4}>$")


def _shape(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(d) for d in t.shape)


def _parse_att_context(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise typer.BadParameter("--att-context must be 'left,right' e.g. '56,0'")
    return [int(parts[0]), int(parts[1])]


def _lang_tag_token_ids(model) -> List[int]:
    ids: List[int] = []
    for i in range(int(model.tokenizer.vocab_size)):
        if _LANG_TAG_RE.match(model.tokenizer.ids_to_tokens([i])[0]):
            ids.append(i)
    return ids


def _save_ir(
    traced: torch.jit.ScriptModule,
    example_input: tuple,
    input_specs: List[tuple],   # (name, [shape], dtype)
    output_names: List[str],
    out_path: Path,
    fp16: bool,
) -> None:
    """OpenVINO equivalent of mobius's `_coreml_convert` + `.save`."""
    inputs = [(name, shape, dtype) for (name, shape, dtype) in input_specs]
    ov_model = ov.convert_model(
        traced,
        example_input=example_input,
        input=[(s[1], s[2]) for s in inputs],
    )
    # Name the I/O so eddy can resolve ports by name.
    for port, (name, _, _) in zip(ov_model.inputs, inputs):
        port.get_tensor().set_names({name})
    for port, name in zip(ov_model.outputs, output_names):
        port.get_tensor().set_names({name})
    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, str(out_path), compress_to_fp16=fp16)
    print(f"  saved {out_path.name}  inputs={[i.get_any_name() for i in ov_model.inputs]}"
          f"  outputs={[o.get_any_name() for o in ov_model.outputs]}")


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    nemo_path: Path = typer.Option(..., "--nemo-path"),
    output_dir: Path = typer.Option(Path("build_ov"), "--output-dir"),
    precision: str = typer.Option("FP32", "--precision", help="FP32 or FP16"),
    att_context: str = typer.Option("56,0", "--att-context"),
    chunk_mel_frames: int = typer.Option(112, "--chunk-mel-frames"),
    pre_encode_cache: int = typer.Option(9, "--pre-encode-cache"),
) -> None:
    import nemo.collections.asr as nemo_asr

    fp16 = precision.upper() == "FP16"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {nemo_path} ...")
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo_path), map_location="cpu"
    )
    model.eval()
    model_class = f"{type(model).__module__}.{type(model).__name__}"
    print(f"  class: {model_class}")
    if not hasattr(model, "prompt_kernel"):
        raise RuntimeError("No prompt_kernel; wrong checkpoint?")

    sample_rate = int(model.cfg.preprocessor.sample_rate)
    mel_features = int(model.cfg.preprocessor.features)

    encoder = model.encoder
    parsed_att = _parse_att_context(att_context)
    encoder.att_context_size = list(parsed_att)
    encoder.setup_streaming_params(att_context_size=parsed_att)

    cache_channel, cache_time, cache_len = encoder.get_initial_cache_state(
        batch_size=1, device="cpu"
    )
    cache_len = cache_len.to(torch.int32)
    cache_channel_b = cache_channel.transpose(0, 1)
    cache_time_b = cache_time.transpose(0, 1)
    print(f"  caches: channel={_shape(cache_channel_b)} time={_shape(cache_time_b)} "
          f"len={_shape(cache_len)}")

    encoder_streaming = EncoderStreamingWithPostPrompt(
        encoder.eval(), model.prompt_kernel.eval(), num_prompts=NUM_PROMPTS
    )
    preprocessor = PreprocessorWrapper(model.preprocessor.eval())
    decoder = DecoderWrapper(model.decoder.eval())
    joint = JointWrapper(model.joint.eval())
    model.decoder._rnnt_export = True

    total_mel_frames = chunk_mel_frames + pre_encode_cache
    max_samples = int(30.0 * sample_rate)

    # === 1. Preprocessor (dynamic audio length) ===
    print("Exporting preprocessor ...")
    audio = torch.randn(1, max_samples)
    traced = torch.jit.trace(
        preprocessor, (audio, torch.tensor([max_samples], dtype=torch.int32)), strict=False
    )
    _save_ir(
        traced,
        (audio, torch.tensor([max_samples], dtype=torch.int32)),
        [("audio", [1, -1], ov.Type.f32), ("audio_length", [1], ov.Type.i32)],
        ["mel", "mel_length"],
        output_dir / "nemotron_preprocessor.xml",
        fp16,
    )

    # === 2. Encoder (prompt-aware streaming, fixed chunk) ===
    print("Exporting encoder ...")
    mel = torch.randn(1, mel_features, total_mel_frames)
    mel_len = torch.tensor([total_mel_frames], dtype=torch.int32)
    prompt_id = torch.tensor([0], dtype=torch.int32)
    enc_ex = (mel, mel_len, cache_channel_b, cache_time_b, cache_len, prompt_id)
    traced = torch.jit.trace(encoder_streaming, enc_ex, strict=False)
    _save_ir(
        traced,
        enc_ex,
        [
            ("mel", list(_shape(mel)), ov.Type.f32),
            ("mel_length", [1], ov.Type.i32),
            ("cache_channel", list(_shape(cache_channel_b)), ov.Type.f32),
            ("cache_time", list(_shape(cache_time_b)), ov.Type.f32),
            ("cache_len", [1], ov.Type.i32),
            ("prompt_id", [1], ov.Type.i32),
        ],
        ["encoded", "encoded_length", "cache_channel_out", "cache_time_out", "cache_len_out"],
        output_dir / "nemotron_encoder.xml",
        fp16,
    )

    # === 3. Decoder (RNNT prediction, single step) ===
    print("Exporting decoder ...")
    dec_hidden = int(model.decoder.pred_hidden)
    dec_layers = int(model.decoder.pred_rnn_layers)
    targets = torch.tensor([[model.decoder.blank_idx]], dtype=torch.int32)
    target_len = torch.tensor([1], dtype=torch.int32)
    h = torch.zeros(dec_layers, 1, dec_hidden)
    c = torch.zeros(dec_layers, 1, dec_hidden)
    dec_ex = (targets, target_len, h, c)
    traced = torch.jit.trace(decoder, dec_ex, strict=False)
    _save_ir(
        traced,
        dec_ex,
        [
            ("token", [1, 1], ov.Type.i32),
            ("token_length", [1], ov.Type.i32),
            ("h_in", list(_shape(h)), ov.Type.f32),
            ("c_in", list(_shape(c)), ov.Type.f32),
        ],
        ["decoder_out", "h_out", "c_out"],
        output_dir / "nemotron_decoder.xml",
        fp16,
    )

    # === 4. Joint (single step) ===
    print("Exporting joint ...")
    with torch.no_grad():
        mel_test, _ = preprocessor(
            audio[:, :sample_rate], torch.tensor([sample_rate], dtype=torch.int32)
        )
        cc, ct_, cl = encoder.get_initial_cache_state(batch_size=1, device="cpu")
        enc_out, _, _, _, _ = encoder_streaming(
            mel_test, torch.tensor([mel_test.shape[2]], dtype=torch.int32),
            cc.transpose(0, 1), ct_.transpose(0, 1), cl.to(torch.int32), prompt_id,
        )
        dec_out, _, _ = decoder(targets, target_len, h, c)
    enc_step = enc_out[:, :, :1].contiguous()
    dec_step = dec_out[:, :, :1].contiguous()
    joint_ex = (enc_step, dec_step)
    traced = torch.jit.trace(joint, joint_ex, strict=False)
    _save_ir(
        traced,
        joint_ex,
        [
            ("encoder", list(_shape(enc_step)), ov.Type.f32),
            ("decoder", list(_shape(dec_step)), ov.Type.f32),
        ],
        ["logits"],
        output_dir / "nemotron_joint.xml",
        fp16,
    )

    # === 5. Vocab + metadata ===
    vocab_size = int(model.tokenizer.vocab_size)
    vocab = {str(i): model.tokenizer.ids_to_tokens([i])[0] for i in range(vocab_size)}
    (output_dir / "nemotron_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2))

    prompt_dict = dict(model.cfg.model_defaults.prompt_dictionary)
    metadata = {
        "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
        "model_class": model_class,
        "precision": precision.upper(),
        "sample_rate": sample_rate,
        "mel_features": mel_features,
        "chunk_mel_frames": chunk_mel_frames,
        "pre_encode_cache": pre_encode_cache,
        "total_mel_frames": total_mel_frames,
        "att_context_size": parsed_att,
        "vocab_size": vocab_size,
        "blank_idx": int(model.decoder.blank_idx),
        "cache_channel_shape": list(cache_channel_b.shape),
        "cache_time_shape": list(cache_time_b.shape),
        "decoder_hidden": dec_hidden,
        "decoder_layers": dec_layers,
        "encoder_dim": int(enc_out.shape[1]),
        "num_prompts": NUM_PROMPTS,
        "prompt_dictionary": prompt_dict,
        "default_prompt_id": prompt_dict.get("auto", 101),
        "lang_tag_token_ids": _lang_tag_token_ids(model),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Done. Exported OpenVINO IR to {output_dir}")


if __name__ == "__main__":
    app()
