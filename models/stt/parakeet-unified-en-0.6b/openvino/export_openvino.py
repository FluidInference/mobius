#!/usr/bin/env python3
"""Export NVIDIA parakeet-unified-en-0.6b to OpenVINO IR (offline / full-context).

Unified FastConformer-RNNT (English, 600M). Unlike the cache-aware Nemotron
models this encoder is STATELESS (no cache tensors, no prompt) — offline mode
runs full self-attention over the whole utterance. Produces the flat IR layout
the eddy-audio backend consumes (decoder/joint identical to Nemotron; the
encoder takes mel+length and returns encoded+length with a dynamic time axis):

    nemotron_preprocessor.xml/.bin   audio -> mel        (dynamic length)
    nemotron_encoder.xml/.bin        mel + mel_length -> encoded + encoded_length
    nemotron_decoder.xml/.bin        token + lstm state -> dec_out + state
    nemotron_joint.xml/.bin          enc_step + dec_step -> logits
    nemotron_vocab.json, metadata.json

Filenames keep the nemotron_* prefix so the eddy backend resolves them with no
special-casing. Requires NeMo + torch + openvino (no coremltools).

    uv run python export_openvino.py --output-dir build_ov --precision FP16
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import openvino as ov
try:  # OpenVINO 2026+ removed the openvino.runtime shim
    import openvino.opset13 as _ovops
    from openvino.utils import replace_node as _ov_replace_node
except ModuleNotFoundError:
    import openvino.runtime.opset13 as _ovops
    from openvino.runtime.utils import replace_node as _ov_replace_node
import torch
import typer

DEFAULT_MODEL_ID = "nvidia/parakeet-unified-en-0.6b"


# --- TorchScript wrappers (inlined from coreml/components.py, minus coremltools).

class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        mel, mel_length = self.module(input_signal=audio_signal, length=length.to(dtype=torch.long))
        return mel, mel_length


class EncoderWrapper(torch.nn.Module):
    """Stateless encoder: mel + length -> encoded + encoded_length. No caches."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, features: torch.Tensor, length: torch.Tensor):
        encoded, encoded_lengths = self.module(audio_signal=features, length=length.to(dtype=torch.long))
        return encoded, encoded_lengths.to(dtype=torch.int32)


class DecoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, targets, target_lengths, h_in, c_in):
        decoder_output, _, new_state = self.module(
            targets=targets.to(dtype=torch.long),
            target_length=target_lengths.to(dtype=torch.long),
            states=[h_in, c_in],
        )
        return decoder_output, new_state[0], new_state[1]


class JointWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, encoder_outputs, decoder_outputs):
        encoder_outputs = encoder_outputs.transpose(1, 2)  # [B, T, D]
        decoder_outputs = decoder_outputs.transpose(1, 2)  # [B, U, D]
        enc_proj = self.module.enc(encoder_outputs)
        dec_proj = self.module.pred(decoder_outputs)
        x = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)
        x = self.module.joint_net[0](x)  # ReLU
        x = self.module.joint_net[1](x)  # Dropout (no-op in eval)
        return self.module.joint_net[2](x)  # Linear -> logits [B, T, U, V+1]


def _shape(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(d) for d in t.shape)


def _make_npu_safe(ov_model: ov.Model) -> int:
    """BitwiseNot(bool) -> LogicalNot so the IR runs on the OpenVINO NPU (the NPU
    plugin does an integer complement on a bool, breaking attention masks)."""
    n = 0
    for op in list(ov_model.get_ordered_ops()):
        if op.get_type_name() == "BitwiseNot":
            repl = _ovops.logical_not(op.input_value(0))
            repl.get_output_tensor(0).set_names(op.get_output_tensor(0).get_names())
            _ov_replace_node(op, repl)
            n += 1
    return n


def _save_ir(traced, example_input, input_specs, output_names, out_path: Path, fp16: bool) -> None:
    ov_model = ov.convert_model(
        traced, example_input=example_input, input=[(s[1], s[2]) for s in input_specs]
    )
    for port, (name, _, _) in zip(ov_model.inputs, input_specs):
        port.get_tensor().set_names({name})
    for port, name in zip(ov_model.outputs, output_names):
        port.get_tensor().set_names({name})
    n_fixed = _make_npu_safe(ov_model)
    if n_fixed:
        print(f"  NPU-safe: rewrote {n_fixed} BitwiseNot -> LogicalNot")
    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, str(out_path), compress_to_fp16=fp16)
    print(f"  saved {out_path.name}")


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def convert(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model-id"),
    nemo_path: str = typer.Option("", "--nemo-path", help="Local .nemo (overrides --model-id)"),
    output_dir: Path = typer.Option(Path("build_ov"), "--output-dir"),
    precision: str = typer.Option("FP32", "--precision", help="FP32 or FP16"),
    streaming_context: str = typer.Option(
        "", "--streaming-context",
        help="left,chunk,right in encoder frames (e.g. 70,13,13) for a fixed-window "
             "streaming encoder (static shapes -> NPU-compatible). Empty = offline/full-context.",
    ),
) -> None:
    import nemo.collections.asr as nemo_asr

    fp16 = precision.upper() == "FP16"
    output_dir.mkdir(parents=True, exist_ok=True)

    if nemo_path:
        print(f"Loading model from {nemo_path} ...")
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(restore_path=nemo_path, map_location="cpu")
    else:
        print(f"Loading model {model_id} ...")
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_id, map_location="cpu")
    model.eval()
    model_class = f"{type(model).__module__}.{type(model).__name__}"

    sample_rate = int(model.cfg.preprocessor.sample_rate)
    mel_features = int(model.cfg.preprocessor.features)
    subsampling = int(getattr(model.encoder, "subsampling_factor", 8))

    ctx = [int(x) for x in streaming_context.split(",")] if streaming_context else None
    if ctx is not None:
        assert len(ctx) == 3, "--streaming-context must be left,chunk,right"
        # Fixed-window chunked attention: bake the mask for [L,C,R] encoder frames.
        model.encoder.set_default_att_context_size(att_context_size=ctx)
        enc_frame_samples = subsampling * int(sample_rate * 0.01 + 0.5)  # 8 * 160 = 1280 (80 ms)
        window_samples = sum(ctx) * enc_frame_samples
    else:
        # Offline: full self-attention over the whole utterance (no chunk limit).
        model.encoder.set_default_att_context_size(att_context_size=[-1, -1, -1])
        window_samples = None

    preprocessor = PreprocessorWrapper(model.preprocessor.eval())
    encoder = EncoderWrapper(model.encoder.eval())
    decoder = DecoderWrapper(model.decoder.eval())
    joint = JointWrapper(model.joint.eval())
    model.decoder._rnnt_export = True

    max_samples = window_samples if window_samples else int(16.0 * sample_rate)

    # === 1. Preprocessor (dynamic audio length) ===
    print("Exporting preprocessor ...")
    audio = torch.randn(1, max_samples)
    pp_ex = (audio, torch.tensor([max_samples], dtype=torch.int32))
    traced = torch.jit.trace(preprocessor, pp_ex, strict=False)
    _save_ir(
        traced, pp_ex,
        [("audio", [1, -1], ov.Type.f32), ("audio_length", [1], ov.Type.i32)],
        ["mel", "mel_length"],
        output_dir / "nemotron_preprocessor.xml", fp16,
    )

    # === 2. Encoder ===
    # Streaming: FIXED window shape (static -> NPU-compatible). Offline: dynamic time.
    print("Exporting encoder ...")
    with torch.no_grad():
        mel, mel_len = preprocessor(audio, torch.tensor([max_samples], dtype=torch.int32))
        mel_len = mel_len.to(torch.int32)
    enc_ex = (mel, mel_len)
    traced = torch.jit.trace(encoder, enc_ex, strict=False)
    mel_time = int(mel.shape[2]) if ctx is not None else -1
    _save_ir(
        traced, enc_ex,
        [("mel", [1, mel_features, mel_time], ov.Type.f32), ("mel_length", [1], ov.Type.i32)],
        ["encoded", "encoded_length"],
        output_dir / "nemotron_encoder.xml", fp16,
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
        traced, dec_ex,
        [
            ("token", [1, 1], ov.Type.i32),
            ("token_length", [1], ov.Type.i32),
            ("h_in", list(_shape(h)), ov.Type.f32),
            ("c_in", list(_shape(c)), ov.Type.f32),
        ],
        ["decoder_out", "h_out", "c_out"],
        output_dir / "nemotron_decoder.xml", fp16,
    )

    # === 4. Joint (single step) ===
    print("Exporting joint ...")
    with torch.no_grad():
        enc_out, _ = encoder(mel, mel_len)
        dec_out, _, _ = decoder(targets, target_len, h, c)
    enc_step = enc_out[:, :, :1].contiguous()
    dec_step = dec_out[:, :, :1].contiguous()
    joint_ex = (enc_step, dec_step)
    traced = torch.jit.trace(joint, joint_ex, strict=False)
    _save_ir(
        traced, joint_ex,
        [("encoder", list(_shape(enc_step)), ov.Type.f32), ("decoder", list(_shape(dec_step)), ov.Type.f32)],
        ["logits"],
        output_dir / "nemotron_joint.xml", fp16,
    )

    # === 5. Vocab + metadata (no caches, no prompt) ===
    vocab_size = int(model.tokenizer.vocab_size)
    vocab = {str(i): model.tokenizer.ids_to_tokens([i])[0] for i in range(vocab_size)}
    (output_dir / "nemotron_vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "model": model_id,
        "model_class": model_class,
        "precision": precision.upper(),
        "mode": "streaming" if ctx is not None else "offline",
        "streaming": ctx is not None,
        "sample_rate": sample_rate,
        "mel_features": mel_features,
        "vocab_size": vocab_size,
        "blank_idx": int(model.decoder.blank_idx),
        "decoder_hidden": dec_hidden,
        "decoder_layers": dec_layers,
        "encoder_dim": int(enc_out.shape[1]),
        "subsampling_factor": subsampling,
    }
    if ctx is not None:
        left, chunk, right = ctx
        metadata.update({
            "context_encoder_frames": {"left": left, "chunk": chunk, "right": right},
            "window_samples": window_samples,
            "window_mel_frames": int(mel.shape[2]),
            "encoder_window_frames": int(enc_out.shape[2]),  # = left+chunk+right
            "chunk_samples": chunk * (subsampling * int(sample_rate * 0.01 + 0.5)),
        })
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Done. Exported OpenVINO IR to {output_dir}")


if __name__ == "__main__":
    app()
