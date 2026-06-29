#!/usr/bin/env python3
"""Export NVIDIA nemotron-speech-streaming-en-0.6b to OpenVINO IR.

Monolingual (English) sibling of nemotron-3.5-asr-streaming-multilingual-0.6b.
Same cache-aware streaming FastConformer-RNNT, but with NO prompt/language
conditioning — so the exported encoder has no `prompt_id` input. Produces the
same flat IR layout the eddy-audio backend consumes (the eddy backend
auto-detects the absent prompt_id and serves both variants):

    nemotron_preprocessor.xml/.bin   audio -> mel
    nemotron_encoder.xml/.bin        mel + caches -> encoded + caches
    nemotron_decoder.xml/.bin        token + lstm state -> dec_out + state
    nemotron_joint.xml/.bin          enc_step + dec_step -> logits
    nemotron_vocab.json, metadata.json

Mirrors ../../nemotron-asr-streaming-multilingual-0.6b/openvino/export_openvino.py.
Requires NeMo + torch + openvino (no coremltools — wrappers are inlined).

    uv run python export_openvino.py --output-dir build_ov --precision FP16
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import openvino as ov
import openvino.runtime.opset13 as _ovops
from openvino.runtime.utils import replace_node as _ov_replace_node
import torch
import typer

DEFAULT_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"


# --- TorchScript wrappers (inlined from the coreml conversion scripts, minus
# coremltools, minus the multilingual prompt kernel). ---

class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor):
        mel, mel_length = self.module(input_signal=audio_signal, length=length.to(dtype=torch.long))
        return mel, mel_length


class EncoderStreamingWrapper(torch.nn.Module):
    """Cache-aware streaming encoder: mel + caches -> encoded + caches.

    No prompt_id (monolingual). Caches are [B, L, ...] externally, transposed to
    [L, B, ...] for NeMo and back on output, matching the multilingual export so
    eddy's cache-carry logic is identical.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        features: torch.Tensor,
        length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ):
        cache_ch_t = cache_last_channel.transpose(0, 1)
        cache_t_t = cache_last_time.transpose(0, 1)
        cache_len_i64 = cache_last_channel_len.to(dtype=torch.int64)
        encoded, encoded_lengths, cache_ch_next, cache_t_next, cache_len_next = self.module(
            audio_signal=features,
            length=length.to(dtype=torch.long),
            cache_last_channel=cache_ch_t,
            cache_last_time=cache_t_t,
            cache_last_channel_len=cache_len_i64,
        )
        return (
            encoded,
            encoded_lengths.to(dtype=torch.int32),
            cache_ch_next.transpose(0, 1),
            cache_t_next.transpose(0, 1),
            cache_len_next.to(dtype=torch.int32),
        )


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
        return self.module.joint_net[2](x)  # Linear -> logits


def _shape(t: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(d) for d in t.shape)


def _make_npu_safe(ov_model: ov.Model) -> int:
    """Rewrite BitwiseNot(bool) -> LogicalNot so the IR runs on the OpenVINO NPU.

    The NPU plugin does an integer complement on a boolean (~0=-1, ~1=-2 are both
    'true'), so the FastConformer attention mask goes all-true -> uniform softmax
    -> the encoder output collapses to ~0 -> empty transcripts. LogicalNot is
    identical on bool and compiles correctly on NPU (no-op for CPU/GPU).
    """
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
) -> None:
    import nemo.collections.asr as nemo_asr

    fp16 = precision.upper() == "FP16"
    output_dir.mkdir(parents=True, exist_ok=True)

    if nemo_path:
        print(f"Loading model from {nemo_path} ...")
        model = nemo_asr.models.ASRModel.restore_from(restore_path=nemo_path, map_location="cpu")
    else:
        print(f"Loading model {model_id} ...")
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_id, map_location="cpu")
    model.eval()
    model_class = f"{type(model).__module__}.{type(model).__name__}"
    print(f"  class: {model_class}")

    sample_rate = int(model.cfg.preprocessor.sample_rate)
    mel_features = int(model.cfg.preprocessor.features)

    encoder = model.encoder
    encoder.setup_streaming_params()
    # Streaming chunk geometry from the model config (chunk_size=[105,112],
    # pre_encode_cache_size=[0,9] for this checkpoint -> 112 + 9 = 121).
    chunk_mel_frames = int(encoder.streaming_cfg.chunk_size[-1])
    pre_encode_cache = int(encoder.streaming_cfg.pre_encode_cache_size[-1])
    total_mel_frames = chunk_mel_frames + pre_encode_cache

    cache_channel, cache_time, cache_len = encoder.get_initial_cache_state(batch_size=1, device="cpu")
    cache_len = cache_len.to(torch.int32)
    cache_channel_b = cache_channel.transpose(0, 1)
    cache_time_b = cache_time.transpose(0, 1)
    print(f"  chunk={chunk_mel_frames} pre_cache={pre_encode_cache} total={total_mel_frames}")
    print(f"  caches: channel={_shape(cache_channel_b)} time={_shape(cache_time_b)}")

    encoder_streaming = EncoderStreamingWrapper(encoder.eval())
    preprocessor = PreprocessorWrapper(model.preprocessor.eval())
    decoder = DecoderWrapper(model.decoder.eval())
    joint = JointWrapper(model.joint.eval())
    model.decoder._rnnt_export = True

    max_samples = int(30.0 * sample_rate)

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

    # === 2. Encoder (cache-aware streaming, NO prompt_id) ===
    print("Exporting encoder ...")
    mel = torch.randn(1, mel_features, total_mel_frames)
    mel_len = torch.tensor([total_mel_frames], dtype=torch.int32)
    enc_ex = (mel, mel_len, cache_channel_b, cache_time_b, cache_len)
    traced = torch.jit.trace(encoder_streaming, enc_ex, strict=False)
    _save_ir(
        traced, enc_ex,
        [
            ("mel", list(_shape(mel)), ov.Type.f32),
            ("mel_length", [1], ov.Type.i32),
            ("cache_channel", list(_shape(cache_channel_b)), ov.Type.f32),
            ("cache_time", list(_shape(cache_time_b)), ov.Type.f32),
            ("cache_len", [1], ov.Type.i32),
        ],
        ["encoded", "encoded_length", "cache_channel_out", "cache_time_out", "cache_len_out"],
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
        mel_test, _ = preprocessor(audio[:, :sample_rate], torch.tensor([sample_rate], dtype=torch.int32))
        cc, ct_, cl = encoder.get_initial_cache_state(batch_size=1, device="cpu")
        enc_out, _, _, _, _ = encoder_streaming(
            mel_test, torch.tensor([mel_test.shape[2]], dtype=torch.int32),
            cc.transpose(0, 1), ct_.transpose(0, 1), cl.to(torch.int32),
        )
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

    # === 5. Vocab + metadata (no prompt fields) ===
    vocab_size = int(model.tokenizer.vocab_size)
    vocab = {str(i): model.tokenizer.ids_to_tokens([i])[0] for i in range(vocab_size)}
    (output_dir / "nemotron_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2))

    metadata = {
        "model": model_id,
        "model_class": model_class,
        "precision": precision.upper(),
        "sample_rate": sample_rate,
        "mel_features": mel_features,
        "chunk_mel_frames": chunk_mel_frames,
        "pre_encode_cache": pre_encode_cache,
        "total_mel_frames": total_mel_frames,
        "vocab_size": vocab_size,
        "blank_idx": int(model.decoder.blank_idx),
        "cache_channel_shape": list(cache_channel_b.shape),
        "cache_time_shape": list(cache_time_b.shape),
        "decoder_hidden": dec_hidden,
        "decoder_layers": dec_layers,
        "encoder_dim": int(enc_out.shape[1]),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Done. Exported OpenVINO IR to {output_dir}")


if __name__ == "__main__":
    app()
