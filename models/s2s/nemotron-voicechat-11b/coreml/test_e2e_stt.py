#!/usr/bin/env python3
"""End-to-end STT smoke test for the converted VoiceChat-11B CoreML models.

Real audio -> NeMo mel (normalize NA) -> per-frame streaming encoder ->
asr_emb -> RNNT greedy decode -> sentencepiece text.

Runs the identical explicit loop through the torch wrappers (parity reference)
and through the CoreML models (encoder fp16, RNNT fp32), then prints both
transcripts. Success = coherent, matching transcripts of the sample audio.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import typer

CHECKPOINT_DIR = Path.home() / "Documents/models/voicechat-11b"
SAMPLE_WAV = CHECKPOINT_DIR / "Speech/examples/speechlm2/sample_audio/sample_general.wav"
BLANK_ID = 1024
MAX_SYMBOLS_PER_FRAME = 10

app = typer.Typer(add_completion=False)


def make_mel(wav_path: Path) -> torch.Tensor:
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

    cfg = json.loads((CHECKPOINT_DIR / "config.json").read_text())
    pre_cfg = dict(cfg["model"]["stt"]["model"]["perception"]["preprocessor"])
    pre_cfg.pop("_target_")
    pre = AudioToMelSpectrogramPreprocessor(**pre_cfg)
    pre.eval()

    audio, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000
    sig = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        mel, mel_len = pre(input_signal=sig, length=torch.tensor([sig.shape[1]]))
    return mel[:, :, : int(mel_len.item())]


def mel_windows(mel: torch.Tensor):
    """Yield 17-mel-frame streaming windows: 9 pre-encode context + 8 new.

    Context is the real mel history, zero-left-padded only for the frames that
    don't exist yet (start=8 gets 1 zero + mel[0:8], not 9 zeros). The final
    partial chunk is zero-right-padded to 8 frames rather than dropped (the
    export has a fixed [1, mel, 17] shape).
    """
    T = mel.shape[2]
    for start in range(0, T, 8):
        n_ctx = min(start, 9)
        ctx = torch.cat(
            [torch.zeros(1, mel.shape[1], 9 - n_ctx), mel[:, :, start - n_ctx : start]], dim=2
        )
        new = mel[:, :, start : start + 8]
        if new.shape[2] < 8:
            new = torch.cat([new, torch.zeros(1, mel.shape[1], 8 - new.shape[2])], dim=2)
        yield torch.cat([ctx, new], dim=2)


def load_tokenizer():
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(CHECKPOINT_DIR / "rnnt_tokenizer/tokenizer.model"))
    return sp


def torch_pipeline(mel: torch.Tensor) -> list[int]:
    from convert_encoder import build_model
    from convert_rnnt import DecoderStepWrapper, JointStepWrapper, build_models

    wrapper, _ = build_model()
    wrapper.encoder.setup_streaming_params()
    decoder, joint = build_models()
    dec_wrap, joint_wrap = DecoderStepWrapper(decoder), JointStepWrapper(joint)

    cch, ct_, clen = wrapper.encoder.get_initial_cache_state(batch_size=1, device="cpu")
    cch, ct_ = cch.transpose(0, 1), ct_.transpose(0, 1)
    clen = clen.to(torch.int32)
    mel_len = torch.tensor([17], dtype=torch.int32)

    h = torch.zeros(2, 1, 640)
    c = torch.zeros(2, 1, 640)
    with torch.no_grad():
        dec_out, h, c = dec_wrap(torch.tensor([[BLANK_ID]], dtype=torch.int32), h, c)

    tokens: list[int] = []
    with torch.no_grad():
        for window in mel_windows(mel):
            out = wrapper(window, mel_len, cch, ct_, clen)
            cch, ct_, clen = out[3], out[4], out[5]
            enc_frame = out[1].transpose(1, 2)  # [1, 1024, 1]
            for _ in range(MAX_SYMBOLS_PER_FRAME):
                logits = joint_wrap(enc_frame, dec_out)
                k = int(logits.flatten().argmax())
                if k == BLANK_ID:
                    break
                tokens.append(k)
                dec_out, h, c = dec_wrap(torch.tensor([[k]], dtype=torch.int32), h, c)
    return tokens


def coreml_pipeline(mel: torch.Tensor) -> list[int]:
    import coremltools as ct

    build = Path("build")
    enc = ct.models.MLModel(str(build / "encoder/encoder_fp16.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    dec = ct.models.MLModel(str(build / "rnnt/decoder.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    joint = ct.models.MLModel(str(build / "rnnt/joint.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)

    from convert_encoder import build_model

    wrapper, _ = build_model()
    wrapper.encoder.setup_streaming_params()
    cch0, ct0, clen0 = wrapper.encoder.get_initial_cache_state(batch_size=1, device="cpu")
    cch = cch0.transpose(0, 1).numpy()
    ctt = ct0.transpose(0, 1).numpy()
    clen = clen0.numpy().astype(np.int32)
    mel_len = np.array([17], dtype=np.int32)

    h = np.zeros((2, 1, 640), dtype=np.float32)
    c = np.zeros((2, 1, 640), dtype=np.float32)
    o = dec.predict({"token": np.array([[BLANK_ID]], dtype=np.int32), "h_in": h, "c_in": c})
    dec_out, h, c = o["decoder_out"], o["h_out"], o["c_out"]

    tokens: list[int] = []
    for window in mel_windows(mel):
        o = enc.predict(
            {
                "mel": window.numpy(),
                "mel_length": mel_len,
                "cache_last_channel": cch,
                "cache_last_time": ctt,
                "cache_last_channel_len": clen,
            }
        )
        cch = o["new_cache_last_channel"].astype(np.float32)
        ctt = o["new_cache_last_time"].astype(np.float32)
        clen = o["new_cache_last_channel_len"].astype(np.int32)
        enc_frame = o["asr_emb"].transpose(0, 2, 1).astype(np.float32)  # [1, 1024, 1]
        for _ in range(MAX_SYMBOLS_PER_FRAME):
            jl = joint.predict({"encoder_step": enc_frame, "decoder_step": dec_out.astype(np.float32)})
            k = int(jl["logits"].flatten().argmax())
            if k == BLANK_ID:
                break
            tokens.append(k)
            od = dec.predict({"token": np.array([[k]], dtype=np.int32), "h_in": h, "c_in": c})
            dec_out, h, c = od["decoder_out"], od["h_out"], od["c_out"]
    return tokens


SAMPLE_EXPECT = "do you know what color the sky is"


@app.command()
def main(
    wav: Path = typer.Option(SAMPLE_WAV, help="16 kHz mono wav"),
    skip_torch: bool = typer.Option(False),
    expect: str = typer.Option("", help="substring the transcript must contain (defaults to the sample's reference)"),
) -> None:
    sp = load_tokenizer()
    mel = make_mel(wav)
    typer.echo(f"mel: {tuple(mel.shape)} ({-(-mel.shape[2] // 8)} streaming steps)")

    if not skip_torch:
        t_tokens = torch_pipeline(mel)
        typer.echo(f"\n[torch ] {sp.decode(t_tokens)}")

    c_tokens = coreml_pipeline(mel)
    text = sp.decode(c_tokens)
    typer.echo(f"\n[coreml] {text}")

    failed = False
    if not skip_torch:
        match = t_tokens == c_tokens
        typer.echo(f"\ntoken sequences identical: {match} (torch {len(t_tokens)} vs coreml {len(c_tokens)} tokens)")
        if not match:
            failed = True
    if not expect and wav == SAMPLE_WAV:
        expect = SAMPLE_EXPECT
    if expect:
        ok = expect.lower() in text.lower()
        typer.echo(f"transcript contains {expect!r}: {ok}")
        if not ok:
            failed = True
    if failed:
        raise typer.Exit(1)
    typer.echo("E2E STT OK")


if __name__ == "__main__":
    app()
