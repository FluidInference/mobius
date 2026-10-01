"""Convert MOSS-Audio-Tokenizer-Nano to CoreML.

    MossNano-CodecDecoder-{tag}.mlpackage   codes [16,1,T≤max] → audio [1,2,T*3840]
    MossNano-CodecEncoder-{tag}.mlpackage   audio [1,2,S≤max*3840] → codes [16,1,S/3840]

Parity: decoder vs upstream decode() on real generated tokens (SNR), encoder vs upstream
encode() on the bundled reference clip (exact code match rate).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.codec_coreml import MossCodecDecoder, MossCodecEncoder  # noqa: E402

CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
FRAME = 3840


def snr_db(got: np.ndarray, want: np.ndarray) -> float:
    noise = np.sum((got - want) ** 2)
    return float(10 * np.log10(np.sum(want**2) / max(noise, 1e-20)))


def precision(fp16: bool):
    if not fp16:
        return ct.precision.FLOAT32
    return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in {"softmax"})


def load_prompt(path: Path) -> torch.Tensor:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)  # [S, C]
    wav = torch.from_numpy(wav.T)
    if sr != 48000:
        import torchaudio

        wav = torchaudio.functional.resample(wav, sr, 48000)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    S = wav.shape[1]
    pad = (-S) % FRAME
    return torch.nn.functional.pad(wav, (0, pad))[None]  # [1,2,S']


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(HERE / "build" / "codec"))
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--max-frames", type=int, default=125, help="decoder RangeDim upper bound in frames (10 s)")
    p.add_argument("--max-prompt-frames", type=int, default=188, help="encoder RangeDim upper bound in frames (15 s)")
    p.add_argument("--skip-decoder", action="store_true")
    p.add_argument("--skip-encoder", action="store_true")
    p.add_argument("--tokens", default=str(HERE / "build" / "ref_audio_token_ids.npy"))
    p.add_argument("--long-tokens", default=str(HERE / "build" / "ref_greedy_audio_token_ids.npy"))
    p.add_argument("--prompt-audio", default=str(HERE / "assets" / "en_2.wav"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "fp16" if args.fp16 else "fp32"
    target = ct.target.macOS14

    from transformers import AutoModel

    print(f"[0] loading {CODEC_REPO}")
    codec = AutoModel.from_pretrained(CODEC_REPO, trust_remote_code=True).eval()
    codes = torch.from_numpy(np.load(args.tokens)).long().T[:, None, :]  # [16,1,T]
    long_codes = torch.from_numpy(np.load(args.long_tokens)).long().T[:, None, : args.max_frames]
    with torch.no_grad():
        want = codec.decode(codes, return_dict=True).audio  # [1,2,S]
        want_long = codec.decode(long_codes, return_dict=True).audio

    if not args.skip_decoder:
        print("[1] decoder")
        dec = MossCodecDecoder(codec).eval()
        with torch.no_grad():
            got = dec(codes.to(torch.int32))
        print(f"      wrapper vs upstream: shape {tuple(got.shape)} SNR={snr_db(got.numpy(), want.numpy()):.1f} dB")
        t0 = time.perf_counter()
        with torch.no_grad():
            traced = torch.jit.trace(dec, (codes.to(torch.int32),), strict=False)
        T = ct.RangeDim(lower_bound=1, upper_bound=args.max_frames, default=codes.shape[-1])
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="codes", shape=(16, 1, T), dtype=np.int32)],
            outputs=[ct.TensorType(name="audio", dtype=np.float32)],
            compute_precision=precision(args.fp16),
            minimum_deployment_target=target,
            convert_to="mlprogram",
        )
        path = out_dir / f"MossNano-CodecDecoder-{tag}.mlpackage"
        ml.save(str(path))
        print(f"      saved {path.name} ({time.perf_counter() - t0:.0f}s)")
        for name, c, w in (("T=57", codes, want), (f"T={long_codes.shape[-1]}", long_codes, want_long)):
            t1 = time.perf_counter()
            pred = ml.predict({"codes": c.numpy().astype(np.int32)})["audio"]
            dt = time.perf_counter() - t1
            print(f"      coreml {name}: shape {pred.shape} SNR={snr_db(pred, w.numpy()):.1f} dB  {dt*1000:.0f} ms")
        del ml, dec

    if not args.skip_encoder:
        print("[2] encoder")
        audio = load_prompt(Path(args.prompt_audio))  # [1,2,S]
        with torch.no_grad():
            ref = codec.encode(audio, return_dict=True).audio_codes  # [16,1,T]
        enc = MossCodecEncoder(codec).eval()
        with torch.no_grad():
            got = enc(audio)
        n = ref.numel()
        print(f"      wrapper vs upstream: {tuple(got.shape)} vs {tuple(ref.shape)}  "
              f"exact={(got.long() == ref).sum().item()}/{n}")
        t0 = time.perf_counter()
        with torch.no_grad():
            traced = torch.jit.trace(enc, (audio,), strict=False)
        S = ct.RangeDim(lower_bound=FRAME, upper_bound=args.max_prompt_frames * FRAME, default=audio.shape[-1])
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name="audio", shape=(1, 2, S), dtype=np.float32)],
            outputs=[ct.TensorType(name="codes", dtype=np.int32)],
            compute_precision=precision(args.fp16),
            minimum_deployment_target=target,
            convert_to="mlprogram",
        )
        path = out_dir / f"MossNano-CodecEncoder-{tag}.mlpackage"
        ml.save(str(path))
        print(f"      saved {path.name} ({time.perf_counter() - t0:.0f}s)")
        t1 = time.perf_counter()
        pred = torch.from_numpy(ml.predict({"audio": audio.numpy()})["codes"]).long()
        dt = time.perf_counter() - t1
        per_cb = [(pred[i] == ref[i]).float().mean().item() for i in range(16)]
        print(f"      coreml: exact={(pred == ref).sum().item()}/{n}  {dt*1000:.0f} ms  "
              f"per-codebook match={[round(v, 3) for v in per_cb]}")
    print("[done]")


if __name__ == "__main__":
    main()
