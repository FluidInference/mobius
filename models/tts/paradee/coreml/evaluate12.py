"""Paradee v1.2 end to end, every voice: Core ML vs PyTorch and vs the released fp32 ONNX (same noise, through the
noise-input copy ref12.py writes), on misaki phonemes (British G2P for bf_/bm_). Writes WAVs (plus the upstream
int8 ONNX's, and the text, for asr_check.py) and prints RTFx.
usage: evaluate12.py [--dir build-v1.2] [--cu CPU_ONLY] [--wav-dir out/wav-v1.2] [--voices 0,1,2,3]"""
import argparse
import pathlib
import re
import time

import coremltools as ct
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from misaki import en, espeak

import ref12
from convert12 import snr
from evaluate import LONG, logmel_l1

SENTS = [
    "Paradee is a small voice that runs anywhere.",
    *[re.sub(r"^\d+\. ", "", s) for s in open(ref12.W12 / "samples/sentences.txt").read().strip().splitlines()],
    "The quirky xylophone cost 1,234 dollars, said Mrs. Zzyzx.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path, default=pathlib.Path("build-v1.2"))
    ap.add_argument("--cu", default="CPU_ONLY")
    ap.add_argument("--wav-dir", type=pathlib.Path, default=pathlib.Path("out/wav-v1.2"))
    ap.add_argument("--voices", default="0,1,2,3")
    A = ap.parse_args()
    A.wav_dir.mkdir(parents=True, exist_ok=True)
    g2p = {b: en.G2P(trf=False, british=b, fallback=espeak.EspeakFallback(british=b), unk="") for b in (False, True)}
    ts, vd, ts_tab, vd_tab = ref12.load12()
    cu = getattr(ct.ComputeUnit, A.cu)
    mt = ct.models.MLModel(str(A.dir / "ParadeeText.mlpackage"), compute_units=cu)
    ma = ct.models.MLModel(str(A.dir / "ParadeeAcoustic.mlpackage"), compute_units=cu)
    onnx_path = ref12.W12 / "paradee_noise_in.onnx"
    if not onnx_path.exists():
        ref12.onnx_with_noise_input(ref12.W12 / "onnx/paradee.onnx", onnx_path)
    onnx = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx8 = ort.InferenceSession(str(ref12.W12 / "onnx/paradee_int8.onnx"), providers=["CPUExecutionProvider"])

    cases = [(s, 1.0) for s in SENTS] + [(LONG, 1.0), (SENTS[2], 0.8), (SENTS[2], 1.3)]
    tot_audio = tot_cm = 0.0
    for v in map(int, A.voices.split(",")):
        voice = ref12.VOICES[v]
        print(f"--- {voice}")
        for i, (text, speed) in enumerate(cases):
            ps = g2p[voice.startswith("b")](text)[0]
            ids = ref12.phonemes_to_ids(ps)
            ref, mid = ref12.reference(ts, vd, ts_tab, vd_tab, ids, v, speed=speed, seed=100 * v + i)
            ref = ref.numpy().ravel()
            vin = np.array([v], np.int32)

            def run():
                o = mt.predict({"input_ids": ids.numpy().astype(np.int32), "voice": vin})
                n = np.maximum(1, np.round(o["duration"][0] / speed)).astype(np.int64)
                z = mid["noise"].numpy()[None]
                if n.sum() * 600 != z.shape[2]:  # quantized durations differ from fp32 torch: fresh noise
                    z = np.random.default_rng(i).standard_normal((1, 1, int(n.sum()) * 600)).astype(np.float32)
                feed = {"en": np.repeat(o["d"], n, -1), "asr": np.repeat(o["asr_tok"], n, -1), "noise": z, "voice": vin}
                return n, z, ma.predict(feed)["audio"].ravel()

            run()
            t0 = time.perf_counter()
            n, z, y = run()
            dt = time.perf_counter() - t0
            tot_audio += len(y) / ref12.SR
            tot_cm += dt
            mism = int((n != ref12.R.durations(mid["dur"], speed).numpy()).sum())
            yo = onnx.run(None, {"input_ids": ids.numpy(), "speed": np.array([speed], np.float32),
                                 "voice": np.array([v], np.int64), "noise": mid["noise"].numpy()})[0].ravel()
            # waveform / log-mel comparisons only line up when every rounded duration matches (int8 may differ)
            cmp = (f"SNR vs torch {snr(ref, y):5.1f} dB  logmel L1 vs torch {logmel_l1(ref, y):.3f} vs onnx "
                   f"{logmel_l1(yo, y):.3f} (onnx vs torch {logmel_l1(yo, ref):.3f})" if len(y) == len(ref) and not mism
                   else f"length {len(y)} vs torch {len(ref)}")
            print(f"{i:2d} T={ids.shape[1]:3d} {len(y) / ref12.SR:5.1f}s x{speed}  dur mism {mism:2d}  {cmp}  "
                  f"RTFx {len(y) / ref12.SR / dt:6.1f}")
            sf.write(A.wav_dir / f"{voice}_{i}_coreml.wav", y, ref12.SR)
            sf.write(A.wav_dir / f"{voice}_{i}_onnx.wav", yo, ref12.SR)
            y8 = onnx8.run(None, {"input_ids": ids.numpy(), "speed": np.array([speed], np.float32),
                                  "voice": np.array([v], np.int64)})[0].ravel()  # upstream int8, its own noise
            sf.write(A.wav_dir / f"{voice}_{i}_onnx8.wav", y8, ref12.SR)
            (A.wav_dir / f"{voice}_{i}.txt").write_text(text)
    print(f"total {tot_audio:.1f} s audio, Core ML {tot_cm:.2f} s -> RTFx {tot_audio / tot_cm:.1f} ({A.cu})")


if __name__ == "__main__":
    main()
