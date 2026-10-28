"""End-to-end Core ML vs PyTorch (matched noise) vs upstream ONNX on real misaki phonemes; writes WAVs + RTFx.
usage: evaluate.py [--dir build] [--cu CPU_ONLY] [--wav-dir out/wav]"""
import argparse
import json
import pathlib
import re
import time

import coremltools as ct
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from misaki import en, espeak

import convert as C
import paradee_ref as R

SENTS = [
    "Paradee is a small voice that runs anywhere.",
    *[re.sub(r"^\d+\. ", "", s) for s in open(R.HERE / "weights/samples/sentences.txt").read().strip().splitlines()],
    "The quirky xylophone cost 1,234 dollars, said Mrs. Zzyzx.",
]
LONG = ("Despite its size, the chickadee survives harsh winters by caching thousands of seeds and remembering where "
        "each one is hidden; researchers have found that its hippocampus grows in autumn, when it needs to store the most "
        "locations, and shrinks again in spring, once the seeds have been eaten or have sprouted, which makes it one of "
        "the clearest examples of seasonal brain plasticity in any animal studied so far, and a favourite subject of "
        "neuroscientists who want to know how memory is built.")


def logmel_l1(a, b):
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(R.SR, n_fft=1024, hop_length=256, n_mels=80)
    k = min(len(a), len(b))
    la, lb = (torch.log(mel(torch.from_numpy(x[:k]).float()).clamp_min(1e-5)) for x in (a, b))
    return (la - lb).abs().mean().item()


def expand(x, n):
    return np.repeat(x, n, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=pathlib.Path, default=pathlib.Path("build"))
    ap.add_argument("--cu", default="CPU_ONLY")
    ap.add_argument("--wav-dir", type=pathlib.Path, default=pathlib.Path("out/wav"))
    A = ap.parse_args()
    A.wav_dir.mkdir(parents=True, exist_ok=True)
    vocab = json.load(open(R.HERE / "weights/config.json"))["vocab"]
    g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk="")
    ts, ds = R.load_halves()
    cu = getattr(ct.ComputeUnit, A.cu)
    mt = ct.models.MLModel(str(A.dir / "ParadeeText.mlpackage"), compute_units=cu)
    ma = ct.models.MLModel(str(A.dir / "ParadeeAcoustic.mlpackage"), compute_units=cu)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    onnx = ort.InferenceSession(str(R.HERE / "weights/onnx/paradee.onnx"), so, providers=["CPUExecutionProvider"])
    onnx8 = ort.InferenceSession(str(R.HERE / "weights/onnx/paradee_int8.onnx"), so, providers=["CPUExecutionProvider"])

    cases = [(s, 1.0) for s in SENTS] + [(LONG, 1.0), (SENTS[2], 0.8), (SENTS[2], 1.3)]
    rows, tot_audio, tot_cm, tot_onnx = [], 0.0, 0.0, 0.0
    for i, (text, speed) in enumerate(cases):
        ps = g2p(text)[0]
        ids = C.phonemes_to_ids(vocab, ps)
        ref, mid = C.reference(ts, ds, ids, speed=speed, seed=i)
        ref = ref.numpy().ravel()

        def run():
            o = mt.predict({"input_ids": ids.numpy().astype(np.int32)})
            n = np.maximum(1, np.round(o["duration"][0] / speed)).astype(np.int64)
            z = mid["z"].numpy()
            if n.sum() * 2 * C.UP != z.shape[2]:  # quantized durations differ from fp32 torch: fresh noise
                z = np.random.default_rng(i).standard_normal((1, 1, int(n.sum()) * 2 * C.UP)).astype(np.float32)
            feed = {"en": expand(o["d"], n), "asr": expand(o["asr_tok"], n), "noise": z}
            return n, ma.predict(feed)["audio"].ravel()

        run()
        t0 = time.perf_counter()
        n, y = run()
        t_cm = time.perf_counter() - t0
        feed = {"input_ids": ids.numpy(), "speed": np.array([speed], np.float32)}
        onnx.run(None, feed)
        t0 = time.perf_counter()
        yo = onnx.run(None, feed)[0].ravel()
        t_onnx = time.perf_counter() - t0
        yo2 = onnx.run(None, feed)[0].ravel()
        yo8 = onnx8.run(None, feed)[0].ravel()
        n_ref = R.durations(mid["dur"], speed).numpy()
        same_len = len(y) == len(ref)
        snr = 10 * np.log10((ref**2).sum() / ((ref - y) ** 2).sum()) if same_len else float("nan")
        row = dict(i=i, speed=speed, T=ids.shape[1], secs=len(y) / R.SR, dur_mismatch=int((n != n_ref).sum()),
                   same_len=same_len, snr_vs_torch=round(float(snr), 1),
                   mel_l1_cm_vs_onnx=round(logmel_l1(y, yo), 4), mel_l1_onnx_vs_onnx=round(logmel_l1(yo, yo2), 4),
                   mel_l1_onnx8_vs_onnx=round(logmel_l1(yo8, yo), 4),
                   rtfx_cm=round(len(y) / R.SR / t_cm, 1), rtfx_onnx_1t=round(len(yo) / R.SR / t_onnx, 1))
        rows.append(row)
        tot_audio += len(y) / R.SR
        tot_cm += t_cm
        tot_onnx += t_onnx
        print(row, flush=True)
        tag = f"{i:02d}" + ("" if speed == 1.0 else f"_s{speed}")
        sf.write(A.wav_dir / f"{tag}_coreml.wav", y, R.SR)
        sf.write(A.wav_dir / f"{tag}_onnx.wav", yo, R.SR)
        sf.write(A.wav_dir / f"{tag}_onnx8.wav", yo8, R.SR)
        with open(A.wav_dir / f"{tag}.txt", "w") as f:
            f.write(text + "\n")
    print(f"TOTAL {tot_audio:.1f}s audio: Core ML ({A.cu}) RTFx {tot_audio / tot_cm:.1f}, ONNX fp32 1-thread RTFx {tot_audio / tot_onnx:.1f}")
    json.dump(rows, open(A.wav_dir / f"results_{A.cu}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
