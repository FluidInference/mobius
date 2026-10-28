"""WER of Core ML vs ONNX WAVs with whisper-large-v3-turbo (mlx). usage: asr_check.py out/wav"""
import sys, glob, os, re, jiwer, mlx_whisper, soundfile as sf, numpy as np
from scipy.signal import resample_poly
load = lambda f: resample_poly(sf.read(f)[0], 2, 3).astype(np.float32)
d = sys.argv[1]; norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", s.lower())).strip()
refs, hyp = [], {k: [] for k in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["coreml", "onnx"])}
for txt in sorted(glob.glob(f"{d}/*.txt")):
    tag = txt[:-4]; refs.append(norm(open(txt).read()))
    for k in hyp:
        hyp[k].append(norm(mlx_whisper.transcribe(load(f"{tag}_{k}.wav"), path_or_hf_repo="mlx-community/whisper-large-v3-turbo")["text"]))
for k, h in hyp.items():
    print(f"{k:7s} WER {jiwer.wer(refs, h)*100:.2f}%")
    for r, x in zip(refs, h):
        if r != x: print("   ref:", r, "\n   hyp:", x)
