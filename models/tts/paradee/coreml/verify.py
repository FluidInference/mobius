"""Core ML vs PyTorch parity + latency on build/ref.npz. usage: verify.py MODEL(text|acoustic) CU [--dir build]"""
import sys, time, pathlib, numpy as np, coremltools as ct
model, cu = sys.argv[1], sys.argv[2]
d = pathlib.Path(sys.argv[4] if len(sys.argv) > 4 else "build")
r = np.load(d / "ref.npz")
if model == "text":
    m = ct.models.MLModel(str(d / "ParadeeText.mlpackage"), compute_units=getattr(ct.ComputeUnit, cu))
    feed = {"input_ids": r["input_ids"]}
else:
    m = ct.models.MLModel(str(d / "ParadeeAcoustic.mlpackage"), compute_units=getattr(ct.ComputeUnit, cu))
    feed = {"en": r["en"], "asr": r["asr"], "noise": r["noise"]}
o = m.predict(feed)
for _ in range(2): m.predict(feed)
t0 = time.perf_counter()
for _ in range(10): m.predict(feed)
ms = (time.perf_counter() - t0) / 10 * 1000
if model == "text":
    msg = " ".join(f"{k} max|diff| {np.abs(o[k] - r[k]).max():.2e}" for k in ("duration", "d", "asr_tok"))
    msg += f"  rounded-dur mismatches {int((np.round(o['duration']) != np.round(r['duration'])).sum())}"
else:
    y, ref = o["audio"].ravel(), r["audio"].ravel()
    msg = f"audio corr {np.corrcoef(ref, y)[0, 1]:.6f} max|diff| {np.abs(ref - y).max():.2e} snr {10*np.log10((ref**2).sum()/((ref-y)**2).sum()):.1f} dB"
print(f"{model:8s} {cu:12s} {ms:7.2f} ms  {msg}")
