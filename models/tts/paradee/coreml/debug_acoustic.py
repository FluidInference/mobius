"""Bisect the acoustic model: convert with intermediate outputs and compare each to torch."""
import sys, numpy as np, torch, coremltools as ct, torch.nn.functional as F
import convert as C, paradee_ref as R
ts, ds = R.load_halves(); am = C.AcousticModel(ts, ds).eval()
r = np.load("build/ref.npz"); en, asr, z = (torch.from_numpy(r[k]) for k in ("en", "asr", "noise"))

class Dbg(torch.nn.Module):
    def __init__(s, am): super().__init__(); s.am = am
    def forward(s, en, asr, z):
        am = s.am; F0, N = am.pred.F0Ntrain(en, am.s_text); g = am.ds.generator
        src = C.source(g, F0, z); mag, ph = g.stft.transform(src)
        rad = F0[:, :, None] * torch.arange(1, 10, dtype=F0.dtype) / 24000; rad = rad - torch.floor(rad)
        phase = torch.cumsum(rad, dim=1) * (2 * 3.141592653589793)
        up = F.interpolate(phase.transpose(1, 2) * 300, scale_factor=300.0, mode="linear")
        return F0, N, phase, up, src, mag, ph, am(en, asr, z)
names = ["F0", "N", "phase", "up", "src", "mag", "ph", "audio"]
m = Dbg(am).eval()
with torch.no_grad(): ref = m(en, asr, z); tr = torch.jit.trace(m, (en, asr, z), strict=False)
Fd = ct.RangeDim(1, 4000, default=en.shape[2]); Sd = ct.RangeDim(600, 2400000, default=z.shape[2])
static = "--static" in sys.argv
ml = ct.convert(tr, inputs=[ct.TensorType(name="en", shape=(1, 224, en.shape[2] if static else Fd)),
                            ct.TensorType(name="asr", shape=(1, 512, en.shape[2] if static else Fd)),
                            ct.TensorType(name="noise", shape=(1, 1, z.shape[2] if static else Sd))],
                outputs=[ct.TensorType(name=n) for n in names], convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT32,
                compute_units=ct.ComputeUnit.CPU_ONLY)
o = ml.predict({"en": r["en"], "asr": r["asr"], "noise": r["noise"]})
for n, t in zip(names, ref):
    a, b = t.numpy(), o[n]
    print(f"{n:6s} shape torch {a.shape} cm {b.shape}", end=" ")
    if a.shape == b.shape: print(f"max|diff| {np.abs(a-b).max():.3e} (max|ref| {np.abs(a).max():.3e})")
    else: print()
print("cm phase[:5,0]", o["phase"][0,:5,0], o["phase"][0,-1,:3]); print("torch", ref[2][0,:5,0].numpy(), ref[2][0,-1,:3].numpy())
