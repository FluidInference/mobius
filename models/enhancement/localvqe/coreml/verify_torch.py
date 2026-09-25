"""PyTorch-only checks: OLA scale, streaming == whole-clip, and parity with the
upstream GGML regression fixture (localvqe-v1.3-4.8M-f32.out.f32)."""
import argparse, sys
import numpy as np, torch
from localvqe_coreml.common import load_model
from localvqe_coreml.streaming import StreamingLocalVQE, run_streaming, HOP

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--upstream", required=True, help="path to LocalVQE git checkout (fixtures)")
p.add_argument("--frames", type=int, default=4)
a = p.parse_args()

m = load_model(a.ckpt)
print("params", sum(v.numel() for v in m.parameters()))

# 1. analysis->synthesis identity with the trained basis: which OLA scale reconstructs?
torch.manual_seed(0)
x = torch.randn(1, 16000)
with torch.no_grad():
    y = m.decoder(m.encoder(x), length=16000)   # upstream: divides by overlap count (=2)
for s in (1.0, 2.0):
    err = ((y * s - x)[:, 512:-512]).abs().max().item()
    print(f"identity: upstream_output*{s}: max|err|={err:.3e}")

# 2. GGML regression fixture parity (input: 16000 mic then 16000 ref, output 16000)
fx = np.fromfile(f"{a.upstream}/ggml/tests/fixtures/regression_input.f32", dtype="<f4")
mic, ref = torch.from_numpy(fx[:16000])[None], torch.from_numpy(fx[16000:])[None]
import os; name = os.path.basename(a.ckpt).replace(".pt", "-f32")
ggml = np.fromfile(f"{a.upstream}/ggml/tests/fixtures/{name}.out.f32", dtype="<f4")
with torch.no_grad():
    full = m.decoder(m(mic, ref), length=16000)[0].numpy()
for s in (1.0, 2.0):
    d = np.abs(full * s - ggml); rel = d.max() / np.abs(ggml).max()
    print(f"ggml fixture vs upstream_torch*{s}: max|diff|={d.max():.3e} rel={rel:.3e}  (ggml peak {np.abs(ggml).max():.3f})")

# 3. streaming wrapper == whole-clip (fp32), for frames=1 and frames=a.frames
for fr in (1, a.frames):
    sm = StreamingLocalVQE(m, frames=fr, ola_scale=1.0).eval()
    ys = run_streaming(sm, mic, ref)[0].numpy()
    d = np.abs(ys - ggml)
    print(f"streaming(frames={fr}, ola_scale=1.0) vs ggml fixture: max|diff|={d.max():.3e} rel={d.max()/np.abs(ggml).max():.3e}")
    d2 = np.abs(ys - full * 2)
    print(f"streaming(frames={fr}) vs upstream_torch*2: max|diff|={d2.max():.3e}")
