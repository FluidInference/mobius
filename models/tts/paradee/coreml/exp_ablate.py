import sys, numpy as np, torch, coremltools as ct, time, torch.nn as nn, torch.nn.functional as F
import convert as C, paradee_ref as R
ts, ds = R.load_halves(); ts.requires_grad_(False); ds.requires_grad_(False)
r = np.load("build/ref.npz"); en, asr, z = (torch.from_numpy(r[k]) for k in ("en", "asr", "noise"))
class Abl(C.AcousticModel):
    def __init__(s, stop): super().__init__(ts, ds); s.stop = stop
    def forward(s, en, asr, z):
        F0, N = s.pred.F0Ntrain(en, s.s_text)
        if s.stop == "pros": return F0 * 1 + N.sum() * 0 + z.sum() * 0 + asr.sum() * 0
        d, st = s.ds, s.s_dec
        f0c = d.F0_conv(F0.unsqueeze(1)); nc = d.N_conv(N.unsqueeze(1))
        x = d.encode(torch.cat([asr, f0c, nc], 1), st); asr_res = d.asr_res(asr)
        for block in d.decode: x = block(torch.cat([x, asr_res, f0c, nc], 1), st)
        g = d.generator; src = C.source(g, F0, z)
        if s.stop == "dec": return x.sum(1) + src.sum() * 0
        if s.stop == "src": return src + x.sum() * 0
        har = torch.cat(g.stft.transform(src), 1)
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, 0.1); xs_ = g.noise_res[i](g.noise_convs[i](har), st); x = g.ups[i](x)
            if i == g.num_upsamples - 1: x = g.reflection_pad(x)
            x = x + xs_; acc = None
            for j in range(g.num_kernels):
                rr = g.resblocks[i * g.num_kernels + j](x, st); acc = rr if acc is None else acc + rr
            x = acc / g.num_kernels
        x = g.conv_post(F.leaky_relu(x)); nb = g.post_n_fft // 2 + 1
        return C.stft_inverse(g.stft, torch.exp(x[:, :nb]), torch.sin(x[:, nb:]))
for stop in sys.argv[1:]:
    m = Abl(stop).eval()
    with torch.no_grad(): tr = torch.jit.trace(m, (en, asr, z), strict=False)
    Fd = ct.RangeDim(1, 4000, default=en.shape[2]); Sd = ct.RangeDim(600, 2400000, default=z.shape[2])
    ml = ct.convert(tr, inputs=[ct.TensorType(name="en", shape=(1, 224, Fd)), ct.TensorType(name="asr", shape=(1, 512, Fd)), ct.TensorType(name="noise", shape=(1, 1, Sd))],
        convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17, compute_precision=ct.precision.FLOAT32)
    p = f"build/exp_{stop}.mlpackage"; ml.save(p)
    feed = {"en": r["en"], "asr": r["asr"], "noise": r["noise"]}
    for cu in ("CPU_ONLY", "CPU_AND_NE"):
        mm = ct.models.MLModel(p, compute_units=getattr(ct.ComputeUnit, cu))
        for _ in range(3): mm.predict(feed)
        t0 = time.perf_counter()
        for _ in range(5): mm.predict(feed)
        print(f"RESULT {stop:5s} {cu:11s} {(time.perf_counter()-t0)/5*1000:8.1f} ms", flush=True)
