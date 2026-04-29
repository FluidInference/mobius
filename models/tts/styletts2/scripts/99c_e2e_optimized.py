"""End-to-end with int8 text_predictor + diffusion fixed at bucket=512."""
from __future__ import annotations
import argparse
import sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _styletts2_lib import (  # noqa: E402
    DEFAULT_CHECKPOINT, COREML_DIR, load_inference_modules, register_coreml_op_shims,
)
register_coreml_op_shims()
import coremltools as ct  # noqa: E402

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."
ALPHA, BETA, SEED = 0.3, 0.7, 0
TOK_BUCKETS = (32, 64, 128, 256, 512)
MEL_BUCKETS = (256, 512, 1024, 2048, 4096)
DIFF_BUCKET = 512  # only bucket retained

def pick(n, b):
    for x in b:
        if n <= x: return x
    raise ValueError

def phonemize(text):
    import phonemizer
    from nltk.tokenize import word_tokenize
    from text_utils import TextCleaner
    backend = phonemizer.backend.EspeakBackend(language="en-us", preserve_punctuation=True, with_stress=True)
    ps = backend.phonemize([text.strip()])
    ps = " ".join(word_tokenize(ps[0]))
    cleaner = TextCleaner(); tokens = cleaner(ps); tokens.insert(0, 0)
    return torch.LongTensor(tokens).unsqueeze(0)

def compute_ref_s(modules, wav_path):
    import librosa, torchaudio
    to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    wave, _ = librosa.load(str(wav_path), sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    mel = to_mel(torch.from_numpy(audio).float())
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) + 4.0) / 4.0
    with torch.no_grad():
        ref_s = modules["style_encoder"](mel.unsqueeze(1))
        ref_p = modules["predictor_encoder"](mel.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)

def karras(n, smin=0.0001, smax=3.0, rho=9.0):
    ri = 1.0/rho; r = np.linspace(0,1,n).astype(np.float64)
    s = (smax**ri + r*(smin**ri - smax**ri))**rho
    return np.concatenate([s, [0.0]]).astype(np.float32)

def adpm2(pf, noise, sigmas, emb, feat):
    x = noise.astype(np.float32)*float(sigmas[0])
    for i in range(len(sigmas)-1):
        s, sn = float(sigmas[i]), float(sigmas[i+1])
        if sn == 0.0:
            d = (x - pf(x, s, emb, feat))/s; x = x + d*(sn-s); continue
        sm = float(np.exp((np.log(s)+np.log(sn))/2))
        d = (x - pf(x, s, emb, feat))/s; xm = x + d*(sm-s)
        dm = (xm - pf(xm, sm, emb, feat))/sm; x = x + dm*(sn-s)
    return x

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--reference-wav", type=Path, required=True,
                    help="Reference WAV for the speaker style encoder.")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--coreml-dir", type=Path, default=COREML_DIR)
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/styletts2-e2e/coreml_int8_diff512.wav"))
    ap.add_argument("--baseline-wav", type=Path, default=None,
                    help="Optional fp16 e2e baseline WAV for spectral cosine "
                    "comparison. Typically the output of 99b_e2e_coreml.py.")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    timings = {}; t_total = time.time()

    t0 = time.time(); modules, cfg = load_inference_modules(args.checkpoint)
    timings["load_pytorch"] = time.time()-t0

    t0 = time.time()
    tokens = phonemize(args.text); T = tokens.shape[-1]; tb = pick(T, TOK_BUCKETS)
    ref_s = compute_ref_s(modules, args.reference_wav)
    timings["frontend"] = time.time()-t0
    print(f"T_tok={T}  tp_bucket={tb}  diff_bucket={DIFF_BUCKET}")

    t0 = time.time()
    tp = ct.models.MLModel(str(args.coreml_dir / f"styletts2_text_predictor_{tb}_int8.mlpackage"), compute_units=ct.ComputeUnit.ALL)
    ds = ct.models.MLModel(str(args.coreml_dir / f"styletts2_diffusion_step_{DIFF_BUCKET}.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    fn = ct.models.MLModel(str(args.coreml_dir / "styletts2_f0n_energy.mlpackage"), compute_units=ct.ComputeUnit.ALL)
    timings["load_models"] = time.time()-t0

    # warmup
    t0 = time.time()
    _wt = np.zeros((1, tb), dtype=np.int32)
    _ws = np.zeros((1, 128), dtype=np.float32)
    tp.predict({"tokens": _wt, "style": _ws})
    ds.predict({
        "x_noisy": np.zeros((1,1,256), dtype=np.float32),
        "sigma": np.array([1.0], dtype=np.float32),
        "embedding": np.zeros((1, DIFF_BUCKET, 768), dtype=np.float32),
        "features": np.zeros((1, 256), dtype=np.float32),
    })
    fn.predict({"en": np.zeros((1, 640, 256), dtype=np.float32), "s": np.zeros((1, 128), dtype=np.float32)})
    timings["warmup_models"] = time.time()-t0

    t0 = time.time()
    tok_pad = np.zeros((1, tb), dtype=np.int32); tok_pad[0, :T] = tokens.numpy().astype(np.int32)[0]
    A = tp.predict({"tokens": tok_pad, "style": ref_s[:,:128].numpy().astype(np.float32)})
    timings["A"] = time.time()-t0

    t_en = A["t_en"][..., :T]; d_full = A["d"][:, :T, :]
    pred_log = A["pred_dur_log"][:, :T, :]; bert_dur = A["bert_dur"][:, :T, :]
    pred_dur = (torch.round(torch.sigmoid(torch.from_numpy(pred_log)).sum(dim=-1).squeeze()).clamp(min=1).long().numpy())
    Tm = int(pred_dur.sum()); mb = pick(Tm, MEL_BUCKETS)
    print(f"T_mel={Tm}  bucket={mb}")

    t0 = time.time()
    np.random.seed(SEED)
    noise = np.random.randn(1,1,256).astype(np.float32)
    # pad bert_dur to DIFF_BUCKET (=512), not to tp bucket
    bp = np.zeros((1, DIFF_BUCKET, bert_dur.shape[-1]), dtype=np.float32); bp[:,:T,:] = bert_dur
    sigmas = karras(5); feats = ref_s.numpy().astype(np.float32)
    def pf(x,s,e,f):
        return ds.predict({"x_noisy":x.astype(np.float32),"sigma":np.array([s],dtype=np.float32),"embedding":e.astype(np.float32),"features":f.astype(np.float32)})["denoised"]
    s_pred = adpm2(pf, noise, sigmas, bp, feats).squeeze(1)
    timings["B"] = time.time()-t0

    s = s_pred[:,128:]; ref = s_pred[:,:128]
    ref = ALPHA*ref + (1-ALPHA)*ref_s[:,:128].numpy()
    s = BETA*s + (1-BETA)*ref_s[:,128:].numpy()

    t0 = time.time()
    aln = np.zeros((T, Tm), dtype=np.float32); c = 0
    for i in range(T):
        n = int(pred_dur[i]); aln[i, c:c+n] = 1.0; c += n
    aln = aln[None]
    en = np.matmul(d_full.transpose(0,2,1), aln); asr = np.matmul(t_en, aln)
    en_s = np.zeros_like(en); en_s[:,:,0]=en[:,:,0]; en_s[:,:,1:]=en[:,:,:-1]
    asr_s = np.zeros_like(asr); asr_s[:,:,0]=asr[:,:,0]; asr_s[:,:,1:]=asr[:,:,:-1]
    en, asr = en_s, asr_s
    timings["align"] = time.time()-t0

    t0 = time.time()
    en_pad = np.zeros((1,640,mb),dtype=np.float32); en_pad[:,:,:Tm] = en
    C = fn.predict({"en":en_pad, "s":s.astype(np.float32)})
    F0 = C["F0"][:,:2*Tm]; N = C["N"][:,:2*Tm]
    timings["C"] = time.time()-t0

    t0 = time.time()
    decoder = ct.models.MLModel(str(args.coreml_dir / f"styletts2_decoder_{mb}.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    timings["D_load"] = time.time()-t0

    # warmup decoder
    t0 = time.time()
    decoder.predict({
        "asr": np.zeros((1, 512, mb), dtype=np.float32),
        "F0_curve": np.zeros((1, 2*mb), dtype=np.float32),
        "N": np.zeros((1, 2*mb), dtype=np.float32),
        "s": np.zeros((1, 128), dtype=np.float32),
    })
    timings["D_warmup"] = time.time()-t0

    t0 = time.time()
    asr_p = np.zeros((1,512,mb),dtype=np.float32); asr_p[:,:,:Tm] = asr
    F0p = np.zeros((1,2*mb),dtype=np.float32); F0p[:,:F0.shape[1]] = F0
    Np = np.zeros((1,2*mb),dtype=np.float32); Np[:,:N.shape[1]] = N
    D = decoder.predict({"asr":asr_p,"F0_curve":F0p,"N":Np,"s":ref.astype(np.float32)})
    timings["D"] = time.time()-t0

    wav = D["waveform"].squeeze()[:Tm*600]
    if wav.shape[-1] > 50: wav = wav[...,:-50]
    sf.write(str(args.out), wav, 24000)

    timings["total"] = time.time()-t_total
    audio = len(wav)/24000.0
    inf = timings["A"]+timings["B"]+timings["align"]+timings["C"]+timings["D"]
    print(f"\n--- Timings ---")
    for k,v in timings.items(): print(f"  {k:14s}  {v*1000:8.1f} ms")
    print(f"  inference        {inf*1000:8.1f} ms")
    print(f"  audio            {audio*1000:8.1f} ms")
    print(f"  RTFx             {audio/inf:8.2f}×")

    # Optional spectral comparison vs fp16 e2e baseline (e.g. output of
    # 99b_e2e_coreml.py). Skipped if --baseline-wav is not provided or the
    # file is missing.
    if args.baseline_wav is not None and args.baseline_wav.exists():
        import torchaudio as ta
        fp16_wav, _ = sf.read(str(args.baseline_wav))
        n = min(len(wav), len(fp16_wav))
        a = wav[:n].astype(np.float64); b = fp16_wav[:n].astype(np.float64)
        to_mel = ta.transforms.MelSpectrogram(sample_rate=24000, n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        ma = to_mel(torch.from_numpy(a).float()).numpy()
        mb_ = to_mel(torch.from_numpy(b).float()).numpy()
        la = np.log(ma+1e-5).flatten(); lb = np.log(mb_+1e-5).flatten()
        mc = np.dot(la,lb)/(np.linalg.norm(la)*np.linalg.norm(lb)+1e-9)
        print(f"\nlog-mel cos(int8+diff512 vs fp16 e2e): {mc:.4f}")
    elif args.baseline_wav is not None:
        print(f"\n[skip] --baseline-wav not found: {args.baseline_wav}")

if __name__ == "__main__":
    main()
