"""Per-stage divergence: walk the full CoreML chain, swap one stage at a time
with its PyTorch counterpart, and measure the audible-band drift introduced.

iSTFT/Tail and Vocoder are already cleared by tail_compare.py and
vocoder_xpre_diff.py (both bit-equivalent to PyTorch given the same inputs).
This script localises the remaining noise contribution stage-by-stage:

    A. Reference: full PyTorch (gen.stft.inverse / torch.istft).
    B. PyTorch through-Vocoder, CoreML Tail.
    C. PyTorch noise_sources, CoreML Vocoder + Tail.
    D. CoreML Noise + CoreML Vocoder + CoreML Tail (pt asr/F0/N).
    E. Full CoreML chain (Albert→PostAlbert→Alignment→Prosody→Noise→Vocoder→Tail).

Diff each pair (A→B, B→C, C→D, D→E) to see which stage introduces the most
HF-band power. The biggest jump wins.

Usage:
    uv run python per_stage_diff.py \
        --models-dir build/ANE-zh \
        --voice zm_009 \
        --out-dir build/per_stage_diff_zm009
"""
import argparse
import math
import pathlib
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

from kokoro import KModel
from kokoro.pipeline import KPipeline


SR = 24000
DEFAULT_REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
DEFAULT_TEXT = "你好世界，今天天气很好。"


def load_voice_pack(models_dir, voice, pipe):
    bin_path = models_dir / "voices" / f"{voice}.bin"
    if not bin_path.exists():
        bin_path = models_dir / f"{voice}.bin"
    if bin_path.exists():
        flat = np.frombuffer(bin_path.read_bytes(), dtype=np.float32).reshape(510, 1, 256)
        return torch.from_numpy(flat.copy())
    return pipe.load_voice(voice)


def hf_db(audio, sr=SR, n_fft=1024, hop=256, hf_cutoff_hz=10000.0):
    from scipy.signal import stft
    f, _, Z = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    S = np.abs(Z) ** 2
    return 10 * math.log10(max(float(S[f >= hf_cutoff_hz].mean()), 1e-30))


def diff_audio(a, b, label):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))
    rms_d = float(np.sqrt(np.mean(diff ** 2)))
    corr = float(np.corrcoef(a, b)[0, 1]) if rms_a and rms_b else 0.0
    diff_hf = hf_db(diff)
    print(f"  [{label}]")
    print(f"    rms_a={rms_a:.4e}  rms_b={rms_b:.4e}  ratio={rms_b/max(rms_a,1e-12):.4f}")
    print(f"    rms_diff={rms_d:.4e}  rel={rms_d/max(rms_a,1e-12):.4e}  corr={corr:.9f}")
    print(f"    HF(diff) = {diff_hf:.2f} dB")
    return {"rms_d": rms_d, "rel": rms_d/max(rms_a,1e-12), "corr": corr,
            "diff_hf_db": diff_hf}


def precompute_noise_sources_pt(generator, F0_curve, style_timbre):
    with torch.no_grad():
        f0_up = generator.f0_upsamp(F0_curve[:, None]).transpose(1, 2)
        har_source, _, _ = generator.m_source(f0_up)
        har_source_flat = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = generator.stft.transform(har_source_flat)
        har = torch.cat([har_spec, har_phase], dim=1)
        sources = []
        for i in range(generator.num_upsamples):
            x_source = generator.noise_convs[i](har)
            x_source = generator.noise_res[i](x_source, style_timbre)
            sources.append(x_source)
        return sources


def pytorch_x_pre(model, asr, F0_pred, N_pred, style_timbre, noise_sources):
    """Run PyTorch decoder.encode/decode + generator → x_pre using the supplied
    noise_sources (so we can swap PyTorch vs CoreML noise side-by-side)."""
    decoder = model.decoder
    gen = decoder.generator
    with torch.no_grad():
        F0 = decoder.F0_conv(F0_pred.unsqueeze(1))
        N_feat = decoder.N_conv(N_pred.unsqueeze(1))
        x = torch.cat([asr, F0, N_feat], dim=1)
        x = decoder.encode(x, style_timbre)
        asr_res = decoder.asr_res(asr)
        res = True
        for block in decoder.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N_feat], dim=1)
            x = block(x, style_timbre)
            if block.upsample_type != "none":
                res = False
        for i in range(gen.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x = gen.ups[i](x)
            if i == gen.num_upsamples - 1:
                x = gen.reflection_pad(x)
            x = x + noise_sources[i]
            xs = None
            for j in range(gen.num_kernels):
                if xs is None:
                    xs = gen.resblocks[i * gen.num_kernels + j](x, style_timbre)
                else:
                    xs = xs + gen.resblocks[i * gen.num_kernels + j](x, style_timbre)
            x = xs / gen.num_kernels
        x_pre = F.leaky_relu(x)
    return x_pre


def pytorch_audio_from_xpre(model, x_pre):
    """Apply generator.conv_post + iSTFT (torch.istft via gen.stft.inverse)."""
    gen = model.decoder.generator
    with torch.no_grad():
        x_post = gen.conv_post(x_pre)
        spec = torch.exp(x_post[:, :gen.post_n_fft // 2 + 1, :])
        phase = torch.sin(x_post[:, gen.post_n_fft // 2 + 1:, :])
        return gen.stft.inverse(spec, phase).squeeze().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("build/per_stage_diff"))
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--phonemes", default=None)
    p.add_argument("--voice", default="zm_009")
    p.add_argument("--lang", default="z")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.phonemes:
        phonemes = args.phonemes
    else:
        print(f"[g2p] phonemizing {args.text!r}", file=sys.stderr)
        m_g = KModel(repo_id=args.repo_id); m_g.eval()
        pipe_g = KPipeline(lang_code=args.lang, repo_id=args.repo_id, model=m_g)
        phonemes = ""
        for _gs, ps, _tks in pipe_g(args.text, voice=args.voice):
            phonemes = ps; break
        if not phonemes:
            p.error(f"empty phonemes for {args.text!r}")
        del m_g, pipe_g

    print(f"[setup] PyTorch reference KModel ({args.repo_id})...", file=sys.stderr)
    model = KModel(repo_id=args.repo_id); model.eval()
    pipe = KPipeline(lang_code=args.lang, repo_id=args.repo_id, model=model)
    voice_pack = load_voice_pack(args.models_dir, args.voice, pipe)

    ids = list(filter(lambda i: i is not None,
                      map(lambda p: model.vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *ids, 0]])
    ref_s = voice_pack[max(min(len(phonemes) - 1, voice_pack.shape[0] - 1), 0)]
    s = ref_s[:, 128:]
    style_timbre = ref_s[:, :128]
    with torch.no_grad():
        input_lengths = torch.LongTensor([input_ids.shape[1]])
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(1, -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        T_a = pred_dur.sum().item()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln = torch.zeros(input_ids.shape[1], T_a)
        pred_aln[indices, torch.arange(T_a)] = 1.0
        pred_aln = pred_aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln
        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln

    print(f"      T_a={T_a}, len(phonemes)={len(phonemes)}", file=sys.stderr)

    voc_path = args.models_dir / "KokoroVocoder.mlpackage"
    tail_path = args.models_dir / "KokoroTail.mlpackage"
    noise_path = args.models_dir / "KokoroNoise.mlpackage"
    for q in [voc_path, tail_path, noise_path]:
        if not q.exists():
            q2 = q.with_suffix(".mlmodelc")
            if q2.exists():
                if q is voc_path: voc_path = q2
                elif q is tail_path: tail_path = q2
                elif q is noise_path: noise_path = q2
            else:
                p.error(f"missing {q}")

    print("[load] CoreML stages...", file=sys.stderr)
    m_voc   = ct.models.MLModel(str(voc_path),   compute_units=ct.ComputeUnit.ALL)
    m_tail  = ct.models.MLModel(str(tail_path),  compute_units=ct.ComputeUnit.ALL)
    m_noise = ct.models.MLModel(str(noise_path), compute_units=ct.ComputeUnit.ALL)

    pt_noise = precompute_noise_sources_pt(model.decoder.generator, F0_pred, style_timbre)

    # ---------- A: full PyTorch ----------
    print("\n[A] full PyTorch reference...", file=sys.stderr)
    pt_x_pre = pytorch_x_pre(model, asr, F0_pred, N_pred, style_timbre, pt_noise)
    audio_A = pytorch_audio_from_xpre(model, pt_x_pre)

    # ---------- B: PyTorch x_pre, CoreML Tail ----------
    print("[B] PT x_pre → CoreML Tail...", file=sys.stderr)
    audio_B = np.array(m_tail.predict(
        {"x_pre": pt_x_pre.numpy().astype(np.float32)})["audio"]).flatten()

    # ---------- C: PT inputs, CoreML Vocoder + Tail ----------
    print("[C] PT inputs → CoreML Vocoder → CoreML Tail...", file=sys.stderr)
    voc_C = m_voc.predict({
        "asr": asr.numpy().astype(np.float32),
        "F0_curve": F0_pred.numpy().astype(np.float32),
        "N_pred": N_pred.numpy().astype(np.float32),
        "x_source_0": pt_noise[0].numpy().astype(np.float32),
        "x_source_1": pt_noise[1].numpy().astype(np.float32),
        "style_timbre": style_timbre.numpy().astype(np.float32),
    })
    audio_C = np.array(m_tail.predict(
        {"x_pre": np.array(voc_C["x_pre"]).astype(np.float32)})["audio"]).flatten()

    # ---------- D: CoreML Noise + CoreML Vocoder + CoreML Tail ----------
    print("[D] CoreML Noise → CoreML Vocoder → CoreML Tail...", file=sys.stderr)
    noise_D = m_noise.predict({
        "F0_curve": F0_pred.numpy().astype(np.float32),
        "style_timbre": style_timbre.numpy().astype(np.float32),
    })
    voc_D = m_voc.predict({
        "asr": asr.numpy().astype(np.float32),
        "F0_curve": F0_pred.numpy().astype(np.float32),
        "N_pred": N_pred.numpy().astype(np.float32),
        "x_source_0": np.array(noise_D["x_source_0"]).astype(np.float32),
        "x_source_1": np.array(noise_D["x_source_1"]).astype(np.float32),
        "style_timbre": style_timbre.numpy().astype(np.float32),
    })
    audio_D = np.array(m_tail.predict(
        {"x_pre": np.array(voc_D["x_pre"]).astype(np.float32)})["audio"]).flatten()

    # Side: noise_sources comparison
    print("\n--- noise sources (PT vs CoreML) ---")
    for i in range(2):
        a = pt_noise[i].numpy().flatten()
        b = np.array(noise_D[f"x_source_{i}"]).flatten()
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        rms_a = float(np.sqrt(np.mean(a ** 2)))
        rms_b = float(np.sqrt(np.mean(b ** 2)))
        rms_d = float(np.sqrt(np.mean((a - b) ** 2)))
        corr = float(np.corrcoef(a, b)[0, 1])
        print(f"  x_source_{i}:  rms_pt={rms_a:.4e}  rms_cm={rms_b:.4e}  "
              f"rms_d={rms_d:.4e}  rel={rms_d/max(rms_a,1e-12):.4e}  corr={corr:.7f}")

    # 1.5x scale-correct everything against PyTorch reference (so all "audio"
    # outputs land on the same magnitude scale as the user-perceived output).
    audio_A_scaled = 1.5 * audio_A

    # ---------- pairwise diffs ----------
    print("\n=== Pairwise audio diffs (descales A by ×1.5) ===")
    metrics_AB = diff_audio(audio_A_scaled, audio_B, "A vs B  (PT-tail vs CoreML-tail, same x_pre)")
    metrics_BC = diff_audio(audio_B, audio_C, "B vs C  (Tail-only vs Voc+Tail, same upstream)")
    metrics_CD = diff_audio(audio_C, audio_D, "C vs D  (PT noise vs CoreML noise upstream)")

    print("\n=== Vs full PyTorch reference ===")
    diff_audio(audio_A_scaled, audio_C, "A vs C  (full PT vs CoreML Voc+Tail w/ PT noise)")
    diff_audio(audio_A_scaled, audio_D, "A vs D  (full PT vs CoreML Noise+Voc+Tail)")

    print("\n=== HF (≥10 kHz) band power ===")
    print(f"  A audio_A      : {hf_db(audio_A_scaled):.2f} dB")
    print(f"  B audio_B      : {hf_db(audio_B):.2f} dB")
    print(f"  C audio_C      : {hf_db(audio_C):.2f} dB")
    print(f"  D audio_D      : {hf_db(audio_D):.2f} dB")

    try:
        import soundfile as sf
        for name, aud in [("A_pt_full", audio_A_scaled), ("B_pt_xpre_cm_tail", audio_B),
                          ("C_pt_noise_cm_voc_tail", audio_C), ("D_full_cm_voc_tail_with_cm_noise", audio_D)]:
            sf.write(str(args.out_dir / f"audio_{name}.wav"), aud, SR, subtype="FLOAT")
        sf.write(str(args.out_dir / "diff_C_minus_D.wav"),
                 audio_C[:len(audio_D)] - audio_D[:len(audio_C)], SR, subtype="FLOAT")
        print(f"\n[wrote] WAVs → {args.out_dir}/", file=sys.stderr)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
