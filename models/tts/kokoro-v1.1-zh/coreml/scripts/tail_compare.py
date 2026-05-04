"""iSTFT/Tail comparison: PyTorch reference vs CoreML KokoroTail.mlmodelc.

Captures the same x_pre tensor for both pipelines and compares per-sample
RMS / max-abs and high-frequency spectrogram band, to determine whether
the residual background noise in the CoreML output originates in the
conv_post + iSTFT tail stage or upstream in the Vocoder graph.

Three reconstructions run on the SAME x_pre:
    1. PyTorch reference iSTFT  : generator.conv_post → exp/sin →
                                  generator.stft.inverse (torch.istft, the
                                  actual training-time formulation).
    2. PyTorch CustomSTFT iSTFT : generator.conv_post → exp/sin →
                                  CustomSTFT.inverse (the conv_transpose1d
                                  rewrite that was traced for CoreML).
    3. CoreML tail              : KokoroTail.mlpackage / .mlmodelc.

Diffs:
    1 vs 2 → does CustomSTFT itself drift from torch.istft in fp32?
    2 vs 3 → does the CoreML conversion of CustomSTFT drift from the trace?
    1 vs 3 → end-to-end tail drift (the user-perceived noise contribution).

Usage:
    uv run python tail_compare.py \
        --models-dir build/ANE-zh \
        --voice zm_009 \
        --text "你好世界，今天天气很好。" \
        --out-dir build/tail_compare_zm009
"""
import argparse
import math
import pathlib
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kokoro import KModel
from kokoro.custom_stft import CustomSTFT
from kokoro.pipeline import KPipeline


SR = 24000
DEFAULT_REPO_ID = "hexgrad/Kokoro-82M-v1.1-zh"
DEFAULT_TEXT = "你好世界，今天天气很好。"


def load_voice_pack(models_dir, voice, pipe):
    """Match inference.py: prefer flat .bin, fall back to pipe.load_voice."""
    bin_path = models_dir / "voices" / f"{voice}.bin"
    if not bin_path.exists():
        bin_path = models_dir / f"{voice}.bin"
    if bin_path.exists():
        flat = np.frombuffer(bin_path.read_bytes(), dtype=np.float32).reshape(510, 1, 256)
        return torch.from_numpy(flat.copy())
    return pipe.load_voice(voice)


def run_pytorch_to_x_pre(model, phonemes, voice_pack):
    """Run PyTorch up to and including decoder.generator MINUS the final
    conv_post + iSTFT. Returns:
        x_pre               : Tensor [1, 128, T_pre] (same as CoreMLVocoder.x_pre)
        intermediate kwargs : dict for downstream tail reconstructions
    """
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

        # Decoder up to and including generator(...) minus conv_post / iSTFT.
        decoder = model.decoder
        F0 = decoder.F0_conv(F0_pred.unsqueeze(1))
        N = decoder.N_conv(N_pred.unsqueeze(1))
        x = torch.cat([asr, F0, N], dim=1)
        x = decoder.encode(x, style_timbre)
        asr_res = decoder.asr_res(asr)
        res = True
        for block in decoder.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], dim=1)
            x = block(x, style_timbre)
            if block.upsample_type != "none":
                res = False

        # Inline-execute generator.forward up to (but not including) conv_post.
        gen = decoder.generator
        f0_up = gen.f0_upsamp(F0_pred[:, None]).transpose(1, 2)
        har_source, _, _ = gen.m_source(f0_up)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(gen.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = gen.noise_convs[i](har)
            x_source = gen.noise_res[i](x_source, style_timbre)
            x = gen.ups[i](x)
            if i == gen.num_upsamples - 1:
                x = gen.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(gen.num_kernels):
                if xs is None:
                    xs = gen.resblocks[i * gen.num_kernels + j](x, style_timbre)
                else:
                    xs = xs + gen.resblocks[i * gen.num_kernels + j](x, style_timbre)
            x = xs / gen.num_kernels
        x_pre = F.leaky_relu(x)

        # Reference audio (full PyTorch path: x_pre → conv_post → iSTFT, using
        # the model's stft attribute — i.e. torch.istft).
        x_post = gen.conv_post(x_pre)
        spec_ref = torch.exp(x_post[:, :gen.post_n_fft // 2 + 1, :])
        phase_ref = torch.sin(x_post[:, gen.post_n_fft // 2 + 1:, :])
        ref_audio = gen.stft.inverse(spec_ref, phase_ref).squeeze().numpy()

    return x_pre, {
        "T_a": int(T_a),
        "ref_audio": ref_audio,
        "post_n_fft": int(gen.post_n_fft),
        "hop_length": int(gen.stft.hop_length),
        "spec_ref": spec_ref.numpy(),
        "phase_ref": phase_ref.numpy(),
    }


def pytorch_custom_stft_tail(model, x_pre):
    """Apply conv_post + exp/sin + CustomSTFT.inverse — the formulation that
    was traced into KokoroTail.mlpackage."""
    gen = model.decoder.generator
    custom = CustomSTFT(filter_length=gen.post_n_fft,
                        hop_length=gen.stft.hop_length,
                        win_length=gen.post_n_fft)
    custom.eval()
    with torch.no_grad():
        x_post = gen.conv_post(x_pre)
        spec = torch.exp(x_post[:, :gen.post_n_fft // 2 + 1, :])
        phase = torch.sin(x_post[:, gen.post_n_fft // 2 + 1:, :])
        audio = custom.inverse(spec, phase)
    return audio.squeeze().numpy()


def coreml_tail(tail_path, x_pre):
    """Run KokoroTail.mlpackage on the same x_pre tensor."""
    ml = ct.models.MLModel(str(tail_path), compute_units=ct.ComputeUnit.ALL)
    out = ml.predict({"x_pre": x_pre.numpy().astype(np.float32)})
    return np.array(out["audio"]).flatten().astype(np.float32)


def diff_metrics(a, b, label, sr=SR):
    """Per-sample RMS / max-abs / corr metrics."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    diff = a - b
    rms_err = float(np.sqrt(np.mean(diff ** 2)))
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    rel_rms = rms_err / max(rms_a, 1e-12)
    corr = float(np.corrcoef(a, b)[0, 1]) if rms_a > 0 and rms_b > 0 else 0.0
    print(f"  [{label}]")
    print(f"    n={n}  rms_a={rms_a:.6e}  rms_b={rms_b:.6e}")
    print(f"    rms_err={rms_err:.6e}  rel_rms={rel_rms:.6e}  max|diff|={max_abs:.6e}")
    print(f"    corr={corr:.9f}")
    return {"rms_err": rms_err, "rel_rms": rel_rms, "max_abs": max_abs, "corr": corr,
            "n": n, "rms_a": rms_a, "rms_b": rms_b}


def hf_band_diff(a, b, label, sr=SR, n_fft=1024, hop=256, hf_cutoff_hz=10000.0):
    """Compare power above hf_cutoff_hz between a and b (the audible-noise band)."""
    from scipy.signal import stft as scipy_stft
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    fa, _, Za = scipy_stft(a, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    fb, _, Zb = scipy_stft(b, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    Sa = np.abs(Za) ** 2
    Sb = np.abs(Zb) ** 2
    hf_mask = fa >= hf_cutoff_hz
    pwr_a = float(Sa[hf_mask].mean())
    pwr_b = float(Sb[hf_mask].mean())
    diff_pwr = float((Sa[hf_mask] - Sb[hf_mask]).mean())
    pwr_a_db = 10 * math.log10(max(pwr_a, 1e-30))
    pwr_b_db = 10 * math.log10(max(pwr_b, 1e-30))
    delta_db = pwr_a_db - pwr_b_db
    diff = a - b
    fd, _, Zd = scipy_stft(diff, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    Sd = np.abs(Zd) ** 2
    diff_hf_db = 10 * math.log10(max(float(Sd[hf_mask].mean()), 1e-30))
    print(f"  [{label}]   HF (≥{hf_cutoff_hz/1000:.0f} kHz) mean power")
    print(f"    a: {pwr_a_db:7.2f} dB    b: {pwr_b_db:7.2f} dB    Δ(a-b): {delta_db:+.2f} dB")
    print(f"    HF mean power of (a-b): {diff_hf_db:.2f} dB")
    return {"pwr_a_db": pwr_a_db, "pwr_b_db": pwr_b_db, "delta_db": delta_db,
            "diff_hf_db": diff_hf_db}


def main():
    p = argparse.ArgumentParser(description="iSTFT/Tail comparison")
    p.add_argument("--models-dir", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("build/tail_compare"))
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--phonemes", default=None,
                   help="Pre-computed phonemes (skip G2P)")
    p.add_argument("--voice", default="zm_009",
                   help="Voice id (default zm_009 — male voice with most audible noise)")
    p.add_argument("--lang", default="z")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tail_pkg = args.models_dir / "KokoroTail.mlpackage"
    if not tail_pkg.exists():
        # Fall back to .mlmodelc if .mlpackage missing.
        tail_pkg = args.models_dir / "KokoroTail.mlmodelc"
        if not tail_pkg.exists():
            p.error(f"Tail not found in {args.models_dir} (.mlpackage / .mlmodelc)")

    # Phonemize.
    if args.phonemes:
        phonemes = args.phonemes
    else:
        print(f"[g2p] phonemizing {args.text!r}", file=sys.stderr)
        m_g = KModel(repo_id=args.repo_id); m_g.eval()
        pipe_g = KPipeline(lang_code=args.lang, repo_id=args.repo_id, model=m_g)
        phonemes = ""
        for _gs, ps, _tks in pipe_g(args.text, voice=args.voice):
            phonemes = ps
            break
        if not phonemes:
            p.error(f"empty phonemes for {args.text!r}")
        print(f"      phonemes ({len(phonemes)}): {phonemes!r}", file=sys.stderr)
        del m_g, pipe_g

    # Load PyTorch model + voice pack.
    print(f"[setup] Loading PyTorch KModel ({args.repo_id})...", file=sys.stderr)
    model = KModel(repo_id=args.repo_id); model.eval()
    pipe = KPipeline(lang_code=args.lang, repo_id=args.repo_id, model=model)
    voice_pack = load_voice_pack(args.models_dir, args.voice, pipe)

    # 1. Capture x_pre from PyTorch.
    print(f"[step 1] Running PyTorch up to x_pre (voice={args.voice})...", file=sys.stderr)
    x_pre, ctx = run_pytorch_to_x_pre(model, phonemes, voice_pack)
    T_pre = x_pre.shape[-1]
    print(f"         x_pre shape: {tuple(x_pre.shape)} (T_a={ctx['T_a']}, T_pre={T_pre})",
          file=sys.stderr)

    # 2. Save x_pre as fp32 .npy for downstream re-use.
    npy_path = args.out_dir / "x_pre.npy"
    np.save(npy_path, x_pre.numpy().astype(np.float32))
    print(f"[step 2] Saved x_pre → {npy_path}", file=sys.stderr)

    # 3a. PyTorch reference iSTFT (torch.istft via gen.stft).
    print("[step 3a] PyTorch reference (gen.stft.inverse, i.e. torch.istft)...",
          file=sys.stderr)
    ref_audio = ctx["ref_audio"]
    print(f"          ref shape: {ref_audio.shape}", file=sys.stderr)

    # 3b. PyTorch CustomSTFT iSTFT (the conv_transpose rewrite).
    print("[step 3b] PyTorch CustomSTFT (conv_transpose1d rewrite)...", file=sys.stderr)
    custom_audio = pytorch_custom_stft_tail(model, x_pre)
    print(f"          custom shape: {custom_audio.shape}", file=sys.stderr)

    # 3c. CoreML tail.
    print(f"[step 3c] CoreML tail ({tail_pkg.name})...", file=sys.stderr)
    coreml_audio = coreml_tail(tail_pkg, x_pre)
    print(f"          coreml shape: {coreml_audio.shape}", file=sys.stderr)

    # Save WAVs for audition.
    try:
        import soundfile as sf
        sf.write(str(args.out_dir / "tail_torch_ref.wav"), ref_audio, SR, subtype="FLOAT")
        sf.write(str(args.out_dir / "tail_torch_custom.wav"), custom_audio, SR, subtype="FLOAT")
        sf.write(str(args.out_dir / "tail_coreml.wav"), coreml_audio, SR, subtype="FLOAT")
        # Also save residuals for audible diff inspection.
        n = min(len(ref_audio), len(custom_audio), len(coreml_audio))
        sf.write(str(args.out_dir / "diff_ref_minus_custom.wav"),
                 ref_audio[:n] - custom_audio[:n], SR, subtype="FLOAT")
        sf.write(str(args.out_dir / "diff_ref_minus_coreml.wav"),
                 ref_audio[:n] - coreml_audio[:n], SR, subtype="FLOAT")
        sf.write(str(args.out_dir / "diff_custom_minus_coreml.wav"),
                 custom_audio[:n] - coreml_audio[:n], SR, subtype="FLOAT")
        print(f"[wrote] WAVs → {args.out_dir}/", file=sys.stderr)
    except ImportError:
        print("[warn] soundfile missing — skipping WAV writes", file=sys.stderr)

    # 4. Diffs.
    print("\n=== Per-sample diffs ===")
    m1 = diff_metrics(ref_audio, custom_audio, "torch_ref vs torch_custom")
    m2 = diff_metrics(custom_audio, coreml_audio, "torch_custom vs coreml")
    m3 = diff_metrics(ref_audio, coreml_audio, "torch_ref vs coreml")

    print("\n=== HF (≥10 kHz) band power ===")
    h1 = hf_band_diff(ref_audio, custom_audio, "torch_ref vs torch_custom")
    h2 = hf_band_diff(custom_audio, coreml_audio, "torch_custom vs coreml")
    h3 = hf_band_diff(ref_audio, coreml_audio, "torch_ref vs coreml")

    # Verdict.
    print("\n=== Verdict ===")
    fp32_eps = 1e-4   # rel_rms threshold for "fp32-precision match"
    pair_custom_vs_coreml = m2["rel_rms"]
    pair_ref_vs_custom = m1["rel_rms"]
    pair_ref_vs_coreml = m3["rel_rms"]
    print(f"  rel_rms torch_ref vs torch_custom : {pair_ref_vs_custom:.4e}")
    print(f"  rel_rms torch_custom vs coreml    : {pair_custom_vs_coreml:.4e}")
    print(f"  rel_rms torch_ref vs coreml       : {pair_ref_vs_coreml:.4e}")
    print(f"  HF Δ(torch_custom - coreml) dB    : {h2['delta_db']:+.2f} dB")
    print(f"  HF Δ(torch_ref - coreml)    dB    : {h3['delta_db']:+.2f} dB")
    if pair_custom_vs_coreml > fp32_eps:
        print("  → torch_custom and coreml DIFFER beyond fp32 precision.")
        print("    The CoreML conversion of CustomSTFT is the culprit.")
        print("    Inspect KokoroTail.mlmodelc weights vs CoreMLCustomSTFT init.")
    elif pair_ref_vs_custom > 1e-2:
        print("  → CustomSTFT itself drifts from torch.istft.")
        print("    Inspect kokoro.custom_stft.CustomSTFT formulation (DC/Nyquist scaling).")
    elif pair_ref_vs_coreml > fp32_eps:
        print("  → CoreML matches torch_custom but both drift from torch_ref.")
        print("    The drift is in CustomSTFT formulation, not the CoreML conversion.")
    else:
        print("  → All three tails agree to fp32 precision.")
        print("    Tail is NOT the noise source — escalate to per-resblock activation diff in Vocoder.")


if __name__ == "__main__":
    main()
