"""Vocoder x_pre divergence: CoreML Vocoder.x_pre vs PyTorch x_pre.

Now that the iSTFT/Tail stage is ruled out (corr=1.000 with torch.istft up to
a deterministic ×1.5 gain), the next question is whether the CoreML Vocoder
produces a different x_pre tensor than PyTorch given the same upstream
inputs. If yes, the noise is in the Vocoder graph (cos/exp/sin op
decomposition, weight quantization, or upsample fusion).

Procedure:
    1. Run PyTorch up through asr / F0_pred / N_pred / style_timbre /
       noise_sources[0..1] (the inputs to KokoroVocoder.mlpackage).
    2. Save those same inputs and feed them as fp32 into KokoroVocoder.
       Record its x_pre output.
    3. Compute PyTorch x_pre by inline-running decoder.encode/decode +
       generator up to leaky_relu before conv_post.
    4. Diff per-channel and per-frame, plus HF-band power on the audio
       reconstructed by KokoroTail from each x_pre.

Usage:
    uv run python vocoder_xpre_diff.py \
        --models-dir build/ANE-zh \
        --voice zm_009 \
        --out-dir build/vocoder_xpre_diff_zm009
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


def precompute_noise_sources(generator, F0_curve, style_timbre):
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
        return sources, har


def pytorch_pipeline(model, phonemes, voice_pack):
    """Run PyTorch up through the full set of inputs the CoreML Vocoder needs."""
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

        # CoreMLVocoderDualOutput-equivalent forward (sin² Snake, not cos).
        decoder = model.decoder
        F0 = decoder.F0_conv(F0_pred.unsqueeze(1))
        N_feat = decoder.N_conv(N_pred.unsqueeze(1))
        x_dec = torch.cat([asr, F0, N_feat], dim=1)
        x_dec = decoder.encode(x_dec, style_timbre)
        asr_res = decoder.asr_res(asr)
        res = True
        for block in decoder.decode:
            if res:
                x_dec = torch.cat([x_dec, asr_res, F0, N_feat], dim=1)
            x_dec = block(x_dec, style_timbre)
            if block.upsample_type != "none":
                res = False

        # Generator inline (sin² Snake1D).
        gen = decoder.generator
        noise_sources, _ = precompute_noise_sources(gen, F0_pred, style_timbre)
        for i in range(gen.num_upsamples):
            x_dec = F.leaky_relu(x_dec, negative_slope=0.1)
            x_dec = gen.ups[i](x_dec)
            if i == gen.num_upsamples - 1:
                x_dec = gen.reflection_pad(x_dec)
            x_dec = x_dec + noise_sources[i]
            xs = None
            for j in range(gen.num_kernels):
                if xs is None:
                    xs = gen.resblocks[i * gen.num_kernels + j](x_dec, style_timbre)
                else:
                    xs = xs + gen.resblocks[i * gen.num_kernels + j](x_dec, style_timbre)
            x_dec = xs / gen.num_kernels
        x_pre = F.leaky_relu(x_dec)

    return {
        "asr": asr,
        "F0_pred": F0_pred,
        "N_pred": N_pred,
        "style_timbre": style_timbre,
        "noise_sources": noise_sources,
        "x_pre": x_pre,
        "T_a": int(T_a),
    }


def hf_band_power_db(audio, sr=SR, n_fft=1024, hop=256, hf_cutoff_hz=10000.0):
    from scipy.signal import stft
    f, _, Z = stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    S = np.abs(Z) ** 2
    return 10 * math.log10(max(float(S[f >= hf_cutoff_hz].mean()), 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("build/vocoder_xpre_diff"))
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
        print(f"      phonemes: {phonemes!r}", file=sys.stderr)
        del m_g, pipe_g

    print(f"[setup] Loading PyTorch KModel ({args.repo_id})...", file=sys.stderr)
    model = KModel(repo_id=args.repo_id); model.eval()
    pipe = KPipeline(lang_code=args.lang, repo_id=args.repo_id, model=model)
    voice_pack = load_voice_pack(args.models_dir, args.voice, pipe)

    print(f"[step 1] PyTorch pipeline → x_pre + Vocoder inputs (voice={args.voice})...",
          file=sys.stderr)
    art = pytorch_pipeline(model, phonemes, voice_pack)
    print(f"         T_a={art['T_a']}, asr={tuple(art['asr'].shape)}, "
          f"F0_pred={tuple(art['F0_pred'].shape)}, x_pre={tuple(art['x_pre'].shape)}",
          file=sys.stderr)

    voc_path = args.models_dir / "KokoroVocoder.mlpackage"
    if not voc_path.exists():
        voc_path = args.models_dir / "KokoroVocoder.mlmodelc"
    print(f"[step 2] CoreML Vocoder ({voc_path.name})...", file=sys.stderr)
    voc = ct.models.MLModel(str(voc_path), compute_units=ct.ComputeUnit.ALL)
    voc_out = voc.predict({
        "asr": art["asr"].numpy().astype(np.float32),
        "F0_curve": art["F0_pred"].numpy().astype(np.float32),
        "N_pred": art["N_pred"].numpy().astype(np.float32),
        "x_source_0": art["noise_sources"][0].numpy().astype(np.float32),
        "x_source_1": art["noise_sources"][1].numpy().astype(np.float32),
        "style_timbre": art["style_timbre"].numpy().astype(np.float32),
    })
    coreml_x_pre = np.array(voc_out["x_pre"]).astype(np.float32)
    print(f"         coreml x_pre: {coreml_x_pre.shape}", file=sys.stderr)

    pt_x_pre = art["x_pre"].numpy().astype(np.float32)
    print(f"         pytorch x_pre: {pt_x_pre.shape}", file=sys.stderr)

    n_pre = min(coreml_x_pre.shape[-1], pt_x_pre.shape[-1])
    a = pt_x_pre[..., :n_pre]
    b = coreml_x_pre[..., :n_pre]

    print("\n=== x_pre per-tensor diff ===")
    diff = a - b
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))
    rms_d = float(np.sqrt(np.mean(diff ** 2)))
    max_d = float(np.max(np.abs(diff)))
    corr = float(np.corrcoef(a.flatten(), b.flatten())[0, 1])
    print(f"  rms(pt)={rms_a:.6e}  rms(cm)={rms_b:.6e}  ratio={rms_b/max(rms_a,1e-12):.4f}")
    print(f"  rms(diff)={rms_d:.6e}  max|diff|={max_d:.4e}  corr={corr:.9f}")
    print(f"  rel-rms = {rms_d/max(rms_a,1e-12):.4e}")

    # Best-gain analysis (in case there is a uniform scale).
    g = float(np.dot(a.flatten(), b.flatten()) / np.dot(a.flatten(), a.flatten()))
    resid = b - g * a
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))
    print(f"  best gain g(pt→cm) = {g:.6f}  rms_resid={rms_resid:.4e}  rel={rms_resid/max(rms_b,1e-12):.4e}")

    # Per-channel diff.
    a_ch = a.reshape(a.shape[1], -1)
    b_ch = b.reshape(b.shape[1], -1)
    rms_per_ch_a = np.sqrt(np.mean(a_ch ** 2, axis=1))
    rms_per_ch_b = np.sqrt(np.mean(b_ch ** 2, axis=1))
    rms_per_ch_d = np.sqrt(np.mean((a_ch - b_ch) ** 2, axis=1))
    rel_per_ch = rms_per_ch_d / np.maximum(rms_per_ch_a, 1e-12)
    print(f"\n  worst 8 channels by rel-rms (channel, rel-rms, rms_pt, rms_cm):")
    worst = np.argsort(-rel_per_ch)[:8]
    for c in worst:
        print(f"    ch[{c:3d}]  rel={rel_per_ch[c]:.3e}  rms_pt={rms_per_ch_a[c]:.3e}  "
              f"rms_cm={rms_per_ch_b[c]:.3e}")

    # Per-frame diff (mean over channels).
    rms_per_frame_a = np.sqrt(np.mean(a[0] ** 2, axis=0))
    rms_per_frame_b = np.sqrt(np.mean(b[0] ** 2, axis=0))
    rms_per_frame_d = np.sqrt(np.mean((a[0] - b[0]) ** 2, axis=0))
    rel_per_frame = rms_per_frame_d / np.maximum(rms_per_frame_a, 1e-12)
    print(f"\n  per-frame stats:  median rel={np.median(rel_per_frame):.3e}  "
          f"p95 rel={np.quantile(rel_per_frame, 0.95):.3e}  "
          f"max rel={np.max(rel_per_frame):.3e}")

    # Reconstruct audio from each x_pre and compare HF band.
    tail_path = args.models_dir / "KokoroTail.mlpackage"
    if not tail_path.exists():
        tail_path = args.models_dir / "KokoroTail.mlmodelc"
    tail = ct.models.MLModel(str(tail_path), compute_units=ct.ComputeUnit.ALL)
    aud_pt = np.array(tail.predict({"x_pre": pt_x_pre.astype(np.float32)})["audio"]).flatten()
    aud_cm = np.array(tail.predict({"x_pre": coreml_x_pre.astype(np.float32)})["audio"]).flatten()
    n = min(len(aud_pt), len(aud_cm))
    aud_pt, aud_cm = aud_pt[:n], aud_cm[:n]
    rms_aud_d = float(np.sqrt(np.mean((aud_pt - aud_cm) ** 2)))
    rms_aud_pt = float(np.sqrt(np.mean(aud_pt ** 2)))
    print(f"\n=== Audio reconstructed from each x_pre (via CoreML Tail) ===")
    print(f"  rms(audio_pt_xpre)={rms_aud_pt:.4e}  "
          f"rms(audio_cm_xpre)={float(np.sqrt(np.mean(aud_cm**2))):.4e}")
    print(f"  rms(diff)={rms_aud_d:.4e}  rel-rms={rms_aud_d/max(rms_aud_pt,1e-12):.4e}")
    pt_hf_db = hf_band_power_db(aud_pt)
    cm_hf_db = hf_band_power_db(aud_cm)
    diff_hf_db = hf_band_power_db(aud_pt - aud_cm)
    print(f"  HF (≥10 kHz) audio_pt_xpre: {pt_hf_db:.2f} dB")
    print(f"  HF (≥10 kHz) audio_cm_xpre: {cm_hf_db:.2f} dB   Δ={cm_hf_db - pt_hf_db:+.2f} dB")
    print(f"  HF (≥10 kHz) (audio_pt - audio_cm): {diff_hf_db:.2f} dB")

    np.save(args.out_dir / "x_pre_pt.npy", pt_x_pre)
    np.save(args.out_dir / "x_pre_cm.npy", coreml_x_pre)
    try:
        import soundfile as sf
        sf.write(str(args.out_dir / "audio_from_pt_xpre.wav"), aud_pt, SR, subtype="FLOAT")
        sf.write(str(args.out_dir / "audio_from_cm_xpre.wav"), aud_cm, SR, subtype="FLOAT")
        sf.write(str(args.out_dir / "audio_diff.wav"), aud_pt - aud_cm, SR, subtype="FLOAT")
        print(f"\n[wrote] x_pre tensors and WAVs → {args.out_dir}/", file=sys.stderr)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
