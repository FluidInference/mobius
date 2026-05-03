"""Localise the divergence inside KokoroNoise.mlpackage.

per_stage_diff.py shows KokoroNoise's x_source_0 diverges from PyTorch by
~44% rel-rms (corr 0.886) — the dominant cause of the residual HF noise.

Probes:
    1. Compute-unit sensitivity (CPU_ONLY vs CPU_AND_NE vs CPU_AND_GPU vs ALL).
    2. Per-channel divergence on x_source_0 (which channels carry the bug).
    3. Sub-stage probe: rebuild the noise pipeline in pure PyTorch using the
       converted CoreMLSineGenV2 / CoreMLForwardSTFT formulation, and check
       which intermediate diverges most:
           a. SineGen sine_wavs (sin(cumsum(rad_values * upsample_scale)))
           b. har_source = tanh(linear(sine_wavs))
           c. STFT magnitude / phase
           d. noise_conv outputs (post-conv, pre-resblock)
           e. noise_res outputs (final x_source_i)

Usage:
    uv run python probe_noise.py --models-dir build/ANE-zh --voice zm_009
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
    bin_path = models_dir / "voices" / f"{voice}.bin"
    if not bin_path.exists():
        bin_path = models_dir / f"{voice}.bin"
    if bin_path.exists():
        flat = np.frombuffer(bin_path.read_bytes(), dtype=np.float32).reshape(510, 1, 256)
        return torch.from_numpy(flat.copy())
    return pipe.load_voice(voice)


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
        return sources, har, har_source_flat


def coreml_sine_gen_v2_pytorch(f0, sine_amp, noise_std, harmonic_num, sampling_rate,
                               voiced_threshold, upsample_scale):
    """Re-implement CoreMLSineGenV2 in pure PyTorch (matches what was traced)."""
    harmonics = torch.arange(1, harmonic_num + 2, device=f0.device, dtype=f0.dtype)
    fn = f0 * harmonics.view(1, 1, -1)
    rad_values = fn / sampling_rate
    rv = rad_values.transpose(1, 2)
    rv_down = F.avg_pool1d(rv, kernel_size=upsample_scale, stride=upsample_scale)
    rad_values_down = rv_down.transpose(1, 2)
    phase = torch.cumsum(rad_values_down, dim=1) * (2 * math.pi)
    ph = phase.transpose(1, 2) * upsample_scale
    ph_up = F.interpolate(ph, scale_factor=float(upsample_scale), mode="linear",
                          align_corners=False)
    phase = ph_up.transpose(1, 2)
    sines = torch.sin(phase) * sine_amp
    uv = (f0 > voiced_threshold).float()
    noise_amp = uv * noise_std + (1 - uv) * sine_amp / 3
    noise = noise_amp * 0.01
    sine_waves = sines * uv + noise
    return sine_waves, uv, noise


def diff_arrays(a, b, label):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))
    rms_d = float(np.sqrt(np.mean((a - b) ** 2)))
    rel = rms_d / max(rms_a, 1e-12)
    corr = float(np.corrcoef(a, b)[0, 1]) if rms_a > 0 and rms_b > 0 else 0.0
    print(f"  [{label:50s}] rms_pt={rms_a:.4e} rms_cm={rms_b:.4e} "
          f"rel={rel:.3e} corr={corr:.7f}")
    return {"rms_a": rms_a, "rms_b": rms_b, "rms_d": rms_d, "rel": rel, "corr": corr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=pathlib.Path, required=True)
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--phonemes", default=None)
    p.add_argument("--voice", default="zm_009")
    p.add_argument("--lang", default="z")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = p.parse_args()

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
            p.error(f"empty phonemes")
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

    pt_sources, pt_har, pt_har_source = precompute_noise_sources_pt(
        model.decoder.generator, F0_pred, style_timbre)
    print(f"  T_a={T_a}, F0_pred={tuple(F0_pred.shape)}, "
          f"har={tuple(pt_har.shape)}, sources=({tuple(pt_sources[0].shape)}, "
          f"{tuple(pt_sources[1].shape)})")

    # ==========================================================================
    # 1. Compute-unit sweep on KokoroNoise
    # ==========================================================================
    noise_path = args.models_dir / "KokoroNoise.mlpackage"
    if not noise_path.exists():
        noise_path = args.models_dir / "KokoroNoise.mlmodelc"
    print(f"\n=== [1] CU sweep on {noise_path.name} ===")
    feed = {
        "F0_curve": F0_pred.numpy().astype(np.float32),
        "style_timbre": style_timbre.numpy().astype(np.float32),
    }
    cu_list = [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
               ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
               ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
               ("ALL", ct.ComputeUnit.ALL)]
    for cu_name, cu in cu_list:
        try:
            ml = ct.models.MLModel(str(noise_path), compute_units=cu)
            out = ml.predict(feed)
            print(f"  -- {cu_name} --")
            for i in range(2):
                a = pt_sources[i].numpy()
                b = np.array(out[f"x_source_{i}"]).astype(np.float32)
                diff_arrays(a, b, f"x_source_{i}")
        except Exception as e:
            print(f"  -- {cu_name} -- FAILED: {str(e)[:120]}")

    # ==========================================================================
    # 2. Per-channel divergence (CPU_ONLY for cleanest signal)
    # ==========================================================================
    print(f"\n=== [2] per-channel divergence (CPU_ONLY) ===")
    ml_cpu = ct.models.MLModel(str(noise_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    out_cpu = ml_cpu.predict(feed)
    for i in range(2):
        a = pt_sources[i].numpy()
        b = np.array(out_cpu[f"x_source_{i}"]).astype(np.float32)
        n = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :n], b[..., :n]
        rms_per_ch_a = np.sqrt(np.mean(a[0] ** 2, axis=1))
        rms_per_ch_d = np.sqrt(np.mean((a[0] - b[0]) ** 2, axis=1))
        rel = rms_per_ch_d / np.maximum(rms_per_ch_a, 1e-12)
        worst = np.argsort(-rel)[:5]
        best = np.argsort(rel)[:3]
        print(f"  x_source_{i} ({a.shape[1]} channels):")
        print(f"    median rel={np.median(rel):.3e}  p95={np.quantile(rel, 0.95):.3e}  "
              f"max={rel.max():.3e}")
        print(f"    worst 5 channels:")
        for c in worst:
            print(f"      ch[{c:3d}]  rel={rel[c]:.3e}  rms_pt={rms_per_ch_a[c]:.3e}")
        print(f"    best 3 channels:")
        for c in best:
            print(f"      ch[{c:3d}]  rel={rel[c]:.3e}  rms_pt={rms_per_ch_a[c]:.3e}")

    # ==========================================================================
    # 3. Sub-stage probe: where inside the noise pipeline does it diverge?
    # ==========================================================================
    # Trace a "stub" graph that exposes intermediate tensors. We do this in
    # PyTorch only — no CoreML conversion of stubs needed because the *full*
    # CoreML noise model is the unit under test, not its sub-stages.
    #
    # But to localise, we run pure-PyTorch and pure-PyTorch-with-CoreML-formulation
    # versions side-by-side. If the upstream CoreML formulation (avg_pool1d/
    # interpolate trick instead of full-rate cumsum) drifts in pure PyTorch,
    # the bug is in the formulation. If it matches PyTorch but the converted
    # CoreML graph drifts, the bug is in the conversion.
    print(f"\n=== [3] sub-stage probe (CoreMLSineGenV2 formulation in pure PT) ===")
    gen = model.decoder.generator
    src = gen.m_source
    sine_gen = src.l_sin_gen
    upsample_scale = sine_gen.upsample_scale
    print(f"  upsample_scale={upsample_scale}, harmonic_num={sine_gen.harmonic_num}, "
          f"sine_amp={sine_gen.sine_amp}, voiced_threshold={sine_gen.voiced_threshold}")

    with torch.no_grad():
        # Reference upstream f0 (same as Generator.forward).
        f0 = gen.f0_upsamp(F0_pred[:, None]).transpose(1, 2)

        # PT teacher SineGen (full sin-of-cumulative-phase at upsampled rate).
        sine_wavs_pt, uv_pt, _ = sine_gen(f0)

        # CoreML-formulation SineGen (avg_pool / cumsum / interpolate trick).
        sine_wavs_cm, uv_cm, _ = coreml_sine_gen_v2_pytorch(
            f0, sine_amp=sine_gen.sine_amp, noise_std=sine_gen.noise_std,
            harmonic_num=sine_gen.harmonic_num, sampling_rate=sine_gen.sampling_rate,
            voiced_threshold=sine_gen.voiced_threshold,
            upsample_scale=upsample_scale)

    diff_arrays(sine_wavs_pt, sine_wavs_cm, "PT SineGen vs PT(CM-formulation) SineGen")
    diff_arrays(uv_pt, uv_cm, "PT uv vs CM-formulation uv")

    # Continue both chains forward.
    with torch.no_grad():
        sine_merge_pt = src.l_tanh(src.l_linear(sine_wavs_pt))
        sine_merge_cm = src.l_tanh(src.l_linear(sine_wavs_cm))
        har_source_pt = sine_merge_pt.transpose(1, 2).squeeze(1)
        har_source_cm = sine_merge_cm.transpose(1, 2).squeeze(1)
    diff_arrays(har_source_pt, har_source_cm, "har_source (after l_tanh∘l_linear)")

    # STFT (using PyTorch's torch.stft via TorchSTFT — same on both sides).
    with torch.no_grad():
        spec_pt, phase_pt = gen.stft.transform(har_source_pt)
        spec_cm, phase_cm = gen.stft.transform(har_source_cm)
        har_pt_full = torch.cat([spec_pt, phase_pt], dim=1)
        har_cm_full = torch.cat([spec_cm, phase_cm], dim=1)
    diff_arrays(spec_pt, spec_cm, "STFT magnitude")
    diff_arrays(phase_pt, phase_cm, "STFT phase")

    with torch.no_grad():
        for i in range(gen.num_upsamples):
            x_post_conv_pt = gen.noise_convs[i](har_pt_full)
            x_post_conv_cm = gen.noise_convs[i](har_cm_full)
            diff_arrays(x_post_conv_pt, x_post_conv_cm,
                        f"noise_convs[{i}] output")
            x_after_res_pt = gen.noise_res[i](x_post_conv_pt, style_timbre)
            x_after_res_cm = gen.noise_res[i](x_post_conv_cm, style_timbre)
            diff_arrays(x_after_res_pt, x_after_res_cm,
                        f"noise_res[{i}] output (= x_source_{i})")

    # CoreML-formulation has lower magnitude than PT — does that match what
    # CoreML produces?
    print("\n  (For reference — directly from CoreML Noise CPU_ONLY:)")
    for i in range(2):
        a = pt_sources[i].numpy()
        b = np.array(out_cpu[f"x_source_{i}"]).astype(np.float32)
        diff_arrays(a, b, f"PT source vs CoreML CPU x_source_{i}")


if __name__ == "__main__":
    main()
