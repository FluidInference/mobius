"""Parity test: PyTorch Kokoro reference vs 7-stage CoreML chain.

Mirrors the validation block at the end of convert-coreml.py, but standalone so
it can be re-run after model edits / re-conversion without rebuilding.

Pass criteria (mobius standard, matches convert.py's reported numbers):
    waveform corr ≥ 0.80
    mel-spectrogram corr ≥ 0.99

Usage:
    uv run python compare-models.py --models-dir build/laishere-kokoro
    uv run python compare-models.py --models-dir build/laishere-kokoro \
        --phonemes "ðə kwɪk bɹaʊn fɑːks dʒʌmps oʊvɚ ðə leɪzi dɑːɡ."
"""
import argparse
import pathlib
import sys

import coremltools as ct
import numpy as np
import torch

from kokoro import KModel
from kokoro.pipeline import KPipeline


SR = 24000
DEFAULT_PHONEMES = "ðə kwɪk bɹaʊn fɑːks dʒʌmps oʊvɚ ðə leɪzi dɑːɡ."


def mel_corr(a, b, sr=SR, n_fft=1024, n_mels=80, hop=256):
    """Mel-spectrogram correlation (matches convert.py's mel_corr)."""
    from scipy.signal import stft
    def _mel(x):
        f, _, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        S = np.abs(Z) ** 2
        # 80-bin linear-frequency approximation (sufficient for MOS-like comparison)
        edges = np.linspace(0, len(f) - 1, n_mels + 1).astype(int)
        mel = np.stack([S[edges[i]:edges[i + 1]].sum(0) for i in range(n_mels)])
        return np.log(mel + 1e-8).flatten()
    return float(np.corrcoef(_mel(a), _mel(b))[0, 1])


def pytorch_reference(phonemes, voice="af_heart", lang="a"):
    """Run Kokoro PyTorch teacher to produce reference waveform + intermediate tensors.

    Returns:
        ref_audio : np.ndarray [N], 24 kHz fp32
        artifacts : dict with input_ids, attention_mask, style_s, style_timbre
    """
    model = KModel(); model.eval()
    pipe = KPipeline(lang_code=lang, model=model)
    voice_pack = pipe.load_voice(voice)

    ids = list(filter(lambda i: i is not None,
                       map(lambda p: model.vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *ids, 0]])
    ref_s = voice_pack[max(min(len(phonemes) - 1, voice_pack.shape[0] - 1), 0)]

    with torch.no_grad():
        input_lengths = torch.LongTensor([input_ids.shape[1]])
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(1, -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        style_timbre = ref_s[:, :128]
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
        ref_audio = model.decoder(asr, F0_pred, N_pred, style_timbre).squeeze().numpy()

    return ref_audio, {
        "input_ids": input_ids.numpy().astype(np.int32),
        "attention_mask": (~text_mask).int().numpy().astype(np.int32),
        "style_s": s.numpy(),
        "style_timbre": style_timbre.numpy(),
        "T_a": int(T_a),
    }


def coreml_chain(models_dir, artifacts):
    """Run the 7-stage CoreML chain. Returns audio waveform [N] fp32."""
    CU_NE = ct.ComputeUnit.CPU_AND_NE
    CU_ALL = ct.ComputeUnit.ALL
    m_albert = ct.models.MLModel(str(models_dir / "KokoroAlbert.mlpackage"),     compute_units=CU_NE)
    m_post   = ct.models.MLModel(str(models_dir / "KokoroPostAlbert.mlpackage"), compute_units=CU_NE)
    m_align  = ct.models.MLModel(str(models_dir / "KokoroAlignment.mlpackage"),  compute_units=CU_NE)
    m_pros   = ct.models.MLModel(str(models_dir / "KokoroProsody.mlpackage"),    compute_units=CU_ALL)
    m_noise  = ct.models.MLModel(str(models_dir / "KokoroNoise.mlpackage"),      compute_units=CU_ALL)
    m_voc    = ct.models.MLModel(str(models_dir / "KokoroVocoder.mlpackage"),    compute_units=CU_NE)
    m_tail   = ct.models.MLModel(str(models_dir / "KokoroTail.mlpackage"),       compute_units=CU_ALL)

    input_ids = artifacts["input_ids"]
    attention_mask = artifacts["attention_mask"]
    style_s_f16 = artifacts["style_s"].astype(np.float16)
    style_timbre = artifacts["style_timbre"]

    o1 = m_albert.predict({"input_ids": input_ids, "attention_mask": attention_mask})
    o2 = m_post.predict({
        "bert_dur": np.array(o1["bert_dur"]).astype(np.float16),
        "input_ids": input_ids,
        "style_s": style_s_f16,
        "speed": np.array([1.0], dtype=np.float16),
        "attention_mask": attention_mask,
    })
    dur = np.array(o2["duration"]).flatten()
    pd = np.round(dur).clip(min=1).astype(np.int32).reshape(1, -1)
    o3 = m_align.predict({
        "pred_dur": pd,
        "d": np.array(o2["d"]).astype(np.float16),
        "t_en": np.array(o2["t_en"]).astype(np.float16),
    })
    o4 = m_pros.predict({
        "en": np.array(o3["en"]).astype(np.float16),
        "style_s": style_s_f16,
    })
    o5 = m_noise.predict({
        "F0_curve": np.array(o4["F0"]).astype(np.float32),
        "style_timbre": style_timbre.astype(np.float32),
    })
    o6 = m_voc.predict({
        "asr": np.array(o3["asr"]).astype(np.float16),
        "F0_curve": np.array(o4["F0"]).astype(np.float16),
        "N_pred": np.array(o4["N"]).astype(np.float16),
        "x_source_0": np.array(o5["x_source_0"]).astype(np.float16),
        "x_source_1": np.array(o5["x_source_1"]).astype(np.float16),
        "style_timbre": style_timbre.astype(np.float16),
    })
    o7 = m_tail.predict({"x_pre": np.array(o6["x_pre"]).astype(np.float32)})
    return np.array(o7["audio"]).flatten().astype(np.float32)


def main():
    p = argparse.ArgumentParser(description="Parity test: PyTorch ref vs CoreML chain")
    p.add_argument("--models-dir", type=pathlib.Path, required=True,
                   help="Directory containing the 7 .mlpackage files")
    p.add_argument("--phonemes", default=DEFAULT_PHONEMES,
                   help="IPA phonemes to synthesize on both sides (default: pangram)")
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--lang", default="a")
    p.add_argument("--corr-threshold", type=float, default=0.80)
    p.add_argument("--mel-corr-threshold", type=float, default=0.99)
    p.add_argument("--save-ref", type=pathlib.Path, help="Optional WAV path for PyTorch reference")
    p.add_argument("--save-coreml", type=pathlib.Path, help="Optional WAV path for CoreML output")
    args = p.parse_args()

    if not args.models_dir.exists():
        p.error(f"models-dir {args.models_dir} does not exist")

    print(f"[ref] PyTorch teacher synthesizing {len(args.phonemes)} phonemes...", file=sys.stderr)
    ref, artifacts = pytorch_reference(args.phonemes, voice=args.voice, lang=args.lang)
    print(f"      ref: {len(ref)} samples ({len(ref) / SR:.2f}s), T_a={artifacts['T_a']}",
          file=sys.stderr)

    print("[coreml] Running 7-stage chain...", file=sys.stderr)
    cm = coreml_chain(args.models_dir, artifacts)
    n = min(len(ref), len(cm))
    ref_n, cm_n = ref[:n], cm[:n]

    corr = float(np.corrcoef(ref_n, cm_n)[0, 1])
    mc = mel_corr(cm_n, ref_n)
    rms_err = float(np.sqrt(np.mean((cm_n - ref_n) ** 2)))
    rms_ref = float(np.sqrt(np.mean(ref_n ** 2)))
    rel_rms = rms_err / max(rms_ref, 1e-9)

    print(f"\n  waveform corr     : {corr:.6f}   (threshold ≥ {args.corr_threshold:.2f})")
    print(f"  mel-spectrogram   : {mc:.6f}   (threshold ≥ {args.mel_corr_threshold:.2f})")
    print(f"  rms err / rms ref : {rel_rms:.4f}")
    print(f"  length ref/coreml : {len(ref)} / {len(cm)} samples")

    if args.save_ref:
        import soundfile as sf
        args.save_ref.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(args.save_ref), ref, SR, subtype="FLOAT")
        print(f"  → wrote {args.save_ref}")
    if args.save_coreml:
        import soundfile as sf
        args.save_coreml.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(args.save_coreml), cm, SR, subtype="FLOAT")
        print(f"  → wrote {args.save_coreml}")

    ok = corr >= args.corr_threshold and mc >= args.mel_corr_threshold
    print(f"\n  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
