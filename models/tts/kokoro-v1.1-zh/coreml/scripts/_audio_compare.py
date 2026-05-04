import math
import sys
import numpy as np
import soundfile as sf
import torch
from kokoro import KModel
from kokoro.pipeline import KPipeline


def hf_db(x, sr=24000, n_fft=1024, hop=256, hf_cutoff=10000):
    from scipy.signal import stft
    f, _, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    S = np.abs(Z) ** 2
    return 10 * math.log10(max(float(S[f >= hf_cutoff].mean()), 1e-30))


def freq_band(x, lo, hi, sr=24000, n_fft=1024, hop=256):
    from scipy.signal import stft
    f, _, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    S = np.abs(Z) ** 2
    return 10 * math.log10(max(float(S[(f >= lo) & (f < hi)].mean()), 1e-30))


def main():
    print("Generating PyTorch reference...", file=sys.stderr)
    m = KModel(repo_id="hexgrad/Kokoro-82M-v1.1-zh"); m.eval()
    pipe = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=m)
    voice_pack = np.frombuffer(
        open("build/ANE-zh/voices/zm_009.bin", "rb").read(),
        dtype=np.float32).reshape(510, 1, 256)
    voice_pack = torch.from_numpy(voice_pack.copy())

    phonemes = ""
    for _gs, ps, _tks in pipe("你好世界，今天天气很好。", voice="zm_009"):
        phonemes = ps; break
    ids = list(filter(lambda i: i is not None, map(lambda p: m.vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *ids, 0]])
    ref_s = voice_pack[max(min(len(phonemes) - 1, voice_pack.shape[0] - 1), 0)]
    s = ref_s[:, 128:]; style_timbre = ref_s[:, :128]

    with torch.no_grad():
        input_lengths = torch.LongTensor([input_ids.shape[1]])
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(1, -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = m.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = m.bert_encoder(bert_dur).transpose(-1, -2)
        d = m.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = m.predictor.lstm(d)
        duration = m.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        T_a = pred_dur.sum().item()
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
        pred_aln = torch.zeros(input_ids.shape[1], T_a)
        pred_aln[indices, torch.arange(T_a)] = 1.0
        pred_aln = pred_aln.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln
        F0_pred, N_pred = m.predictor.F0Ntrain(en, s)
        t_en = m.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln
        pt = m.decoder(asr, F0_pred, N_pred, style_timbre).squeeze().numpy()
    sf.write("build/audio_compare_zm009/pytorch_ref_zm009.wav", pt, 24000, subtype="FLOAT")

    before = sf.read("build/audio_compare_zm009/before_fix_zm009.wav")[0]
    after = sf.read("build/audio_compare_zm009/after_fix_zm009.wav")[0]
    n = min(len(pt), len(before), len(after))
    pt, before, after = pt[:n], before[:n], after[:n]

    print()
    print("=== HF (>=10 kHz) audible-noise band power ===")
    pt_hf = hf_db(pt)
    before_hf = hf_db(before)
    after_hf = hf_db(after)
    print(f"  PyTorch ref       : {pt_hf:7.2f} dB")
    print(f"  CoreML BEFORE fix : {before_hf:7.2f} dB   (delta vs ref: {before_hf - pt_hf:+.2f} dB)")
    print(f"  CoreML AFTER  fix : {after_hf:7.2f} dB   (delta vs ref: {after_hf - pt_hf:+.2f} dB)")
    print()
    print("=== Per-band power (dB) ===")
    for lo, hi in [(0, 1000), (1000, 4000), (4000, 8000), (8000, 10000), (10000, 12000)]:
        print(f"  {lo:5d}-{hi:5d} Hz:  PT={freq_band(pt,lo,hi):+6.2f}  "
              f"before={freq_band(before,lo,hi):+6.2f}  "
              f"after={freq_band(after,lo,hi):+6.2f}")
    print()
    print("=== Residual (CoreML - PT) HF power ===")
    def residual_hf(a, b):
        return hf_db(a - b)
    rb = residual_hf(before, pt); ra = residual_hf(after, pt)
    print(f"  BEFORE fix: HF(coreml - pt) = {rb:.2f} dB")
    print(f"  AFTER  fix: HF(coreml - pt) = {ra:.2f} dB")
    print(f"  Improvement: {rb - ra:.2f} dB")
    print()
    print("=== Waveform corr (CoreML vs PT) ===")
    print(f"  BEFORE: {float(np.corrcoef(before, pt)[0,1]):.6f}")
    print(f"  AFTER:  {float(np.corrcoef(after, pt)[0,1]):.6f}")


if __name__ == "__main__":
    main()
