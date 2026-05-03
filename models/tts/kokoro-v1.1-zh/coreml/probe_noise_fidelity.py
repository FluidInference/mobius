"""Test conversion fidelity of KokoroNoise.mlpackage.

Run CoreMLFullNoiseModel.forward (the SAME nn.Module that was traced) in
pure PyTorch with the same fp32 inputs the CoreML model receives. If the
results MATCH within fp32 precision → the conversion is faithful and the
44% gap to upstream SineGen is purely the missing rand_ini. If they
DIFFER → the bug is in coremltools/CoreML's compilation of one of the
sub-ops (conv1d STFT, atan2, sqrt, leaky_relu, etc.).
"""
import importlib.util
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


# Re-import classes from convert-coreml.py
spec = importlib.util.spec_from_file_location("convert_coreml", "convert-coreml.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def diff(a, b, label):
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rms_a = float(np.sqrt(np.mean(a ** 2)))
    rms_b = float(np.sqrt(np.mean(b ** 2)))
    rms_d = float(np.sqrt(np.mean((a - b) ** 2)))
    rel = rms_d / max(rms_a, 1e-12)
    corr = float(np.corrcoef(a, b)[0, 1]) if rms_a > 0 and rms_b > 0 else 0.0
    max_d = float(np.max(np.abs(a - b)))
    print(f"  [{label:55s}] rel={rel:.3e} corr={corr:.7f} max|d|={max_d:.3e}")


def main():
    print("[setup] Loading PyTorch KModel + voice ...", file=sys.stderr)
    model = KModel(repo_id="hexgrad/Kokoro-82M-v1.1-zh"); model.eval()
    pipe = KPipeline(lang_code="z", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=model)

    voice_pack = np.frombuffer(
        open("build/ANE-zh/voices/zm_009.bin", "rb").read(),
        dtype=np.float32).reshape(510, 1, 256)
    voice_pack = torch.from_numpy(voice_pack.copy())

    phonemes = ""
    for _gs, ps, _tks in pipe("你好世界，今天天气很好。", voice="zm_009"):
        phonemes = ps; break
    ids = list(filter(lambda i: i is not None,
                       map(lambda p: model.vocab.get(p), phonemes)))
    input_ids = torch.LongTensor([[0, *ids, 0]])
    ref_s = voice_pack[max(min(len(phonemes) - 1, voice_pack.shape[0] - 1), 0)]
    s = ref_s[:, 128:]; style_timbre = ref_s[:, :128]

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

    print(f"[setup] T_a={T_a}, F0_pred shape={tuple(F0_pred.shape)}",
          file=sys.stderr)

    # Remove weight_norm to match what was traced for conversion.
    cc._remove_weight_norm(model.decoder)

    # Build CoreMLFullNoiseModel from the upstream generator.
    nm = cc.CoreMLFullNoiseModel(model.decoder.generator)
    nm.eval()

    # Run PyTorch.
    with torch.no_grad():
        pt_x_source_0, pt_x_source_1 = nm(F0_pred, style_timbre)

    # Run CoreML.
    ml = ct.models.MLModel("build/ANE-zh/KokoroNoise.mlpackage",
                            compute_units=ct.ComputeUnit.CPU_ONLY)
    out = ml.predict({
        "F0_curve": F0_pred.numpy().astype(np.float32),
        "style_timbre": style_timbre.numpy().astype(np.float32),
    })
    cm_x_source_0 = np.array(out["x_source_0"]).astype(np.float32)
    cm_x_source_1 = np.array(out["x_source_1"]).astype(np.float32)

    print("\n=== Conversion fidelity: PT(CoreMLFullNoiseModel) vs CoreML ===")
    diff(pt_x_source_0.numpy(), cm_x_source_0, "x_source_0")
    diff(pt_x_source_1.numpy(), cm_x_source_1, "x_source_1")

    # Also run sub-stages in PyTorch and compare to a hypothetically-converted
    # version of the noise stage. Specifically, run the source module + STFT,
    # so we can localize which sub-stage breaks.
    with torch.no_grad():
        f0_up = nm.f0_upsamp(F0_pred[:, None]).transpose(1, 2)
        sine_merge_pt, _, _ = nm.m_source(f0_up)
        har_source_pt = sine_merge_pt.transpose(1, 2).squeeze(1)
        spec_pt, phase_pt = nm.stft.transform(har_source_pt)
        har_pt = torch.cat([spec_pt, phase_pt], dim=1)
        x_post_conv_0_pt = nm.noise_convs[0](har_pt)
        x_after_res_0_pt = nm.noise_res[0](x_post_conv_0_pt, style_timbre)
        x_post_conv_1_pt = nm.noise_convs[1](har_pt)
        x_after_res_1_pt = nm.noise_res[1](x_post_conv_1_pt, style_timbre)

    print(f"  PT shapes: sine_merge={tuple(sine_merge_pt.shape)} "
          f"har_source={tuple(har_source_pt.shape)} har={tuple(har_pt.shape)}")
    print(f"  PT noise_convs[0]={tuple(x_post_conv_0_pt.shape)} "
          f"noise_convs[1]={tuple(x_post_conv_1_pt.shape)}")

    # Trace + convert just the source-module portion → har, then check fidelity.
    print("\n=== Sub-stage fidelity: SourceModule + STFT, no noise_conv/res ===")
    class JustSourceSTFT(nn.Module):
        def __init__(self, full_noise_model):
            super().__init__()
            self.f0_upsamp = full_noise_model.f0_upsamp
            self.m_source = full_noise_model.m_source
            self.stft = full_noise_model.stft
        def forward(self, F0_curve):
            f0 = self.f0_upsamp(F0_curve[:, None]).transpose(1, 2)
            har_source, _, _ = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            spec, phase = self.stft.transform(har_source)
            return torch.cat([spec, phase], dim=1)

    js = JustSourceSTFT(nm); js.eval()
    with torch.no_grad():
        traced = torch.jit.trace(js, (F0_pred,), strict=False)
    T2_dim = ct.RangeDim(lower_bound=2, upper_bound=4000, default=F0_pred.shape[1])
    ml_js = ct.convert(traced,
        inputs=[ct.TensorType(name="F0_curve", shape=(1, T2_dim), dtype=np.float32)],
        outputs=[ct.TensorType(name="har")],
        convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY)
    pathlib.Path("build/probe").mkdir(parents=True, exist_ok=True)
    ml_js.save("build/probe/JustSourceSTFT.mlpackage")
    ml_js = ct.models.MLModel("build/probe/JustSourceSTFT.mlpackage",
                               compute_units=ct.ComputeUnit.CPU_ONLY)
    out_js = ml_js.predict({"F0_curve": F0_pred.numpy().astype(np.float32)})
    cm_har = np.array(out_js["har"]).astype(np.float32)
    diff(har_pt.numpy(), cm_har, "har (sine_merge → STFT → cat)")

    # Split into spec and phase.
    n_spec = spec_pt.shape[1]
    diff(spec_pt.numpy(), cm_har[:, :n_spec, :], "spec (magnitude)")
    diff(phase_pt.numpy(), cm_har[:, n_spec:, :], "phase (atan2)")

    # If har matches but x_source_0 doesn't, the bug is in noise_convs[0]/noise_res[0].
    # If har doesn't match, the bug is in source module / STFT.

    # Trace + convert just the source-module portion (stop before STFT)
    print("\n=== Sub-stage fidelity: just SineGen output (sine_merge) ===")
    class JustSineMerge(nn.Module):
        def __init__(self, full_noise_model):
            super().__init__()
            self.f0_upsamp = full_noise_model.f0_upsamp
            self.m_source = full_noise_model.m_source
        def forward(self, F0_curve):
            f0 = self.f0_upsamp(F0_curve[:, None]).transpose(1, 2)
            har_source, _, _ = self.m_source(f0)
            return har_source.transpose(1, 2).squeeze(1)

    jsm = JustSineMerge(nm); jsm.eval()
    with torch.no_grad():
        traced = torch.jit.trace(jsm, (F0_pred,), strict=False)
    ml_jsm = ct.convert(traced,
        inputs=[ct.TensorType(name="F0_curve", shape=(1, T2_dim), dtype=np.float32)],
        outputs=[ct.TensorType(name="har_source")],
        convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY)
    ml_jsm.save("build/probe/JustSineMerge.mlpackage")
    ml_jsm = ct.models.MLModel("build/probe/JustSineMerge.mlpackage",
                                compute_units=ct.ComputeUnit.CPU_ONLY)
    out_jsm = ml_jsm.predict({"F0_curve": F0_pred.numpy().astype(np.float32)})
    cm_har_source = np.array(out_jsm["har_source"]).astype(np.float32)
    diff(har_source_pt.numpy(), cm_har_source, "har_source (post-tanh-linear)")


if __name__ == "__main__":
    main()
