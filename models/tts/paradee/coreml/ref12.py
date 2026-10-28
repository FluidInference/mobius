"""PyTorch reference for Paradee-8M v1.2 (four voices): upstream export_paradee_v1_1.Paradee2 (text side + decoder A
trunk + frame-rate source-filter head, vhead.SourceFilterHead2), with the voice choosing one row of each half's
`style` table (text side [4, 32], decoder [4, 16]).

usage: uv run python ref12.py    checks the reference against the released fp32 ONNX graph with its
                                 RandomNormalLike node turned into a `noise` input, for every voice
"""
import json
import pathlib
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream/training/scripts"))

import paradee_ref as R  # noqa: E402  (strip, text_forward, durations, alignment)
from student import TextStudent  # noqa: E402
from vhead import FHOP, SR, VDecoder  # noqa: E402

W12 = HERE / "weights-v1.2"
CONFIG = json.load(open(W12 / "config.json"))
VOICES = CONFIG["voices"]  # model order: af_heart, bf_isabella, am_puck, bm_fable


def load12():
    """-> text side, decoder, text style table [V, 32], decoder style table [V, 16]."""
    t = torch.load(W12 / "pytorch/text_side.pt", map_location="cpu")
    d = torch.load(W12 / "pytorch/decoder.pt", map_location="cpu")
    assert t["voices"] == VOICES == d["voices"], (t["voices"], d["voices"], VOICES)
    ts = TextStudent("s+mlp").eval()
    sd = dict(t["model"])
    ts_tab = sd.pop("style")
    res = ts.load_state_dict(sd, strict=False)
    assert res.missing_keys == ["style"] and not res.unexpected_keys, res
    vd = VDecoder(head="v2").eval()
    sd = dict(d["model"])
    vd_tab = sd.pop("style")
    res = vd.load_state_dict(sd, strict=False)
    assert res.missing_keys == ["style"] and not res.unexpected_keys, res
    R.strip(ts)
    R.strip(vd)
    return ts, vd, ts_tab.float(), vd_tab.float()


def set_voice(ts, vd, ts_tab, vd_tab, voice):
    ts.style = nn.Parameter(ts_tab[voice : voice + 1].clone(), requires_grad=False)
    vd.style = nn.Parameter(vd_tab[voice : voice + 1].clone(), requires_grad=False)


def reference(ts, vd, ts_tab, vd_tab, ids, voice, speed=1.0, noise=None, seed=0):
    """Paradee2.forward for one voice with the head's noise given (or drawn from `seed`).
    -> audio [1, S], intermediates (d, dur, asr_tok, aln, en, asr, F0, noise)."""
    set_voice(ts, vd, ts_tab, vd_tab, voice)
    with torch.no_grad():
        d, dur, asr_tok = R.text_forward(ts, ids)
        aln = R.alignment(R.durations(dur, speed))
        en = d.transpose(-1, -2) @ aln
        asr = asr_tok @ aln
        F0, N = ts.predictor.F0Ntrain(en, ts.style)
        if noise is None:
            noise = torch.randn(1, F0.shape[1] * FHOP, generator=torch.Generator().manual_seed(seed))
        y = vd(asr, F0, N, noise=noise)
    return y, dict(d=d, dur=dur, asr_tok=asr_tok, aln=aln, en=en, asr=asr, F0=F0, N=N, noise=noise)


def phonemes_to_ids(ps):
    vocab = CONFIG["vocab"]
    return torch.LongTensor([[0] + [vocab[c] for c in ps if c in vocab][:510] + [0]])


def onnx_with_noise_input(src, dst):
    """Replace the graph's RandomNormalLike with a graph input `noise` [1, S] so runs are deterministic."""
    import onnx
    from onnx import TensorProto, helper

    g = onnx.load(src)
    rnd = [n for n in g.graph.node if n.op_type in ("RandomNormalLike", "RandomNormal")]
    assert len(rnd) == 1, [n.op_type for n in rnd]
    out = rnd[0].output[0]
    g.graph.node.remove(rnd[0])
    g.graph.input.append(helper.make_tensor_value_info("noise", TensorProto.FLOAT, [1, "S"]))
    for n in g.graph.node:
        for i, x in enumerate(n.input):
            if x == out:
                n.input[i] = "noise"
    onnx.save(g, dst)
    return dst


def main():
    import onnxruntime as ort

    ts, vd, ts_tab, vd_tab = load12()
    print(f"voices {VOICES}; style tables text {tuple(ts_tab.shape)} decoder {tuple(vd_tab.shape)}")
    lines = ["pəɹˈædi ɪz ɐ smˈɔl vˈYs ðæt ɹˈʌnz ˈɛniwˌɛɹ.", "ðə bˈɑːθ ɪz tˈuː hˈɒt, sˈɛd ðə ɡˈɜːl."]
    sessions = {}
    # fp32 only: int8 rounds some durations differently, so its noise length differs from the reference's
    for name in ("paradee.onnx",):
        p = onnx_with_noise_input(W12 / "onnx" / name, HERE / f"weights-v1.2/{name[:-5]}_noise_in.onnx")
        sessions[name] = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    for ps in lines:
        ids = phonemes_to_ids(ps)
        for v, voice in enumerate(VOICES):
            y, mid = reference(ts, vd, ts_tab, vd_tab, ids, v, speed=1.0, seed=v)
            for name, sess in sessions.items():
                feed = {"input_ids": ids.numpy(), "speed": np.ones(1, np.float32), "voice": np.array([v], np.int64),
                        "noise": mid["noise"].numpy()}
                yo = sess.run(None, feed)[0]
                n = min(yo.shape[1], y.shape[1])
                err = np.abs(yo[:, :n] - y.numpy()[:, :n]).max()
                snr = 10 * np.log10((y.numpy() ** 2).sum() / max(((yo[:, :n] - y.numpy()[:, :n]) ** 2).sum(), 1e-20))
                print(f"{voice:12s} T={ids.shape[1]:3d} {name:18s} len onnx {yo.shape[1]} torch {y.shape[1]}  "
                      f"max|diff| {err:.2e}  SNR {snr:5.1f} dB")


if __name__ == "__main__":
    main()
