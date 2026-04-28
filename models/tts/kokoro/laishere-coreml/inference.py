"""Standalone end-to-end CoreML inference for Kokoro TTS.

Loads the 7 .mlpackage files produced by convert-coreml.py, runs G2P (or
takes pre-computed phonemes), and writes a 24 kHz mono WAV.

Usage:
    uv run python inference.py \
        --models-dir build/laishere-kokoro \
        --text "Hello world" \
        --output /tmp/out.wav

    # Skip G2P with pre-computed IPA phonemes (matches iOS app flow):
    uv run python inference.py --models-dir build/laishere-kokoro \
        --phonemes "həlˈoʊ wˈɜːld" --output /tmp/out.wav

Per-stage timings are printed to stderr.
"""
import argparse
import json
import pathlib
import sys
import time

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

from kokoro import KModel
from kokoro.pipeline import KPipeline

SR = 24000


def encode_phonemes(vocab, phonemes):
    """phonemes (str) → int32 [1, T_enc] with BOS=0/EOS=0 wrap."""
    ids = [vocab[ch] for ch in phonemes if ch in vocab]
    return np.array([[0, *ids, 0]], dtype=np.int32)


def load_vocab(models_dir, model):
    """Prefer vocab.json next to the models; fall back to PyTorch model.vocab."""
    vp = models_dir / "vocab.json"
    if vp.exists():
        return json.loads(vp.read_text())
    return dict(model.vocab)


def load_voice_pack(models_dir, voice, pipe=None):
    """Load voice pack as torch [510, 1, 256]. Prefer flat .bin if present."""
    bin_path = models_dir / f"{voice}.bin"
    if bin_path.exists():
        flat = np.frombuffer(bin_path.read_bytes(), dtype=np.float32).reshape(510, 1, 256)
        return torch.from_numpy(flat.copy())
    if pipe is None:
        raise FileNotFoundError(
            f"{bin_path} not found and no PyTorch pipeline provided to fall back to load_voice"
        )
    return pipe.load_voice(voice)


def main():
    p = argparse.ArgumentParser(description="End-to-end CoreML synth for Kokoro")
    p.add_argument("--models-dir", type=pathlib.Path, required=True,
                   help="Directory containing the 7 .mlpackage files")
    p.add_argument("--text", help="Text to synthesize (uses Kokoro G2P)")
    p.add_argument("--phonemes", help="Pre-computed IPA phonemes (skips G2P)")
    p.add_argument("--voice", default="af_heart", help="Voice id (default: af_heart)")
    p.add_argument("--lang", default="a", help="Kokoro lang code for G2P (default: a)")
    p.add_argument("--output", type=pathlib.Path, required=True,
                   help="Output WAV path (24 kHz mono float32)")
    p.add_argument("--n-warmup", type=int, default=2, help="Warmup runs before timed run")
    args = p.parse_args()

    if not args.text and not args.phonemes:
        p.error("must provide --text or --phonemes")
    if not args.models_dir.exists():
        p.error(f"models-dir {args.models_dir} does not exist")

    models_dir = args.models_dir

    # Phonemize (only if --text and no PyTorch model needed for vocab fallback either)
    needs_pipe = bool(args.text) or not (models_dir / "vocab.json").exists() \
        or not (models_dir / f"{args.voice}.bin").exists()
    pipe = None
    pt_model = None
    if needs_pipe:
        print("[setup] Loading Kokoro PyTorch (G2P / vocab fallback)...", file=sys.stderr)
        pt_model = KModel(); pt_model.eval()
        pipe = KPipeline(lang_code=args.lang, model=pt_model)

    if args.phonemes:
        phonemes = args.phonemes
    else:
        # Reuse benchmark.phonemize_for_benchmark for parity with dump-benchmark-data.py
        from benchmark import phonemize_for_benchmark
        phonemes = phonemize_for_benchmark(pipe, args.text)
    print(f"[g2p] phonemes ({len(phonemes)} chars): {phonemes!r}", file=sys.stderr)

    vocab = load_vocab(models_dir, pt_model)
    voice_pack = load_voice_pack(models_dir, args.voice, pipe=pipe)

    input_ids_np = encode_phonemes(vocab, phonemes)
    T_enc = input_ids_np.shape[1]
    row = max(min(len(phonemes) - 1, voice_pack.shape[0] - 1), 0)
    ref_s = voice_pack[row]                   # [1, 256]
    style_s = ref_s[:, 128:].numpy().astype(np.float16)
    style_timbre = ref_s[:, :128].numpy()
    attention_mask = np.ones((1, T_enc), dtype=np.int32)

    print(f"[setup] Loading 7 CoreML models from {models_dir}...", file=sys.stderr)
    CU_NE = ct.ComputeUnit.CPU_AND_NE
    CU_ALL = ct.ComputeUnit.ALL
    m_albert = ct.models.MLModel(str(models_dir / "KokoroAlbert.mlpackage"),     compute_units=CU_NE)
    m_post   = ct.models.MLModel(str(models_dir / "KokoroPostAlbert.mlpackage"), compute_units=CU_NE)
    m_align  = ct.models.MLModel(str(models_dir / "KokoroAlignment.mlpackage"),  compute_units=CU_NE)
    m_pros   = ct.models.MLModel(str(models_dir / "KokoroProsody.mlpackage"),    compute_units=CU_ALL)
    m_noise  = ct.models.MLModel(str(models_dir / "KokoroNoise.mlpackage"),      compute_units=CU_ALL)
    m_voc    = ct.models.MLModel(str(models_dir / "KokoroVocoder.mlpackage"),    compute_units=CU_NE)
    m_tail   = ct.models.MLModel(str(models_dir / "KokoroTail.mlpackage"),       compute_units=CU_ALL)

    def run_chain(timings=None):
        def _t():
            return time.perf_counter() if timings is not None else None

        t0 = _t()
        o1 = m_albert.predict({"input_ids": input_ids_np, "attention_mask": attention_mask})
        if timings is not None: timings["Albert"] = (time.perf_counter() - t0) * 1000

        t0 = _t()
        o2 = m_post.predict({
            "bert_dur": np.array(o1["bert_dur"]).astype(np.float16),
            "input_ids": input_ids_np,
            "style_s": style_s,
            "speed": np.array([1.0], dtype=np.float16),
            "attention_mask": attention_mask,
        })
        if timings is not None: timings["PostAlbert"] = (time.perf_counter() - t0) * 1000

        dur = np.array(o2["duration"]).flatten()
        pd = np.round(dur).clip(min=1).astype(np.int32).reshape(1, -1)

        t0 = _t()
        o3 = m_align.predict({
            "pred_dur": pd,
            "d": np.array(o2["d"]).astype(np.float16),
            "t_en": np.array(o2["t_en"]).astype(np.float16),
        })
        if timings is not None: timings["Alignment"] = (time.perf_counter() - t0) * 1000

        t0 = _t()
        o4 = m_pros.predict({
            "en": np.array(o3["en"]).astype(np.float16),
            "style_s": style_s,
        })
        if timings is not None: timings["Prosody"] = (time.perf_counter() - t0) * 1000

        t0 = _t()
        o5 = m_noise.predict({
            "F0_curve": np.array(o4["F0"]).astype(np.float32),
            "style_timbre": style_timbre.astype(np.float32),
        })
        if timings is not None: timings["Noise"] = (time.perf_counter() - t0) * 1000

        t0 = _t()
        o6 = m_voc.predict({
            "asr": np.array(o3["asr"]).astype(np.float16),
            "F0_curve": np.array(o4["F0"]).astype(np.float16),
            "N_pred": np.array(o4["N"]).astype(np.float16),
            "x_source_0": np.array(o5["x_source_0"]).astype(np.float16),
            "x_source_1": np.array(o5["x_source_1"]).astype(np.float16),
            "style_timbre": style_timbre.astype(np.float16),
        })
        if timings is not None: timings["Vocoder"] = (time.perf_counter() - t0) * 1000

        t0 = _t()
        o7 = m_tail.predict({"x_pre": np.array(o6["x_pre"]).astype(np.float32)})
        if timings is not None: timings["Tail"] = (time.perf_counter() - t0) * 1000

        return o7

    print(f"[chain] warmup x{args.n_warmup}...", file=sys.stderr)
    for _ in range(args.n_warmup):
        run_chain()

    timings = {}
    t0 = time.perf_counter()
    o7 = run_chain(timings=timings)
    chain_ms = (time.perf_counter() - t0) * 1000

    audio = np.array(o7["audio"]).flatten().astype(np.float32)
    audio_dur_s = len(audio) / SR
    speedup = audio_dur_s / (chain_ms / 1000)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, SR, subtype="FLOAT")

    print(f"\n[result] T_enc={T_enc}  audio={audio_dur_s:.2f}s  chain={chain_ms:.1f}ms  "
          f"speed={speedup:.1f}x", file=sys.stderr)
    for name in ["Albert", "PostAlbert", "Alignment", "Prosody", "Noise", "Vocoder", "Tail"]:
        print(f"  {name:11s} {timings[name]:6.2f}ms", file=sys.stderr)
    print(f"[wrote] {args.output} ({len(audio)} samples @ {SR} Hz)", file=sys.stderr)


if __name__ == "__main__":
    main()
