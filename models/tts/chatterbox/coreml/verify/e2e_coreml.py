"""End-to-end Chatterbox Multilingual synthesis through the CoreML chain.

CoreML models: T3-Prefill + T3-Decode (I/O KV or MLState) + Flow + HiFT.
Host (this script — the Swift-port spec): tokenization, embedding prep
(tables from the PyTorch checkpoint), CFG combine, AlignmentAnalyzerPort fed
by the exported attention rows, HF logits processors, sampling, SineGen
randomness, and bucket padding/cropping.

Usage:
    uv run python verify/e2e_coreml.py --lang en --text "..." [--stateful]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from verify.analyzer_port import AlignmentAnalyzerPort  # noqa: E402

T_PREFILL = 256
MAX_LEN = 1024
N_TOKENS = 500

SENTENCES = {
    "en": "The quick brown fox jumps over the lazy dog near the river bank.",
    "de": "Der schnelle braune Fuchs springt über den faulen Hund am Flussufer.",
    "fr": "Le renard brun rapide saute par-dessus le chien paresseux près de la rivière.",
}


def load_model():
    from chatterbox.mtl_tts import REPO_ID, ChatterboxMultilingualTTS
    from huggingface_hub import snapshot_download

    ckpt_dir = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                        "grapheme_mtl_merged_expanded_v1.json", "conds.pt",
                        "Cangjie5_TC.json"],
    )
    return ChatterboxMultilingualTTS.from_local(ckpt_dir, "cpu")


def build_prefill(model, text: str, lang: str, cfg_weight: float = 0.5):
    import torch.nn.functional as F
    from chatterbox.mtl_tts import punc_norm

    t3 = model.t3
    text_tokens = model.tokenizer.text_to_tokens(punc_norm(text), language_id=lang)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    sot, eot = t3.hp.start_text_token, t3.hp.stop_text_token
    text_tokens = F.pad(F.pad(text_tokens, (1, 0), value=sot), (0, 1), value=eot)
    with torch.no_grad():
        embeds, len_cond = t3.prepare_input_embeds(
            t3_cond=model.conds.t3, text_tokens=text_tokens,
            speech_tokens=t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1]),
            cfg_weight=cfg_weight)
        bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long)
        bos_embed = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
        embeds = torch.cat([embeds, torch.cat([bos_embed, bos_embed])], dim=1)
    return embeds, len_cond, text_tokens.shape[1]


def t3_generate(model, t3_dir: Path, embeds, len_cond, text_len, *,
                stateful=False, cfg_weight=0.5, temperature=0.8,
                repetition_penalty=2.0, min_p=0.05, top_p=1.0,
                max_new_tokens=1000, seed=42):
    from transformers.generation.logits_process import (
        MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper)

    t3 = model.t3
    T0 = embeds.shape[1]
    padded = torch.nn.functional.pad(embeds, (0, 0, 0, T_PREFILL - T0))
    input_len = np.array([T0], dtype=np.int32)

    mp = ct.models.MLModel(str(t3_dir / f"T3-Prefill-T{T_PREFILL}-M{MAX_LEN}-fp16.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_AND_GPU)
    t_pre0 = time.perf_counter()
    out = mp.predict({"inputs_embeds": padded.numpy(), "input_len": input_len})
    t_prefill = time.perf_counter() - t_pre0

    if stateful:
        raise NotImplementedError("stateful e2e needs Swift-side state seeding")
    md = ct.models.MLModel(str(t3_dir / f"T3-Decode-M{MAX_LEN}-fp16.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_AND_GPU)

    analyzer = AlignmentAnalyzerPort((len_cond, len_cond + text_len),
                                     eos_idx=t3.hp.stop_speech_token)
    rep = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
    min_p_warper = MinPLogitsWarper(min_p=min_p)
    top_p_warper = TopPLogitsWarper(top_p=top_p)
    gen = torch.Generator().manual_seed(seed)

    logits2 = torch.from_numpy(out["logits"])
    align = torch.from_numpy(out["align_attn"])[:, :, :T0]     # (3, 2, T0)
    kv_k, kv_v = out["kv_k"], out["kv_v"]
    generated_ids = torch.tensor([[t3.hp.start_speech_token]])
    predicted = []
    step_times = []

    for i in range(max_new_tokens):
        cond, uncond = logits2[0:1], logits2[1:2]
        logits = cond + cfg_weight * (cond - uncond)
        last_token = int(generated_ids[0, -1])
        logits = analyzer.step(logits, align, next_token=last_token)
        logits = rep(generated_ids[:1], logits)
        if temperature != 1.0:
            logits = logits / temperature
        logits = min_p_warper(generated_ids[:1], logits)
        logits = top_p_warper(generated_ids[:1], logits)
        probs = torch.softmax(logits, dim=-1)
        next_token = int(torch.multinomial(probs, 1, generator=gen))
        predicted.append(next_token)
        generated_ids = torch.cat([generated_ids,
                                   torch.tensor([[next_token]])], dim=1)
        if next_token == t3.hp.stop_speech_token:
            break
        with torch.no_grad():
            emb = t3.speech_emb(torch.tensor([[next_token]])) \
                + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        emb = torch.cat([emb, emb]).numpy()
        t0 = time.perf_counter()
        out = md.predict({"inputs_embeds": emb, "kv_k": kv_k, "kv_v": kv_v,
                          "cur_len": np.array([T0 + i], dtype=np.int32)})
        step_times.append(time.perf_counter() - t0)
        kv_k, kv_v = out["kv_k_out"], out["kv_v_out"]
        logits2 = torch.from_numpy(out["logits"])
        align = torch.from_numpy(out["align_attn"])[:, :T0 + i + 1]

    tokens = torch.tensor(predicted)
    tokens = tokens[tokens < 6561]  # drop_invalid_tokens (SPEECH_VOCAB_SIZE)
    med = sorted(step_times)[len(step_times) // 2] if step_times else 0.0
    print(f"  t3: {len(predicted)} tokens, prefill {t_prefill:.2f}s, "
          f"median decode {med * 1000:.0f} ms/step")
    return tokens


def s3gen_synthesize(model, s3gen_dir: Path, speech_tokens, seed=1234):
    ref = model.conds.gen
    P = ref["prompt_token"].shape[1]
    tokens_real = torch.cat([ref["prompt_token"],
                             speech_tokens.view(1, -1)], dim=1)
    T_real = tokens_real.shape[1]
    assert T_real <= N_TOKENS, f"{T_real} tokens exceed flow bucket {N_TOKENS}"
    M = N_TOKENS * 2

    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(1, 80, M)
    z[:, :, :2 * T_real] = torch.randn(1, 80, 2 * T_real, generator=g)

    tokens = torch.nn.functional.pad(tokens_real, (0, N_TOKENS - T_real)).to(torch.int32)
    prompt_feat = torch.zeros(1, M, 80)
    prompt_feat[:, :ref["prompt_feat"].shape[1]] = ref["prompt_feat"]

    mf = ct.models.MLModel(str(s3gen_dir / f"Flow-N{N_TOKENS}-fp16.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_AND_GPU)
    t0 = time.perf_counter()
    out = mf.predict({
        "tokens": tokens.numpy(),
        "token_len": np.array([T_real], dtype=np.int32),
        "prompt_len": np.array([P], dtype=np.int32),
        "prompt_feat": prompt_feat.numpy(),
        "embedding": ref["embedding"].numpy(),
        "z": z.numpy()})
    t_flow = time.perf_counter() - t0
    mel_full = torch.from_numpy(out["mel"])
    T_mel = 2 * (T_real - P)

    # HiFT bucket is M mel frames; valid mel occupies [2P, 2T_real)
    mel_pad = torch.zeros(1, 80, M)
    mel_pad[:, :, :T_mel] = mel_full[:, :, 2 * P:2 * T_real]
    phase = torch.zeros(1, 9, 1)
    phase[0, 1:, 0] = torch.rand(8, generator=g) * 2 * np.pi - np.pi
    noise = torch.randn(1, 9, M * 480, generator=g)

    mh = ct.models.MLModel(str(s3gen_dir / f"HiFT-T{M}-fp16.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_AND_GPU)
    t0 = time.perf_counter()
    out = mh.predict({"mel": mel_pad.numpy(), "phase_vec": phase.numpy(),
                      "noise": noise.numpy()})
    t_hift = time.perf_counter() - t0
    wav = torch.from_numpy(out["audio"])[:, :T_mel * 480]

    # trim_fade from S3Token2Wav
    n_trim = 24000 // 50
    fade = torch.zeros(2 * n_trim)
    fade[n_trim:] = (torch.cos(torch.linspace(np.pi, 0, n_trim)) + 1) / 2
    wav[:, :2 * n_trim] *= fade
    print(f"  s3gen: flow {t_flow:.2f}s, hift {t_hift:.2f}s, "
          f"{wav.shape[-1] / 24000:.2f}s audio")
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t3-dir", type=Path, default=HERE / "build" / "t3")
    ap.add_argument("--s3gen-dir", type=Path, default=HERE / "build" / "s3gen")
    ap.add_argument("--out-dir", type=Path, default=HERE / "build" / "e2e")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--text", default=None)
    ap.add_argument("--stateful", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    text = args.text or SENTENCES[args.lang]
    print(f"[e2e] loading checkpoint tables...")
    model = load_model()
    print(f"[e2e] {args.lang}: {text}")
    embeds, len_cond, text_len = build_prefill(model, text, args.lang)
    tokens = t3_generate(model, args.t3_dir, embeds, len_cond, text_len,
                         stateful=args.stateful, seed=args.seed)
    wav = s3gen_synthesize(model, args.s3gen_dir, tokens)
    out = args.out_dir / f"e2e_{args.lang}.wav"
    sf.write(out, wav.squeeze(0).numpy(), 24000)
    print(f"[e2e] wrote {out}")


if __name__ == "__main__":
    main()
