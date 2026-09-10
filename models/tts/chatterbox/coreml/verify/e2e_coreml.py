"""End-to-end Chatterbox Multilingual synthesis through the CoreML chain.

Runs WITHOUT the upstream PyTorch checkpoint: models, embedding tables, the
default voice, and the tokenizer all come from the published CoreML repo
(FluidInference/chatterbox-multilingual-coreml) — ~2.3 GB on first run.

CoreML models: T3-Prefill + T3-Decode (I/O KV) + Flow + HiFT.
Host (this script — the Swift-port spec): tokenization, embedding prep from
tables.safetensors, CFG combine, AlignmentAnalyzerPort fed by the exported
attention rows, HF logits processors, sampling, SineGen randomness, and
bucket padding/cropping.

Usage:
    uv run python verify/e2e_coreml.py --lang en [--text "..."] [--seed 42]
    uv run python verify/e2e_coreml.py --models-dir build/upload/chatterbox-multilingual-coreml
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

HF_REPO = "FluidInference/chatterbox-multilingual-coreml"
T_PREFILL = 256
MAX_LEN = 1024
N_TOKENS = 500

START_TEXT, STOP_TEXT = 255, 0
START_SPEECH, STOP_SPEECH = 6561, 6562
SPEECH_VOCAB = 6561
LEN_COND = 34

SENTENCES = {
    "en": "The quick brown fox jumps over the lazy dog near the river bank.",
    "de": "Der schnelle braune Fuchs springt über den faulen Hund am Flussufer.",
    "fr": "Le renard brun rapide saute par-dessus le chien paresseux près de la rivière.",
}


class Host:
    """Tables-only host: tokenizer + embedding tables + default voice."""

    def __init__(self, root: Path):
        from chatterbox.models.tokenizers.tokenizer import MTLTokenizer
        from safetensors.torch import load_file

        self.tokenizer = MTLTokenizer(str(root / "tokenizer" / "grapheme_mtl_merged_expanded_v1.json"))
        t = load_file(root / "tables" / "tables.safetensors")
        self.text_emb = t["text_emb"].float()
        self.speech_emb = t["speech_emb"].float()
        self.text_pos = t["text_pos_emb"].float()
        self.speech_pos = t["speech_pos_emb"].float()
        v = load_file(root / "tables" / "voice-default.safetensors")
        self.cond_emb = v["t3_cond_emb"].float()          # (1, 34, 1024)
        self.prompt_token = v["prompt_token"].to(torch.int64)
        self.prompt_feat = v["prompt_feat"].float()
        self.embedding = v["embedding"].float()

    def prefill_embeds(self, text: str, lang: str):
        """Replicates T3.prepare_input_embeds + inference()'s BOS append,
        including the stock double-BOS tail and the CFG text-zeroing order
        (text emb zeroed BEFORE positional add)."""
        from chatterbox.mtl_tts import punc_norm

        ids = self.tokenizer.text_to_tokens(punc_norm(text), language_id=lang).view(-1)
        ids = torch.cat([torch.tensor([START_TEXT]), ids, torch.tensor([STOP_TEXT])])
        text_len = ids.shape[0]

        te = self.text_emb[ids]                            # (T, C)
        pos = self.text_pos[:text_len]
        text_cond = te + pos
        text_uncond = torch.zeros_like(te) + pos           # CFG row: zeroed text, kept pos

        bos = self.speech_emb[START_SPEECH] + self.speech_pos[0]   # (C,)
        rows = []
        for txt in (text_cond, text_uncond):
            rows.append(torch.cat([self.cond_emb[0], txt, bos.view(1, -1), bos.view(1, -1)]))
        embeds = torch.stack(rows)                         # (2, 34+T+2, C)
        return embeds, LEN_COND, text_len

    def step_embed(self, token: int, step: int):
        e = self.speech_emb[token] + self.speech_pos[step + 1]
        return e.view(1, 1, -1).expand(2, 1, -1).contiguous()


def resolve_models(models_dir: Path | None) -> Path:
    if models_dir is not None:
        return models_dir
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        HF_REPO,
        allow_patterns=[
            "T3-Prefill-T256-M1024-fp16.mlpackage/*",
            "T3-Decode-M1024-fp16.mlpackage/*",
            f"Flow-N{N_TOKENS}-fp16.mlpackage/*",
            f"HiFT-T{N_TOKENS * 2}-fp16.mlpackage/*",
            "tables/*", "tokenizer/*",
        ]))


def t3_generate(host: Host, root: Path, embeds, len_cond, text_len, *,
                cfg_weight=0.5, temperature=0.8, repetition_penalty=2.0,
                min_p=0.05, top_p=1.0, max_new_tokens=1000, seed=42):
    from transformers.generation.logits_process import (
        MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper)

    T0 = embeds.shape[1]
    padded = torch.nn.functional.pad(embeds, (0, 0, 0, T_PREFILL - T0))

    mp = ct.models.MLModel(str(root / f"T3-Prefill-T{T_PREFILL}-M{MAX_LEN}-fp16.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_AND_GPU)
    t_pre0 = time.perf_counter()
    out = mp.predict({"inputs_embeds": padded.numpy(),
                      "input_len": np.array([T0], dtype=np.int32)})
    t_prefill = time.perf_counter() - t_pre0
    md = ct.models.MLModel(str(root / f"T3-Decode-M{MAX_LEN}-fp16.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_AND_GPU)

    analyzer = AlignmentAnalyzerPort((len_cond, len_cond + text_len),
                                     eos_idx=STOP_SPEECH)
    rep = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
    min_p_warper = MinPLogitsWarper(min_p=min_p)
    top_p_warper = TopPLogitsWarper(top_p=top_p)
    gen = torch.Generator().manual_seed(seed)

    logits2 = torch.from_numpy(out["logits"])
    align = torch.from_numpy(out["align_attn"])[:, :, :T0]
    kv_k, kv_v = out["kv_k"], out["kv_v"]
    generated_ids = torch.tensor([[START_SPEECH]])
    predicted = []
    step_times = []

    for i in range(max_new_tokens):
        cond, uncond = logits2[0:1], logits2[1:2]
        logits = cond + cfg_weight * (cond - uncond)
        logits = analyzer.step(logits, align, next_token=int(generated_ids[0, -1]))
        logits = rep(generated_ids[:1], logits)
        if temperature != 1.0:
            logits = logits / temperature
        logits = min_p_warper(generated_ids[:1], logits)
        logits = top_p_warper(generated_ids[:1], logits)
        probs = torch.softmax(logits, dim=-1)
        next_token = int(torch.multinomial(probs, 1, generator=gen))
        predicted.append(next_token)
        generated_ids = torch.cat([generated_ids, torch.tensor([[next_token]])], dim=1)
        if next_token == STOP_SPEECH:
            break
        emb = host.step_embed(next_token, i).numpy()
        t0 = time.perf_counter()
        out = md.predict({"inputs_embeds": emb, "kv_k": kv_k, "kv_v": kv_v,
                          "cur_len": np.array([T0 + i], dtype=np.int32)})
        step_times.append(time.perf_counter() - t0)
        kv_k, kv_v = out["kv_k_out"], out["kv_v_out"]
        logits2 = torch.from_numpy(out["logits"])
        align = torch.from_numpy(out["align_attn"])[:, :T0 + i + 1]

    tokens = torch.tensor(predicted)
    tokens = tokens[tokens < SPEECH_VOCAB]
    med = sorted(step_times)[len(step_times) // 2] if step_times else 0.0
    print(f"  t3: {len(predicted)} tokens, prefill {t_prefill:.2f}s, "
          f"median decode {med * 1000:.0f} ms/step")
    return tokens


def s3gen_synthesize(host: Host, root: Path, speech_tokens, seed=1234):
    P = host.prompt_token.shape[1]
    tokens_real = torch.cat([host.prompt_token, speech_tokens.view(1, -1)], dim=1)
    T_real = tokens_real.shape[1]
    assert T_real <= N_TOKENS, f"{T_real} tokens exceed flow bucket {N_TOKENS}"
    M = N_TOKENS * 2

    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(1, 80, M)
    z[:, :, :2 * T_real] = torch.randn(1, 80, 2 * T_real, generator=g)

    tokens = torch.nn.functional.pad(tokens_real, (0, N_TOKENS - T_real)).to(torch.int32)
    prompt_feat = torch.zeros(1, M, 80)
    prompt_feat[:, :host.prompt_feat.shape[1]] = host.prompt_feat

    mf = ct.models.MLModel(str(root / f"Flow-N{N_TOKENS}-fp16.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_AND_GPU)
    t0 = time.perf_counter()
    out = mf.predict({
        "tokens": tokens.numpy(),
        "token_len": np.array([T_real], dtype=np.int32),
        "prompt_len": np.array([P], dtype=np.int32),
        "prompt_feat": prompt_feat.numpy(),
        "embedding": host.embedding.numpy(),
        "z": z.numpy()})
    t_flow = time.perf_counter() - t0
    mel_full = torch.from_numpy(out["mel"])
    T_mel = 2 * (T_real - P)

    mel_pad = torch.zeros(1, 80, M)
    mel_pad[:, :, :T_mel] = mel_full[:, :, 2 * P:2 * T_real]
    phase = torch.zeros(1, 9, 1)
    phase[0, 1:, 0] = torch.rand(8, generator=g) * 2 * np.pi - np.pi
    noise = torch.randn(1, 9, M * 480, generator=g)

    mh = ct.models.MLModel(str(root / f"HiFT-T{M}-fp16.mlpackage"),
                           compute_units=ct.ComputeUnit.CPU_AND_GPU)
    t0 = time.perf_counter()
    out = mh.predict({"mel": mel_pad.numpy(), "phase_vec": phase.numpy(),
                      "noise": noise.numpy()})
    t_hift = time.perf_counter() - t0
    wav = torch.from_numpy(out["audio"])[:, :T_mel * 480]

    n_trim = 24000 // 50
    fade = torch.zeros(2 * n_trim)
    fade[n_trim:] = (torch.cos(torch.linspace(np.pi, 0, n_trim)) + 1) / 2
    wav[:, :2 * n_trim] *= fade
    print(f"  s3gen: flow {t_flow:.2f}s, hift {t_hift:.2f}s, "
          f"{wav.shape[-1] / 24000:.2f}s audio")
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=None,
                    help="local model dir (default: download from HF)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "build" / "e2e")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--text", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_models(args.models_dir)
    host = Host(root)
    text = args.text or SENTENCES.get(args.lang, SENTENCES["en"])
    print(f"[e2e] {args.lang}: {text}")
    embeds, len_cond, text_len = host.prefill_embeds(text, args.lang)
    tokens = t3_generate(host, root, embeds, len_cond, text_len, seed=args.seed)
    wav = s3gen_synthesize(host, root, tokens)
    out = args.out_dir / f"e2e_{args.lang}.wav"
    sf.write(out, wav.squeeze(0).numpy(), 24000)
    print(f"[e2e] wrote {out}")


if __name__ == "__main__":
    main()
