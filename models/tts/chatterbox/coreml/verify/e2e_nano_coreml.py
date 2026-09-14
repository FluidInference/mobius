"""End-to-end Chatterbox Nano synthesis through the CoreML chain.

Runs WITHOUT the upstream PyTorch checkpoint once tables are exported:
models, embedding tables, the default voice, and the BPE tokenizer all come
from the CoreML artifact directory (or the published HF repo).

CoreML models: T3Nano-Prefill + T3Nano-Decode (I/O KV) + FlowMean + HiFT.
Host (this script — the Swift-port spec): BPE tokenization, embedding prep
from tables.safetensors (no positional add — wpe is in-graph), turbo
sampling (temperature/top-k/top-p/repetition-penalty, no CFG, no alignment
analyzer), SineGen randomness, bucket padding/cropping.

Usage:
    uv run python verify/e2e_nano_coreml.py --text "..." [--seed 42]
    uv run python verify/e2e_nano_coreml.py --models-dir build/upload/chatterbox-nano-coreml
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

HF_REPO = "FluidInference/chatterbox-nano-coreml"
T_PREFILL = 512
MAX_LEN = 1536
N_TOKENS = 500

START_SPEECH, STOP_SPEECH = 6561, 6562
SPEECH_VOCAB = 6561

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog near the river bank."


class Host:
    """Tables-only host: BPE tokenizer + embedding tables + default voice."""

    def __init__(self, root: Path):
        from transformers import AutoTokenizer
        from safetensors.torch import load_file

        self.tokenizer = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
        t = load_file(root / "tables" / "tables.safetensors")
        self.text_emb = t["text_emb"].float()              # (50276, 768)
        self.speech_emb = t["speech_emb"].float()          # (6563, 768)
        v = load_file(root / "tables" / "voice-default.safetensors")
        self.cond_emb = v["t3_cond_emb"].float()           # (1, 376, 768)
        self.prompt_token = v["prompt_token"].to(torch.int64)
        self.prompt_feat = v["prompt_feat"].float()
        self.embedding = v["embedding"].float()

    def prefill_embeds(self, text: str):
        """Replicates T3.prepare_input_embeds for inference_turbo: cond ++
        text ++ single BOS. No positional add (wpe is in-graph)."""
        from chatterbox.tts_turbo import punc_norm

        ids = self.tokenizer(punc_norm(text), return_tensors="pt").input_ids.view(-1)
        text_len = ids.shape[0]
        te = self.text_emb[ids]                            # (T, C)
        bos = self.speech_emb[START_SPEECH].view(1, -1)
        embeds = torch.cat([self.cond_emb[0], te, bos]).unsqueeze(0)  # (1, L, C)
        return embeds, self.cond_emb.shape[1], text_len

    def step_embed(self, token: int):
        return self.speech_emb[token].view(1, 1, -1).contiguous()


def resolve_models(models_dir: Path | None) -> Path:
    if models_dir is not None:
        return models_dir
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        HF_REPO,
        allow_patterns=[
            f"T3Nano-Prefill-T{T_PREFILL}-M{MAX_LEN}-fp16.mlpackage/*",
            f"T3Nano-Decode-M{MAX_LEN}-fp16.mlpackage/*",
            f"FlowMean-N{N_TOKENS}-fp16.mlpackage/*",
            f"HiFT-T{N_TOKENS * 2}-fp16.mlpackage/*",
            "tables/*", "tokenizer/*",
        ]))


def t3_generate(host: Host, root: Path, embeds, *,
                temperature=0.8, top_k=1000, top_p=0.95,
                repetition_penalty=1.2, max_new_tokens=1000, seed=42,
                compute_units="CPU_AND_GPU"):
    from transformers.generation.logits_process import (
        RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
        TopKLogitsWarper, TopPLogitsWarper)

    cu = getattr(ct.ComputeUnit, compute_units)
    T0 = embeds.shape[1]
    assert T0 <= T_PREFILL, f"prefill {T0} > bucket {T_PREFILL}"
    padded = torch.nn.functional.pad(embeds, (0, 0, 0, T_PREFILL - T0))

    mp = ct.models.MLModel(str(root / f"T3Nano-Prefill-T{T_PREFILL}-M{MAX_LEN}-fp16.mlpackage"),
                           compute_units=cu)
    t_pre0 = time.perf_counter()
    out = mp.predict({"inputs_embeds": padded.numpy(),
                      "input_len": np.array([T0], dtype=np.int32)})
    t_prefill = time.perf_counter() - t_pre0
    md = ct.models.MLModel(str(root / f"T3Nano-Decode-M{MAX_LEN}-fp16.mlpackage"),
                           compute_units=cu)

    # inference_turbo's processor order: temperature, top-k, top-p, rep-penalty
    processors = []
    if temperature > 0 and temperature != 1.0:
        processors.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        processors.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p))
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    gen = torch.Generator().manual_seed(seed)

    logits = torch.from_numpy(out["logits"])
    kv_k, kv_v = out["kv_k"], out["kv_v"]
    generated = []
    step_times = []

    for i in range(max_new_tokens):
        ids = (torch.tensor(generated, dtype=torch.int64).view(1, -1)
               if generated else torch.zeros(1, 0, dtype=torch.int64))
        proc = logits
        for p in processors:
            proc = p(ids, proc)
        probs = torch.softmax(proc, dim=-1)
        next_token = int(torch.multinomial(probs, 1, generator=gen))
        generated.append(next_token)
        if next_token == STOP_SPEECH:
            break
        emb = host.step_embed(next_token).numpy()
        t0 = time.perf_counter()
        out = md.predict({"inputs_embeds": emb, "kv_k": kv_k, "kv_v": kv_v,
                          "cur_len": np.array([T0 + i], dtype=np.int32)})
        step_times.append(time.perf_counter() - t0)
        kv_k, kv_v = out["kv_k_out"], out["kv_v_out"]
        logits = torch.from_numpy(out["logits"])

    tokens = torch.tensor(generated)
    tokens = tokens[tokens < SPEECH_VOCAB]
    med = sorted(step_times)[len(step_times) // 2] if step_times else 0.0
    print(f"  t3-nano: {len(generated)} tokens, prefill {t_prefill:.2f}s, "
          f"median decode {med * 1000:.0f} ms/step")
    return tokens, med


def s3gen_synthesize(host: Host, root: Path, speech_tokens, seed=1234,
                     compute_units="CPU_AND_GPU"):
    cu = getattr(ct.ComputeUnit, compute_units)
    # stock appends 3 silence tokens before the vocoder
    from chatterbox.models.s3gen.const import S3GEN_SIL
    silence = torch.tensor([S3GEN_SIL] * 3, dtype=speech_tokens.dtype)
    speech_tokens = torch.cat([speech_tokens, silence])

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

    mf = ct.models.MLModel(str(root / f"FlowMean-N{N_TOKENS}-fp16.mlpackage"),
                           compute_units=cu)
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

    mh = ct.models.MLModel(str(root / f"HiFT-T{M}-fp16.mlpackage"), compute_units=cu)
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
    return wav, t_flow, t_hift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=None,
                    help="local model dir (default: download from HF)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "build" / "e2e-nano")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--tag", default="en", help="output filename tag")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compute-units", default="CPU_AND_GPU",
                    choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU"])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_models(args.models_dir)
    host = Host(root)
    print(f"[e2e-nano] {args.text}")
    t_wall0 = time.perf_counter()
    embeds, len_cond, text_len = host.prefill_embeds(args.text)
    tokens, _ = t3_generate(host, root, embeds, seed=args.seed,
                            compute_units=args.compute_units)
    wav, _, _ = s3gen_synthesize(host, root, tokens,
                                 compute_units=args.compute_units)
    wall = time.perf_counter() - t_wall0
    dur = wav.shape[-1] / 24000
    print(f"[e2e-nano] wall {wall:.2f}s for {dur:.2f}s audio (RTFx {dur / wall:.2f})")
    out = args.out_dir / f"e2e_nano_{args.tag}.wav"
    sf.write(out, wav.squeeze(0).numpy(), 24000)
    print(f"[e2e-nano] wrote {out}")


if __name__ == "__main__":
    main()
