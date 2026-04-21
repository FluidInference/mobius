"""Full end-to-end CoreML TTS test.

Pipeline (everything except text-frontend runs through CoreML):
    text + prompt_wav
      └─ Python frontend  → token IDs, lm_input_embeds, prompt_mel, spk_emb
      └─ LLM-Prefill CoreML → initial KV + first logits
      └─ loop: LLM-Decode CoreML + ras_sampling → speech tokens
      └─ Flow CoreML → mel
      └─ HiFT CoreML → WAV

Usage:
    uv run python verify/test_coreml_e2e.py \
        --tts-text '希望你以后能够做的比我还好用' \
        --output-wav build/wavs/e2e_coreml.wav
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "CosyVoice"))
sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))

import coremltools as ct

from src.text_frontend import build_frontend_inputs


BUILD = ROOT / "build"

# Compile-time constants (must match the mlpackages we've converted).
T_PREFILL = 256            # LLM-Prefill T
LLM_MAX_LEN = 768          # LLM KV cache max_len
FLOW_N_TOTAL = 125         # Flow mlpackage total tokens
HIFT_T = 250               # HiFT mlpackage frames
SPEECH_VOCAB = 6761
SPEECH_TOKEN_SIZE = 6561   # EOS is 6561, stop ids are [6561, 6561+200)
STOP_IDS = set(range(SPEECH_TOKEN_SIZE, SPEECH_TOKEN_SIZE + 200))


# --------------------------------------------------------------------------- #
#  ras_sampling (pure torch, port of CosyVoice's utils.common.ras_sampling)
# --------------------------------------------------------------------------- #

def _nucleus_sampling(logp: torch.Tensor, top_p: float, top_k: int) -> int:
    probs = logp.softmax(dim=0)
    sorted_value, sorted_idx = probs.sort(descending=True, stable=True)
    sel_idx, sel_prob, cum = [], [], 0.0
    for v, i in zip(sorted_value.tolist(), sorted_idx.tolist()):
        if cum < top_p and len(sel_prob) < top_k:
            cum += v
            sel_prob.append(v)
            sel_idx.append(i)
        else:
            break
    sel_prob_t = torch.tensor(sel_prob)
    pick = int(torch.multinomial(sel_prob_t, 1, replacement=True).item())
    return sel_idx[pick]


def ras_sampling(logp: torch.Tensor, decoded_tokens: list[int], *,
                 top_p: float = 0.8, top_k: int = 25,
                 win_size: int = 10, tau_r: float = 0.1) -> int:
    """Repetition-aware sampling (VALL-E 2 style)."""
    top_id = _nucleus_sampling(logp, top_p, top_k)
    recent = decoded_tokens[-win_size:]
    rep = sum(1 for t in recent if t == top_id)
    if rep >= win_size * tau_r:
        logp_masked = logp.clone()
        logp_masked[top_id] = -float("inf")
        # fall back to pure random sampling
        probs = logp_masked.softmax(dim=0)
        top_id = int(torch.multinomial(probs, 1, replacement=True).item())
    return top_id


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _load_mlmodels(compute_units: str):
    cu = {
        "CPU_ONLY":    ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE":  ct.ComputeUnit.CPU_AND_NE,
        "ALL":         ct.ComputeUnit.ALL,
    }[compute_units]
    pre_mlp  = next(iter((BUILD / "llm-fp16").glob("LLM-Prefill-*.mlpackage")))
    dec_mlp  = next(iter((BUILD / "llm-fp16").glob("LLM-Decode-*.mlpackage")))
    flow_mlp = BUILD / "flow-fp32" / "Flow-N125-fp32.mlpackage"
    hift_mlp = BUILD / "hift-fp32" / "HiFT-T250-fp32.mlpackage"
    print(f"[load] prefill : {pre_mlp.name}")
    print(f"[load] decode  : {dec_mlp.name}")
    print(f"[load] flow    : {flow_mlp.name}")
    print(f"[load] hift    : {hift_mlp.name}")
    return (
        ct.models.MLModel(str(pre_mlp),  compute_units=cu),
        ct.models.MLModel(str(dec_mlp),  compute_units=cu),
        ct.models.MLModel(str(flow_mlp), compute_units=cu),
        ct.models.MLModel(str(hift_mlp), compute_units=cu),
    )


def _pad_prefill(lm_input: torch.Tensor, t_pre: int, t_target: int):
    """Right-pad lm_input (1, t_pre, 896) to (1, t_target, 896) with zeros."""
    if t_pre > t_target:
        raise SystemExit(f"lm_input len {t_pre} > T_PREFILL {t_target}; rebuild with a bigger T")
    pad = torch.zeros(1, t_target - t_pre, lm_input.shape[-1], dtype=lm_input.dtype)
    return torch.cat([lm_input, pad], dim=1)


def _speech_embed_table(model_dir: Path) -> torch.Tensor:
    """Load the 6761×896 speech embedding table from llm.pt (no Qwen2 cost)."""
    sd = torch.load(str(model_dir / "llm.pt"), map_location="cpu", weights_only=False)
    w = sd["speech_embedding.weight"].to(torch.float32)
    assert w.shape == (6761, 896), f"unexpected speech_embedding.weight shape: {w.shape}"
    return w


# --------------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-text", default="希望你以后能够做的比我还好用")
    ap.add_argument("--prompt-text",
                    default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。")
    ap.add_argument("--prompt-wav",
                    default=str(HERE / "CosyVoice" / "asset" / "zero_shot_prompt.wav"))
    ap.add_argument("--output-wav", default=str(BUILD / "wavs" / "e2e_coreml.wav"))
    ap.add_argument("--compute-units", default="CPU_ONLY",
                    choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="Hard cap on generated speech tokens (0=derive from Flow N_total)")
    ap.add_argument("--min-new-tokens", type=int, default=0,
                    help="Disallow EOS before this many tokens")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_path = Path(args.output_wav)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_dir = ROOT / "cosyvoice3_dl"

    # ---------------------------------------------------------------- #
    # [1/5] Text frontend (Python)
    # ---------------------------------------------------------------- #
    print(f"[1/5] Text frontend (Python)…")
    fr = build_frontend_inputs(args.tts_text, args.prompt_text, args.prompt_wav,
                               model_dir=model_dir)
    print(f"      {fr.summary()}")

    # Flow can only absorb FLOW_N_TOTAL speech tokens total.
    N_prompt = int(fr.llm_prompt_speech_ids.shape[1])
    max_new = FLOW_N_TOTAL - N_prompt
    if args.max_new_tokens > 0:
        max_new = min(max_new, args.max_new_tokens)
    if max_new <= 0:
        raise SystemExit(f"No room for new tokens: N_prompt={N_prompt} >= FLOW_N_TOTAL={FLOW_N_TOTAL}")
    print(f"      N_prompt={N_prompt}  max_new_tokens={max_new}")

    speech_w = _speech_embed_table(model_dir)  # [6761, 896]

    # ---------------------------------------------------------------- #
    # [2/5] Load CoreML models
    # ---------------------------------------------------------------- #
    print(f"[2/5] Loading CoreML models (compute_units={args.compute_units})…")
    pre_model, dec_model, flow_model, hift_model = _load_mlmodels(args.compute_units)

    # ---------------------------------------------------------------- #
    # [3/5] LLM Prefill + Decode loop
    # ---------------------------------------------------------------- #
    lm_input_padded = _pad_prefill(fr.lm_input_embeds, fr.t_pre, T_PREFILL)
    input_len = np.array([fr.t_pre], dtype=np.int32)
    print(f"[3/5] LLM prefill  T_pre={fr.t_pre}  (padded→{T_PREFILL})…")
    t0 = time.time()
    pre_out = pre_model.predict({
        "inputs_embeds": lm_input_padded.detach().numpy().astype(np.float32),
        "input_len": input_len,
    })
    sl_pre = torch.from_numpy(pre_out["speech_logits"])  # [1, T_PREFILL, 6761]
    kv_k = pre_out["kv_k"]  # [L, 1, Hkv, M, D]
    kv_v = pre_out["kv_v"]
    t_prefill = time.time() - t0

    logp0 = sl_pre[0, fr.t_pre - 1].log_softmax(dim=-1)
    print(f"      prefill done in {t_prefill:.2f}s  logits max argmax={int(logp0.argmax())}")

    decoded: list[int] = []
    min_new = max(args.min_new_tokens, 0)

    def _pick(logp: torch.Tensor, step: int) -> int:
        lp = logp.clone()
        # Block EOS / stop-range while step < min_new
        if step < min_new:
            for sid in STOP_IDS:
                lp[sid] = -float("inf")
        return ras_sampling(lp, decoded)

    # First token from prefill output
    top_id = _pick(logp0, 0)
    if top_id in STOP_IDS:
        print(f"      [early stop] first token {top_id} is a stop token; generating silence?")
    else:
        decoded.append(top_id)

    cur_len = fr.t_pre  # next decode step writes cache position cur_len
    print(f"[4/5] LLM decode up to {max_new} tokens…")
    t0 = time.time()
    for step in range(1, max_new):
        next_emb = speech_w[top_id].reshape(1, 1, 896)
        dec_out = dec_model.predict({
            "inputs_embeds": next_emb.numpy().astype(np.float32),
            "kv_k": kv_k, "kv_v": kv_v,
            "cur_len": np.array([cur_len], dtype=np.int32),
        })
        kv_k = dec_out["kv_k_out"]
        kv_v = dec_out["kv_v_out"]
        logits = torch.from_numpy(dec_out["speech_logits"])[0, 0]  # [6761]
        logp = logits.log_softmax(dim=-1)
        top_id = _pick(logp, step)
        cur_len += 1
        if top_id in STOP_IDS:
            print(f"      EOS at step {step} (token={top_id})")
            break
        decoded.append(top_id)
    t_decode = time.time() - t0
    print(f"      decoded {len(decoded)} tokens in {t_decode:.2f}s "
          f"({len(decoded) / max(t_decode, 1e-6):.1f} tok/s)")
    if not decoded:
        raise SystemExit("no speech tokens generated")

    # ---------------------------------------------------------------- #
    # [5/5] Flow + HiFT
    # ---------------------------------------------------------------- #
    N_new = len(decoded)
    N_total = N_prompt + N_new
    token_total = np.zeros((1, FLOW_N_TOTAL), dtype=np.int32)
    token_total[:, :N_prompt] = fr.llm_prompt_speech_ids.numpy().astype(np.int32)
    token_total[:, N_prompt:N_total] = np.asarray(decoded, dtype=np.int32)

    prompt_feat_padded = np.zeros((1, FLOW_N_TOTAL * 2, 80), dtype=np.float32)
    M_prompt = int(fr.prompt_mel.shape[1])
    prompt_feat_padded[:, :M_prompt] = fr.prompt_mel.detach().numpy().astype(np.float32)

    print(f"[5a/5] Flow: N_prompt={N_prompt}  N_new={N_new}  N_total={N_total}")
    t0 = time.time()
    flow_out = flow_model.predict({
        "token_total": token_total,
        "num_prompt_tokens": np.array([N_prompt], dtype=np.int32),
        "prompt_feat": prompt_feat_padded,
        "embedding": fr.spk_embedding.detach().numpy().astype(np.float32),
    })
    t_flow = time.time() - t0
    full_mel = np.asarray(flow_out["mel"])  # (1, 80, 250)
    n_prompt_mel = int(np.asarray(flow_out["num_prompt_mel"]).ravel()[0])
    new_mel = full_mel[:, :, n_prompt_mel : n_prompt_mel + N_new * 2]
    print(f"      mel {full_mel.shape}  new-mel {new_mel.shape}  flow {t_flow:.2f}s")

    T_new = new_mel.shape[-1]
    T_hift = min(T_new, HIFT_T)
    mel_for_hift = np.zeros((1, 80, HIFT_T), dtype=np.float32)
    mel_for_hift[:, :, :T_hift] = new_mel[:, :, :T_hift]
    print(f"[5b/5] HiFT: {T_hift}/{HIFT_T} valid frames")
    t0 = time.time()
    hift_out = hift_model.predict({
        "mel": mel_for_hift,
        "num_valid_frames": np.array([T_hift], dtype=np.int32),
    })
    t_hift = time.time() - t0
    audio = np.asarray(hift_out["audio"]).flatten()
    alen = int(np.asarray(hift_out["audio_length_samples"]).ravel()[0])
    audio = audio[:alen]
    print(f"      audio {audio.shape}  {alen} samples  hift {t_hift:.2f}s")

    sf.write(str(out_path), audio, 24000)
    print(f"\n[out] saved: {out_path}  ({alen / 24000:.2f}s)")

    # Optional Whisper transcription for sanity.
    try:
        import whisper as whisper_asr
        print("\n[asr] Whisper base on output…")
        w = whisper_asr.load_model("base")
        txt = w.transcribe(str(out_path), language="zh", verbose=False)["text"].strip()
        print(f"      expected : {args.tts_text}")
        print(f"      whisper  : {txt}")
    except Exception as e:
        print(f"[asr] skipped ({e})")


if __name__ == "__main__":
    main()
