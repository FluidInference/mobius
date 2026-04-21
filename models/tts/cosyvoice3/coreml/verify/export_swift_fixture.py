"""Export a frontend fixture + reference WAV + split speech embedding table for
the FluidAudio Swift parity harness.

Runs the full Python pipeline (text frontend → LLM prefill → AR decode with RAS
sampling → Flow → HiFT) and writes:

  build/frontend/shipping.safetensors
      lm_input_embeds        [1, T_pre, 896]  fp32
      t_pre                  [1]              int32
      llm_prompt_speech_ids  [1, N_speech]    int32
      prompt_mel             [1, 2*N_speech, 80] fp32
      spk_embedding          [1, 192]         fp32
      decoded_tokens         [1, N_new]       int32
      seed                   [1]              int32

  build/embeddings/speech_embedding-fp16.safetensors
      speech_embedding       [6761, 896]      fp16   (mmap'd row-lookup on device)

  build/wavs/e2e_shipping.wav
      24 kHz mono int16 WAV (reference for Swift byte-equality tolerance test)

Usage:
    uv run python verify/export_swift_fixture.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.numpy import save_file

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "CosyVoice"))
sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))

import coremltools as ct  # noqa: E402

from src.text_frontend import build_frontend_inputs  # noqa: E402


BUILD = ROOT / "build"

T_PREFILL = 256
FLOW_N_TOTAL = 250
HIFT_T = 500
SPEECH_VOCAB = 6761
SPEECH_TOKEN_SIZE = 6561
STOP_IDS = set(range(SPEECH_TOKEN_SIZE, SPEECH_TOKEN_SIZE + 200))


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


def ras_sampling(logp: torch.Tensor, decoded: list[int]) -> int:
    top_id = _nucleus_sampling(logp, 0.8, 25)
    recent = decoded[-10:]
    rep = sum(1 for t in recent if t == top_id)
    if rep >= 10 * 0.1:
        lp = logp.clone()
        lp[top_id] = -float("inf")
        probs = lp.softmax(dim=0)
        top_id = int(torch.multinomial(probs, 1, replacement=True).item())
    return top_id


def _speech_embed_table(model_dir: Path) -> torch.Tensor:
    sd = torch.load(str(model_dir / "llm.pt"), map_location="cpu", weights_only=False)
    w = sd["speech_embedding.weight"].to(torch.float32)
    assert w.shape == (6761, 896), f"unexpected shape: {w.shape}"
    return w


def _pad_prefill(lm_input: torch.Tensor, t_pre: int, t_target: int):
    if t_pre > t_target:
        raise SystemExit(f"lm_input {t_pre} > T_PREFILL {t_target}")
    pad = torch.zeros(1, t_target - t_pre, lm_input.shape[-1], dtype=lm_input.dtype)
    return torch.cat([lm_input, pad], dim=1)


def _find_mlp(subdir: str, prefix: str) -> Path:
    return next(iter((BUILD / subdir).glob(f"{prefix}*.mlpackage")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-text", default="希望你以后能够做的比我还好用")
    ap.add_argument("--prompt-text",
                    default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。")
    ap.add_argument("--prompt-wav",
                    default=str(HERE / "CosyVoice" / "asset" / "zero_shot_prompt.wav"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(BUILD / "frontend"))
    ap.add_argument("--emb-dir", default=str(BUILD / "embeddings"))
    ap.add_argument("--wav-dir", default=str(BUILD / "wavs"))
    ap.add_argument("--fixture-name", default="shipping.safetensors")
    ap.add_argument("--wav-name", default="e2e_shipping.wav")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = Path(args.emb_dir); emb_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = Path(args.wav_dir); wav_dir.mkdir(parents=True, exist_ok=True)

    model_dir = ROOT / "cosyvoice3_dl"

    print("[1/6] Text frontend (Python)…")
    fr = build_frontend_inputs(args.tts_text, args.prompt_text, args.prompt_wav,
                               model_dir=model_dir)
    print(f"      {fr.summary()}")
    N_prompt = int(fr.llm_prompt_speech_ids.shape[1])
    max_new = FLOW_N_TOTAL - N_prompt
    assert max_new > 0

    print("[2/6] Speech embedding table → split safetensors…")
    speech_w = _speech_embed_table(model_dir)
    emb_path = emb_dir / "speech_embedding-fp16.safetensors"
    save_file(
        {"speech_embedding": speech_w.to(torch.float16).numpy()},
        str(emb_path),
    )
    print(f"      saved: {emb_path}  ({speech_w.shape} fp16)")

    print("[3/6] Loading CoreML models (CPU_ONLY, ship config)…")
    pre_mlp = _find_mlp("llm-fp16", "LLM-Prefill-")
    dec_mlp = _find_mlp("llm-fp16", "LLM-Decode-")
    flow_mlp = BUILD / "flow-fp32-n250" / "Flow-N250-fp32.mlpackage"
    hift_mlp = BUILD / "hift-fp16-t500" / "HiFT-T500-fp16.mlpackage"
    for p in (pre_mlp, dec_mlp, flow_mlp, hift_mlp):
        print(f"      {p.name}")
    cu = ct.ComputeUnit.CPU_ONLY
    pre = ct.models.MLModel(str(pre_mlp), compute_units=cu)
    dec = ct.models.MLModel(str(dec_mlp), compute_units=cu)
    flow = ct.models.MLModel(str(flow_mlp), compute_units=cu)
    hift = ct.models.MLModel(str(hift_mlp), compute_units=cu)

    print(f"[4/6] LLM prefill T_pre={fr.t_pre}…")
    lm_padded = _pad_prefill(fr.lm_input_embeds, fr.t_pre, T_PREFILL)
    t0 = time.time()
    pre_out = pre.predict({
        "inputs_embeds": lm_padded.detach().numpy().astype(np.float32),
        "input_len": np.array([fr.t_pre], dtype=np.int32),
    })
    sl_pre = torch.from_numpy(pre_out["speech_logits"])
    kv_k = pre_out["kv_k"]; kv_v = pre_out["kv_v"]
    print(f"      prefill {time.time()-t0:.2f}s")

    logp0 = sl_pre[0, fr.t_pre - 1].log_softmax(dim=-1)
    decoded: list[int] = []
    top_id = ras_sampling(logp0, decoded)
    if top_id in STOP_IDS:
        raise SystemExit("first token was stop")
    decoded.append(top_id)

    cur_len = fr.t_pre
    print(f"[5/6] LLM decode (max {max_new})…")
    t0 = time.time()
    for step in range(1, max_new):
        next_emb = speech_w[top_id].reshape(1, 1, 896)
        out = dec.predict({
            "inputs_embeds": next_emb.numpy().astype(np.float32),
            "kv_k": kv_k, "kv_v": kv_v,
            "cur_len": np.array([cur_len], dtype=np.int32),
        })
        kv_k = out["kv_k_out"]; kv_v = out["kv_v_out"]
        logits = torch.from_numpy(out["speech_logits"])[0, 0]
        logp = logits.log_softmax(dim=-1)
        top_id = ras_sampling(logp, decoded)
        cur_len += 1
        if top_id in STOP_IDS:
            print(f"      EOS at step {step} (token={top_id})")
            break
        decoded.append(top_id)
    print(f"      decoded {len(decoded)} in {time.time()-t0:.2f}s")
    N_new = len(decoded)

    # Flow + HiFT (for reference WAV)
    token_total = np.zeros((1, FLOW_N_TOTAL), dtype=np.int32)
    token_total[:, :N_prompt] = fr.llm_prompt_speech_ids.numpy().astype(np.int32)
    token_total[:, N_prompt:N_prompt + N_new] = np.asarray(decoded, dtype=np.int32)
    prompt_feat_padded = np.zeros((1, FLOW_N_TOTAL * 2, 80), dtype=np.float32)
    M_prompt = int(fr.prompt_mel.shape[1])
    prompt_feat_padded[:, :M_prompt] = fr.prompt_mel.detach().numpy().astype(np.float32)

    print(f"[6/6] Flow + HiFT → reference WAV…")
    flow_out = flow.predict({
        "token_total": token_total,
        "num_prompt_tokens": np.array([N_prompt], dtype=np.int32),
        "prompt_feat": prompt_feat_padded,
        "embedding": fr.spk_embedding.detach().numpy().astype(np.float32),
    })
    full_mel = np.asarray(flow_out["mel"])
    n_prompt_mel = int(np.asarray(flow_out["num_prompt_mel"]).ravel()[0])
    new_mel = full_mel[:, :, n_prompt_mel: n_prompt_mel + N_new * 2]
    T_new = new_mel.shape[-1]
    T_hift = min(T_new, HIFT_T)
    mel_for_hift = np.zeros((1, 80, HIFT_T), dtype=np.float32)
    mel_for_hift[:, :, :T_hift] = new_mel[:, :, :T_hift]
    hift_out = hift.predict({
        "mel": mel_for_hift,
        "num_valid_frames": np.array([T_hift], dtype=np.int32),
    })
    audio = np.asarray(hift_out["audio"]).flatten()
    alen = int(np.asarray(hift_out["audio_length_samples"]).ravel()[0])
    audio = audio[:alen]
    wav_path = wav_dir / args.wav_name
    sf.write(str(wav_path), audio, 24000)
    print(f"      saved reference: {wav_path}  ({alen/24000:.2f}s)")

    # Dump fixture
    fixture_path = out_dir / args.fixture_name
    tensors = {
        "lm_input_embeds":       fr.lm_input_embeds.detach().to(torch.float32).numpy(),
        "t_pre":                 np.array([fr.t_pre], dtype=np.int32),
        "llm_prompt_speech_ids": fr.llm_prompt_speech_ids.detach().to(torch.int32).numpy(),
        "prompt_mel":            fr.prompt_mel.detach().to(torch.float32).numpy(),
        "spk_embedding":         fr.spk_embedding.detach().to(torch.float32).numpy(),
        "decoded_tokens":        np.asarray(decoded, dtype=np.int32).reshape(1, -1),
        "seed":                  np.array([args.seed], dtype=np.int32),
        "num_prompt_mel":        np.array([n_prompt_mel], dtype=np.int32),
        "audio_length_samples":  np.array([alen], dtype=np.int32),
    }
    save_file(tensors, str(fixture_path))
    print(f"      saved fixture : {fixture_path}")
    for k, v in tensors.items():
        print(f"        {k:<22} {v.dtype}  {v.shape}")

    # JSON sidecar for Phase 2 text frontend parity (safetensors can't hold
    # variable-length strings).
    import json
    sidecar = fixture_path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "prompt_text": args.prompt_text,
        "tts_text": args.tts_text,
    }, ensure_ascii=False, indent=2))
    print(f"      saved sidecar : {sidecar}")


if __name__ == "__main__":
    main()
