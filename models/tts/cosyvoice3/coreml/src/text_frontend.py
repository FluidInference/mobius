"""Thin Python wrapper around CosyVoice3's zero-shot frontend.

Goal: expose a single entry point that, given (tts_text, prompt_text, prompt_wav),
returns everything the CoreML pipeline needs:

    - Qwen2 text token IDs (prompt_text and tts_text)
    - prompt speech tokens  (from speech_tokenizer_v3.onnx)
    - prompt mel            (24 kHz 80-bin log-mel for Flow)
    - CAMPPlus speaker embedding  (192-d from campplus.onnx)
    - LM input embedding    ([1, T_pre, 896] the LLM Prefill expects)

This is the Swift boundary: on-device, Swift will need to replicate the
tokenizer + audio preprocessing.  We document each piece below so the Swift
port has a reference implementation.

Usage (standalone):
    uv run python -m src.text_frontend \
        --tts-text "希望你以后能够做的比我还好用" \
        --prompt-text "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
        --prompt-wav  verify/CosyVoice/asset/zero_shot_prompt.wav
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


HERE = Path(__file__).parent
ROOT = HERE.parent


# --------------------------------------------------------------------------- #
#  Lazy-initialised singleton (avoids re-loading CosyVoice3 every call).
# --------------------------------------------------------------------------- #

_CV_SINGLETON = None


def _get_cv(model_dir: Path):
    global _CV_SINGLETON
    if _CV_SINGLETON is None:
        # Delayed imports so test environments don't pay the cost when unused.
        sys.path.insert(0, str(ROOT / "verify" / "CosyVoice"))
        sys.path.insert(0, str(ROOT / "verify" / "CosyVoice" / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import CosyVoice3
        _CV_SINGLETON = CosyVoice3(str(model_dir), load_trt=False, load_vllm=False, fp16=False)
        _CV_SINGLETON.model.llm.float()
    return _CV_SINGLETON


# --------------------------------------------------------------------------- #
#  Result dataclass
# --------------------------------------------------------------------------- #

@dataclass
class FrontendResult:
    """Everything downstream CoreML models will need.

    Shapes (all torch tensors unless noted):
        prompt_text_ids        : [1, N_prompt_txt] int32 — Qwen2 token IDs
                                  INCLUDES <|endofprompt|> (151646)
        tts_text_ids           : [1, N_tts_txt]    int32
        llm_prompt_speech_ids  : [1, N_speech]     int32 — IDs in [0, 6560]
        prompt_mel             : [1, 2*N_speech, 80] float32 — for Flow
        spk_embedding          : [1, 192]          float32 — CAMPPlus output
        lm_input_embeds        : [1, T_pre, 896]   float32 — ready for Qwen2Prefill

    `T_pre` = 1 (sos) + (N_prompt_txt + N_tts_txt) + 1 (task_id) + N_speech.
    """
    prompt_text_ids: torch.Tensor
    tts_text_ids: torch.Tensor
    llm_prompt_speech_ids: torch.Tensor
    prompt_mel: torch.Tensor
    spk_embedding: torch.Tensor
    lm_input_embeds: torch.Tensor
    t_pre: int

    def summary(self) -> str:
        return (
            f"prompt_text_ids   : {tuple(self.prompt_text_ids.shape)}  "
            f"(last 5: {self.prompt_text_ids[0, -5:].tolist()})\n"
            f"tts_text_ids      : {tuple(self.tts_text_ids.shape)}  "
            f"(first 5: {self.tts_text_ids[0, :5].tolist()})\n"
            f"llm_prompt_speech : {tuple(self.llm_prompt_speech_ids.shape)}\n"
            f"prompt_mel        : {tuple(self.prompt_mel.shape)}\n"
            f"spk_embedding     : {tuple(self.spk_embedding.shape)}\n"
            f"lm_input_embeds   : {tuple(self.lm_input_embeds.shape)}  "
            f"(T_pre={self.t_pre})"
        )


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #

def build_frontend_inputs(
    tts_text: str,
    prompt_text: str,
    prompt_wav: str,
    *,
    model_dir: Optional[Path] = None,
) -> FrontendResult:
    """Run CosyVoice3's zero-shot frontend and package the outputs.

    This delegates entirely to the upstream frontend (so text normalisation /
    tokenization / speech tokenization stays bit-exact with the reference
    implementation).  We additionally construct the ``lm_input`` embedding so
    the caller can feed it straight into the LLM-Prefill CoreML model without
    needing the speech-embedding and text-embedding tables on the Swift side.
    """
    if model_dir is None:
        model_dir = ROOT / "cosyvoice3_dl"

    cv = _get_cv(model_dir)
    fe = cv.frontend
    llm = cv.model.llm

    mi = fe.frontend_zero_shot(
        tts_text, prompt_text, prompt_wav,
        cv.sample_rate, zero_shot_spk_id="",
    )

    prompt_ids = mi["prompt_text"].to(torch.int32)
    text_ids   = mi["text"].to(torch.int32)
    pst        = mi["llm_prompt_speech_token"].to(torch.int32)
    prompt_mel = mi["prompt_speech_feat"].to(torch.float32)
    spk_emb    = mi["llm_embedding"].to(torch.float32)

    # Build lm_input exactly as CosyVoice3LM.inference does (llm.py:474..494).
    full_text = torch.concat([prompt_ids.long(), text_ids.long()], dim=1)
    text_emb  = llm.llm.model.model.embed_tokens(full_text)
    sos_emb   = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
    tid_emb   = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
    if pst.shape[1] > 0:
        pst_emb = llm.speech_embedding(pst.long())
    else:
        pst_emb = torch.zeros(1, 0, llm.speech_embedding.embedding_dim, dtype=text_emb.dtype)

    lm_input = torch.concat([sos_emb, text_emb, tid_emb, pst_emb], dim=1).to(torch.float32)
    t_pre = int(lm_input.shape[1])

    return FrontendResult(
        prompt_text_ids=prompt_ids,
        tts_text_ids=text_ids,
        llm_prompt_speech_ids=pst,
        prompt_mel=prompt_mel,
        spk_embedding=spk_emb,
        lm_input_embeds=lm_input,
        t_pre=t_pre,
    )


# --------------------------------------------------------------------------- #
#  Swift-port notes (kept with the wrapper on purpose)
# --------------------------------------------------------------------------- #

SWIFT_PORT_NOTES = """
Pieces of this frontend that must be re-implemented on device (Swift):

1. Qwen2 tokenizer
     * HuggingFace Qwen2Tokenizer = tiktoken BPE with a 58 836-token base vocab
       (`multilingual_zh_ja_yue_char_del.tiktoken`) plus added special tokens.
     * Critical special token: <|endofprompt|> = 151646. Zero-shot fails if
       absent (hard assert in CosyVoice3LM.inference).
     * swift-transformers has a working Qwen2 tokenizer.

2. Text normalisation (Chinese / English)
     * Currently uses wetext; result is passed to tokenizer.
     * Recommendation for v1: do normalisation in Python/server side and ship
       pre-normalised text with the request.  Porting wetext to Swift is a
       significant project.

3. Audio preprocessing for prompt_wav
     * 16 kHz path (for speech_tokenizer_v3 and CAMPPlus):
         - load audio @ 16 kHz, mono, float32 [-1, 1]
         - Whisper log-mel: n_fft=400, hop=160, n_mels=128
         - kaldi fbank (for CAMPPlus): n_mels=80, 25 ms frame, 10 ms hop,
           mean-normalised over the clip.
     * 24 kHz path (for Flow mel-spec prompt):
         - load @ 24 kHz
         - mel-spec: n_fft=1920, hop=480, n_mels=80, center=True
     * All of this is straightforward in Accelerate.framework / vDSP.

4. ONNX / CoreML models
     * speech_tokenizer_v3.onnx      -> needs CoreML port (not yet done)
     * campplus.onnx                 -> CAMPPlus-T300-fp32.mlpackage (shipped)
     * Qwen2 LLM (prefill + decode)  -> LLM-Prefill / LLM-Decode mlpackages
     * Flow (CFM DiT + UPS)          -> Flow-N125-{fp16,fp32}.mlpackage
     * HiFT vocoder                  -> HiFT-T250-fp32.mlpackage

5. LM input construction
     * Swift will need access to two embedding tables:
         a. Qwen2 `model.embed_tokens`  (151 936 x 896)
         b. Speech `speech_embedding`   (6 761 x 896)
       These should be exported as .safetensors or .npz alongside the
       mlpackages so Swift can mmap them.
     * lm_input layout:
         [ sos_emb(1) , text_emb(N_prompt_txt + N_tts_txt) , task_id_emb(1) ,
           speech_emb(N_prompt_speech) ]
       T_pre = 1 + (N_prompt_txt + N_tts_txt) + 1 + N_prompt_speech.

6. Sampling loop (after LLM-Decode)
     * ras_sampling: top_p=0.8, top_k=25, repetition window=10, tau_r=0.1.
     * Implement in Swift; no CoreML op needed.
     * Stop tokens: 6561..6760 (speech EOS range).
"""


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-text", required=True)
    ap.add_argument("--prompt-text", required=True)
    ap.add_argument("--prompt-wav", required=True)
    ap.add_argument("--save", type=Path, default=None,
                    help="Optional .pt path to save the FrontendResult dict")
    ap.add_argument("--show-notes", action="store_true")
    args = ap.parse_args()

    res = build_frontend_inputs(args.tts_text, args.prompt_text, args.prompt_wav)
    print(res.summary())

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "prompt_text_ids": res.prompt_text_ids,
            "tts_text_ids": res.tts_text_ids,
            "llm_prompt_speech_ids": res.llm_prompt_speech_ids,
            "prompt_mel": res.prompt_mel,
            "spk_embedding": res.spk_embedding,
            "lm_input_embeds": res.lm_input_embeds,
            "t_pre": res.t_pre,
        }, str(args.save))
        print(f"\nsaved: {args.save}")

    if args.show_notes:
        print("\n" + SWIFT_PORT_NOTES)


if __name__ == "__main__":
    _cli()
