"""End-to-end TTS → ASR test.

Pipeline:
  1. Load CosyVoice3 end-to-end (LLM + Flow + HiFT, all PyTorch)
  2. Synthesize zero-shot Chinese TTS from prompt_wav + tts_text → ref_audio
  3. Monkey-patch flow.inference() to capture its inputs (prompt_token, new_token,
     prompt_feat, embedding)
  4. Rerun captured Flow inputs through our Flow CoreML + HiFT CoreML → our_audio
  5. Whisper-transcribe prompt, ref_audio, our_audio; compare

Success = ref_audio text ≈ our_audio text ≈ tts_text.
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "CosyVoice"))
sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))

import os
import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import coremltools as ct
import whisper as whisper_asr

MODEL_DIR = ROOT / "cosyvoice3_dl"
BUILD = ROOT / "build"
OUT = BUILD / "wavs"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    from cosyvoice.cli.cosyvoice import CosyVoice3

    tts_text = "希望你以后能够做的比我还好用"
    # CosyVoice3 LLM requires <|endofprompt|> (id 151646) in the text.
    # Standard prefix used throughout upstream examples:
    prompt_text = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    prompt_wav = HERE / "CosyVoice" / "asset" / "zero_shot_prompt.wav"

    print(f"[1/7] Load CosyVoice3 end-to-end from {MODEL_DIR}")
    cv = CosyVoice3(str(MODEL_DIR), load_trt=False, load_vllm=False, fp16=False)
    # Qwen2 base is stored in BFloat16; force FP32 on CPU to avoid dtype mismatch.
    cv.model.llm.float()
    flow = cv.model.flow
    hift = cv.model.hift
    print(f"      ok. device={cv.model.device}")

    # ---------------------------------------------------------------- #
    # [2/7] Monkey-patch flow.inference to capture inputs + output
    # ---------------------------------------------------------------- #
    captured = {}
    orig_inf = flow.inference

    def capturing_inference(token, token_len, prompt_token, prompt_token_len,
                            prompt_feat, prompt_feat_len, embedding,
                            streaming, finalize, **kw):
        captured["token"] = token.detach().cpu().clone()
        captured["token_len"] = token_len.detach().cpu().clone()
        captured["prompt_token"] = prompt_token.detach().cpu().clone()
        captured["prompt_token_len"] = prompt_token_len.detach().cpu().clone()
        captured["prompt_feat"] = prompt_feat.detach().cpu().clone()
        captured["prompt_feat_len"] = prompt_feat_len.detach().cpu().clone()
        captured["embedding"] = embedding.detach().cpu().clone()
        captured["streaming"] = streaming
        captured["finalize"] = finalize
        out = orig_inf(token, token_len, prompt_token, prompt_token_len,
                       prompt_feat, prompt_feat_len, embedding, streaming, finalize)
        mel = out[0].detach().cpu().clone()
        captured.setdefault("mels", []).append(mel)
        return out

    flow.inference = capturing_inference

    # ---------------------------------------------------------------- #
    # [3/7] Run upstream synthesis
    # ---------------------------------------------------------------- #
    print(f"[2/7] Synthesizing zero-shot: text={tts_text!r}")
    chunks = []
    for rec in cv.inference_zero_shot(tts_text, prompt_text, str(prompt_wav), stream=False):
        chunks.append(rec["tts_speech"].cpu().numpy().flatten())
    ref_audio = np.concatenate(chunks)
    ref_path = OUT / "rt_ref_upstream.wav"
    sf.write(str(ref_path), ref_audio, cv.sample_rate)
    print(f"      saved ref: {ref_path}  ({len(ref_audio)/cv.sample_rate:.2f}s)")
    print(f"      captured {len(captured.get('mels', []))} Flow call(s)")

    # Pick the (final) Flow call to reproduce.
    N_prompt = int(captured["prompt_token_len"].item())
    N_new = int(captured["token_len"].item())
    N_total = N_prompt + N_new
    print(f"      Flow inputs: N_prompt={N_prompt}  N_new={N_new}  N_total={N_total}")
    if N_total > 125:
        print(f"      WARNING: Flow mlpackage is N=125; total={N_total} — will truncate new_token")
        N_new = 125 - N_prompt
    # ---------------------------------------------------------------- #
    # [4/7] Pad captured inputs for our static-shape Flow mlpackage (N=125)
    # ---------------------------------------------------------------- #
    N = 125
    M = N * 2
    token_total = torch.zeros(1, N, dtype=torch.int32)
    token_total[:, :N_prompt] = captured["prompt_token"][:, :N_prompt].to(torch.int32)
    token_total[:, N_prompt:N_prompt + N_new] = captured["token"][:, :N_new].to(torch.int32)
    num_prompt_tokens = torch.tensor([N_prompt], dtype=torch.int32)

    prompt_feat_padded = torch.zeros(1, M, 80, dtype=torch.float32)
    M_prompt = int(captured["prompt_feat_len"].item())
    prompt_feat_padded[:, :M_prompt] = captured["prompt_feat"][:, :M_prompt].to(torch.float32)

    embedding = captured["embedding"].to(torch.float32)

    # ---------------------------------------------------------------- #
    # [5/7] Flow CoreML → mel  (then HiFT CoreML → audio)
    # ---------------------------------------------------------------- #
    print("[3/7] Loading Flow CoreML mlpackage...")
    flow_mlp = BUILD / "flow-fp32" / "Flow-N125-fp32.mlpackage"
    flow_ml = ct.models.MLModel(str(flow_mlp), compute_units=ct.ComputeUnit.CPU_ONLY)

    out = flow_ml.predict({
        "token_total": token_total.numpy(),
        "num_prompt_tokens": num_prompt_tokens.numpy(),
        "prompt_feat": prompt_feat_padded.numpy(),
        "embedding": embedding.numpy(),
    })
    cm_full_mel = np.asarray(out["mel"])             # (1, 80, 250)
    cm_num_prompt_mel = int(np.asarray(out["num_prompt_mel"]).ravel()[0])
    print(f"      mel: {cm_full_mel.shape}  num_prompt_mel={cm_num_prompt_mel}")

    # slice to the "new" portion actually consumed by HiFT
    new_mel = cm_full_mel[:, :, cm_num_prompt_mel : cm_num_prompt_mel + N_new * 2]
    print(f"      new-mel for HiFT: {new_mel.shape}")

    # HiFT CoreML expects T=250 with num_valid_frames trimming
    T_HIFT = 250
    T_new_actual = new_mel.shape[-1]
    if T_new_actual > T_HIFT:
        new_mel = new_mel[:, :, :T_HIFT]
        T_new_actual = T_HIFT
    mel_for_hift = np.zeros((1, 80, T_HIFT), dtype=np.float32)
    mel_for_hift[:, :, :T_new_actual] = new_mel
    num_valid_frames = np.array([T_new_actual], dtype=np.int32)

    print("[4/7] Loading HiFT CoreML mlpackage...")
    hift_mlp = BUILD / "hift-fp32" / "HiFT-T250-fp32.mlpackage"
    hift_ml = ct.models.MLModel(str(hift_mlp), compute_units=ct.ComputeUnit.CPU_ONLY)
    hift_out = hift_ml.predict({"mel": mel_for_hift, "num_valid_frames": num_valid_frames})
    our_audio = np.asarray(hift_out["audio"]).flatten()
    alen = int(np.asarray(hift_out["audio_length_samples"]).ravel()[0])
    our_audio = our_audio[:alen]

    our_path = OUT / "rt_coreml.wav"
    sf.write(str(our_path), our_audio, cv.sample_rate)
    print(f"      saved our: {our_path}  ({len(our_audio)/cv.sample_rate:.2f}s)")

    # ---------------------------------------------------------------- #
    # [6/7] Whisper ASR
    # ---------------------------------------------------------------- #
    print("[5/7] Running Whisper ASR...")
    w = whisper_asr.load_model("base")

    def asr(path):
        r = w.transcribe(str(path), language="zh", verbose=False)
        return r["text"].strip()

    txt_prompt = asr(prompt_wav)
    txt_ref = asr(ref_path)
    txt_ours = asr(our_path)
    print(f"\nTranscriptions:")
    print(f"  tts_text   : {tts_text}")
    print(f"  prompt_wav : {txt_prompt}")
    print(f"  upstream   : {txt_ref}")
    print(f"  our CoreML : {txt_ours}")

    # ---------------------------------------------------------------- #
    # [7/7] Report
    # ---------------------------------------------------------------- #
    print("\nSummary:")
    print(f"  upstream ↔ ours character overlap: "
          f"{len(set(txt_ref) & set(txt_ours)) / max(1, len(set(txt_ref) | set(txt_ours))):.2%}")


if __name__ == "__main__":
    main()
