"""End-to-end MOSS-TTS-Nano synthesis with the CoreML LM (prefill + step + frame).

Modes:
  --replay N   greedy, teacher-forced replay of the PyTorch greedy reference: reports how many
               of the first N frames the CoreML chain reproduces token-for-token.
  default      free-running sampled generation; decodes with the CoreML codec if --codec is
               given, else the PyTorch codec, and writes a 48 kHz stereo WAV.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
TEXT = "The quick brown fox jumps over the lazy dog near the riverbank."

CU = {
    "all": ct.ComputeUnit.ALL,
    "cpu": ct.ComputeUnit.CPU_ONLY,
    "gpu": ct.ComputeUnit.CPU_AND_GPU,
    "ane": ct.ComputeUnit.CPU_AND_NE,
}


class CoreMLMossLM:
    def __init__(self, lm_dir: Path, tag: str, t_prefill: int, max_len: int, cu: dict[str, str]) -> None:
        def load(name: str, unit: str):
            path = lm_dir / name
            t0 = time.perf_counter()
            m = ct.models.MLModel(str(path), compute_units=CU[unit])
            print(f"      loaded {path.name} on {unit} ({time.perf_counter() - t0:.1f}s)")
            return m

        self.prefill = load(f"MossNano-Prefill-T{t_prefill}-M{max_len}-{tag}.mlpackage", cu["prefill"])
        self.step = load(f"MossNano-Step-M{max_len}-{tag}.mlpackage", cu["step"])
        self.frame = load(f"MossNano-Frame-{tag}.mlpackage", cu["frame"])
        self.t_prefill = t_prefill
        self.max_len = max_len

    def run_prefill(self, input_ids: np.ndarray, pad_text: int, pad_audio: int):
        n = input_ids.shape[1]
        padded = np.full((1, self.t_prefill, input_ids.shape[2]), pad_audio, np.int32)
        padded[:, :, 0] = pad_text
        padded[:, :n] = input_ids
        out = self.prefill.predict({"input_ids": padded, "input_len": np.array([n], np.int32)})
        return out["hidden"], out["kv_k"], out["kv_v"]

    def run_step(self, row: np.ndarray, kv_k, kv_v, cur_len: int):
        out = self.step.predict(
            {"input_ids": row.astype(np.int32), "kv_k": kv_k, "kv_v": kv_v, "cur_len": np.array([cur_len], np.int32)}
        )
        return out["hidden"], out["kv_k_out"], out["kv_v_out"]

    def run_frame(self, hidden, rng, seen, greedy: bool, sampling: dict):
        out = self.frame.predict(
            {
                "global_hidden": hidden.astype(np.float32),
                "text_u": rng.random(1).astype(np.float32),
                "audio_u": rng.random((1, 16)).astype(np.float32),
                "text_temperature": np.array([sampling["text_temperature"]], np.float32),
                "audio_temperature": np.array([sampling["audio_temperature"]], np.float32),
                "audio_top_p": np.array([sampling["audio_top_p"]], np.float32),
                "repetition_penalty": np.array([sampling["repetition_penalty"]], np.float32),
                "seen": seen,
                "greedy": np.array([1.0 if greedy else 0.0], np.float32),
            }
        )
        return int(out["should_continue"].reshape(-1)[0]), out["frame"].reshape(-1).astype(np.int64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lm-dir", default=str(HERE / "build" / "lm"))
    p.add_argument("--tag", default="fp16")
    p.add_argument("--t-prefill", type=int, default=512)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--codec", default=None, help="CoreML full codec decoder mlpackage (default: PyTorch codec)")
    p.add_argument("--codec-step", default=None, help="CoreML streaming codec step mlpackage (decode per frame)")
    p.add_argument("--cu-codec", default="gpu")
    p.add_argument("--text", default=TEXT)
    p.add_argument("--prompt-audio", default=str(HERE / "assets" / "en_2.wav"))
    p.add_argument("--prompt-tokens", default=str(HERE / "build" / "ref_greedy_prompt_audio_token_ids.npy"))
    p.add_argument("--replay", type=int, default=0, help="greedy replay of the first N reference frames")
    p.add_argument("--ref-tokens", default=str(HERE / "build" / "ref_greedy_audio_token_ids.npy"))
    p.add_argument("--max-new-frames", type=int, default=375)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=str(HERE / "build" / "coreml_out.wav"))
    p.add_argument("--cu-prefill", default="all")
    p.add_argument("--cu-step", default="all")
    p.add_argument("--cu-frame", default="all")
    p.add_argument("--text-temperature", type=float, default=1.5)
    p.add_argument("--audio-temperature", type=float, default=1.7)
    p.add_argument("--audio-top-p", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM

    print("[0] loading upstream model for prompt construction")
    model = AutoModelForCausalLM.from_pretrained(TTS_REPO, trust_remote_code=True, dtype=torch.float32).eval()
    cfg = model.config
    tokenizer = model._load_text_tokenizer(text_tokenizer=None, text_tokenizer_path=None)
    prompt_codes = torch.from_numpy(np.load(args.prompt_tokens)).long()
    input_ids, _ = model.build_inference_input_ids(
        text=args.text, text_tokenizer=tokenizer, mode="voice_clone", prompt_audio_codes=prompt_codes, device="cpu"
    )
    ids = input_ids.numpy().astype(np.int32)
    n_prompt = ids.shape[1]
    print(f"      prompt rows={n_prompt}")

    print("[1] loading CoreML LM")
    lm = CoreMLMossLM(
        Path(args.lm_dir), args.tag, args.t_prefill, args.max_len,
        {"prefill": args.cu_prefill, "step": args.cu_step, "frame": args.cu_frame},
    )
    sampling = {
        "text_temperature": args.text_temperature,
        "audio_temperature": args.audio_temperature,
        "audio_top_p": args.audio_top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    rng = np.random.default_rng(args.seed)
    seen = np.zeros((1, cfg.n_vq, cfg.audio_vocab_size), np.float32)

    t0 = time.perf_counter()
    hidden, kv_k, kv_v = lm.run_prefill(ids, cfg.pad_token_id, cfg.audio_pad_token_id)
    hidden, kv_k, kv_v = lm.run_prefill(ids, cfg.pad_token_id, cfg.audio_pad_token_id)  # warm
    t_prefill = time.perf_counter() - t0
    print(f"      prefill: {t_prefill*1000/2:.0f} ms (avg of 2)")

    greedy = args.replay > 0
    ref = np.load(args.ref_tokens) if greedy else None
    frames: list[np.ndarray] = []
    cur_len = n_prompt
    t_step = t_frame = 0.0
    mismatched = 0
    limit = args.replay if greedy else args.max_new_frames
    t_gen = time.perf_counter()
    for i in range(limit):
        t1 = time.perf_counter()
        cont, frame = lm.run_frame(hidden, rng, seen, greedy, sampling)
        t_frame += time.perf_counter() - t1
        if not cont:
            print(f"      stop token at frame {i}")
            break
        if greedy:
            ok = np.array_equal(frame, ref[i])
            mismatched += 0 if ok else 1
            if not ok:
                print(f"      frame {i}: {int((frame == ref[i]).sum())}/16 match  got={frame.tolist()}")
            frame = ref[i]  # teacher force
        frames.append(frame)
        seen[0, np.arange(cfg.n_vq), frame] = 1.0
        row = np.full((1, 1, cfg.n_vq + 1), cfg.audio_pad_token_id, np.int32)
        row[0, 0, 0] = cfg.audio_assistant_slot_token_id
        row[0, 0, 1:] = frame
        t1 = time.perf_counter()
        hidden, kv_k, kv_v = lm.run_step(row, kv_k, kv_v, cur_len)
        t_step += time.perf_counter() - t1
        cur_len += 1
    n = len(frames)
    t_total = time.perf_counter() - t_gen
    print(f"      frames={n}  step {t_step*1000/max(n,1):.1f} ms  frame {t_frame*1000/max(n,1):.1f} ms  "
          f"total {t_total:.2f}s  ({n/12.5:.2f}s audio, LM RTFx={n/12.5/t_total:.2f})")
    if greedy:
        print(f"      replay: {n - mismatched}/{n} frames match the PyTorch greedy reference exactly")
        return

    tokens = np.stack(frames, 0)  # [T,16]
    codes = torch.from_numpy(tokens.T[:, None, :]).long()  # [16,1,T]
    t2 = time.perf_counter()
    if args.codec_step:
        from src.codec_coreml import MossCodecStepDecoder  # geometry only; weights unused

        step_ml = ct.models.MLModel(args.codec_step, compute_units=CU[args.cu_codec])
        spec = step_ml.get_spec()
        cache_names = [i.name for i in spec.description.input if i.name not in ("codes", "frame_index")]
        caches = {
            i.name: np.zeros([d for d in i.type.multiArrayType.shape], np.float32)
            for i in spec.description.input
            if i.name in cache_names
        }
        chunks = []
        t_first = None
        for t in range(codes.shape[-1]):
            feed = {"codes": codes[:, :, t : t + 1].numpy().astype(np.int32),
                    "frame_index": np.array([t], np.int32), **caches}
            out = step_ml.predict(feed)
            if t_first is None:
                t_first = time.perf_counter() - t2
            chunks.append(out["audio"])
            caches = {n: out[f"{n}_out"] for n in cache_names}
        audio = torch.from_numpy(np.concatenate(chunks, axis=-1))
        print(f"      codec step[{args.cu_codec}]: first frame {t_first*1000:.0f} ms, "
              f"{(time.perf_counter()-t2)*1000/codes.shape[-1]:.1f} ms/frame")
    elif args.codec:
        codec = ct.models.MLModel(args.codec, compute_units=ct.ComputeUnit.ALL)
        audio = codec.predict({"codes": codes.numpy().astype(np.int32)})["audio"]
        audio = torch.from_numpy(audio)
    else:
        from transformers import AutoModel

        codec = AutoModel.from_pretrained(CODEC_REPO, trust_remote_code=True).eval()
        with torch.no_grad():
            audio = codec.decode(codes, return_dict=True).audio
    t_codec = time.perf_counter() - t2
    import soundfile as sf

    wav = audio[0].T.numpy()
    sf.write(args.output, wav, 48000)
    dur = wav.shape[0] / 48000
    print(f"      codec: {t_codec*1000:.0f} ms  audio={dur:.2f}s  wrote {args.output}")
    print(f"      end-to-end RTFx={dur/(t_total + t_codec + t_prefill/2):.2f}")


if __name__ == "__main__":
    main()
