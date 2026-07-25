"""End-to-end NeuTTS-2E inference on CoreML.

tokenizer → LM prefill → sampled decode loop (stateful or pass-through KV)
→ NeuCodec decoder → 24 kHz wav. No watermark (upstream applies Perth
post-hoc; that is host-side postprocessing, not part of the models).

Modes:
  * default        — sample with temperature/top-k like upstream
  * --teacher-force — replay build/ref/gen_ids.json through the decode loop
                      and report next-token agreement vs the PyTorch run,
                      then decode the reference codes with the CoreML codec
                      (deterministic parity path)

Usage:
    uv run python inference.py \
        --lm-dir ./build/lm-fp16 --codec ./build/codec/NeuCodec-Decoder-fp16.mlpackage \
        --text "I can't believe it's finally here!" --speaker emily --emotion happy \
        --output ./build/out.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from src.prompt import MAX_CONTEXT, SAMPLE_RATE, build_prompt_ids, extract_speech_codes  # noqa: E402

BACKBONE_REPO = "neuphonic/neutts-2e"
MIN_NEW_TOKENS = 50


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {directory}, found {matches}")
    return matches[0]


def sample_top_k(logits: np.ndarray, temperature: float, top_k: int, rng) -> int:
    logits = logits.astype(np.float64) / max(temperature, 1e-6)
    top = np.argpartition(logits, -top_k)[-top_k:]
    z = logits[top] - logits[top].max()
    p = np.exp(z) / np.exp(z).sum()
    return int(rng.choice(top, p=p))


class StatefulDecoder:
    def __init__(self, model: ct.models.MLModel, num_layers: int):
        self.model = model
        self.state = model.make_state()
        self.num_layers = num_layers

    def seed(self, kv_k: np.ndarray, kv_v: np.ndarray) -> None:
        # write_state only accepts float32 (converts to the fp16 state itself).
        for i in range(self.num_layers):
            self.state.write_state(f"kv_k_{i}", np.ascontiguousarray(kv_k[i], dtype=np.float32))
            self.state.write_state(f"kv_v_{i}", np.ascontiguousarray(kv_v[i], dtype=np.float32))

    def step(self, token: int, cur_len: int) -> np.ndarray:
        out = self.model.predict(
            {
                "input_ids": np.array([[token]], dtype=np.int32),
                "cur_len": np.array([cur_len], dtype=np.int32),
            },
            state=self.state,
        )
        return out["logits"][0]


class PassthroughDecoder:
    def __init__(self, model: ct.models.MLModel, kv_k: np.ndarray, kv_v: np.ndarray):
        self.model = model
        self.kv_k = kv_k.astype(np.float32)
        self.kv_v = kv_v.astype(np.float32)

    def seed(self, kv_k: np.ndarray, kv_v: np.ndarray) -> None:
        self.kv_k = kv_k.astype(np.float32)
        self.kv_v = kv_v.astype(np.float32)

    def step(self, token: int, cur_len: int) -> np.ndarray:
        out = self.model.predict(
            {
                "input_ids": np.array([[token]], dtype=np.int32),
                "kv_k": self.kv_k,
                "kv_v": self.kv_v,
                "cur_len": np.array([cur_len], dtype=np.int32),
            }
        )
        self.kv_k = out["kv_k_out"]
        self.kv_v = out["kv_v_out"]
        return out["logits"][0]


def linear_overlap_add(frames: list[np.ndarray], stride: int) -> np.ndarray:
    """Upstream neutts._linear_overlap_add (triangular weights)."""
    assert len(frames)
    total_size = 0
    for i, frame in enumerate(frames):
        total_size = max(total_size, stride * i + frame.shape[-1])
    sum_weight = np.zeros(total_size, dtype=frames[0].dtype)
    out = np.zeros(total_size, dtype=frames[0].dtype)
    offset = 0
    for frame in frames:
        n = frame.shape[-1]
        t = np.linspace(0, 1, n + 2, dtype=frames[0].dtype)[1:-1]
        weight = 0.5 - np.abs(t - 0.5)
        out[offset : offset + n] += weight * frame
        sum_weight[offset : offset + n] += weight
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


class StreamingVocoder:
    """Upstream _infer_stream_ggml windowing over the CoreML codec.

    The speaker's reference codes provide lookback context for the first
    windows, exactly as upstream does; emitted samples cover only newly
    generated codes.
    """

    CHUNK = 25       # frames yielded per step (0.5 s)
    LOOKBACK = 50
    LOOKFORWARD = 5
    OVERLAP = 1

    def __init__(self, codec: ct.models.MLModel, ref_codes: list[int]):
        self.codec = codec
        self.stride = self.CHUNK * 480
        self.token_cache: list[int] = list(ref_codes)
        self.n_decoded_tokens = len(ref_codes)
        self.n_decoded_samples = 0
        self.audio_cache: list[np.ndarray] = []
        # One-time warmup at the window size used per chunk — the first
        # predict at a new RangeDim size pays a specialization cost.
        w = self.LOOKBACK + self.OVERLAP + self.CHUNK + self.LOOKFORWARD + self.OVERLAP
        t0 = time.perf_counter()
        self._decode((self.token_cache * ((w // len(ref_codes)) + 1))[:w])
        print(f"vocoder warmup: {(time.perf_counter() - t0) * 1000:.0f} ms (one-time)")

    def _decode(self, window: list[int]) -> np.ndarray:
        return self.codec.predict({"codes": np.array([window], dtype=np.int32)})["audio"][0]

    def push(self, code: int) -> np.ndarray | None:
        self.token_cache.append(code)
        if len(self.token_cache) - self.n_decoded_tokens < self.CHUNK + self.LOOKFORWARD:
            return None
        start = max(self.n_decoded_tokens - self.LOOKBACK - self.OVERLAP, 0)
        end = self.n_decoded_tokens + self.CHUNK + self.LOOKFORWARD + self.OVERLAP
        sample_start = (self.n_decoded_tokens - start) * 480
        sample_end = sample_start + (self.CHUNK + 2 * self.OVERLAP) * 480
        recon = self._decode(self.token_cache[start:end])[sample_start:sample_end]
        self.audio_cache.append(recon)
        processed = linear_overlap_add(self.audio_cache, self.stride)
        new_end = len(self.audio_cache) * self.stride
        out = processed[self.n_decoded_samples : new_end]
        self.n_decoded_samples = new_end
        self.n_decoded_tokens += self.CHUNK
        return out

    def flush(self) -> np.ndarray | None:
        remaining = len(self.token_cache) - self.n_decoded_tokens
        if remaining <= 0:
            return None
        start = max(
            len(self.token_cache) - (self.LOOKBACK + self.OVERLAP + remaining), 0
        )
        sample_start = (
            len(self.token_cache) - start - remaining - self.OVERLAP
        ) * 480
        recon = self._decode(self.token_cache[start:])[sample_start:]
        self.audio_cache.append(recon)
        processed = linear_overlap_add(self.audio_cache, self.stride)
        return processed[self.n_decoded_samples :]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lm-dir", required=True)
    p.add_argument("--codec", required=True)
    p.add_argument("--text", default="I can't believe it's finally here!")
    p.add_argument("--speaker", default="emily")
    p.add_argument("--emotion", default="happy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", default="build/out-coreml.wav")
    p.add_argument("--passthrough-kv", action="store_true",
                   help="use the macOS14 pass-through-KV decode model instead of stateful")
    p.add_argument("--teacher-force", action="store_true")
    p.add_argument("--stream", action="store_true",
                   help="windowed streaming vocoder (upstream 25-frame chunks); reports TTFA")
    p.add_argument("--compute-units", default="ALL",
                   choices=["ALL", "CPU_AND_GPU", "CPU_ONLY", "CPU_AND_NE"],
                   help="LM compute units (stateful decode cannot use CPU_AND_NE)")
    p.add_argument("--codec-compute-units", default="CPU_AND_NE",
                   choices=["ALL", "CPU_AND_GPU", "CPU_ONLY", "CPU_AND_NE"],
                   help="codec compute units (ANE is ~2x faster than GPU here)")
    args = p.parse_args()

    cu = getattr(ct.ComputeUnit, args.compute_units)
    codec_cu = getattr(ct.ComputeUnit, args.codec_compute_units)
    lm_dir = Path(args.lm_dir)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_REPO)
    if args.teacher_force:
        # Replay must use the exact prompt the reference run saw.
        prompt_ids = json.loads((HERE / "build" / "ref" / "prompt_ids.json").read_text())
    else:
        prompt_ids = build_prompt_ids(tokenizer, args.text, args.speaker, args.emotion)
    eos_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
    print(f"prompt: {len(prompt_ids)} tokens")

    print("loading prefill...")
    prefill = ct.models.MLModel(str(find_one(lm_dir, "LM-Prefill-*.mlpackage")), compute_units=cu)
    t_prefill = prefill.get_spec().description.input[0].type.multiArrayType.shape[1]
    if len(prompt_ids) > t_prefill:
        raise SystemExit(f"prompt ({len(prompt_ids)}) > prefill window ({t_prefill})")

    t0 = time.perf_counter()
    padded = prompt_ids + [0] * (t_prefill - len(prompt_ids))
    out = prefill.predict(
        {
            "input_ids": np.array([padded], dtype=np.int32),
            "input_len": np.array([len(prompt_ids)], dtype=np.int32),
        }
    )
    t_pre = time.perf_counter() - t0
    logits = out["logits_last"][0]
    kv_k, kv_v = out["kv_k"], out["kv_v"]
    num_layers = kv_k.shape[0]
    max_len = kv_k.shape[3]  # decode state capacity; generation must stop here
    print(f"prefill: {t_pre * 1000:.0f} ms ({num_layers} layers, kv {kv_k.shape})")

    print("loading decode...")
    if args.passthrough_kv:
        decode_path = find_one(lm_dir, "LM-Decode-M*[!l].mlpackage")
        decoder = PassthroughDecoder(ct.models.MLModel(str(decode_path), compute_units=cu), kv_k, kv_v)
    else:
        decode_path = find_one(lm_dir, "LM-Decode-*-stateful.mlpackage")
        decoder = StatefulDecoder(ct.models.MLModel(str(decode_path), compute_units=cu), num_layers)
        decoder.seed(kv_k, kv_v)

    rng = np.random.default_rng(args.seed)
    cur_len = len(prompt_ids)
    gen_ids: list[int] = []

    if args.teacher_force:
        ref_ids = json.loads((HERE / "build" / "ref" / "gen_ids.json").read_text())
        agree_argmax = agree_topk = 0
        step_logits = logits
        t0 = time.perf_counter()
        for i, ref_tok in enumerate(ref_ids):
            order = np.argsort(step_logits)[::-1]
            agree_argmax += int(order[0] == ref_tok)
            agree_topk += int(ref_tok in order[: args.top_k])
            if i + 1 < len(ref_ids):
                step_logits = decoder.step(ref_tok, cur_len)
                cur_len += 1
        dt = time.perf_counter() - t0
        n = len(ref_ids)
        print(f"teacher-force: argmax agreement {agree_argmax}/{n} "
              f"({100.0 * agree_argmax / n:.1f}%), ref token in CoreML top-{args.top_k}: "
              f"{agree_topk}/{n} ({100.0 * agree_topk / n:.1f}%)")
        print(f"decode: {1000.0 * dt / max(n - 1, 1):.1f} ms/token")
        codes = json.loads((HERE / "build" / "ref" / "ref_codes.json").read_text())
    elif args.stream:
        from src.prompt import load_speaker

        print("loading codec (streaming)...")
        codec = ct.models.MLModel(str(Path(args.codec)), compute_units=codec_cu)
        ref_codes, _ = load_speaker(args.speaker)
        voc = StreamingVocoder(codec, ref_codes)
        speech_0 = tokenizer.convert_tokens_to_ids("<|speech_0|>")

        pieces: list[np.ndarray] = []
        chunk_times: list[float] = []
        t_start = t_gen0 = time.perf_counter()
        ttfa = None
        step_logits = logits
        while cur_len < min(MAX_CONTEXT, max_len) - 1:
            if len(gen_ids) < MIN_NEW_TOKENS:
                step_logits[eos_id] = -1e9
            tok = sample_top_k(step_logits, args.temperature, args.top_k, rng)
            gen_ids.append(tok)
            if tok == eos_id:
                break
            if speech_0 <= tok < speech_0 + 65_536:
                piece = voc.push(tok - speech_0)
                if piece is not None:
                    now = time.perf_counter()
                    if ttfa is None:
                        ttfa = now - t_start
                    chunk_times.append(now)
                    pieces.append(piece)
            step_logits = decoder.step(tok, cur_len)
            cur_len += 1
        tail = voc.flush()
        if tail is not None:
            pieces.append(tail)
        dt = time.perf_counter() - t_gen0
        codes = extract_speech_codes(tokenizer, gen_ids)
        audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        dur = len(audio) / SAMPLE_RATE
        gaps = np.diff(chunk_times) if len(chunk_times) > 1 else np.array([0.0])
        # TTFA includes prefill, which happened before t_start; add it back.
        print(f"stream: TTFA {(t_pre + (ttfa or 0)) * 1000:.0f} ms "
              f"(prefill {t_pre * 1000:.0f} + gen-to-first-chunk {(ttfa or 0) * 1000:.0f}), "
              f"{len(pieces)} chunks, inter-chunk mean {gaps.mean() * 1000:.0f} ms "
              f"(budget 500 ms), max {gaps.max() * 1000:.0f} ms")
        print(f"generated {len(codes)} codes ({dur:.2f}s) in {dt:.1f}s wall "
              f"(incl. interleaved vocoding) → {dur / dt:.2f}x RT overall")
    else:
        t0 = time.perf_counter()
        step_logits = logits
        while cur_len < min(MAX_CONTEXT, max_len) - 1:
            if len(gen_ids) < MIN_NEW_TOKENS:
                step_logits[eos_id] = -1e9
            tok = sample_top_k(step_logits, args.temperature, args.top_k, rng)
            gen_ids.append(tok)
            if tok == eos_id:
                break
            step_logits = decoder.step(tok, cur_len)
            cur_len += 1
        dt = time.perf_counter() - t0
        codes = extract_speech_codes(tokenizer, gen_ids)
        print(f"generated {len(gen_ids)} tokens ({len(codes)} codes, "
              f"{len(codes) / 50.0:.2f}s) in {dt:.1f}s "
              f"({1000.0 * dt / max(len(gen_ids), 1):.1f} ms/token)")
        if not codes:
            raise SystemExit("no speech codes generated")

    if not args.stream:
        print("loading codec...")
        codec = ct.models.MLModel(str(Path(args.codec)), compute_units=codec_cu)
        t0 = time.perf_counter()
        audio = codec.predict({"codes": np.array([codes], dtype=np.int32)})["audio"][0]
        t_dec = time.perf_counter() - t0
        dur = len(audio) / SAMPLE_RATE
        print(f"codec: {t_dec * 1000:.0f} ms for {dur:.2f}s audio ({dur / t_dec:.1f}x RT)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio.astype(np.float32), SAMPLE_RATE)
    print(f"wrote {out_path} ({dur:.2f}s, peak {np.abs(audio).max():.3f})")

    if args.teacher_force:
        ref_wav_path = HERE / "build" / "ref" / "ref.wav"
        if ref_wav_path.exists():
            ref_wav, _ = sf.read(ref_wav_path, dtype="float32")
            n = min(len(ref_wav), len(audio))
            noise = audio[:n] - ref_wav[:n]
            snr = 10.0 * np.log10((ref_wav[:n] ** 2).sum() / max((noise**2).sum(), 1e-12))
            print(f"audio vs PyTorch ref: SNR {snr:.1f} dB, max|Δ| {np.abs(noise).max():.4f}")


if __name__ == "__main__":
    main()
