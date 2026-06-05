#!/usr/bin/env python3
"""WER + RTFx benchmark for the B1-fused (decoder+joint) Nemotron streaming path.

Same streaming/cache logic as test_coreml_streaming.py, but the inner decode loop
makes ONE fused CoreML call (decoder_joint.mlpackage) per step instead of two.
Argmax stays in Python (== Swift in the real pipeline). CPU_AND_NE (ship config).
"""
import argparse
import glob
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf

from test_coreml_streaming import NemotronCoreMLStreaming, load_ground_truth, compute_wer


class FusedStreaming(NemotronCoreMLStreaming):
    def load_fused(self, model_dir):
        md = Path(model_dir)
        cu = ct.ComputeUnit.CPU_AND_NE
        self.preprocessor = ct.models.MLModel(str(md / "preprocessor.mlpackage"), compute_units=cu)
        self.encoder = ct.models.MLModel(str(md / "encoder.mlpackage"), compute_units=cu)
        self.fused = ct.models.MLModel(str(md / "decoder_joint.mlpackage"), compute_units=cu)

    def transcribe_streaming(self, audio: np.ndarray) -> str:
        audio = audio.astype(np.float32)
        total = len(audio)
        cache_channel, cache_time, cache_len = self._get_initial_cache()
        h, c = self._get_initial_decoder_state()
        last_token = self.blank_idx
        all_tokens = []
        mel_cache = None
        offset = 0
        while offset < total:
            end = min(offset + self.chunk_samples, total)
            chunk = audio[offset:end]
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))
            chunk = chunk.reshape(1, -1)
            pre = self.preprocessor.predict({"audio": chunk, "audio_length": np.array([chunk.shape[1]], dtype=np.int32)})
            cmel = pre["mel"]
            if mel_cache is not None:
                imel = np.concatenate([mel_cache, cmel], axis=2)
            else:
                imel = np.pad(cmel, ((0, 0), (0, 0), (self.pre_encode_cache, 0)), mode="constant")
            cf = imel.shape[2]
            if cf < self.total_mel_frames:
                imel = np.pad(imel, ((0, 0), (0, 0), (0, self.total_mel_frames - cf)), mode="constant")
            elif cf > self.total_mel_frames:
                imel = imel[:, :, : self.total_mel_frames]
            mel_cache = cmel[:, :, -self.pre_encode_cache:] if cmel.shape[2] >= self.pre_encode_cache else cmel

            enc = self.encoder.predict({
                "mel": imel.astype(np.float32), "mel_length": np.array([self.total_mel_frames], dtype=np.int32),
                "cache_channel": cache_channel, "cache_time": cache_time, "cache_len": cache_len})
            encoded = enc["encoded"]
            cache_channel, cache_time, cache_len = enc["cache_channel_out"], enc["cache_time_out"], enc["cache_len_out"]

            for t in range(encoded.shape[2]):
                enc_step = encoded[:, :, t:t + 1].astype(np.float32)
                for _ in range(10):
                    out = self.fused.predict({
                        "token": np.array([[last_token]], dtype=np.int32),
                        "token_length": np.array([1], dtype=np.int32),
                        "h_in": h, "c_in": c, "encoder": enc_step})
                    pred = int(np.argmax(out["logits"][0, 0, 0, :]))
                    if pred == self.blank_idx:
                        break
                    all_tokens.append(pred)
                    last_token = pred
                    h, c = out["h_out"], out["c_out"]
            offset += self.chunk_samples
        return self._decode_tokens(all_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--num-files", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    inf = FusedStreaming(args.model_dir)
    inf.load_fused(args.model_dir)
    gt = load_ground_truth(args.dataset)
    files = sorted(glob.glob(f"{args.dataset}/**/*.flac", recursive=True))[: args.num_files + args.warmup]

    te = tw = 0
    audio_s = compute_s = 0.0
    n = 0
    for i, p in enumerate(files):
        fid = Path(p).stem
        a, sr = sf.read(p, dtype="float32")
        t0 = time.perf_counter()
        hyp = inf.transcribe_streaming(a)
        dt = time.perf_counter() - t0
        if i < args.warmup:
            continue
        n += 1
        audio_s += len(a) / sr
        compute_s += dt
        if fid in gt:
            e, w = compute_wer(gt[fid], hyp)
            te += e
            tw += w
    print("=" * 60)
    print(f"model-dir : {args.model_dir} (B1 fused decoder+joint)")
    print(f"chunk_mel : {inf.chunk_mel_frames} ({inf.chunk_mel_frames*10}ms)")
    print(f"files     : {n}")
    print(f"WER       : {100*te/max(tw,1):.2f}%")
    print(f"RTFx      : {audio_s/compute_s:.1f}")


if __name__ == "__main__":
    main()
