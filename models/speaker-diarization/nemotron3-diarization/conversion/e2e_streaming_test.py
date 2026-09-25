"""Closed-loop streaming comparison on real audio: torch reference vs CoreML forward.

Both paths share NeMo's mel frontend, chunk loader, and ``streaming_update_async``
state logic (async mode = fixed-capacity states, the semantics a fixed-shape host
uses). Only the per-chunk network forward differs, so any drift here is fp16 error
compounding through the spkcache/FIFO feedback loop — the thing single-chunk parity
can't see.

Usage: uv run python e2e_streaming_test.py --variant low --wav build/test_120s.wav
"""

import argparse
import math

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

import config
from convert import apply_variant, load_model


def prep(model, v):
    apply_variant(model, v)
    model.async_streaming = True
    model.async_pad_to_max = True


def torch_streaming(model, mel, mel_len):
    with torch.no_grad():
        return model.forward_streaming(mel, mel_len)


def coreml_streaming(model, mlmodel, v, mel, mel_len):
    """forward_streaming_step loop with the CoreML model as the network forward."""
    sm = model.sortformer_modules
    mel_frames_fixed = config.chunk_mel_frames(v)
    state = sm.init_streaming_state(batch_size=1, async_streaming=True, device=torch.device("cpu"))
    total_preds = torch.zeros((1, 0, sm.n_spk))
    offset = torch.zeros((1,), dtype=torch.long)

    with torch.no_grad():
        for _, chunk_t, feat_lengths, left_offset, right_offset in sm.streaming_feat_loader(
            feat_seq=mel, feat_seq_length=mel_len, feat_seq_offset=offset
        ):
            T = chunk_t.shape[1]
            chunk_fixed = torch.zeros((1, mel_frames_fixed, config.FEAT_DIM))
            chunk_fixed[:, :T] = chunk_t

            out = mlmodel.predict(
                {
                    "chunk": chunk_fixed.numpy(),
                    "chunk_lengths": np.array([int(feat_lengths[0])], dtype=np.int32),
                    "spkcache": state.spkcache.numpy(),
                    "spkcache_lengths": state.spkcache_lengths.numpy().astype(np.int32),
                    "fifo": state.fifo.numpy(),
                    "fifo_lengths": state.fifo_lengths.numpy().astype(np.int32),
                }
            )
            preds = torch.from_numpy(out["speaker_preds"])
            hires = torch.from_numpy(out["speaker_preds_10ms"])
            chunk_embs = torch.from_numpy(out["chunk_pre_encode_embs"])
            chunk_enc_lengths = torch.tensor([int(out["chunk_pre_encode_lengths"].ravel()[0])])

            lc_enc = round(left_offset / model.encoder.subsampling_factor)
            rc_enc = math.ceil(right_offset / model.encoder.subsampling_factor)

            saved_sc = state.spkcache_lengths.clone()
            saved_fifo = state.fifo_lengths.clone()
            state, chunk_preds = sm.streaming_update_async(
                streaming_state=state,
                chunk=chunk_embs,
                chunk_lengths=chunk_enc_lengths,
                preds=preds,
                lc=lc_enc,
                rc=rc_enc,
            )
            max_chunk_len = chunk_embs.shape[1] - lc_enc - rc_enc
            chunk_lengths_eff = (chunk_enc_lengths - lc_enc).clamp(min=0, max=max_chunk_len)
            chunk_preds = model._extract_async_high_resolution_chunk_preds(
                high_resolution_preds=hires,
                spkcache_lengths=saved_sc,
                fifo_lengths=saved_fifo,
                chunk_lengths=chunk_lengths_eff,
                max_chunk_len=max_chunk_len,
                lc_enc=lc_enc,
            )
            total_preds = torch.cat([total_preds, chunk_preds], dim=1)

    output_frames = math.ceil(mel.shape[2] / model.output_subsampling_factor)
    return total_preds[:, :output_frames]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="low")
    parser.add_argument("--wav", default="build/test_120s.wav")
    parser.add_argument("--build-dir", default="build")
    args = parser.parse_args()

    v = config.VARIANTS[args.variant]
    model = load_model()
    prep(model, v)

    audio, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000
    sig = torch.from_numpy(audio).unsqueeze(0)
    sig_len = torch.tensor([sig.shape[1]], dtype=torch.long)
    with torch.no_grad():
        mel, mel_len = model.process_signal(sig, sig_len)

    ref = torch_streaming(model, mel.clone(), mel_len.clone())

    prep(model, v)  # reset any state-dependent attrs before the CoreML pass
    mlmodel = ct.models.MLModel(
        f"{args.build_dir}/Nemotron3Diarizer_{args.variant}.mlpackage",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    got = coreml_streaming(model, mlmodel, v, mel, mel_len)

    n = min(ref.shape[1], got.shape[1])
    r, g = ref[0, :n].numpy(), got[0, :n].numpy()
    diff = np.abs(r - g)
    agree = ((r > 0.5) == (g > 0.5)).mean()
    active_r, active_g = (r > 0.5).mean(), (g > 0.5).mean()
    print(f"frames compared: {n} (10 ms each, {n / 100:.1f}s)")
    print(f"max abs diff:    {diff.max():.4f}")
    print(f"mean abs diff:   {diff.mean():.6f}")
    print(f"p99 abs diff:    {np.percentile(diff, 99):.4f}")
    print(f"binary (0.5) frame agreement: {agree * 100:.3f}%")
    print(f"active rate ref/cml: {active_r:.4f} / {active_g:.4f}")
    per_spk = ((r > 0.5) != (g > 0.5)).mean(axis=0)
    print("per-speaker disagreement:", np.array2string(per_spk, precision=5))


if __name__ == "__main__":
    main()
