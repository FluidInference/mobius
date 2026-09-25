"""W8A8 quantization of a split-graph variant, calibrated with real streaming states.

Drives the state loop with the monolithic model of the same shape, constructs
split-graph inputs at each sampled step, then applies activation quantization
scoped to linear/matmul/conv (int32-free split graphs only) + int8 weights.

Usage: uv run python w8a8_split.py --variant s32
"""
import argparse, math
import numpy as np, torch, soundfile as sf, coremltools as ct
import coremltools.optimize.coreml as ctc
from coremltools.optimize.coreml.experimental import OpActivationLinearQuantizerConfig, linear_quantize_activations
import config
from convert import load_model
from e2e_streaming_test import prep

NEG = -30000.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="s32")
    ap.add_argument("--sample-every", type=int, default=8)
    args = ap.parse_args()

    v = config.VARIANTS[args.variant]
    T = config.packed_frames(v)
    model = load_model(); prep(model, v)
    sm = model.sortformer_modules
    proj_t = np.fromfile("build/pre_encode_proj_t.bin", np.float32).reshape(1024, 512)
    audio, _ = sf.read("build/test_120s.wav", dtype="float32")
    sig = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        mel, mel_len = model.process_signal(sig, torch.tensor([sig.shape[1]]))

    mono = ct.models.MLModel(
        f"build/Nemotron3Diarizer_{args.variant}.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = sm.init_streaming_state(batch_size=1, async_streaming=True, device=torch.device("cpu"))
    samples = []
    offset = torch.zeros((1,), dtype=torch.long)
    i = 0
    with torch.no_grad():
        for _, chunk_t, feat_lengths, lo, ro in sm.streaming_feat_loader(mel, mel_len, offset):
            Tm = chunk_t.shape[1]
            cf = torch.zeros((1, config.chunk_mel_frames(v), 128)); cf[:, :Tm] = chunk_t
            stacked = cf[0].numpy().reshape(-1, 1024)
            embs = stacked @ proj_t
            enc_len = (int(feat_lengths[0]) + 7) // 8
            sc_n, f_n = int(state.spkcache_lengths[0]), int(state.fifo_lengths[0])
            packed = np.zeros((1, T, 512), np.float32); pos = 0
            for part, n in ((state.spkcache[0].numpy(), sc_n), (state.fifo[0].numpy(), f_n), (embs, enc_len)):
                packed[0, pos:pos+n] = part[:n]; pos += n
            bias = np.zeros((1, 1, 1, T), np.float32); bias[..., pos:] = NEG
            omask = np.zeros((1, T, 1), np.float32); omask[0, :pos] = 1.0
            if i % args.sample_every == 0:
                samples.append({"packed": packed, "attn_bias": bias, "output_mask": omask})
            out = mono.predict({
                "chunk": cf.numpy(), "chunk_lengths": np.array([int(feat_lengths[0])], np.int32),
                "spkcache": state.spkcache.numpy(),
                "spkcache_lengths": state.spkcache_lengths.numpy().astype(np.int32),
                "fifo": state.fifo.numpy(), "fifo_lengths": state.fifo_lengths.numpy().astype(np.int32)})
            state, _ = sm.streaming_update_async(
                streaming_state=state,
                chunk=torch.from_numpy(out["chunk_pre_encode_embs"]),
                chunk_lengths=torch.tensor([int(out["chunk_pre_encode_lengths"].ravel()[0])]),
                preds=torch.from_numpy(out["speaker_preds"]), lc=round(lo/8), rc=math.ceil(ro/8))
            i += 1
    print(f"captured {len(samples)} calibration samples from {i} chunks")

    act_cfg = ctc.OptimizationConfig(global_config=None, op_type_configs={
        "linear": OpActivationLinearQuantizerConfig(mode="linear_symmetric"),
        "matmul": OpActivationLinearQuantizerConfig(mode="linear_symmetric"),
        "conv": OpActivationLinearQuantizerConfig(mode="linear_symmetric")})
    m_act = linear_quantize_activations(
        ct.models.MLModel(f"build/Nemotron3Diarizer_{args.variant}_split.mlpackage"),
        act_cfg, sample_data=samples)
    w_cfg = ctc.OptimizationConfig(
        global_config=ctc.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8"))
    m_w8a8 = ctc.linear_quantize_weights(m_act, config=w_cfg)
    m_w8a8.save(f"build/Nemotron3Diarizer_{args.variant}_split_w8a8.mlpackage")
    print(f"saved {args.variant}_split_w8a8")

if __name__ == "__main__":
    main()
