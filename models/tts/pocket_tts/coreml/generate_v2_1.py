#!/usr/bin/env python3
"""End-to-end audio generation with the v2.1 optimized stack (validation).

Reuses generate_coreml_v4's asset/introspection helpers but runs the v2.1
contracts: text prefill via cond_prefill (one call), flow via the fused
decoder (one call), fp16 flowlm. Voice = v2 KV snapshot (unchanged). Writes a
24 kHz wav so it can be Whisper-checked.

Usage:
    python generate_v2_1.py --pack v2.1/english --voice alba \
        --text "Hello, this is a text to speech system." --output /tmp/v21_en.wav
"""
import argparse, os, time, sys
import numpy as np
from scipy.io import wavfile
import coremltools as ct
import sentencepiece as sp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import generate_coreml_v4 as g  # helpers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="v2.1/english", help="path to a v2.1/<lang> pack dir")
    ap.add_argument("--voice", default="alba")
    ap.add_argument("--text", default="Hello, this is a text to speech system.")
    ap.add_argument("--output", default="/tmp/v21_en.wav")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pack = os.path.join(_HERE, args.pack) if not os.path.isabs(args.pack) else args.pack
    const_dir = os.path.join(pack, "constants")
    model_dir = pack

    prepared_text, frames_after_eos = g.prepare_text_prompt(args.text)
    frames_after_eos += 2
    print(f"text: '{prepared_text}'")

    # text embeddings
    tok_path = next((p for p in [os.path.join(model_dir, "constants_bin", "tokenizer.model"),
                                 os.path.join(const_dir, "tokenizer.model")] if os.path.isfile(p)), None)
    tk = sp.SentencePieceProcessor(); tk.load(tok_path)
    token_ids = tk.encode(prepared_text)
    embed_table = np.load(os.path.join(const_dir, "text_embed_table.npy"))
    text_emb = embed_table[token_ids][np.newaxis, :, :].astype(np.float32)  # [1, T_text, 1024]
    bos_emb = np.load(os.path.join(const_dir, "bos_emb.npy")).astype(np.float32)
    voice_v2_path = g._locate_voice_v2(const_dir, model_dir, args.voice)
    if voice_v2_path is None:
        raise FileNotFoundError(f"no v2 voice for {args.voice} in {const_dir}")

    # ---- load v2.1 models ----
    CU = ct.ComputeUnit
    print("loading v2.1 models...")
    cond = ct.models.MLModel(os.path.join(model_dir, "cond_prefill.mlpackage"), compute_units=CU.CPU_AND_GPU)
    step = ct.models.MLModel(os.path.join(model_dir, "flowlm_step.mlpackage"), compute_units=CU.ALL)
    flow = ct.models.MLModel(os.path.join(model_dir, "flow_decoder_fused.mlpackage"), compute_units=CU.ALL)
    mimi = ct.models.MLModel(os.path.join(model_dir, "mimi_decoder.mlpackage"), compute_units=CU.CPU_ONLY)

    num_layers, cache_slots = g._introspect_cache_shape(step)
    cond_cache_keys, cond_pos_keys = g._introspect_cond_output_keys(cond, num_layers)  # same schema as cond_step
    sx, se, scache, spos = g._introspect_step_output_keys(step, num_layers)

    # ---- voice KV snapshot ----
    vs = g._load_voice_state_v2(voice_v2_path, cache_slots, num_layers)
    caches, positions = vs["caches"], vs["positions"]
    print(f"voice-seeded; text tokens to prefill: {text_emb.shape[1]}")

    # ---- text prefill via cond_prefill (ONE call) ----
    T_MAX = 256
    t_len = text_emb.shape[1]
    assert t_len <= T_MAX, f"text {t_len} > T_max {T_MAX}"
    cond_block = np.zeros((1, T_MAX, 1024), np.float32)
    cond_block[:, :t_len, :] = text_emb
    feed = {"conditioning": cond_block, "valid_len": np.array([float(t_len)], np.float32),
            **caches, **positions}
    t0 = time.time()
    cout = cond.predict(feed)
    for i in range(num_layers):
        caches[f"cache{i}"] = cout[cond_cache_keys[i]]
        positions[f"position{i}"] = cout[cond_pos_keys[i]]
    print(f"cond_prefill 1 call: {(time.time()-t0)*1000:.1f}ms; pos0={positions['position0'][0]}")

    # ---- AR loop: flowlm -> fused decoder (1 call) -> mimi ----
    max_gen = int((len(args.text.split())*1 + 2.0) * 12.5)
    np.random.seed(args.seed)
    # mimi state init (copied from generate_coreml_v4)
    mstate = {}; min_order = []
    msp = mimi.get_spec()
    for inp in msp.description.input:
        if inp.name == "latent": continue
        shp = tuple(int(d) for d in inp.type.multiArrayType.shape)
        mstate[inp.name] = (np.ones(shp, np.float32) if inp.name.endswith("_first") else np.zeros(shp, np.float32))
        min_order.append(inp.name)
    mout_names = [o.name for o in msp.description.output]

    seq = np.full((1,1,32), np.nan, np.float32); temp=0.7
    eos_step=None; chunks=[]
    tg=time.time()
    for sstep in range(max_gen):
        so = step.predict({"sequence": seq, "bos_emb": bos_emb, **caches, **positions})
        tout = so[sx]; eos = float(so[se].flatten()[0])
        for i in range(num_layers):
            caches[f"cache{i}"]=so[scache[i]]; positions[f"position{i}"]=so[spos[i]]
        if eos > -4.0 and eos_step is None:
            eos_step=sstep; print(f"  EOS at {sstep} (logit {eos:+.2f})")
        if eos_step is not None and sstep >= eos_step + frames_after_eos: break
        # fused decoder: ONE call, z0 -> z_N
        z0 = (np.random.randn(1,32).astype(np.float32) * (temp**0.5))
        fo = flow.predict({"transformer_out": tout.reshape(1,1024), "latent_init": z0})
        latent = list(fo.values())[0]
        mo = mimi.predict({"latent": latent.astype(np.float32), **mstate})
        chunks.append(mo[mout_names[0]])
        for sn, on in zip(min_order, mout_names[1:]): mstate[sn]=mo[on]
        seq = latent.reshape(1,1,32)
    gen=time.time()-tg
    n=len(chunks); secs=n*0.08
    print(f"generated {n} frames, {secs:.2f}s audio in {gen:.2f}s (RTFx {secs/max(gen,1e-6):.2f}x)")

    audio=np.concatenate(chunks,axis=-1)[0,0]
    audio=audio/(np.abs(audio).max()+1e-8)*0.9
    wavfile.write(args.output, 24000, (audio*32767).astype(np.int16))
    print(f"saved {args.output} ({len(audio)/24000:.2f}s)")


if __name__ == "__main__":
    main()
