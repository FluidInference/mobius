#!/usr/bin/env python3
"""Trial 23: full-pipeline MLState design — prefill + fused step + voice writer.

Trial 22 proved MLState-resident KV caches save 1.6 ms/frame (44%) on the
step model alone, but flagged the integration wall: the host cannot hand a
voice snapshot or a prefill result to an opaque MLState, so EVERY producer
of the cache (cond_prefill, the step model, the Trial 21 fused step) must
write the SAME state. This script prototypes and measures the full design:

1. Three stateful functions over one shared 12-buffer fp16 KV state
   (`k_cache0..v_cache5`, each [1, L, 16, 64]):
     - `write_state`: KVStateWriter — slice-assigns host-provided k/v
       tensors into the state (voice-snapshot injection + utterance reset
       in one call: it overwrites ALL slots, so no fresh make_state needed
       per utterance).
     - `prefill`   : StatefulCondPrefillANE — Trial 20's hoisted prefill
       graph writing the state instead of returning 12 cache tensors.
     - `generate`  : StatefulFlowLMFusedANE — Trial 21's fused
       flowlm+flowdec (one dispatch per frame) writing the state; returns
       latent_final + is_eos (+ transformer_out kept as a debug output for
       parity, [1,1,1024] fp32 ≈ 4 KB — drop for ship).
   Merged into ONE multifunction mlpackage via
   `ct.utils.MultiFunctionDescriptor` + `save_multifunction` (iOS18).

2. State-sharing experiments: (a) one MLState across the three functions
   of the multifunction package (the design requirement), (b) one MLState
   across two SEPARATE stateful mlpackages (expected to be rejected —
   verified + documented).

3. Voice-snapshot injection round-trip: write the real alba v2 snapshot
   (126 positions, fp32) through `write_state`, read the state back
   (st.read_state) and require bit-exact fp16(snapshot); then run one
   generate step and compare vs the cache-as-IO models fed the same caches.
   Also measures the no-model alternative: `MLState.write_state` (the
   python binding of the Swift `MLState.withMultiArray(for:)` mutable
   accessor).

4. Endgame benchmark — one simulated utterance per variant
   (voice reset + 1 prefill + N generation frames, default 40):
     A  io-pair  : cond_prefill_ane + (flowlm_step_ane + flow_decoder_fused)
     B  io-fused : cond_prefill_ane + flowlm_flow_fused
     C  state    : write_state + prefill + generate   (one shared MLState)
     C' state/ws : like C but voice reset via MLState.write_state x12
   plus `make_state()` cost (median), since a per-utterance state reset via
   make_state would eat the win for short utterances.

5. Parity over the full simulated utterance (same per-frame inputs both
   sides): worst |d_transformer_out| / |d_eos| / |d_latent_final|,
   (a) CPU_ONLY with the fp16-rounded snapshot on both sides (algorithmic)
   and (b) CPU_AND_NE with each side's deployment inputs (fp16 band).

Usage (from models/tts/pocket_tts):
    uv run python coreml/bench_pipeline_mlstate.py [--language english]
        [--phases convert,merge,share,writer,bench,parity,profile]
        [--frames 40] [--utterances 15] [--skip-convert]
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time

import coremltools as ct
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../coreml
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "convert_models", "traceable"))

H, D = 16, 64
NUM_LAYERS = 6
L = 512
T_MAX = 256
LDIM = 32
NUM_FLOW_STEPS = 8

STATE_NAMES = [f"{kind}_cache{i}" for i in range(NUM_LAYERS) for kind in ("k", "v")]

# Real assets: alba v2 voice snapshot (per-layer [2,1,126,16,64] KV) — the
# exact file the FluidAudio host ships — and a fixed benchmark sentence.
ALBA_REPO = "kyutai/pocket-tts-without-voice-cloning"
ALBA_FILE = "languages/english/embeddings/alba.safetensors"
ALBA_REV = "e041936c75475d350b405bc870bcf7c22da4e9e6"
BENCH_TEXT = "Hello, this is a full pipeline state test of the pocket text to speech system."


# --------------------------------------------------------------------------
# Stateful torch modules (built lazily — torch only needed for `convert`)
# --------------------------------------------------------------------------


def _build_stateful_modules(language: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from traceable_cond_prefill_ane import TraceableCondPrefillANE
    from traceable_flow_decoder_fused import TraceableFlowDecoderFused
    from traceable_flowlm_step_ane import TraceableFlowLMStepANE

    class StatefulCondPrefillANE(TraceableCondPrefillANE):
        """Trial 20 prefill writing 12 state buffers instead of cache I/O.

        Same weights/graph; `_shared_block_values` + `_streaming_attention_block`
        are the parent's verbatim. Single shared `position` input (the host
        contract already requires all per-layer positions equal).
        """

        def __init__(self, num_layers: int = 6, max_seq_len: int = 512, t_max: int = 256):
            super().__init__(num_layers=num_layers, max_seq_len=max_seq_len, t_max=t_max)
            for i in range(num_layers):
                self.register_buffer(f"k_cache{i}", torch.zeros(1, max_seq_len, H, D), persistent=False)
                self.register_buffer(f"v_cache{i}", torch.zeros(1, max_seq_len, H, D), persistent=False)

        def forward(self, conditioning, valid_len, position):  # type: ignore[override]
            x = conditioning
            pos0 = position.float() if position.dtype != torch.float32 else position
            valid_f = valid_len.float() if valid_len.dtype != torch.float32 else valid_len
            rotr, roti, assign, covered, mask_bias = self._shared_block_values(pos0, valid_f, x.dtype)

            for i in range(self.num_layers):
                residual = x
                x_norm = getattr(self, f"norm{i}_1")(x)
                attn_out, new_k, new_v, _ = self._streaming_attention_block(
                    x_norm,
                    getattr(self, f"attn{i}_in_proj"),
                    getattr(self, f"attn{i}_out_proj"),
                    getattr(self, f"k_cache{i}"),
                    getattr(self, f"v_cache{i}"),
                    position,
                    valid_f,
                    rotr,
                    roti,
                    assign,
                    covered,
                    mask_bias,
                )
                # buf[:] = new — the only in-place pattern the jit frontend
                # lowers to coreml_update_state (see bench_flowlm_mlstate.py).
                getattr(self, f"k_cache{i}")[:] = new_k
                getattr(self, f"v_cache{i}")[:] = new_v
                x = residual + attn_out

                residual = x
                x_norm = getattr(self, f"norm{i}_2")(x)
                ffn_out = getattr(self, f"linear{i}_2")(F.gelu(getattr(self, f"linear{i}_1")(x_norm)))
                x = residual + ffn_out

            new_position = pos0 + valid_f
            return new_position

    class StatefulFlowLMFusedANE(TraceableFlowLMStepANE):
        """Trial 21 fused flowlm+flowdec writing 12 state buffers.

        Step math is the parent's `_streaming_attention_t1` verbatim; the
        flow decoder is the verified `TraceableFlowDecoderFused` submodule.
        `transformer_out` is returned as a debug output for parity checks
        (drop it for ship; it is the only non-essential I/O left).
        """

        def __init__(self, flow_decoder, num_layers: int = 6, max_seq_len: int = 512):
            super().__init__(num_layers=num_layers, max_seq_len=max_seq_len)
            self.flow_decoder = flow_decoder
            for i in range(num_layers):
                self.register_buffer(f"k_cache{i}", torch.zeros(1, max_seq_len, H, D), persistent=False)
                self.register_buffer(f"v_cache{i}", torch.zeros(1, max_seq_len, H, D), persistent=False)

        def forward(self, sequence, latent_init, position):  # type: ignore[override]
            x = self.input_linear(sequence)
            for i in range(self.num_layers):
                residual = x
                x_norm = getattr(self, f"norm{i}_1")(x)
                attn_out, new_k, new_v, _ = self._streaming_attention_t1(
                    x_norm,
                    getattr(self, f"attn{i}_in_proj"),
                    getattr(self, f"attn{i}_out_proj"),
                    getattr(self, f"k_cache{i}"),
                    getattr(self, f"v_cache{i}"),
                    position,
                )
                getattr(self, f"k_cache{i}")[:] = new_k
                getattr(self, f"v_cache{i}")[:] = new_v
                x = residual + attn_out

                residual = x
                x_norm = getattr(self, f"norm{i}_2")(x)
                ffn_out = getattr(self, f"linear{i}_2")(F.gelu(getattr(self, f"linear{i}_1")(x_norm)))
                x = residual + ffn_out

            x = self.out_norm(x)
            is_eos = self.out_eos(x)
            latent_final = self.flow_decoder(x.reshape(1, 1024), latent_init)
            return latent_final, is_eos, x

    class KVStateWriter(nn.Module):
        """Voice-snapshot injection + utterance reset: overwrite ALL state slots.

        Inputs are the host's zero-padded fp32 snapshot tensors (k_in{i} /
        v_in{i} [1, L, H, D]); each is slice-assigned over the entire state
        buffer, so one call both resets the cache and seeds the voice — no
        per-utterance make_state required. `ack` is a tiny input-dependent
        output (CoreML requires >= 1 output).
        """

        def __init__(self, num_layers: int = 6, max_seq_len: int = 512):
            super().__init__()
            self.num_layers = num_layers
            for i in range(num_layers):
                self.register_buffer(f"k_cache{i}", torch.zeros(1, max_seq_len, H, D), persistent=False)
                self.register_buffer(f"v_cache{i}", torch.zeros(1, max_seq_len, H, D), persistent=False)

        def forward(self, *kv: torch.Tensor):
            for i in range(self.num_layers):
                getattr(self, f"k_cache{i}")[:] = kv[2 * i]
                getattr(self, f"v_cache{i}")[:] = kv[2 * i + 1]
            ack = kv[0].reshape(-1)[:1]
            return ack

    print(f"Loading model (language={language})...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(language=language, lsd_decode_steps=NUM_FLOW_STEPS)
    model.eval()

    base_prefill = TraceableCondPrefillANE.from_flowlm(model.flow_lm, max_seq_len=L, t_max=T_MAX)
    prefill = StatefulCondPrefillANE(num_layers=base_prefill.num_layers, max_seq_len=L, t_max=T_MAX)
    prefill.load_state_dict(base_prefill.state_dict(), strict=False)
    prefill.eval()

    base_step = TraceableFlowLMStepANE.from_flowlm(model.flow_lm, max_seq_len=L)
    flowdec = TraceableFlowDecoderFused.from_flowlm(model.flow_lm, num_steps=NUM_FLOW_STEPS)
    fused = StatefulFlowLMFusedANE(flowdec, num_layers=base_step.num_layers, max_seq_len=L)
    fused.load_state_dict(base_step.state_dict(), strict=False)
    fused.eval()

    writer = KVStateWriter(num_layers=NUM_LAYERS, max_seq_len=L)
    writer.eval()

    return model, prefill, fused, writer


def _state_types():
    return [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, L, H, D), dtype=np.float16),
            name=name,
        )
        for name in STATE_NAMES
    ]


def _export_assets(model, build_dir: str) -> str:
    """Dump real bos_emb + text embeddings for the fixed benchmark sentence."""
    import sentencepiece as sp

    sys.path.insert(0, _SCRIPT_DIR)
    from generate_coreml_v4 import prepare_text_prompt

    tok_path = os.path.join(_SCRIPT_DIR, "v2.1", "english", "constants_bin", "tokenizer.model")
    tk = sp.SentencePieceProcessor()
    tk.load(tok_path)
    prepared, _ = prepare_text_prompt(BENCH_TEXT)
    token_ids = tk.encode(prepared)
    embed_table = model.flow_lm.conditioner.embed.weight.detach().numpy().astype(np.float32)
    text_emb = embed_table[token_ids][np.newaxis].astype(np.float32)  # [1, T_text, 1024]
    bos_emb = model.flow_lm.bos_emb.data.numpy().astype(np.float32).reshape(32)

    path = os.path.join(build_dir, "trial23_assets.npz")
    np.savez(path, text_emb=text_emb, bos_emb=bos_emb)
    print(f"assets: {path} (text tokens: {text_emb.shape[1]})")
    return path


def phase_convert(language: str, build_dir: str, skip_convert: bool) -> None:
    import torch

    paths = artifact_paths(build_dir)
    if skip_convert and all(os.path.exists(p) for p in paths.values()):
        print("convert: all artifacts present, skipping")
        return

    model, prefill, fused, writer = _build_stateful_modules(language)
    _export_assets(model, build_dir)

    def _skip(key: str) -> bool:
        if skip_convert and os.path.exists(paths[key]):
            print(f"convert: {paths[key]} exists, skipping")
            return True
        return False

    # --- prefill ---
    if not _skip("prefill"):
        _convert_prefill(prefill, paths["prefill"])
    if not _skip("generate"):
        _convert_generate(fused, paths["generate"])
    if not _skip("writer"):
        _convert_writer(writer, paths["writer"])


def _convert_prefill(prefill, out_path: str) -> None:
    import torch

    print("Tracing + converting prefill (stateful)...")
    with torch.no_grad():
        traced = torch.jit.trace(
            prefill,
            (torch.randn(1, T_MAX, 1024), torch.tensor([141.0]), torch.tensor([126.0])),
        )
    m = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="conditioning", shape=(1, T_MAX, 1024), dtype=np.float32),
            ct.TensorType(name="valid_len", shape=(1,), dtype=np.float32),
            ct.TensorType(name="position", shape=(1,), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="new_position", dtype=np.float32)],
        states=_state_types(),
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    m.save(out_path)
    print(f"saved {out_path}")


def _convert_generate(fused, out_path: str) -> None:
    import torch

    print("Tracing + converting fused generate (stateful)...")
    with torch.no_grad():
        traced = torch.jit.trace(
            fused, (torch.randn(1, 1, 32), torch.randn(1, 32), torch.tensor([136.0]))
        )
    m = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sequence", shape=(1, 1, 32), dtype=np.float32),
            ct.TensorType(name="latent_init", shape=(1, 32), dtype=np.float32),
            ct.TensorType(name="position", shape=(1,), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="latent_final", dtype=np.float32),
            ct.TensorType(name="is_eos", dtype=np.float32),
            ct.TensorType(name="transformer_out", dtype=np.float32),
        ],
        states=_state_types(),
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    m.save(out_path)
    print(f"saved {out_path}")


def _convert_writer(writer, out_path: str) -> None:
    import torch

    # --- writer ---
    # fp16 inputs: with fp32 inputs the in-graph fp32->fp16 boundary cast
    # rounds 1 ULP differently from numpy on ~18/524288 snapshot values
    # (measured), breaking bit-exactness. fp16 in -> fp16 state is a pure
    # copy (bit-exact) and halves the marshalled bytes (25.2 -> 12.6 MB).
    # The host converts the snapshot to fp16 once at voice-load time.
    print("Tracing + converting kv state writer (fp16 inputs)...")
    kv_inputs = []
    ct_inputs = []
    for i in range(NUM_LAYERS):
        kv_inputs.extend([torch.randn(1, L, H, D), torch.randn(1, L, H, D)])
        ct_inputs.append(ct.TensorType(name=f"k_in{i}", shape=(1, L, H, D), dtype=np.float16))
        ct_inputs.append(ct.TensorType(name=f"v_in{i}", shape=(1, L, H, D), dtype=np.float16))
    with torch.no_grad():
        traced = torch.jit.trace(writer, tuple(kv_inputs))
    m = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=[ct.TensorType(name="ack", dtype=np.float32)],
        states=_state_types(),
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    m.save(out_path)
    print(f"saved {out_path}")


def artifact_paths(build_dir: str) -> dict:
    return {
        "prefill": os.path.join(build_dir, "cond_prefill_ane_state.mlpackage"),
        "generate": os.path.join(build_dir, "flowlm_fused_state.mlpackage"),
        "writer": os.path.join(build_dir, "kv_state_writer.mlpackage"),
    }


def _dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / 1e6


def phase_merge(build_dir: str) -> str:
    paths = artifact_paths(build_dir)
    mf_path = os.path.join(build_dir, "pocket_flowlm_mf_state.mlpackage")

    desc = ct.utils.MultiFunctionDescriptor()
    desc.add_function(paths["prefill"], "main", "prefill")
    desc.add_function(paths["generate"], "main", "generate")
    desc.add_function(paths["writer"], "main", "write_state")
    desc.default_function_name = "generate"
    print("Merging into multifunction package...")
    ct.utils.save_multifunction(desc, mf_path)

    sizes = {k: _dir_size_mb(p) for k, p in paths.items()}
    mf_size = _dir_size_mb(mf_path)
    print(f"sizes: prefill={sizes['prefill']:.1f} MB, generate={sizes['generate']:.1f} MB, "
          f"writer={sizes['writer']:.1f} MB (sum {sum(sizes.values()):.1f} MB)")
    print(f"multifunction package: {mf_size:.1f} MB "
          f"(weight dedup saves {sum(sizes.values()) - mf_size:.1f} MB)")
    return mf_path


def _load_mf(build_dir: str, compute_units=None):
    cu = compute_units or ct.ComputeUnit.CPU_AND_NE
    mf_path = os.path.join(build_dir, "pocket_flowlm_mf_state.mlpackage")
    mp = ct.models.MLModel(mf_path, compute_units=cu, function_name="prefill")
    mg = ct.models.MLModel(mf_path, compute_units=cu, function_name="generate")
    mw = ct.models.MLModel(mf_path, compute_units=cu, function_name="write_state")
    return mp, mg, mw


def _load_snapshot() -> tuple[list[np.ndarray], list[np.ndarray], int]:
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    path = hf_hub_download(ALBA_REPO, ALBA_FILE, revision=ALBA_REV)
    tensors = load_file(path)
    ks, vs = [], []
    prompt_len = None
    for i in range(NUM_LAYERS):
        raw = tensors[f"transformer.layers.{i}.self_attn/cache"].astype(np.float32)
        prompt_len = raw.shape[2]
        k = np.zeros((1, L, H, D), dtype=np.float32)
        v = np.zeros((1, L, H, D), dtype=np.float32)
        k[:, :prompt_len] = raw[0]
        v[:, :prompt_len] = raw[1]
        ks.append(k)
        vs.append(v)
    return ks, vs, int(prompt_len)


def _writer_feed(ks, vs) -> dict:
    """Writer-model feed: fp16 (bit-exact copy into the fp16 state)."""
    feed = {}
    for i in range(NUM_LAYERS):
        feed[f"k_in{i}"] = ks[i].astype(np.float16)
        feed[f"v_in{i}"] = vs[i].astype(np.float16)
    return feed


def phase_share(build_dir: str) -> None:
    """Q1: which models/functions can share one MLState?"""
    print("\n=== state sharing ===")
    mp, mg, mw = _load_mf(build_dir)
    ks, vs, prompt_len = _load_snapshot()

    # (a) one state across the three functions of the multifunction package.
    st = mg.make_state()
    seq = np.zeros((1, 1, 32), np.float32)
    z0 = np.zeros((1, 32), np.float32)
    try:
        mw.predict(_writer_feed(ks, vs), state=st)
        got_k0 = st.read_state("k_cache0")
        ok_write = np.array_equal(got_k0.astype(np.float32), ks[0].astype(np.float16).astype(np.float32))
        mp.predict(
            {"conditioning": np.zeros((1, T_MAX, 1024), np.float32),
             "valid_len": np.array([4.0], np.float32),
             "position": np.array([float(prompt_len)], np.float32)},
            state=st,
        )
        mg.predict({"sequence": seq, "latent_init": z0,
                    "position": np.array([float(prompt_len) + 4.0], np.float32)}, state=st)
        print(f"  [PASS] one MLState shared across write_state -> prefill -> generate "
              f"(multifunction; writer round-trip exact: {ok_write})")
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] multifunction cross-function state sharing: {type(e).__name__}: {e}")

    # (b) one state across two SEPARATE stateful mlpackages. Expected to be
    # rejected (MLState is documented per-model) — verify FUNCTIONALLY:
    # seed two states identically via write_state, run the same generate
    # call on the own-state and the foreign-state, require bit-equality.
    sep_step_path = os.path.join(build_dir, "flowlm_step_ane_state.mlpackage")
    if os.path.exists(sep_step_path):
        m_sep = ct.models.MLModel(sep_step_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        st_sep = m_sep.make_state()  # made by the Trial 22 step model
        st_own = mg.make_state()  # made by the multifunction generate fn
        try:
            for i, name in enumerate(STATE_NAMES):
                src = (ks if name.startswith("k") else vs)[i // 2]
                st_sep.write_state(name, src)
                st_own.write_state(name, src)
            feed = {"sequence": seq, "latent_init": z0,
                    "position": np.array([float(prompt_len)], np.float32)}
            ref = mg.predict(feed, state=st_own)
            got = mg.predict(feed, state=st_sep)
            d = float(np.abs(got["transformer_out"] - ref["transformer_out"]).max())
            updated = bool(np.abs(st_sep.read_state("k_cache0")[:, prompt_len]).max() > 0)
            print(f"  [UNEXPECTED PASS] a foreign model's MLState is accepted AND functional "
                  f"(d_transformer_out vs own state: {d:.3e}; foreign state updated: {updated}). "
                  f"Undocumented — do not design around it; ship the multifunction package.")
        except Exception as e:  # noqa: BLE001
            print(f"  [EXPECTED FAIL] separate stateful models cannot share an MLState: "
                  f"{type(e).__name__}: {str(e)[:200]}")
    else:
        print(f"  (skip separate-model test: {sep_step_path} missing — run bench_flowlm_mlstate.py)")


def phase_writer(build_dir: str) -> None:
    """Q2: voice-snapshot injection round-trip + write_state alternative."""
    print("\n=== voice-snapshot injection ===")
    cu = ct.ComputeUnit.CPU_AND_NE
    mp, mg, mw = _load_mf(build_dir, cu)
    ks, vs, prompt_len = _load_snapshot()
    assets = np.load(os.path.join(build_dir, "trial23_assets.npz"))
    bos = assets["bos_emb"].reshape(1, 1, 32).astype(np.float32)

    # --- writer-model path ---
    st = mg.make_state()
    feed = _writer_feed(ks, vs)
    t0 = time.perf_counter()
    mw.predict(feed, state=st)
    t_writer = (time.perf_counter() - t0) * 1000.0
    worst_mismatch = 0
    for i, name in enumerate(STATE_NAMES):
        src = (ks if name.startswith("k") else vs)[i // 2]
        got = st.read_state(name)
        ref = src.astype(np.float16)
        worst_mismatch = max(worst_mismatch, int((got != ref).sum()))
    print(f"  writer model: {t_writer:.2f} ms/call; read_state vs fp16(snapshot): "
          f"{'bit-exact' if worst_mismatch == 0 else f'{worst_mismatch} mismatches'}")

    # --- MLState.write_state path (Swift: MLState.withMultiArray mutable) ---
    # NB: the python binding only converts fp32 ndarrays ("value type not
    # convertible" for np.float16); its internal fp32->fp16 cast matches
    # numpy rounding (verified bit-exact below).
    st2 = mg.make_state()
    t0 = time.perf_counter()
    for i, name in enumerate(STATE_NAMES):
        src = (ks if name.startswith("k") else vs)[i // 2]
        st2.write_state(name, src)
    t_ws = (time.perf_counter() - t0) * 1000.0
    ok2 = all(
        np.array_equal(st2.read_state(name),
                       (ks if name.startswith("k") else vs)[i // 2].astype(np.float16))
        for i, name in enumerate(STATE_NAMES)
    )
    print(f"  MLState.write_state x12: {t_ws:.2f} ms; round-trip exact: {ok2}")

    # --- one generate step vs the IO models fed the same caches ---
    z0 = np.random.default_rng(0).standard_normal((1, 32)).astype(np.float32)
    pos = np.array([float(prompt_len)], np.float32)
    got = mg.predict({"sequence": bos, "latent_init": z0, "position": pos}, state=st)

    io_step = ct.models.MLModel(os.path.join(build_dir, "flowlm_step_ane.mlpackage"), compute_units=cu)
    io_flow = ct.models.MLModel(os.path.join(build_dir, "flow_decoder_fused.mlpackage"), compute_units=cu)
    io_feed = {"sequence": bos}
    for i in range(NUM_LAYERS):
        io_feed[f"k_cache{i}"] = ks[i]
        io_feed[f"v_cache{i}"] = vs[i]
        io_feed[f"position{i}"] = pos
    ref = io_step.predict(io_feed)
    ref_latent = io_flow.predict(
        {"transformer_out": ref["transformer_out"].reshape(1, 1024), "latent_init": z0}
    )["latent_final"]
    d_t = float(np.abs(got["transformer_out"] - ref["transformer_out"]).max())
    d_e = float(np.abs(got["is_eos"] - ref["is_eos"]).max())
    d_l = float(np.abs(got["latent_final"] - ref_latent).max())
    print(f"  1-step vs IO pipeline (same caches, CPU_AND_NE): "
          f"d_transformer_out={d_t:.3e} d_eos={d_e:.3e} d_latent={d_l:.3e}")


def _make_utterance_inputs(build_dir: str, frames: int, seed: int = 0):
    assets = np.load(os.path.join(build_dir, "trial23_assets.npz"))
    text_emb = assets["text_emb"].astype(np.float32)
    bos = assets["bos_emb"].reshape(1, 1, 32).astype(np.float32)
    t_len = text_emb.shape[1]
    cond_block = np.zeros((1, T_MAX, 1024), np.float32)
    cond_block[:, :t_len] = text_emb
    rng = np.random.default_rng(seed)
    z0s = rng.standard_normal((frames, 1, 32)).astype(np.float32) * (0.7**0.5)
    return cond_block, float(t_len), bos, z0s


def phase_bench(build_dir: str, frames: int, utterances: int) -> None:
    """Q3: endgame benchmark — full simulated utterance per variant."""
    print(f"\n=== endgame benchmark ({frames} frames, median of {utterances} utterances) ===")
    cu = ct.ComputeUnit.CPU_AND_NE
    ks, vs, prompt_len = _load_snapshot()
    cond_block, t_len, bos, z0s = _make_utterance_inputs(build_dir, frames)
    pos_prefill = float(prompt_len)
    pos_gen0 = pos_prefill + t_len

    # IO models (shipped v2.1 contracts). prefill loaded at .ALL per Trial
    # 20's Mac ship recommendation; step/flow/fused at CPU_AND_NE (Trial 21).
    io_prefill = ct.models.MLModel(
        os.path.join(build_dir, "cond_prefill_ane.mlpackage"), compute_units=ct.ComputeUnit.ALL)
    io_step = ct.models.MLModel(os.path.join(build_dir, "flowlm_step_ane.mlpackage"), compute_units=cu)
    io_flow = ct.models.MLModel(os.path.join(build_dir, "flow_decoder_fused.mlpackage"), compute_units=cu)
    io_fused_path = os.path.join(build_dir, "flowlm_flow_fused.mlpackage")
    io_fused = ct.models.MLModel(io_fused_path, compute_units=cu) if os.path.exists(io_fused_path) else None
    if io_fused is not None:
        fused_spec = io_fused.get_spec()
        fused_in_names = {i.name for i in fused_spec.description.input}
        fused_out_order = [o.name for o in fused_spec.description.output]
    mp, mg, mw = _load_mf(build_dir, cu)


    def run_io_pair():
        t0 = time.perf_counter()
        caches = {}
        for i in range(NUM_LAYERS):
            caches[f"k_cache{i}"] = ks[i].copy()
            caches[f"v_cache{i}"] = vs[i].copy()
        positions = {f"position{i}": np.array([pos_prefill], np.float32) for i in range(NUM_LAYERS)}
        t1 = time.perf_counter()
        out = io_prefill.predict(
            {"conditioning": cond_block, "valid_len": np.array([t_len], np.float32), **caches, **positions})
        for i in range(NUM_LAYERS):
            caches[f"k_cache{i}"] = out[f"new_k_cache{i}"]
            caches[f"v_cache{i}"] = out[f"new_v_cache{i}"]
            positions[f"position{i}"] = out[f"new_position{i}"]
        t2 = time.perf_counter()
        seq = bos
        for f in range(frames):
            so = io_step.predict({"sequence": seq, **caches, **positions})
            for i in range(NUM_LAYERS):
                caches[f"k_cache{i}"] = so[f"new_k_cache{i}"]
                caches[f"v_cache{i}"] = so[f"new_v_cache{i}"]
                positions[f"position{i}"] = so[f"new_position{i}"]
            fo = io_flow.predict(
                {"transformer_out": so["transformer_out"].reshape(1, 1024), "latent_init": z0s[f]})
            seq = fo["latent_final"].reshape(1, 1, 32).astype(np.float32)
        t3 = time.perf_counter()
        return (t1 - t0, t2 - t1, t3 - t2)

    def run_io_fused():
        t0 = time.perf_counter()
        caches = {}
        for i in range(NUM_LAYERS):
            caches[f"k_cache{i}"] = ks[i].copy()
            caches[f"v_cache{i}"] = vs[i].copy()
        positions = {f"position{i}": np.array([pos_prefill], np.float32) for i in range(NUM_LAYERS)}
        t1 = time.perf_counter()
        out = io_prefill.predict(
            {"conditioning": cond_block, "valid_len": np.array([t_len], np.float32), **caches, **positions})
        for i in range(NUM_LAYERS):
            caches[f"k_cache{i}"] = out[f"new_k_cache{i}"]
            caches[f"v_cache{i}"] = out[f"new_v_cache{i}"]
            positions[f"position{i}"] = out[f"new_position{i}"]
        t2 = time.perf_counter()
        seq = bos
        extra = {"bos_emb": bos.reshape(32)} if "bos_emb" in fused_in_names else {}
        for f in range(frames):
            so = io_fused.predict(
                {"sequence": seq, "latent_init": z0s[f], **extra, **caches, **positions})
            # Output names are mangled in the pre-Trial-22 artifact; use the
            # traced positional order (latent, eos, then per-layer triples).
            vals = [so[name] for name in fused_out_order]
            latent = vals[0]
            for i in range(NUM_LAYERS):
                caches[f"k_cache{i}"] = vals[2 + 3 * i]
                caches[f"v_cache{i}"] = vals[3 + 3 * i]
                positions[f"position{i}"] = vals[4 + 3 * i]
            seq = latent.reshape(1, 1, 32).astype(np.float32)
        t3 = time.perf_counter()
        return (t1 - t0, t2 - t1, t3 - t2)

    writer_feed = _writer_feed(ks, vs)

    def run_state(st, reset_via_write_state: bool):
        t0 = time.perf_counter()
        if reset_via_write_state:
            # python binding only converts fp32 ndarrays (fp16 rejected)
            for i, name in enumerate(STATE_NAMES):
                st.write_state(name, (ks if name.startswith("k") else vs)[i // 2])
        else:
            mw.predict(writer_feed, state=st)
        t1 = time.perf_counter()
        mp.predict({"conditioning": cond_block, "valid_len": np.array([t_len], np.float32),
                    "position": np.array([pos_prefill], np.float32)}, state=st)
        t2 = time.perf_counter()
        seq = bos
        for f in range(frames):
            go = mg.predict({"sequence": seq, "latent_init": z0s[f],
                             "position": np.array([pos_gen0 + f], np.float32)}, state=st)
            seq = go["latent_final"].reshape(1, 1, 32).astype(np.float32)
        t3 = time.perf_counter()
        return (t1 - t0, t2 - t1, t3 - t2)

    # make_state cost (risk register): fresh state per utterance.
    t_ms = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = mg.make_state()
        t_ms.append((time.perf_counter() - t0) * 1000.0)
    print(f"make_state(): median {statistics.median(t_ms):.2f} ms over 20 calls "
          f"(not needed per utterance: write_state overwrites all slots)")

    st = mg.make_state()  # persistent state, reset per utterance by the writer
    variants = [
        ("A  io-pair (prefill+step+flowdec)", run_io_pair),
        ("C  state (writer+prefill+generate)", lambda: run_state(st, False)),
        ("C' state (write_state reset)", lambda: run_state(st, True)),
    ]
    if io_fused is not None:
        variants.insert(1, ("B  io-fused (prefill+fused)", run_io_fused))

    print(f"{'variant':<37s} {'reset':>8s} {'prefill':>8s} {'frames':>9s} {'ms/frame':>9s} {'total':>9s}")
    print("-" * 86)
    results = {}
    for name, fn in variants:
        for _ in range(2):
            fn()  # warmup
        rs, ps, fs = [], [], []
        for _ in range(utterances):
            r, p, f = fn()
            rs.append(r * 1000)
            ps.append(p * 1000)
            fs.append(f * 1000)
        r, p, f = statistics.median(rs), statistics.median(ps), statistics.median(fs)
        results[name[:1]] = r + p + f
        print(f"{name:<37s} {r:>6.2f}ms {p:>6.2f}ms {f:>7.1f}ms {f / frames:>7.2f}ms {r + p + f:>7.1f}ms")

    if "A" in results and "C" in results:
        a, c = results["A"], results["C"]
        print(f"\nstateful pipeline saves {a - c:+.1f} ms/utterance vs io-pair "
              f"({(1 - c / a) * 100:.1f}%) at {frames} frames")


def phase_parity(build_dir: str, frames: int) -> None:
    """Q4: stateful vs IO pipeline over a full simulated utterance."""
    print(f"\n=== parity over {frames} frames ===")
    ks, vs, prompt_len = _load_snapshot()
    cond_block, t_len, bos, z0s = _make_utterance_inputs(build_dir, frames)
    pos_prefill = float(prompt_len)
    pos_gen0 = pos_prefill + t_len
    # fp16-rounded snapshot for the algorithmic flavor (both sides identical).
    ks_r = [k.astype(np.float16).astype(np.float32) for k in ks]
    vs_r = [v.astype(np.float16).astype(np.float32) for v in vs]

    for flavor, cu, use_rounded in (
        ("algorithmic (CPU_ONLY, fp16-rounded snapshot both sides)", ct.ComputeUnit.CPU_ONLY, True),
        ("deployment (CPU_AND_NE, each side's shipped inputs)", ct.ComputeUnit.CPU_AND_NE, False),
    ):
        mp, mg, mw = _load_mf(build_dir, cu)
        io_prefill = ct.models.MLModel(os.path.join(build_dir, "cond_prefill_ane.mlpackage"), compute_units=cu)
        io_step = ct.models.MLModel(os.path.join(build_dir, "flowlm_step_ane.mlpackage"), compute_units=cu)
        io_flow = ct.models.MLModel(os.path.join(build_dir, "flow_decoder_fused.mlpackage"), compute_units=cu)

        kk = ks_r if use_rounded else ks
        vv = vs_r if use_rounded else vs

        st = mg.make_state()
        mw.predict(_writer_feed(kk, vv), state=st)
        mp.predict({"conditioning": cond_block, "valid_len": np.array([t_len], np.float32),
                    "position": np.array([pos_prefill], np.float32)}, state=st)

        caches = {}
        for i in range(NUM_LAYERS):
            caches[f"k_cache{i}"] = kk[i].copy()
            caches[f"v_cache{i}"] = vv[i].copy()
        positions = {f"position{i}": np.array([pos_prefill], np.float32) for i in range(NUM_LAYERS)}
        out = io_prefill.predict(
            {"conditioning": cond_block, "valid_len": np.array([t_len], np.float32), **caches, **positions})
        for i in range(NUM_LAYERS):
            caches[f"k_cache{i}"] = out[f"new_k_cache{i}"]
            caches[f"v_cache{i}"] = out[f"new_v_cache{i}"]
            positions[f"position{i}"] = out[f"new_position{i}"]

        worst_t = worst_e = worst_l = 0.0
        seq = bos  # both sides get the SAME sequence each frame (no compounding inputs)
        for f in range(frames):
            go = mg.predict({"sequence": seq, "latent_init": z0s[f],
                             "position": np.array([pos_gen0 + f], np.float32)}, state=st)
            so = io_step.predict({"sequence": seq, **caches, **positions})
            for i in range(NUM_LAYERS):
                caches[f"k_cache{i}"] = so[f"new_k_cache{i}"]
                caches[f"v_cache{i}"] = so[f"new_v_cache{i}"]
                positions[f"position{i}"] = so[f"new_position{i}"]
            fo = io_flow.predict(
                {"transformer_out": so["transformer_out"].reshape(1, 1024), "latent_init": z0s[f]})
            worst_t = max(worst_t, float(np.abs(go["transformer_out"] - so["transformer_out"]).max()))
            worst_e = max(worst_e, float(np.abs(go["is_eos"] - so["is_eos"]).max()))
            worst_l = max(worst_l, float(np.abs(go["latent_final"] - fo["latent_final"]).max()))
            seq = go["latent_final"].reshape(1, 1, 32).astype(np.float32)
        print(f"  {flavor}:")
        print(f"    worst d_transformer_out={worst_t:.3e} d_eos={worst_e:.3e} d_latent={worst_l:.3e}")


def phase_profile(build_dir: str) -> None:
    """ANE placement of every stateful artifact (/tmp/ane_profile on mlmodelc)."""
    print("\n=== placement (/tmp/ane_profile) ===")
    if not os.path.exists("/tmp/ane_profile"):
        print("  /tmp/ane_profile missing — skipping")
        return
    targets = list(artifact_paths(build_dir).values()) + [
        os.path.join(build_dir, "pocket_flowlm_mf_state.mlpackage")]
    for pkg in targets:
        if not os.path.exists(pkg):
            continue
        mlc = pkg.replace(".mlpackage", ".mlmodelc")
        if not os.path.exists(mlc) or os.path.getmtime(mlc) < os.path.getmtime(pkg):
            # ct.utils.compile_model: works with CommandLineTools-only setups
            # (no xcrun coremlcompiler available).
            import shutil
            if os.path.exists(mlc):
                shutil.rmtree(mlc)
            ct.utils.compile_model(pkg, destination_path=mlc)
        r = subprocess.run(["/tmp/ane_profile", mlc], capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if ln.strip() and not ln.startswith(("model", "---"))]
        msg = lines[-1] if lines else (r.stderr.strip()[:200] or "(no output)")
        print(f"  {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="english")
    parser.add_argument("--phases", default="convert,merge,share,writer,bench,parity,profile")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--utterances", type=int, default=15)
    parser.add_argument("--skip-convert", action="store_true")
    args = parser.parse_args()

    build_dir = os.path.join(_SCRIPT_DIR, "build", args.language)
    os.makedirs(build_dir, exist_ok=True)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    if "convert" in phases:
        phase_convert(args.language, build_dir, args.skip_convert)
    if "merge" in phases:
        phase_merge(build_dir)
    if "share" in phases:
        phase_share(build_dir)
    if "writer" in phases:
        phase_writer(build_dir)
    if "bench" in phases:
        phase_bench(build_dir, args.frames, args.utterances)
    if "parity" in phases:
        phase_parity(build_dir, args.frames)
    if "profile" in phases:
        phase_profile(build_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
