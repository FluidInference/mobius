#!/usr/bin/env python3
"""Phase 2b: VoiceChat-11B TTS one-step decoder (gemma3 backbone + MoG head) -> CoreML.

Per 80 ms frame (batch 2 = CFG cond+uncond rows, guidance scale 0.2):
  host prep: code_embeds = embed_code(depthsum_embedding(prev_code)) ; cond =
    subword embedding (cond row) / null_emb (uncond row) ; inputs_embeds =
    gated_fusion(code_embeds, cond)                       [tiny — Phase 4 Swift]
  CoreML A (stateful): gemma3_text backbone step [2,1,1152] -> [2,1,1152]
    28L, h1152, 16 heads, head_dim 72, q/k RMSNorm, sandwich norms,
    layer_types 5x sliding_attention + full_attention (i%6==5), RoPE theta
    10k (sliding) / 1e6 (full), scale 256**-0.5, gelu-tanh MLP. KV window is a
    rolling KV_WINDOW-slot state with pos-derived validity masking (exact
    below KV_WINDOW frames; full-attention layers see the last KV_WINDOW).
  CoreML B (x8 iterations): MoG dense pass — mlp_stack + CFG combine ->
    mixture logits [1,1024], log-std [1,1] (clamped -4), mu_res [1,512],
    guided hidden [1,1152].
  host: top-p+gumbel (or argmax) mixture pick; mu = low_mat[idx] @
    (proj_mus[idx] @ x); z = mu*exp(logs)+mu_res (+noise); RVQ
    depthsum_encoding_step fills the scheduled quantizers (self-inverse
    power schedule, exponent 3, num_iter 8).

Commands: convert | parity | bench   (parity exits nonzero on failure)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
import typer
from safetensors.torch import load_file

CHECKPOINT_DIR = Path.home() / "Documents/models/voicechat-11b"
COMPONENTS_DIR = CHECKPOINT_DIR / "components"
BUILD = Path("build/tts")

HID, NH, HD, NL, EPS = 1152, 16, 72, 28, 1e-6
SCALE = 256**-0.5
ROPE_LOCAL, ROPE_GLOBAL = 10000.0, 1000000.0
KV_WINDOW = 1024
NQ, VCB, LAT = 31, 1024, 512
GUIDANCE = 0.2

app = typer.Typer(add_completion=False)


def tts_config() -> dict:
    cfg = json.loads((CHECKPOINT_DIR / "config.json").read_text())
    return cfg["model"]["speech_generation"]["model"]["tts_config"]


def build_tts():
    """Build RVQEARTTSModel with real weights. Returns (model, has_subword)."""
    from omegaconf import DictConfig

    cfg = dict(tts_config())
    tok = None
    try:
        from nemo.collections.common.tokenizers import AutoTokenizer

        tok = AutoTokenizer(cfg["cas_config"]["pretrained_tokenizer_name"])
        # installed nemo predates the branch's kwarg filter — strip it ourselves
        cfg["cas_config"] = {k: v for k, v in cfg["cas_config"].items() if k != "pretrained_tokenizer_name"}
    except Exception as e:  # offline etc. — build without the subword encoder
        typer.echo(f"WARN: tokenizer unavailable ({type(e).__name__}) — embed_subword disabled")
        cfg["cas_config"] = None

    from nemo.collections.speechlm2.modules.ear_tts_model import RVQEARTTSModel

    model = RVQEARTTSModel(DictConfig(cfg), tokenizer=tok)
    sd = load_file(COMPONENTS_DIR / "tts.safetensors")
    prefix = "tts_model.tts_model."
    stripped = {k[len(prefix) :]: v.float() for k, v in sd.items() if k.startswith(prefix)}
    model.set_rvq_embs(stripped.pop("rvq_embs"))
    # RVQEARTTSModel overrides load_state_dict (returns None) — use the base one
    missing, unexpected = torch.nn.Module.load_state_dict(model, stripped, strict=False)
    ok_missing = [k for k in missing if k == "rvq_embs"]
    assert len(missing) == len(ok_missing), f"missing: {[k for k in missing if k not in ok_missing][:5]}"
    if tok is None:
        unexpected = [k for k in unexpected if not k.startswith("embed_subword.")]
    # installed nemo predates the frozen audio-prompt projection buffer; it is
    # session-init-only (speaker prompt path), not part of the per-frame step
    unexpected = [k for k in unexpected if k != "audio_prompt_projection_W"]
    assert not unexpected, f"unexpected: {unexpected[:5]}"
    model.eval()
    return model, tok is not None


def grms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w)


def layer_is_full(i: int) -> bool:
    return i % 6 == 5  # gemma3 default: every 6th layer is full_attention


class Gemma3Step(torch.nn.Module):
    """Single-frame backbone step, batch 2, rolling stateful KV window."""

    def __init__(self, backbone):
        super().__init__()
        g = lambda t: torch.nn.Parameter(t.detach().float(), requires_grad=False)
        for i, layer in enumerate(backbone.layers):
            a = layer.self_attn
            for name, t in (
                ("ln_in", layer.input_layernorm.weight),
                ("qw", a.q_proj.weight), ("kw", a.k_proj.weight), ("vw", a.v_proj.weight),
                ("ow", a.o_proj.weight), ("qn", a.q_norm.weight), ("kn", a.k_norm.weight),
                ("ln_pa", layer.post_attention_layernorm.weight),
                ("ln_pf", layer.pre_feedforward_layernorm.weight),
                ("gate", layer.mlp.gate_proj.weight), ("up", layer.mlp.up_proj.weight),
                ("down", layer.mlp.down_proj.weight),
                ("ln_ff", layer.post_feedforward_layernorm.weight),
            ):
                setattr(self, f"{name}_{i}", g(t))
            self.register_buffer(f"k_cache_{i}", torch.zeros(2, NH, KV_WINDOW, HD))
            self.register_buffer(f"v_cache_{i}", torch.zeros(2, NH, KV_WINDOW, HD))
        self.norm_f = g(backbone.norm.weight)
        self.register_buffer("pos", torch.zeros(1))
        inv = lambda base: torch.nn.Parameter(
            1.0 / base ** (torch.arange(0, HD, 2).float() / HD), requires_grad=False
        )
        self.inv_local, self.inv_global = inv(ROPE_LOCAL), inv(ROPE_GLOBAL)

    def reset(self):
        for _, b in self.named_buffers():
            b.zero_()

    def forward(self, x):  # x [2, 1, 1152]
        idx = torch.arange(KV_WINDOW, dtype=torch.float32)
        valid = torch.clamp(self.pos + 1.0, max=float(KV_WINDOW))
        neg = torch.where(idx < (KV_WINDOW - valid), torch.full_like(idx, -3e4), torch.zeros_like(idx))
        for i in range(NL):
            h = grms(x, getattr(self, f"ln_in_{i}"))
            q = (h @ getattr(self, f"qw_{i}").T).reshape(2, 1, NH, HD).transpose(1, 2)
            k = (h @ getattr(self, f"kw_{i}").T).reshape(2, 1, NH, HD).transpose(1, 2)
            v = (h @ getattr(self, f"vw_{i}").T).reshape(2, 1, NH, HD).transpose(1, 2)
            q = grms(q, getattr(self, f"qn_{i}"))
            k = grms(k, getattr(self, f"kn_{i}"))
            inv_freq = self.inv_global if layer_is_full(i) else self.inv_local
            ang = self.pos * inv_freq  # [36]
            cos = torch.cat([ang.cos(), ang.cos()], -1)
            sin = torch.cat([ang.sin(), ang.sin()], -1)
            rot = lambda t: torch.cat([-t[..., HD // 2 :], t[..., : HD // 2]], -1)
            q = q * cos + rot(q) * sin
            k = k * cos + rot(k) * sin
            k_cache = getattr(self, f"k_cache_{i}")
            v_cache = getattr(self, f"v_cache_{i}")
            ks = torch.cat([k_cache[:, :, 1:], k], dim=2)
            vs = torch.cat([v_cache[:, :, 1:], v], dim=2)
            k_cache[:, :, :, :] = ks
            v_cache[:, :, :, :] = vs
            att = torch.softmax((q @ ks.transpose(-1, -2)) * SCALE + neg, dim=-1)
            o = (att @ vs).transpose(1, 2).reshape(2, 1, HID) @ getattr(self, f"ow_{i}").T
            x = x + grms(o, getattr(self, f"ln_pa_{i}"))
            h = grms(x, getattr(self, f"ln_pf_{i}"))
            m = (F.gelu(h @ getattr(self, f"gate_{i}").T, approximate="tanh") * (h @ getattr(self, f"up_{i}").T)) @ getattr(self, f"down_{i}").T
            x = x + grms(m, getattr(self, f"ln_ff_{i}"))
        self.pos[:] = self.pos + 1.0
        return grms(x, self.norm_f)


class MoGDense(torch.nn.Module):
    """mlp_stack + CFG combine + dense heads. Sampling + low-rank mu are host-side.

    disable_eos_prediction is set for VoiceChat (no lm_head in the checkpoint):
    end-of-speech comes from the duplex LLM's text channel, not the TTS."""

    def __init__(self, mog_head):
        super().__init__()
        self.mlp_stack = mog_head.mlp_stack
        g = lambda t: torch.nn.Parameter(t.detach().float(), requires_grad=False)
        self.w_logits = g(mog_head.proj_logits.weight)
        self.w_logs = g(mog_head.proj_logs.weight)
        self.w_else = g(mog_head.proj_else.weight)

    def forward(self, x2, gscale):  # x2 [2,1,1152], gscale [1]
        y = self.mlp_stack(x2)
        xg = y[0:1] + gscale * (y[0:1] - y[1:2])
        logits = xg @ self.w_logits.T
        logs = torch.clamp_min(xg @ self.w_logs.T, -4.0)
        mu_res = xg @ self.w_else.T
        return logits, logs, mu_res, xg


def convert_module(mod, name: str, precision, stateful: bool):
    ex = (torch.zeros(2, 1, HID),) if stateful else (torch.zeros(2, 1, HID), torch.tensor([GUIDANCE]))
    with torch.no_grad():
        traced = torch.jit.trace(mod.eval(), ex)
    states = [
        ct.StateType(wrapped_type=ct.TensorType(shape=tuple(b.shape)), name=n)
        for n, b in mod.named_buffers()
    ] if stateful else None
    inputs = [ct.TensorType(name="x", shape=(2, 1, HID), dtype=np.float32)]
    if not stateful:
        inputs.append(ct.TensorType(name="gscale", shape=(1,), dtype=np.float32))
    outputs = (
        [ct.TensorType(name="hidden", dtype=np.float32)]
        if stateful
        else [ct.TensorType(name=n, dtype=np.float32) for n in ("logits", "logs", "mu_res", "xg")]
    )
    mlm = ct.convert(
        traced, inputs=inputs, outputs=outputs, states=states,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=precision, compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    mlm.save(str(BUILD / f"{name}.mlpackage"))
    typer.echo(f"saved {BUILD}/{name}.mlpackage")


@app.command()
def convert(fp32: bool = typer.Option(True)) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    model, _ = build_tts()

    np.save(BUILD / "proj_mus_w.npy", model.mog_head.proj_mus.weight.detach().numpy().reshape(VCB, 64, HID).astype(np.float16))
    np.save(BUILD / "low_mat.npy", model.mog_head.low_mat.detach().numpy().astype(np.float16))
    np.save(BUILD / "embed_code_w.npy", model.embed_code.weight.detach().numpy())
    np.save(BUILD / "rvq_embs.npy", model.rvq_embs.numpy())
    fus = model.gated_fusion_audio_text
    np.savez(
        BUILD / "prep_params.npz",
        audio_w=fus.audio_proj.weight.detach().numpy(), audio_b=fus.audio_proj.bias.detach().numpy(),
        text_w=fus.text_proj.weight.detach().numpy(), text_b=fus.text_proj.bias.detach().numpy(),
        gate=fus.gate.detach().numpy(), residual_scale=fus.residual_scale.detach().numpy(),
        final_norm_w=fus.final_norm.weight.detach().numpy(),
        null_emb=model.null_emb.detach().numpy(), bos_emb=model.bos_emb.detach().numpy(),
    )
    typer.echo("exported host-side npys (proj_mus, low_mat, embed_code, rvq_embs, prep_params)")

    # CoreML states are fp16-only — no fp32 stateful variant exists
    step = Gemma3Step(model.backbone)
    convert_module(step, "backbone_step_fp16", ct.precision.FLOAT16, stateful=True)
    mog = MoGDense(model.mog_head)
    convert_module(mog, "mog_dense_fp16", ct.precision.FLOAT16, stateful=False)
    convert_module(mog, "mog_dense_fp32", ct.precision.FLOAT32, stateful=False)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------
def host_mu(xg: np.ndarray, logs: np.ndarray, mu_res: np.ndarray, idx: int, mus_w, low_mat) -> np.ndarray:
    mu64 = mus_w[idx].astype(np.float32) @ xg.reshape(HID)
    mu512 = low_mat[idx].astype(np.float32) @ mu64
    return mu512 * np.exp(logs.reshape(1)) + mu_res.reshape(LAT)


def frame_codes_deterministic(hidden2, mog_call, model_np):
    """One frame of generate_step, deterministic (argmax pick, noise 0).

    hidden2: [2,1,1152] backbone output. mog_call(x2)->(logits, logs, mu_res, xg).
    model_np: dict with rvq (np [31,1024,512]), embed_code_w, mus_w, low_mat.
    """
    from nemo.collections.speechlm2.modules.ear_tts_model import get_masking_rate

    rvq, ecw = model_np["rvq"], model_np["embed_code_w"]
    code = np.full((NQ,), VCB, dtype=np.int64)
    rates = torch.linspace(0.0, 1.0, 9)[:-1].unsqueeze(-1)
    num_maskings = torch.ceil(get_masking_rate(rates, exponent=3.0) * NQ).long()
    ks = (num_maskings - F.pad(num_maskings[1:], [0, 0, 0, 1])).flatten().tolist()
    cnt = 0
    for k in ks:
        if k == 0:
            continue
        emb = np.zeros(LAT, dtype=np.float32)
        for i in range(NQ):
            if code[i] < VCB:
                emb += rvq[i, code[i]]
        mog_embed = (ecw @ emb).astype(np.float32)  # [1152]
        x2 = np.stack([mog_embed + hidden2[0, 0], mog_embed + hidden2[1, 0]])[:, None, :]
        logits, logs, mu_res, xg = mog_call(x2.astype(np.float32))
        assert np.isfinite(logits).all() and np.isfinite(xg).all(), "NaN/inf in MoG outputs"
        idx = int(np.argmax(logits))
        z = host_mu(xg, logs, mu_res, idx, model_np["mus_w"], model_np["low_mat"])
        # depthsum_encoding_step: greedy nearest per depth over cnt..cnt+k
        r = z.copy()
        for i in range(cnt, cnt + k):
            d = (rvq[i] ** 2).sum(-1) - 2.0 * (rvq[i] @ r)
            sel = int(np.argmin(d))
            r -= rvq[i, sel]
            code[i] = sel
        cnt += k
    return code


@app.command()
def parity(frames: int = typer.Option(4), prefill_t: int = typer.Option(12)) -> None:
    model, has_subword = build_tts()
    failed = False

    # 1. manual step stack vs HF backbone, full prefill, fp32
    torch.manual_seed(0)
    embeds = torch.randn(2, prefill_t, HID) * 0.05
    step = Gemma3Step(model.backbone)  # single shared instance (595M fp32) — reset between uses
    with torch.no_grad():
        ref = model.backbone(inputs_embeds=embeds, return_dict=True).last_hidden_state
        outs = [step(embeds[:, t : t + 1]) for t in range(prefill_t)]
    d1 = float((torch.cat(outs, 1) - ref).abs().max())
    ok1 = d1 < 2e-4
    typer.echo(f"manual step vs HF backbone ({prefill_t} steps): max|Δ| {d1:.3e} -> {'OK' if ok1 else 'FAIL'}")
    failed |= not ok1

    # torch reference hiddens for the CoreML backbone check (run BEFORE any
    # coremltools predict — torch-after-CoreML segfaults with a GIL fatal here)
    step.reset()
    torch.manual_seed(1)
    frames_in = [torch.randn(2, 1, HID) * 0.05 for _ in range(frames)]
    with torch.no_grad():
        ref_hidden = [step(f) for f in frames_in]

    # 4. e2e deterministic frame loop: torch replica vs CoreML fp32 replica -> identical codes
    model_np = {
        "rvq": model.rvq_embs.numpy(),
        "embed_code_w": model.embed_code.weight.detach().numpy(),
        "mus_w": np.load(BUILD / "proj_mus_w.npy"),
        "low_mat": np.load(BUILD / "low_mat.npy"),
    }
    sd = load_file(COMPONENTS_DIR / "tts.safetensors")
    silence = sd["tts_model.codec_silence_tokens"].numpy().reshape(-1)[:NQ]

    def prep(prev_code):
        c = torch.from_numpy(prev_code[None, None, :])
        ce = model.embed_code(model.depthsum_embedding(c))
        ce2 = torch.cat([ce, ce], 0)
        flags = torch.tensor([False, True]).view(2, 1, 1)
        import inspect
        kw = {"asr_speech_tokens_emb": None} if "asr_speech_tokens_emb" in inspect.signature(model._prepare_conditioning).parameters else {}
        cond = model._prepare_conditioning(None, None, None, flags, **kw)
        return model.gated_fusion_audio_text(ce2, cond) if model.config.use_gated_fusion_for_text_audio else ce2 + cond

    pp = np.load(BUILD / "prep_params.npz")

    def prep_np(prev_code):
        # numpy replica of prep() — keeps the CoreML replica torch-free (the
        # coremltools predict + torch interleave segfaults on this machine)
        emb = np.zeros(LAT, dtype=np.float32)
        for i in range(NQ):
            if prev_code[i] < VCB:
                emb += model_np["rvq"][i, prev_code[i]]
        ce = (model_np["embed_code_w"] @ emb).astype(np.float32)
        rows = []
        for cond_vec in (np.zeros(HID, dtype=np.float32), pp["null_emb"]):
            a = pp["audio_w"] @ (ce / NQ) + pp["audio_b"]
            t = pp["text_w"] @ cond_vec + pp["text_b"]
            g = 1.0 / (1.0 + np.exp(-pp["gate"]))
            r = 1.0 / (1.0 + np.exp(-pp["residual_scale"]))
            h = r * (g * a + (1.0 - g) * t)
            h = h / np.sqrt((h**2).mean() + EPS) * (1.0 + pp["final_norm_w"])  # NeMo RMSNorm is (1+w)
            rows.append(h)
        return np.stack(rows).astype(np.float32)[:, None, :]

    def mog_torch(x2np):
        with torch.no_grad():
            o = mog_t(torch.from_numpy(x2np), torch.tensor([GUIDANCE]))
        return tuple(t.numpy() for t in o[:4])


    def mog_coreml(mlm):
        def call(x2np):
            o = mlm.predict({"x": x2np, "gscale": np.array([GUIDANCE], dtype=np.float32)})
            return o["logits"], o["logs"], o["mu_res"], o["xg"]
        return call

    # one-time: numpy prep replica must match torch prep exactly
    with torch.no_grad():
        d_prep = float(np.abs(prep_np(silence.astype(np.int64)) - prep(silence.astype(np.int64)).numpy()).max())
    ok_prep = d_prep < 1e-5
    typer.echo(f"numpy prep vs torch prep: max|Δ| {d_prep:.3e} -> {'OK' if ok_prep else 'FAIL'}")
    failed |= not ok_prep

    # torch reference for the MoG dense check
    mog_t = MoGDense(model.mog_head)
    torch.manual_seed(2)
    x2 = torch.randn(2, 1, HID) * 0.3
    with torch.no_grad():
        ref_out = mog_t(x2, torch.tensor([GUIDANCE]))

    # 4a. gated: torch-only pass first (hiddens + codes), then CoreML-fp32-MoG on the
    # SAME hiddens -> identical codes. Two clean phases: the coremltools predict +
    # torch interleave segfaults (GIL fault), and fp16-only states preclude an fp32
    # CoreML backbone, so this gate isolates MoG + host RVQ math at fp32.
    step.reset()
    prev = silence.astype(np.int64)
    hiddens, codes_t = [], []
    with torch.no_grad():
        for fi in range(frames):
            h_t = step(prep(prev)).numpy()
            code_t = frame_codes_deterministic(h_t, mog_torch, model_np)
            hiddens.append(h_t)
            codes_t.append(code_t)
            prev = code_t
    del step  # release the fp32 torch stack before the CoreML phase
    mlm_m = ct.models.MLModel(str(BUILD / "mog_dense_fp32.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    mlm_m16 = ct.models.MLModel(str(BUILD / "mog_dense_fp16.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    # 2. CoreML backbone chained vs torch manual
    mlm_b16 = ct.models.MLModel(str(BUILD / "backbone_step_fp16.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    st16 = mlm_b16.make_state()
    outs = [mlm_b16.predict({"x": f.numpy()}, state=st16)["hidden"] for f in frames_in]
    d = max(float(np.abs(o - r.numpy()).max()) for o, r in zip(outs, ref_hidden))
    ok = d < 8e-2
    typer.echo(f"backbone_step_fp16 chained x{frames}: max|Δ| {d:.3e} (gate 0.08) -> {'OK' if ok else 'FAIL'}")
    failed |= not ok

    # 3. MoG dense CoreML vs torch
    for name, gate in (("mog_dense_fp32", 1e-4), ("mog_dense_fp16", 8e-2)):
        mlm = ct.models.MLModel(str(BUILD / f"{name}.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
        o = mlm.predict({"x": x2.numpy(), "gscale": np.array([GUIDANCE], dtype=np.float32)})
        d = max(
            float(np.abs(o[n] - r.numpy()).max())
            for n, r in zip(("logits", "logs", "mu_res", "xg"), ref_out)
        )
        ok = d < gate
        typer.echo(f"{name}: max|Δ| {d:.3e} (gate {gate:g}) -> {'OK' if ok else 'FAIL'}")
        failed |= not ok

    match = True
    for fi in range(frames):
        code_c = frame_codes_deterministic(hiddens[fi], mog_coreml(mlm_m), model_np)
        same = bool((codes_t[fi] == code_c).all())
        typer.echo(f"frame {fi}: codes identical (fp32 MoG): {same}" + ("" if same else f" ({int((codes_t[fi] != code_c).sum())}/31 differ)"))
        match &= same
    failed |= not match

    # 4b. informational: full fp16 CoreML chain (numpy prep, torch-free), teacher-forced
    # with the torch codes so per-frame inputs are comparable
    mlm_b16b = ct.models.MLModel(str(BUILD / "backbone_step_fp16.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
    st_b = mlm_b16b.make_state()
    agree = total = 0
    for fi in range(frames):
        prev16 = silence.astype(np.int64) if fi == 0 else codes_t[fi - 1]
        h16 = mlm_b16b.predict({"x": prep_np(prev16)}, state=st_b)["hidden"]
        code16 = frame_codes_deterministic(h16, mog_coreml(mlm_m16), model_np)
        agree += int((code16 == codes_t[fi]).sum()); total += NQ
    typer.echo(f"full fp16 chain code agreement (teacher-forced per frame): {agree}/{total}")
    typer.echo(f"subword conditioning available in this run: {has_subword}")
    if failed:
        raise typer.Exit(1)
    typer.echo("TTS PARITY OK")


@app.command()
def bench(runs: int = typer.Option(40)) -> None:
    x = np.random.randn(2, 1, HID).astype(np.float32) * 0.05
    for name, feed in (("backbone_step_fp16", True), ("mog_dense_fp16", False)):
        for units in (ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_NE):
            try:
                mlm = ct.models.MLModel(str(BUILD / f"{name}.mlpackage"), compute_units=units)
                st = mlm.make_state() if feed else None
                inp = {"x": x} if feed else {"x": x, "gscale": np.array([GUIDANCE], dtype=np.float32)}
                for _ in range(3):
                    mlm.predict(inp, state=st) if feed else mlm.predict(inp)
                times = []
                for _ in range(runs):
                    t0 = time.perf_counter()
                    mlm.predict(inp, state=st) if feed else mlm.predict(inp)
                    times.append((time.perf_counter() - t0) * 1e3)
                med = sorted(times)[len(times) // 2]
                typer.echo(f"{name} {units.name}: {med:.2f} ms median")
            except Exception as e:
                typer.echo(f"{name} {units.name}: REJECTED ({type(e).__name__}: {str(e)[:80]})")


if __name__ == "__main__":
    app()
