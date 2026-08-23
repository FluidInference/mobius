#!/usr/bin/env python3
"""Calibrated sub-8-bit quantization quality for the fine-tuned VoiceChat-11B LLM.

Extends measure_int4_quality.py (same dual-track fp32-vs-dequant methodology,
same validated layer semantics) with:

  * batched layer-major streaming: all sequences advance through a layer
    together, so each weight is read from llm.safetensors exactly once per
    pass (right-padding is safe: every op is causal, pads sit at the end)
  * AWQ-style activation-aware per-input-channel scales, grid-searched per
    linear on a small conversational-text calibration set. CAVEAT: deployment
    feeds fused audio embeddings; plain text is a proxy for that regime.
  * a per-linear sensitivity proxy (relative output MSE on calibration
    activations) recorded during calibration, used to drive mixed-precision
    promotion (int8/int6) under an effective-bits budget
  * exact effective-bits accounting (weight bits + fp16 scale overhead)

Quantization scheme matches the deployment plan and the old harness exactly:
linear_symmetric intN per-block along the input axis (coremltools
OpLinearQuantizerConfig semantics). AWQ scales are measured as
Q(W*s)/s — at deployment 1/s folds into the preceding norm weight
(in/up/q/k/v: layer norm.weight; out_proj: mixer gated norm) or is a free
elementwise multiply before the matmul (heads, o_proj, down_proj).

Subcommands:
  calibrate  -> writes calib_scales.npz (scales + per-linear sensitivity)
  evaluate   -> dual-track metrics vs fp32; --method rtn|fp16|awq|mixed

Typical use:
  uv run python measure_calibrated_quant.py calibrate
  uv run python measure_calibrated_quant.py evaluate --method rtn  --eval-set orig   # sanity vs 74.3%
  uv run python measure_calibrated_quant.py evaluate --method awq  --eval-set both
  uv run python measure_calibrated_quant.py evaluate --method mixed --budget 5.0 --eval-set both
"""
from __future__ import annotations

import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import typer
from safetensors import safe_open

COMPONENTS = Path.home() / "Documents/models/voicechat-11b/components/llm.safetensors"
TOKENIZER = Path.home() / "Documents/models/nemotron-nano-9b-mlx-4bit/tokenizer.json"
SCALES_DEFAULT = Path(__file__).parent / "calib_scales.npz"

D_MODEL = 4480
N_LAYERS = 56
ATTN_LAYERS = {14, 21, 30, 39}
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 40, 8, 128
N_HEADS, HEADDIM, D_STATE, N_GROUPS, D_INNER, D_CONV = 128, 80, 128, 8, 10240, 4
CONV_DIM = D_INNER + 2 * N_GROUPS * D_STATE

HEAD_KEYS = ("stt_model.lm_head.weight", "stt_model.function_head.weight")
LINEAR_NAMES = {
    "mamba": ("in_proj", "out_proj"),
    "mlp": ("up_proj", "down_proj"),
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
}
LINEAR_SHAPES = {
    "in_proj": (22656, 4480),
    "out_proj": (4480, 10240),
    "up_proj": (15680, 4480),
    "down_proj": (4480, 15680),
    "q_proj": (5120, 4480),
    "k_proj": (1024, 4480),
    "v_proj": (1024, 4480),
    "o_proj": (4480, 5120),
}
HEAD_SHAPE = (131072, 4480)

# ---------------------------------------------------------------------------
# Prompt sets.  ORIG_PROMPTS are byte-identical to measure_int4_quality.py so
# the int4-RTN sanity number is directly comparable to the 2026-08-03 runs.
# CALIB_TEXTS (calibration) and HELDOUT_PROMPTS (extra eval) are disjoint.
# ---------------------------------------------------------------------------
ORIG_PROMPTS = [
    "You are an AI voice assistant developed by NVIDIA. Your name is NVIDIA Voice Chat. "
    "Your job is to be helpful and harmless and have engaging conversations in English.",
    "Hello, do you know what color the sky is? I was wondering about that earlier today.",
    "Please call the weather tool for San Francisco and tell me if I need an umbrella.",
]

CALIB_TEXTS = [
    "Hi there! How are you doing today? I hope you're having a great morning.",
    "Can you set a timer for ten minutes? I'm boiling eggs and always forget them.",
    "What's the capital of France? My daughter asked me and I want to double check.",
    "Please look up the weather in New York City for tomorrow afternoon.",
    "I'd like to schedule a meeting with the design team next Tuesday at three.",
    "Tell me a short joke about computers, something my coworkers would enjoy.",
    "How do I convert two hundred fahrenheit to celsius? I'm following a recipe.",
    "Remind me to call my mother this evening after dinner, around seven o'clock.",
    "What's a good restaurant near downtown that serves vegetarian food?",
    "Can you explain what machine learning is in one or two simple sentences?",
    "Play some relaxing jazz music, please. I'm trying to focus on my reading.",
    "How long does it take to fly from Los Angeles to Tokyo on a direct flight?",
    "I think my internet connection is slow today. Can you run a speed test?",
    "What time is it right now in London? I need to call a client there.",
    "Add milk, eggs, and bread to my shopping list for this weekend.",
    "Who wrote the novel Pride and Prejudice? I can't remember the author.",
    "Can you summarize the main news headlines for me this morning?",
    "Turn off the living room lights and lock the front door, please.",
    "What's the square root of one hundred forty four? Quick math question.",
    "I'm feeling a bit stressed today. Do you have any relaxation tips?",
    "Book a table for four at an Italian restaurant on Friday evening.",
    "How many ounces are there in a gallon? I always mix up the units.",
    "Tell me an interesting fact about the ocean that most people don't know.",
    "Please send a message to Sarah saying I'll be fifteen minutes late.",
    "What movies are playing at the theater near me this weekend?",
    "Can you help me draft a short thank you note for my colleague?",
    "How do you make a classic margherita pizza from scratch at home?",
    "What's the battery level on my phone? Should I charge it before leaving?",
    "Translate the phrase good morning, how are you, into Spanish for me.",
    "I need directions to the nearest gas station. My tank is almost empty.",
    "What's the difference between a crocodile and an alligator, briefly?",
    "Please call the calendar tool and check my appointments for Thursday.",
    "Wake me up at six thirty tomorrow morning. I have an early flight.",
    "How tall is Mount Everest? A friend and I were debating this yesterday.",
    "Can you recommend a good podcast about history for my commute?",
    "What should I wear today? Is it going to be cold this afternoon?",
    "Start a three minute countdown for my plank exercise, please.",
    "Who won the basketball game last night? I fell asleep before the end.",
    "Please use the search tool to find reviews of the new electric sedan.",
    "How do I remove a coffee stain from a white cotton shirt?",
    "What's my next meeting today, and do I need to prepare anything for it?",
    "Tell me about the tallest building in the world and where it is located.",
    "Can you check if my package from the bookstore has shipped yet?",
    "I want to learn to play guitar. How should a complete beginner start?",
    "What's the phone number for the dentist office on Main Street?",
    "Please dim the bedroom lights to thirty percent and play white noise.",
    "How many calories are in a medium sized banana, roughly speaking?",
    "Call the stock tool and tell me how the market is doing this morning.",
    "What year did the first person walk on the moon? I think it was the sixties.",
    "Can you suggest a fun weekend activity for a family with two young kids?",
    "My flight got delayed by two hours. Can you update my airport pickup?",
    "What does the idiom break the ice actually mean in conversation?",
    "Please turn up the thermostat by two degrees. It feels chilly in here.",
    "How often should I water a small succulent plant kept indoors?",
    "Give me a quick rundown of tomorrow's schedule before I go to bed.",
    "Is it going to rain this weekend? We're planning a picnic at the lake.",
]

EXTRA_CALIB_TEXTS = [
    "Could you walk me through how to set up a new email account on my phone?",
    "What's the weather looking like for the marathon on Saturday morning?",
    "Please add a dentist appointment to my calendar for next Wednesday at nine.",
    "How do airplanes stay in the air? Explain it like I'm ten years old.",
    "Set the volume to forty percent and skip to the next song, please.",
    "What's a good substitute for buttermilk if I don't have any at home?",
    "Can you check the score of the soccer match that started an hour ago?",
    "I need to renew my passport soon. What documents do I need to bring?",
    "Tell me the three biggest cities in Brazil and roughly how large they are.",
    "Please text my brother that dinner is moved to seven thirty tonight.",
    "Why does the moon look bigger when it's near the horizon at night?",
    "Find me a hardware store that's open right now within ten minutes of here.",
    "How many teaspoons are in a tablespoon? I can never remember that.",
    "Turn on do not disturb until my meeting ends at four o'clock.",
    "What are some easy stretches I can do at my desk during work breaks?",
    "Call the navigation tool and start a route to the airport, avoiding tolls.",
    "Who invented the telephone, and roughly when did that happen?",
    "My smoke detector keeps chirping. What does that usually mean?",
    "Give me a thirty second summary of what photosynthesis actually does.",
    "Please order my usual coffee for pickup at the shop on Fifth Avenue.",
    "What time does the sun set tonight? We want to walk after dinner.",
    "How should I store fresh basil so it lasts more than a couple days?",
    "Check my heart rate from this morning's run and compare it to last week.",
    "Can you spell the word necessary for me? I always get it wrong.",
]

HELDOUT_PROMPTS = [
    "Good evening! Could you tell me a little about how volcanoes are formed?",
    "Please call the timer tool and set a reminder for my laundry in forty minutes.",
    "What's the fastest land animal, and how fast can it actually run?",
    "I'm planning a trip to Seattle next month. What should I pack for the weather?",
    "Can you explain the rules of chess to someone who has never played before?",
    "Turn on the kitchen lights and preheat the oven to four hundred degrees.",
    "How much should I tip at a restaurant in the United States, typically?",
    "Who painted the Mona Lisa, and where is the painting displayed today?",
    "Please check the traffic on my route to work and tell me when to leave.",
    "What's a simple dinner I can cook tonight with chicken, rice, and broccoli?",
    "My laptop won't connect to the wifi network. What should I try first?",
    "Tell me something encouraging. I have a big job interview in an hour.",
    "Use the music tool to queue up my workout playlist and set volume to sixty.",
]

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Quantization primitives (identical math to measure_int4_quality.py)
# ---------------------------------------------------------------------------
def quant_blockwise(w: torch.Tensor, nbits: int, block: int) -> torch.Tensor:
    """linear_symmetric intN, per-block along the input (last) axis."""
    if nbits >= 16:  # fp16 cast — noise-floor reference
        return w.half().float()
    out, inp = w.shape
    block = block or inp  # block=0 -> per-channel (one scale per row)
    qmax = 2 ** (nbits - 1) - 1
    pad = (-inp) % block
    if pad:
        w = F.pad(w, (0, pad))
    blocks = w.reshape(out, -1, block)
    scale = blocks.abs().amax(-1, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(blocks / scale), -qmax - 1, qmax)
    return (q * scale).reshape(out, -1)[:, :inp]


def tensor_eff_bits(out: int, inp: int, nbits: int, block: int) -> int:
    """Total stored bits: intN payload + one fp16 scale per block."""
    n = out * inp
    if nbits >= 16:
        return n * 16
    block = block or inp
    return n * nbits + out * math.ceil(inp / block) * 16


class Quantizer:
    """Maps full tensor key -> dequantized weight per the active config."""

    def __init__(self, nbits: int, block: int, head_nbits: int, head_block: int,
                 scales: dict[str, np.ndarray] | None = None,
                 promote: dict[str, int] | None = None):
        self.nbits, self.block = nbits, block
        self.head_nbits, self.head_block = head_nbits, head_block
        self.scales = scales or {}
        self.promote = promote or {}

    def bits_for(self, key: str) -> tuple[int, int]:
        is_head = key in HEAD_KEYS
        nb = self.promote.get(key, self.head_nbits if is_head else self.nbits)
        blk = self.head_block if is_head else self.block
        return nb, blk

    def dequant(self, key: str, w: torch.Tensor) -> torch.Tensor:
        nb, blk = self.bits_for(key)
        if nb >= 16:
            return w.half().float()
        s = self.scales.get(key)
        if s is None:
            return quant_blockwise(w, nb, blk)
        s_t = torch.from_numpy(s).to(w.dtype)
        return quant_blockwise(w * s_t, nb, blk) / s_t


# ---------------------------------------------------------------------------
# Model forward (batched version of the validated LayerRunner semantics)
# ---------------------------------------------------------------------------
def rms_norm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w


def fwd_mamba(w, H, collect=None):
    Bsz, T, _ = H.shape
    x = rms_norm(H, w["norm"])
    if collect is not None:
        collect["in_proj"] = x
    zxbcdt = x @ w["in_proj"].T  # [B, T, 22656]
    z = zxbcdt[..., :D_INNER]
    xBC = zxbcdt[..., D_INNER : D_INNER + CONV_DIM]
    dt = zxbcdt[..., D_INNER + CONV_DIM :]

    xBC = F.conv1d(
        F.pad(xBC.transpose(1, 2), (D_CONV - 1, 0)),
        w["conv_w"].unsqueeze(1), bias=w["conv_b"], groups=CONV_DIM,
    ).transpose(1, 2)
    xBC = F.silu(xBC)

    xs = xBC[..., :D_INNER].reshape(Bsz, T, N_HEADS, HEADDIM)
    Bm = xBC[..., D_INNER : D_INNER + N_GROUPS * D_STATE].reshape(Bsz, T, N_GROUPS, D_STATE)
    Cm = xBC[..., D_INNER + N_GROUPS * D_STATE :].reshape(Bsz, T, N_GROUPS, D_STATE)
    Bm = Bm.repeat_interleave(N_HEADS // N_GROUPS, dim=2)  # [B, T, 128, 128]
    Cm = Cm.repeat_interleave(N_HEADS // N_GROUPS, dim=2)

    dt = F.softplus(dt + w["dt_bias"])  # [B, T, 128]
    dA = torch.exp(dt * -torch.exp(w["A_log"]))
    dtB = dt.unsqueeze(-1) * Bm  # [B, T, 128, 128]

    state = torch.zeros(Bsz, N_HEADS, HEADDIM, D_STATE)
    ys = torch.empty(Bsz, T, N_HEADS, HEADDIM)
    for t in range(T):
        state = state * dA[:, t, :, None, None] + xs[:, t].unsqueeze(-1) * dtB[:, t].unsqueeze(2)
        ys[:, t] = (state * Cm[:, t].unsqueeze(2)).sum(-1) + w["D"][:, None] * xs[:, t]
    y = ys.reshape(Bsz, T, D_INNER)
    y = rms_norm(y * F.silu(z), w["gnorm"])
    if collect is not None:
        collect["out_proj"] = y
    return H + y @ w["out_proj"].T


def fwd_mlp(w, H, collect=None):
    x = rms_norm(H, w["norm"])
    if collect is not None:
        collect["up_proj"] = x
    u = F.relu(x @ w["up_proj"].T).pow(2)
    if collect is not None:
        collect["down_proj"] = u
    return H + u @ w["down_proj"].T


def fwd_attn(w, H, collect=None):
    Bsz, T, _ = H.shape
    x = rms_norm(H, w["norm"])
    if collect is not None:
        collect["q_proj"] = collect["k_proj"] = collect["v_proj"] = x
    q = (x @ w["q_proj"].T).reshape(Bsz, T, N_Q_HEADS, HEAD_DIM).transpose(1, 2)
    k = (x @ w["k_proj"].T).reshape(Bsz, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = (x @ w["v_proj"].T).reshape(Bsz, T, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
    k = k.repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=1)
    v = v.repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=1)
    att = (q @ k.transpose(-1, -2)) / HEAD_DIM**0.5
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    att = torch.softmax(att + mask, dim=-1)
    out = (att @ v).transpose(1, 2).reshape(Bsz, T, N_Q_HEADS * HEAD_DIM)
    if collect is not None:
        collect["o_proj"] = out
    return H + out @ w["o_proj"].T


FWD = {"mamba": fwd_mamba, "mlp": fwd_mlp, "attn": fwd_attn}


def layer_kind(keyset: set[str], i: int) -> str:
    if i in ATTN_LAYERS:
        return "attn"
    return "mamba" if f"stt_model.llm.layers.{i}.mixer.A_log" in keyset else "mlp"


def full_key(i: int, name: str) -> str:
    return f"stt_model.llm.layers.{i}.mixer.{name}.weight"


def load_layer(f, i: int, kind: str) -> dict[str, torch.Tensor]:
    def g(k):
        return f.get_tensor(k).to(torch.float32)

    p = f"stt_model.llm.layers.{i}.mixer."
    w = {"norm": g(f"stt_model.llm.layers.{i}.norm.weight")}
    if kind == "mamba":
        w.update(
            in_proj=g(p + "in_proj.weight"), conv_w=g(p + "conv1d.weight").squeeze(1),
            conv_b=g(p + "conv1d.bias"), A_log=g(p + "A_log"), D=g(p + "D"),
            dt_bias=g(p + "dt_bias"), gnorm=g(p + "norm.weight"),
            out_proj=g(p + "out_proj.weight"),
        )
    elif kind == "mlp":
        w.update(up_proj=g(p + "up_proj.weight"), down_proj=g(p + "down_proj.weight"))
    else:
        w.update(
            q_proj=g(p + "q_proj.weight"), k_proj=g(p + "k_proj.weight"),
            v_proj=g(p + "v_proj.weight"), o_proj=g(p + "o_proj.weight"),
        )
    return w


def encode_batch(texts: list[str], cap: int) -> list[list[int]]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    return [tok.encode(t).ids[:cap] for t in texts]


def embed_batch(f, seqs: list[list[int]]):
    embed = f.get_tensor("stt_model.embed_tokens.weight").to(torch.float32)
    B, T = len(seqs), max(len(s) for s in seqs)
    H = torch.zeros(B, T, D_MODEL)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b, s in enumerate(seqs):
        H[b, : len(s)] = embed[s]
        mask[b, : len(s)] = True
    del embed
    return H, mask


def report_footprint(t0: float) -> str:
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    return f"wall {time.time() - t0:.0f}s  peak RSS {rss_gb:.1f} GB"


# ---------------------------------------------------------------------------
# AWQ scale search
# ---------------------------------------------------------------------------
def _chunked_out_mse(sub: torch.Tensor, W: torch.Tensor, s: torch.Tensor | None,
                     nbits: int, block: int, ref: torch.Tensor, chunk: int = 32768) -> float:
    """MSE(sub @ Wdq.T, ref) with Wdq materialized in row chunks (bounds memory)."""
    err, n = 0.0, 0
    for r0 in range(0, W.shape[0], chunk):
        Wc = W[r0 : r0 + chunk]
        if s is None:
            dq = quant_blockwise(Wc, nbits, block)
        else:
            dq = quant_blockwise(Wc * s, nbits, block) / s
        d = sub @ dq.T - ref[:, r0 : r0 + chunk]
        err += d.pow(2).sum().item()
        n += d.numel()
    return err / n


def awq_search(W: torch.Tensor, X: torch.Tensor, nbits: int, block: int,
               rows: int, dev: str, gen: torch.Generator) -> dict:
    """Grid-search per-input-channel scale s; objective = output MSE on rows."""
    idx = torch.randperm(X.shape[0], generator=gen)[:rows]
    sub = X[idx].to(dev)
    Wd = W.to(dev)
    ref = sub @ Wd.T
    ref_ms = ref.pow(2).mean().item()
    act = X.abs().mean(0).clamp(min=1e-5).to(dev)
    wmag = Wd.abs().mean(0).clamp(min=1e-5)

    def norm_s(s):
        s = s.clamp(1e-4, 1e4)
        return s / (s.max() * s.min()).sqrt()

    cands: list[tuple[str, torch.Tensor | None]] = [("rtn", None)]
    for a in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        cands.append((f"act^{a}", norm_s(act**a)))
    for a in (0.25, 0.5, 0.75):
        cands.append((f"act^{a}/w^{1 - a}", norm_s(act**a / wmag ** (1 - a))))

    errs = [ _chunked_out_mse(sub, Wd, s, nbits, block, ref) for _, s in cands ]
    err_rtn = errs[0]
    best = int(np.argmin(errs))
    name, s_best = cands[best]
    out = {
        "scale": (s_best.cpu().numpy().astype(np.float32) if s_best is not None
                  else np.ones(W.shape[1], dtype=np.float32)),
        "alpha_desc": name,
        "err_rtn": err_rtn / ref_ms,
        "err_awq": errs[best] / ref_ms,
    }
    del sub, Wd, ref, act, wmag
    if dev == "mps":
        torch.mps.empty_cache()
    return out


# ---------------------------------------------------------------------------
# GPTQ (error-compensated rounding, per-block-32 scales along input axis)
# ---------------------------------------------------------------------------
def gptq_quantize(W: torch.Tensor, X: torch.Tensor, nbits: int, block: int,
                  percdamp: float = 0.01, blocksize: int = 128) -> torch.Tensor:
    """Official-GPTQ column loop with group (per-block) scales recomputed on the
    partially-compensated weights. X: [N, in] calibration inputs (thin-Hessian:
    N << in here, damping regularizes the null space toward RTN)."""
    out, inp = W.shape
    if nbits >= 16:
        return W.half().float()
    block = block or inp
    per_channel = block >= inp
    assert per_channel or (inp % block == 0 and blocksize % block == 0)
    qmax = 2 ** (nbits - 1) - 1

    H = X.T @ X
    diag = H.diagonal()
    dead = diag == 0
    diag[dead] = 1.0
    W = W.clone()
    W[:, dead] = 0
    damp = percdamp * diag.mean()
    for _ in range(4):
        try:
            H.diagonal().add_(damp)
            U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
            break
        except torch.linalg.LinAlgError:
            damp *= 10.0
    else:
        raise RuntimeError("cholesky failed after damping escalation")

    Q = torch.empty_like(W)
    scale = torch.ones(out)
    if per_channel:  # one scale per row, from the uncompensated weights
        scale = W.abs().amax(1) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    for i1 in range(0, inp, blocksize):
        i2 = min(i1 + blocksize, inp)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            if not per_channel and col % block == 0:
                grp = W1[:, j : j + block]
                scale = grp.abs().amax(1) / qmax
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            wcol = W1[:, j]
            q = torch.clamp(torch.round(wcol / scale), -qmax - 1, qmax) * scale
            Q[:, j + i1] = q
            err = (wcol - q) / U1[j, j]
            W1[:, j + 1 :] -= err[:, None] * U1[j, j + 1 :][None, :]
            Err1[:, j] = err
        W[:, i2:] -= Err1 @ U[i1:i2, i2:]
    return Q


def gptq_dequant_for(qz: Quantizer, key: str, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """GPTQ with optional AWQ pre-scale folded in (quantize W*s on inputs x/s)."""
    nb, blk = qz.bits_for(key)
    if nb >= 16:
        return W.half().float()
    s = qz.scales.get(key)
    if s is None:
        return gptq_quantize(W, X, nb, blk)
    s_t = torch.from_numpy(s)
    return gptq_quantize(W * s_t, X / s_t, nb, blk) / s_t


def gptq_layer(w: dict, kind: str, i: int, H_cal: torch.Tensor, cmask: torch.Tensor,
               qz: Quantizer) -> tuple[dict, torch.Tensor]:
    """Sequentially GPTQ-quantize one layer's linears; the calibration track
    H_cal advances through the QUANTIZED weights (standard GPTQ convention)."""
    wq = dict(w)
    if kind == "mamba":
        x = rms_norm(H_cal, w["norm"])
        wq["in_proj"] = gptq_dequant_for(qz, full_key(i, "in_proj"), w["in_proj"], x[cmask])
        coll: dict = {}
        fwd_mamba(wq, H_cal, collect=coll)  # out_proj value unused below
        y = coll["out_proj"]
        wq["out_proj"] = gptq_dequant_for(qz, full_key(i, "out_proj"), w["out_proj"], y[cmask])
        H_cal = H_cal + y @ wq["out_proj"].T
    elif kind == "mlp":
        x = rms_norm(H_cal, w["norm"])
        wq["up_proj"] = gptq_dequant_for(qz, full_key(i, "up_proj"), w["up_proj"], x[cmask])
        u = F.relu(x @ wq["up_proj"].T).pow(2)
        wq["down_proj"] = gptq_dequant_for(qz, full_key(i, "down_proj"), w["down_proj"], u[cmask])
        H_cal = H_cal + u @ wq["down_proj"].T
    else:
        x = rms_norm(H_cal, w["norm"])
        for n in ("q_proj", "k_proj", "v_proj"):
            wq[n] = gptq_dequant_for(qz, full_key(i, n), w[n], x[cmask])
        coll = {}
        fwd_attn(wq, H_cal, collect=coll)
        out = coll["o_proj"]
        wq["o_proj"] = gptq_dequant_for(qz, full_key(i, "o_proj"), w["o_proj"], out[cmask])
        H_cal = H_cal + out @ wq["o_proj"].T
    return wq, H_cal


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------
@app.command()
def calibrate(
    nbits: int = typer.Option(4),
    block: int = typer.Option(32),
    out: Path = typer.Option(SCALES_DEFAULT),
    n_texts: int = typer.Option(56),
    cap: int = typer.Option(48, help="max tokens per calibration text"),
    rows: int = typer.Option(192, help="activation rows for the search objective"),
    search_device: str = typer.Option("auto", help="auto|cpu|mps for the alpha grid search"),
) -> None:
    t0 = time.time()
    torch.set_grad_enabled(False)
    dev = search_device
    if dev == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    gen = torch.Generator().manual_seed(0)
    seqs = encode_batch(CALIB_TEXTS[:n_texts], cap)
    n_tok = sum(len(s) for s in seqs)
    typer.echo(f"calibrating int{nbits} pb{block} on {len(seqs)} texts, {n_tok} tokens, search on {dev}")

    results: dict[str, dict] = {}
    with safe_open(COMPONENTS, framework="pt", device="cpu") as f:
        keyset = set(f.keys())
        H, mask = embed_batch(f, seqs)
        for i in range(N_LAYERS):
            kind = layer_kind(keyset, i)
            w = load_layer(f, i, kind)
            coll: dict = {}
            H = FWD[kind](w, H, collect=coll)
            for name in LINEAR_NAMES[kind]:
                X = coll[name][mask]  # [n_tok, in]
                res = awq_search(w[name], X, nbits, block, rows, dev, gen)
                results[full_key(i, name)] = res
                typer.echo(
                    f"L{i:02d} {kind:5s} {name:9s} best={res['alpha_desc']:<14s} "
                    f"rel_err rtn={res['err_rtn']:.5f} -> awq={res['err_awq']:.5f}"
                )
            del w, coll

        norm_f = f.get_tensor("stt_model.llm.norm_f.weight").to(torch.float32)
        Xf = rms_norm(H, norm_f)[mask]
        for hk in HEAD_KEYS:
            W = f.get_tensor(hk).to(torch.float32)
            res = awq_search(W, Xf, nbits, block, min(rows, 128), dev, gen)
            results[hk] = res
            typer.echo(
                f"HEAD {hk.split('.')[-2]:14s} best={res['alpha_desc']:<14s} "
                f"rel_err rtn={res['err_rtn']:.5f} -> awq={res['err_awq']:.5f}"
            )
            del W

    payload = {}
    for k, r in results.items():
        payload[f"{k}::scale"] = r["scale"]
        payload[f"{k}::meta"] = np.array([r["err_rtn"], r["err_awq"]], dtype=np.float64)
        payload[f"{k}::alpha"] = np.array(r["alpha_desc"])
    np.savez_compressed(out, **payload)
    typer.echo(f"saved {len(results)} scale vectors -> {out}")
    typer.echo(report_footprint(t0))


def load_scales(path: Path) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]]]:
    z = np.load(path, allow_pickle=False)
    scales, meta = {}, {}
    for k in z.files:
        if k.endswith("::scale"):
            scales[k[: -len("::scale")]] = z[k]
        elif k.endswith("::meta"):
            meta[k[: -len("::meta")]] = (float(z[k][0]), float(z[k][1]))
    return scales, meta


# ---------------------------------------------------------------------------
# effective bits + mixed-precision selection
# ---------------------------------------------------------------------------
def all_quant_tensors(keyset: set[str]) -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}
    for i in range(N_LAYERS):
        kind = layer_kind(keyset, i)
        for name in LINEAR_NAMES[kind]:
            shapes[full_key(i, name)] = LINEAR_SHAPES[name]
    for hk in HEAD_KEYS:
        shapes[hk] = HEAD_SHAPE
    return shapes


def effective_bits(qz: Quantizer, shapes: dict[str, tuple[int, int]]) -> float:
    tp, tb = 0, 0
    for key, (o, i) in shapes.items():
        nb, blk = qz.bits_for(key)
        tp += o * i
        tb += tensor_eff_bits(o, i, nb, blk)
    return tb / tp


def choose_promotions(meta: dict[str, tuple[float, float]], shapes: dict[str, tuple[int, int]],
                      base: Quantizer, budget: float, promote_bits: int) -> dict[str, int]:
    """Greedily promote the most-sensitive body linears while eff bits <= budget."""
    body = [k for k in shapes if k not in HEAD_KEYS]
    order = sorted(body, key=lambda k: meta.get(k, (0, 0))[1], reverse=True)
    promote: dict[str, int] = {}
    for k in order:
        trial = Quantizer(base.nbits, base.block, base.head_nbits, base.head_block,
                          promote={**promote, k: promote_bits})
        if effective_bits(trial, shapes) <= budget:
            promote[k] = promote_bits
    return promote


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
@app.command()
def evaluate(
    method: str = typer.Option("rtn", help="rtn|fp16|awq|mixed|gptq|gptq-mixed"),
    nbits: int = typer.Option(4),
    block: int = typer.Option(32),
    head_nbits: int = typer.Option(0, help="0 -> same as --nbits"),
    head_block: int = typer.Option(-1, help="-1 -> same as --block; 0 -> per-channel"),
    eval_set: str = typer.Option("orig", help="orig|heldout|both"),
    scales_path: Path = typer.Option(SCALES_DEFAULT),
    budget: float = typer.Option(5.0, help="effective-bits budget for --method mixed"),
    promote_bits: int = typer.Option(8, help="bits for promoted body linears (mixed)"),
    per_layer_cos: bool = typer.Option(False),
) -> None:
    t0 = time.time()
    torch.set_grad_enabled(False)
    head_nbits = head_nbits or nbits
    if head_block < 0:
        head_block = block

    scales, meta = (None, {})
    if method in ("awq", "mixed", "gptq", "gptq-mixed"):
        scales, meta = load_scales(scales_path)
    if method == "fp16":
        nbits = head_nbits = 16
    gptq_mode = method.startswith("gptq")

    prompts = {"orig": ORIG_PROMPTS, "heldout": HELDOUT_PROMPTS,
               "both": ORIG_PROMPTS + HELDOUT_PROMPTS}[eval_set]
    seqs = encode_batch(prompts, 64)

    with safe_open(COMPONENTS, framework="pt", device="cpu") as f:
        keyset = set(f.keys())
        shapes = all_quant_tensors(keyset)
        qz = Quantizer(nbits, block, head_nbits, head_block, scales=scales)
        if method in ("mixed", "gptq-mixed"):
            qz.promote = choose_promotions(meta, shapes, qz, budget, promote_bits)
            n_pro = len(qz.promote)
            pro_params = sum(shapes[k][0] * shapes[k][1] for k in qz.promote)
            typer.echo(f"mixed: promoted {n_pro} body linears ({pro_params / 1e6:.0f}M params) to int{promote_bits}:")
            for k in sorted(qz.promote):
                typer.echo(f"  int{promote_bits} <- {k}  (rel_err {meta.get(k, (0, 0))[1]:.5f})")
        eff = effective_bits(qz, shapes)
        typer.echo(
            f"method={method} body int{nbits} pb{block}, heads int{head_nbits} "
            f"pb{head_block or 'chan'}  ->  effective bits/weight = {eff:.3f}"
        )

        H_fp, mask = embed_batch(f, seqs)
        H_q = H_fp.clone()
        H_cal, cmask = (None, None)
        if gptq_mode:
            calib_seqs = encode_batch(CALIB_TEXTS + EXTRA_CALIB_TEXTS, 48)
            H_cal, cmask = embed_batch(f, calib_seqs)
            typer.echo(f"gptq: {len(calib_seqs)} calib texts, {int(cmask.sum())} tokens for Hessians")
        cos_by_layer = []
        for i in range(N_LAYERS):
            kind = layer_kind(keyset, i)
            w = load_layer(f, i, kind)
            if gptq_mode:
                wq, H_cal = gptq_layer(w, kind, i, H_cal, cmask, qz)
            else:
                wq = {n: (qz.dequant(full_key(i, n), t) if n in LINEAR_NAMES[kind] else t)
                      for n, t in w.items()}
            H_fp = FWD[kind](w, H_fp)
            H_q = FWD[kind](wq, H_q)
            cos = F.cosine_similarity(H_fp[mask], H_q[mask], dim=-1).mean().item()
            cos_by_layer.append(cos)
            if per_layer_cos:
                typer.echo(f"L{i:02d} {kind:5s} cos {cos:.5f}")
            del w, wq

        norm_f = f.get_tensor("stt_model.llm.norm_f.weight").to(torch.float32)
        xf, xq = rms_norm(H_fp, norm_f), rms_norm(H_q, norm_f)
        x_cal = rms_norm(H_cal, norm_f)[cmask] if gptq_mode else None

        all_top1, all_top5, all_kl, all_fn_top1 = [], [], [], []
        for hk, store in ((HEAD_KEYS[0], "lm"), (HEAD_KEYS[1], "fn")):
            W = f.get_tensor(hk).to(torch.float32)
            Wq = gptq_dequant_for(qz, hk, W, x_cal) if gptq_mode else qz.dequant(hk, W)
            for b, s in enumerate(seqs):
                L = len(s)
                lf = xf[b, :L] @ W.T
                lq = xq[b, :L] @ Wq.T
                if store == "lm":
                    top1 = (lf.argmax(-1) == lq.argmax(-1)).float().mean().item()
                    top5_fp = lf.topk(5, dim=-1).indices
                    top5 = (lq.argmax(-1, keepdim=True) == top5_fp).any(-1).float().mean().item()
                    kl = F.kl_div(
                        F.log_softmax(lq, -1), F.log_softmax(lf, -1),
                        log_target=True, reduction="batchmean",
                    ).item()
                    all_top1.append(top1)
                    all_top5.append(top5)
                    all_kl.append(kl)
                    typer.echo(f"prompt[{L} tok]: top1 {top1:.3f}  top5 {top5:.3f}  KL {kl:.4f}")
                else:
                    all_fn_top1.append((lf.argmax(-1) == lq.argmax(-1)).float().mean().item())
            del W, Wq

    typer.echo(
        f"\nOVERALL {method} (eval_set={eval_set}, eff_bits={eff:.3f}): "
        f"top1 {np.mean(all_top1):.3f}  top5 {np.mean(all_top5):.3f}  "
        f"KL {np.mean(all_kl):.4f}  fn_top1 {np.mean(all_fn_top1):.3f}  "
        f"final cos {cos_by_layer[-1]:.5f}  min cos {min(cos_by_layer):.5f}"
    )
    typer.echo(report_footprint(t0))


if __name__ == "__main__":
    app()
