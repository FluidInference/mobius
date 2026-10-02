"""Fixed-shape, mask-aware JointSchemaHead for Core ML (the student's head weights).

Clef's head loops over questions and option spans in Python. Here every span mean is a host-built matrix and the
question/option counts are padded to fixed Q / O with masks, so one trace covers any schema that fits:

  states    [L, D]   post-norm hidden states (LMRows output; rows past the record's n tokens are padding)
  mem_mask  [L]      additive key mask over tokens: 0 real, -1e4 padding
  last      [L]      one-hot of the last real token (global vector)
  q_mean    [Q, L]   row i averages question i's instruction span (zero rows for padded questions)
  o_mean    [O, L]   row j averages option j's span
  o2q       [O, Q]   one-hot: option j belongs to question i
  lexical   [O, D]   host-gathered mean output-embedding rows of option j's tokens
  type_oh   [Q, 3]   one-hot question type (noul / choice / score)
  q_mask    [Q]      additive key mask over questions for the field self-attention: 0 real, -1e4 padded
  o_mask    [O]      additive mask for the per-question option softmax: 0 real, -1e4 padded
  -> logits [O]      (padded options carry garbage; the host reads rows [0, n_options))
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, queries, memory, key_mask):
        """queries [Nq, W], memory [Nk, W], key_mask [Nk] additive."""
        nq, width = queries.shape
        d = width // self.heads
        q = self.q_proj(queries).reshape(nq, self.heads, d).permute(1, 0, 2)
        k = self.k_proj(memory).reshape(-1, self.heads, d).permute(1, 0, 2)
        v = self.v_proj(memory).reshape(-1, self.heads, d).permute(1, 0, 2)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d) + key_mask[None, None, :]
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        return self.out_proj(out.permute(1, 0, 2).reshape(nq, width))


class EvidenceLayer(nn.Module):
    def __init__(self, width, heads, ff):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        self.ff1 = nn.Linear(width, ff)
        self.ff2 = nn.Linear(ff, width)

    def forward(self, q, memory, mem_mask):
        m = self.memory_norm(memory)
        q = q + self.attention(self.query_norm(q), m, mem_mask)
        return q + self.ff2(F.gelu(self.ff1(self.feedforward_norm(q))))


class DecoderLayer(nn.Module):
    def __init__(self, width, heads, ff):
        super().__init__()
        self.self_attn = Attention(width, heads)
        self.multihead_attn = Attention(width, heads)
        self.linear1 = nn.Linear(width, ff)
        self.linear2 = nn.Linear(ff, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)

    def forward(self, x, memory, mem_mask, q_mask):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, q_mask)
        x = x + self.multihead_attn(self.norm2(x), memory, mem_mask)
        return x + self.linear2(F.gelu(self.linear1(self.norm3(x))))


class FixedHead(nn.Module):
    def __init__(self, hidden_size, width, routing_layers, layers, heads, feedforward, dropout=0.0, **unexpected):
        if unexpected:  # a head config with options this rewrite does not implement must not export silently
            raise ValueError(f"unsupported joint_head_config keys: {sorted(unexpected)}")
        super().__init__()
        self.width = width
        self.hidden_norm = nn.LayerNorm(hidden_size)
        for name in ("memory", "question", "option_question", "global", "option_context", "option_lexical"):
            setattr(self, f"{name}_projection", nn.Linear(hidden_size, width, bias=False))
        self.type_embedding = nn.Linear(3, width, bias=False)  # one-hot @ table == embedding lookup
        self.evidence_layers = nn.ModuleList([EvidenceLayer(width, heads, feedforward) for _ in range(routing_layers)])
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList([DecoderLayer(width, heads, feedforward) for _ in range(layers)])
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.scorer1 = nn.Linear(width * 4, width)
        self.scorer2 = nn.Linear(width, 1)
        self.prior_logit_scale = nn.Parameter(torch.zeros(()))
        self.joint_logit_scale = nn.Parameter(torch.zeros(()))
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(self, states, mem_mask, last, q_mean, o_mean, o2q, lexical, type_oh, q_mask, o_mask):
        normed = self.hidden_norm(states)                       # [L, D]
        memory = self.memory_projection(normed)                 # [L, W]
        global_vector = torch.matmul(last, normed)              # [D]
        question_vectors = torch.matmul(q_mean, normed)         # [Q, D]
        context = torch.matmul(o_mean, normed)                  # [O, D]
        option_queries = (self.option_context_projection(context) + self.option_lexical_projection(lexical)
                          + torch.matmul(o2q, self.option_question_projection(question_vectors)))
        routed = option_queries
        for layer in self.evidence_layers:
            routed = layer(routed, memory, mem_mask)            # [O, W]
        base_fields = self.question_projection(question_vectors)  # [Q, W]
        field_per_option = torch.matmul(o2q, base_fields)       # [O, W]
        scores = (routed * field_per_option).sum(-1) / math.sqrt(self.width) + o_mask  # [O]
        # softmax within each question's options: subtract the per-question max (a global max would underflow the
        # weaker questions' exponentials), then normalise by the per-question sum.
        # constants stay fp16-representable (-1e9 / 1e-30 would turn into inf*0 = NaN under --precision fp16)
        group_max = (scores.unsqueeze(-1) + (1.0 - o2q) * -1e4).max(dim=0).values  # [Q]
        exp = torch.exp(scores - torch.matmul(o2q, group_max.unsqueeze(-1)).squeeze(-1))
        group = torch.matmul(o2q.transpose(0, 1), exp.unsqueeze(-1)).squeeze(-1)  # [Q]; >= 1 for real questions
        weights = exp / torch.clamp(torch.matmul(o2q, group.unsqueeze(-1)).squeeze(-1), min=1e-4)  # [O]
        summaries = torch.matmul(o2q.transpose(0, 1), weights.unsqueeze(-1) * routed)  # [Q, W]
        fields = (base_fields + self.option_summary_norm(summaries) + self.global_projection(global_vector)[None, :]
                  + self.type_embedding(type_oh))
        for layer in self.layers:
            fields = layer(fields, memory, mem_mask, q_mask)
        fields = self.field_norm(fields)                        # [Q, W]
        anchor = F.normalize(question_vectors + global_vector[None, :], dim=-1)  # [Q, D]
        anchor_per_option = torch.matmul(o2q, anchor)           # [O, D]
        prior_scale = torch.exp(torch.clamp(self.prior_logit_scale, max=math.log(100.0)))
        prior = prior_scale * (F.normalize(lexical, dim=-1) * anchor_per_option).sum(-1)
        options = self.option_norm(routed)
        field_o = torch.matmul(o2q, fields)
        cosine = F.cosine_similarity(field_o, options, dim=-1)
        features = torch.cat([field_o, options, field_o * options, torch.abs(field_o - options)], dim=-1)
        residual = self.scorer2(F.gelu(self.scorer1(features))).squeeze(-1)
        joint_scale = torch.exp(torch.clamp(self.joint_logit_scale, max=math.log(100.0)))
        return prior + torch.sigmoid(self.residual_gate) * (joint_scale * cosine + residual)

    def load_clef_head(self, state: dict[str, torch.Tensor]) -> None:
        own = {}
        width = self.width

        def attention(src, dst):
            w, b = state[f"{src}.in_proj_weight"], state[f"{src}.in_proj_bias"]
            for i, name in enumerate(("q_proj", "k_proj", "v_proj")):
                own[f"{dst}.{name}.weight"] = w[i * width:(i + 1) * width]
                own[f"{dst}.{name}.bias"] = b[i * width:(i + 1) * width]
            own[f"{dst}.out_proj.weight"] = state[f"{src}.out_proj.weight"]
            own[f"{dst}.out_proj.bias"] = state[f"{src}.out_proj.bias"]

        for key, value in state.items():
            if ".in_proj_" in key or (".out_proj." in key and ("attention" in key or "attn" in key)):
                continue
            new = key
            for a, b in ((".feedforward.0.", ".ff1."), (".feedforward.3.", ".ff2."),
                         ("residual_scorer.0.", "scorer1."), ("residual_scorer.3.", "scorer2.")):
                new = new.replace(a, b)
            if new == "type_embedding.weight":
                value = value.t().contiguous()  # Embedding [3, W] -> Linear weight [W, 3]
            own[new] = value
        for i in range(len(self.evidence_layers)):
            attention(f"evidence_layers.{i}.attention", f"evidence_layers.{i}.attention")
        for i in range(len(self.layers)):
            attention(f"layers.{i}.self_attn", f"layers.{i}.self_attn")
            attention(f"layers.{i}.multihead_attn", f"layers.{i}.multihead_attn")
        missing, unexpected = self.load_state_dict({k: v.float() for k, v in own.items()}, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"head weight mismatch: missing={missing[:6]} unexpected={unexpected[:6]}")


def head_inputs(encoded, states: torch.Tensor, output_embed: torch.Tensor, seq_len: int, max_q: int, max_o: int):
    """Host matrices for one Clef ``EncodedRecord``. ``states`` [seq_len, D] already padded; returns tensors + n_options."""
    n = len(encoded.input_ids)
    questions = encoded.questions
    n_q = len(questions)
    n_o = sum(len(q.option_spans) for q in questions)
    if n_q > max_q or n_o > max_o or n > seq_len:
        raise ValueError(f"record needs Q={n_q} O={n_o} L={n}; bucket Q={max_q} O={max_o} L={seq_len}")
    ids = torch.tensor(encoded.input_ids)
    mem_mask = torch.zeros(seq_len); mem_mask[n:] = -1e4
    last = torch.zeros(seq_len); last[n - 1] = 1.0
    q_mean = torch.zeros(max_q, seq_len); o_mean = torch.zeros(max_o, seq_len); o2q = torch.zeros(max_o, max_q)
    lexical = torch.zeros(max_o, output_embed.shape[1]); type_oh = torch.zeros(max_q, 3)
    q_mask = torch.full((max_q,), -1e4); o_mask = torch.full((max_o,), -1e4)
    j = 0
    for i, q in enumerate(questions):
        s, e = q.question_span; q_mean[i, s:e] = 1.0 / (e - s); type_oh[i, q.question_type] = 1.0; q_mask[i] = 0.0
        for s, e in q.option_spans:
            o_mean[j, s:e] = 1.0 / (e - s); o2q[j, i] = 1.0; o_mask[j] = 0.0
            lexical[j] = output_embed[ids[s:e]].float().mean(0); j += 1
    return (states, mem_mask, last, q_mean, o_mean, o2q, lexical, type_oh, q_mask, o_mask), n_o


def load_head(student_dir: Path):
    from safetensors.torch import load_file

    config = json.loads((student_dir / "joint_head_config.json").read_text())
    head = FixedHead(**config)
    head.load_clef_head(load_file(str(student_dir / "joint_head.safetensors")))
    return head.eval(), config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", type=Path, default=Path("build/student-r1/best"))
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--max-q", type=int, default=16)
    ap.add_argument("--max-o", type=int, default=64)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp32")
    ap.add_argument("--out", type=Path, default=Path("build/coreml"))
    args = ap.parse_args()
    import coremltools as ct

    head, config = load_head(args.student)
    L, Q, O, D = args.length, args.max_q, args.max_o, config["hidden_size"]
    example = (torch.zeros(L, D), torch.zeros(L), torch.zeros(L), torch.zeros(Q, L), torch.zeros(O, L), torch.zeros(O, Q),
               torch.zeros(O, D), torch.zeros(Q, 3), torch.zeros(Q), torch.zeros(O))
    names = ["states", "mem_mask", "last", "q_mean", "o_mean", "o2q", "lexical", "type_oh", "q_mask", "o_mask"]
    started = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(head, example, check_trace=False)
    model = ct.convert(
        traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[ct.TensorType(name=n, shape=tuple(t.shape), dtype=np.float32) for n, t in zip(names, example)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    )
    model.short_description = "clef-vision-0.8b joint schema head: one logit per option, fixed Q/O with masks"
    model.license = "Apache-2.0"
    out = args.out / f"Head_L{L}_Q{Q}_O{O}"
    out.mkdir(parents=True, exist_ok=True)
    package = out / f"Head_{args.precision}.mlpackage"
    model.save(str(package))
    (out / "config.json").write_text(json.dumps({"length": L, "max_q": Q, "max_o": O, **config,
                                                  "precision": args.precision}, indent=2) + "\n")
    print(f"saved {package} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
