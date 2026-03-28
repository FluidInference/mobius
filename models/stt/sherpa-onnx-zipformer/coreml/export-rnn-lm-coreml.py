#!/usr/bin/env python3
"""Export a trained RNN-LM checkpoint to CoreML (step-by-step format).

Takes a PyTorch checkpoint (.pt) from train-rnn-lm.py and produces a
CoreML .mlpackage that scores one BPE token at a time with LSTM state.

I/O format:
  Inputs:  token_id [1] int32, h_in [num_layers, 1, hidden_dim] float32,
           c_in [num_layers, 1, hidden_dim] float32
  Outputs: log_probs [1, vocab_size] float32, h_out [...], c_out [...]

Usage:
    python export-rnn-lm-coreml.py \
        --checkpoint lm_weights.pt \
        --vocab-size 500 \
        --embedding-dim 256 --hidden-dim 512 --num-layers 2 \
        --output rnn_lm.mlpackage

    # Also export compiled .mlmodelc:
    xcrun coremlcompiler compile rnn_lm.mlpackage ./
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnLmModel(nn.Module):
    """Must match train-rnn-lm.py exactly."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, vocab_size)


class RnnLmStep(nn.Module):
    """Step-by-step wrapper for CoreML export.

    Scores one token at a time, carrying LSTM state between calls.
    """

    def __init__(self, model: RnnLmModel):
        super().__init__()
        self.embedding = model.input_embedding
        self.rnn = model.rnn
        self.output_linear = model.output_linear

    def forward(
        self, token_id: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_id: [1] int32 — single token ID
            h_in: [num_layers, 1, hidden_dim] float32
            c_in: [num_layers, 1, hidden_dim] float32
        Returns:
            log_probs: [1, vocab_size] float32
            h_out: [num_layers, 1, hidden_dim] float32
            c_out: [num_layers, 1, hidden_dim] float32
        """
        emb = self.embedding(token_id.long())  # [1, embedding_dim]
        emb = emb.unsqueeze(1)  # [1, 1, embedding_dim]
        out, (h_out, c_out) = self.rnn(emb, (h_in, c_in))
        logits = self.output_linear(out.squeeze(1))  # [1, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, h_out, c_out


def main():
    parser = argparse.ArgumentParser(description="Export RNN-LM checkpoint to CoreML")
    parser.add_argument("--checkpoint", required=True, help="PyTorch weights (.pt) from train-rnn-lm.py")
    parser.add_argument("--output", default="rnn_lm.mlpackage", help="Output CoreML path")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()

    try:
        import coremltools as ct
        import numpy as np
    except ImportError:
        print("pip install coremltools numpy", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = RnnLmModel(args.vocab_size, args.embedding_dim, args.hidden_dim, args.num_layers)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()

    step = RnnLmStep(model)
    step.eval()

    print("Tracing...")
    token_id = torch.tensor([0], dtype=torch.int32)
    h_in = torch.zeros(args.num_layers, 1, args.hidden_dim)
    c_in = torch.zeros(args.num_layers, 1, args.hidden_dim)

    traced = torch.jit.trace(step, (token_id, h_in, c_in))

    print("Converting to CoreML...")
    ct_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="token_id", shape=(1,), dtype=np.int32),
            ct.TensorType(name="h_in", shape=(args.num_layers, 1, args.hidden_dim), dtype=np.float32),
            ct.TensorType(name="c_in", shape=(args.num_layers, 1, args.hidden_dim), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="log_probs", dtype=np.float32),
            ct.TensorType(name="h_out", dtype=np.float32),
            ct.TensorType(name="c_out", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,
        skip_model_load=True,
    )

    ct_model.save(args.output)
    size_mb = sum(
        f.stat().st_size for f in __import__("pathlib").Path(args.output).rglob("*") if f.is_file()
    ) / 1024 / 1024
    print(f"Exported: {args.output} ({size_mb:.1f} MB)")
    print(f"\nTo compile: xcrun coremlcompiler compile {args.output} ./")


if __name__ == "__main__":
    main()
