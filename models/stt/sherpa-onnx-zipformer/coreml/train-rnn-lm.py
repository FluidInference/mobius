#!/usr/bin/env python3
"""Train a small RNN language model compatible with sherpa-onnx.

Trains a BPE-level LSTM language model from text data and exports to ONNX
in the format expected by sherpa-onnx's offline recognizer.

Input: text file (one utterance per line) + BPE tokens.txt from the ASR model
Output: lm.onnx compatible with sherpa-onnx --lm flag

Usage:
    python train-rnn-lm.py \
        --text training_text.txt \
        --tokens /path/to/tokens.txt \
        --output lm.onnx \
        --epochs 10

    # Use with sherpa-onnx:
    swift run benchmark predict --engine sherpa \
        --sherpa-lm lm.onnx --sherpa-lm-scale 0.5

Text format: one sentence per line, lowercased. Example:
    lufthansa four three nine zero descend flight level one hundred
    speed bird triple five turn left heading three six zero
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("pip install torch", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# BPE tokenizer (matches icefall/sherpa-onnx tokens.txt)
# ---------------------------------------------------------------------------

class BpeTokenizer:
    """Minimal BPE tokenizer using tokens.txt from icefall/sherpa-onnx."""

    WORD_BOUNDARY = "\u2581"  # SentencePiece ▁

    def __init__(self, tokens_path: str):
        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}
        with open(tokens_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    tok, tid = parts[0], int(parts[1])
                    self.id_to_token[tid] = tok
                    self.token_to_id[tok] = tid
        self.vocab_size = len(self.id_to_token)
        self.blank_id = self.token_to_id.get("<blk>", 0)
        self.sos_id = self.token_to_id.get("<sos/eos>", 1)
        self.unk_id = self.token_to_id.get("<unk>", 2)

        # Build sorted token list for greedy BPE encoding (longest match first)
        self._tokens_by_length = sorted(
            [(tok, tid) for tok, tid in self.token_to_id.items()
             if not tok.startswith("<")],
            key=lambda x: -len(x[0])
        )

    def encode(self, text: str) -> list[int]:
        """Encode text to BPE token IDs using greedy longest-match."""
        # Prepend ▁ to mark word boundaries (SentencePiece convention)
        text = self.WORD_BOUNDARY + text.replace(" ", self.WORD_BOUNDARY)
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for tok, tid in self._tokens_by_length:
                if text[i:i + len(tok)] == tok:
                    ids.append(tid)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ids.append(self.unk_id)
                i += 1
        return ids


# ---------------------------------------------------------------------------
# RNN-LM model (matches icefall RnnLmModel architecture)
# ---------------------------------------------------------------------------

class RnnLmModel(nn.Module):
    """LSTM language model compatible with sherpa-onnx ONNX LM format.

    Architecture matches icefall's rnn_lm/model.py exactly:
        input_embedding → LSTM → output_linear
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_linear = nn.Linear(hidden_dim, vocab_size)

        if tie_weights:
            assert embedding_dim == hidden_dim
            self.output_linear.weight = self.input_embedding.weight

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Rescoring forward pass matching icefall RnnLmModel.forward().

        Args:
            x: Input with SOS prepended, shape (N, L)
            y: Target with EOS appended, shape (N, L)
            lengths: Sequence lengths before padding, shape (N,)

        Returns:
            Per-token NLL, shape (N, L). Padding positions are zeroed.
        """
        embedding = self.input_embedding(x)
        rnn_out, _ = self.rnn(embedding)
        logits = self.output_linear(rnn_out)
        nll_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            y.reshape(-1),
            reduction="none",
        )
        # Mask padding positions
        N, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        mask = positions >= lengths.unsqueeze(1)
        nll_loss = nll_loss.reshape(N, L)
        nll_loss.masked_fill_(mask, 0)
        return nll_loss

    def forward_train(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Training forward pass (full sequence).

        Args:
            x: Input token IDs, shape (batch, seq_len)

        Returns:
            logits: shape (batch, seq_len, vocab_size)
        """
        emb = self.input_embedding(x)
        out, _ = self.rnn(emb)
        return self.output_linear(out)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_text_data(text_path: str, tokenizer: BpeTokenizer) -> list[list[int]]:
    """Load and tokenize text file, one sentence per line."""
    sequences = []
    with open(text_path) as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            ids = tokenizer.encode(line)
            if len(ids) >= 2:
                sequences.append(ids)
    return sequences


def make_batches(
    sequences: list[list[int]], batch_size: int, sos_id: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create training batches with SOS prepended.

    Input:  [t1, t2, t3, ...]
    Target: [t1, t2, t3, ...]
    Input with SOS: [SOS, t1, t2, ...]
    """
    random.shuffle(sequences)
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        max_len = max(len(s) for s in batch_seqs)

        inputs = torch.zeros(len(batch_seqs), max_len + 1, dtype=torch.long)
        targets = torch.full((len(batch_seqs), max_len), -100, dtype=torch.long)

        for j, seq in enumerate(batch_seqs):
            inputs[j, 0] = sos_id
            for k, tid in enumerate(seq):
                inputs[j, k + 1] = tid
                targets[j, k] = tid

        batches.append((inputs[:, :-1], targets))  # align lengths
    return batches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: RnnLmModel,
    sequences: list[list[int]],
    sos_id: int,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
):
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        batches = make_batches(sequences, batch_size, sos_id)
        total_loss = 0.0
        total_tokens = 0

        for inputs, targets in batches:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model.forward_train(inputs)
            # logits: (batch, seq_len, vocab), targets: (batch, seq_len)
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            n_tokens = (targets != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

        scheduler.step()
        ppl = np.exp(total_loss / max(total_tokens, 1))
        print(f"  Epoch {epoch}/{epochs}  loss={total_loss/max(total_tokens,1):.4f}  ppl={ppl:.1f}  lr={scheduler.get_last_lr()[0]:.6f}")


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

class RnnLmRescoringWrapper(torch.nn.Module):
    """Wraps RnnLmModel for batch rescoring export (sherpa-onnx format).

    sherpa-onnx expects:
        Inputs:  x (N, L) token sequences, x_lens (N,) lengths
        Outputs: nll (N,) negative log-likelihoods per sequence

    The wrapper prepends SOS and appends EOS internally.
    """

    def __init__(self, model: RnnLmModel, sos_id: int, eos_id: int):
        super().__init__()
        self.model = model
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        """Exact match with icefall RnnLmModelWrapper.forward()."""
        N = x.size(0)

        sos_tensor = torch.full(
            (1,), fill_value=self.sos_id, dtype=x.dtype
        ).expand(N, 1)
        sos_x = torch.cat([sos_tensor, x], dim=1)

        pad_col = torch.zeros((1,), dtype=x.dtype).expand(N, 1)
        x_eos = torch.cat([x, pad_col], dim=1)

        row_index = torch.arange(0, N, dtype=x.dtype)
        x_eos[row_index, x_lens] = self.eos_id

        # use x_lens + 1 here since we prepended x with sos
        return (
            self.model(x=sos_x, y=x_eos, lengths=x_lens + 1)
            .to(torch.float32)
            .sum(dim=1)
        )


def export_onnx(model: RnnLmModel, output_path: str, sos_id: int = 1, eos_id: int = 1):
    """Export to ONNX in sherpa-onnx batch-rescoring format.

    Matches icefall rnn_lm/export-onnx.py export_without_state() exactly.
    """
    model.eval()
    model.cpu()

    wrapper = RnnLmRescoringWrapper(model, sos_id=sos_id, eos_id=eos_id)
    wrapper.eval()

    N = 1
    L = 20
    x = torch.randint(low=1, high=model.vocab_size, size=(N, L), dtype=torch.int64)
    x_lens = torch.full((N,), fill_value=L, dtype=torch.int64)

    torch.onnx.export(
        wrapper,
        (x, x_lens),
        output_path,
        verbose=False,
        opset_version=13,
        input_names=["x", "x_lens"],
        output_names=["nll"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},
            "x_lens": {0: "N"},
            "nll": {0: "N"},
        },
        dynamo=False,
    )

    # Add metadata matching icefall convention
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        for key, value in {
            "model_type": "rnnlm",
            "version": "1",
            "comment": "rnnlm without state",
            "sos_id": str(sos_id),
            "eos_id": str(eos_id),
            "vocab_size": str(model.vocab_size),
        }.items():
            meta = onnx_model.metadata_props.add()
            meta.key = key
            meta.value = value
        onnx.save(onnx_model, output_path)
    except ImportError:
        pass  # onnx not installed, skip metadata
    print(f"  Exported: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train an RNN-LM for sherpa-onnx from text data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From raw text (one sentence per line):
  python train-rnn-lm.py --text atc_text.txt --tokens tokens.txt --output lm.onnx

  # From ARPA LM (extracts n-gram text for training):
  python train-rnn-lm.py --arpa en-atco.lm --tokens tokens.txt --output lm.onnx

  # With custom hyperparameters:
  python train-rnn-lm.py --text atc_text.txt --tokens tokens.txt \\
      --embedding-dim 256 --hidden-dim 512 --num-layers 2 \\
      --epochs 20 --batch-size 128 --output lm.onnx

  # Use the trained LM with sherpa-onnx benchmark:
  swift run -c release benchmark predict --engine sherpa \\
      --sherpa-lm lm.onnx --sherpa-lm-scale 0.5
        """,
    )

    # Input (one of --text or --arpa required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Training text file (one sentence per line)")
    input_group.add_argument("--arpa", type=str, help="ARPA LM file (extracts unigram words for training)")

    parser.add_argument("--tokens", type=str, required=True, help="tokens.txt from the ASR model")
    parser.add_argument("--output", type=str, default="lm.onnx", help="Output ONNX model path")
    parser.add_argument("--save-checkpoint", type=str, default=None, help="Save PyTorch weights (.pt) for later re-export")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Load PyTorch weights (.pt) — skip training, export only")

    # Model hyperparameters
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tie-weights", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size (default: from tokens.txt)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading tokenizer: {args.tokens}")
    tokenizer = BpeTokenizer(args.tokens)
    print(f"  vocab_size={tokenizer.vocab_size}, sos_id={tokenizer.sos_id}")

    # Load training text
    if args.text:
        print(f"Loading text: {args.text}")
        text_lines = []
        with open(args.text) as f:
            for line in f:
                line = line.strip()
                if line:
                    text_lines.append(line)
    else:
        # Extract sentences from ARPA unigrams (generate synthetic n-gram text)
        print(f"Extracting training text from ARPA: {args.arpa}")
        text_lines = extract_text_from_arpa(args.arpa)

    print(f"  {len(text_lines)} sentences")

    print("Tokenizing...")
    sequences = []
    for line in text_lines:
        ids = tokenizer.encode(line.lower())
        if len(ids) >= 2:
            sequences.append(ids)
    print(f"  {len(sequences)} sequences, avg length {np.mean([len(s) for s in sequences]):.1f} tokens")

    effective_vocab_size = args.vocab_size or tokenizer.vocab_size
    print(f"  Using vocab_size={effective_vocab_size}")

    model = RnnLmModel(
        vocab_size=effective_vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        tie_weights=args.tie_weights,
    )

    if args.load_checkpoint:
        print(f"Loading checkpoint: {args.load_checkpoint}")
        state = torch.load(args.load_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    else:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"\nTraining RNN-LM (embed={args.embedding_dim}, hidden={args.hidden_dim}, "
              f"layers={args.num_layers}, device={args.device})")
        print(f"  parameters: {param_count:,}")

        train(
            model, sequences, tokenizer.sos_id,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, device=args.device,
        )

        if args.save_checkpoint:
            torch.save(model.state_dict(), args.save_checkpoint)
            print(f"  Saved checkpoint: {args.save_checkpoint}")

    print(f"\nExporting ONNX: {args.output}")
    export_onnx(model, args.output, sos_id=tokenizer.sos_id, eos_id=tokenizer.sos_id)
    print("Done.")


def extract_text_from_arpa(arpa_path: str) -> list[str]:
    """Extract training sentences from ARPA bigrams.

    Reconstructs sentences from bigram sequences in the ARPA file.
    This provides reasonable training data when no raw text corpus is available.
    """
    unigrams = []
    bigrams: dict[str, list[str]] = {}
    section = ""

    with open(arpa_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("\\data\\"):
                continue
            if line == "\\end\\":
                break
            if line.startswith("\\"):
                section = line
                continue
            if line.startswith("ngram "):
                continue

            parts = line.split("\t")
            if section == "\\1-grams:" and len(parts) >= 2:
                word = parts[1]
                if not word.startswith("<") and word.isalpha():
                    unigrams.append(word)
            elif section == "\\2-grams:" and len(parts) >= 3:
                ctx, word = parts[1], parts[2]
                if not ctx.startswith("<") and not word.startswith("<"):
                    bigrams.setdefault(ctx, []).append(word)

    # Generate sentences by following bigram chains
    sentences = []
    for start_word in unigrams[:5000]:  # Limit starting words
        sentence = [start_word]
        current = start_word
        for _ in range(15):  # Max sentence length
            nexts = bigrams.get(current, [])
            if not nexts:
                break
            current = random.choice(nexts)
            sentence.append(current)
            if current in (".", "?", "!"):
                break
        if len(sentence) >= 3:
            sentences.append(" ".join(sentence))

    # Also add raw unigram combinations as short phrases
    for _ in range(len(sentences)):
        n = random.randint(3, 10)
        phrase = " ".join(random.choices(unigrams, k=n))
        sentences.append(phrase)

    random.shuffle(sentences)
    return sentences


if __name__ == "__main__":
    main()
