#!/usr/bin/env python3
"""Wrapper modules for exporting CTC decoder head from pure CTC model to CoreML.

The pure CTC model (EncDecCTCBPEModel) exposes .decoder as the CTC projection
layer, unlike the hybrid model which uses .ctc_decoder.
"""
from __future__ import annotations

import torch


class CTCDecoderWrapper(torch.nn.Module):
    """CTC decoder head: encoder_output -> log-probabilities.

    Wraps the CTC decoder module for torch.jit.trace compatibility.
    Input:  encoder_output [B, D, T]  (raw encoder features)
    Output: ctc_logits     [B, T, V+1] (CTC log-probabilities)
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        return self.module(encoder_output=encoder_output)
