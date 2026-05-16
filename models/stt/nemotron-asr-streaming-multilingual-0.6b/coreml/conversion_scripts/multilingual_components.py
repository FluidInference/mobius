#!/usr/bin/env python3
"""Prompt-aware streaming-encoder wrapper for Nemotron-3.5-ASR-Multilingual.

This wraps `model.encoder` so the resulting CoreML graph takes the same
inputs as the English Nemotron encoder plus one extra int32 `prompt_id`
input, and applies the language `prompt_kernel` adapter INSIDE the
graph. Swift only needs to pass the int prompt id resolved from
`metadata.json["prompt_dictionary"]`.

## Layout (verified against the live NeMo class)

The model class (`EncDecRNNTBPEModelWithPrompt`) applies the prompt
AFTER the full encoder runs, not between pre_encode and the conformer
body. From `conformer_stream_step`:

    encoded, encoded_len, caches = self.encoder.cache_aware_stream_step(...)
    encoded = self._apply_prompt_to_encoded(encoded)   # <-- post-encoder

Where `_apply_prompt_to_encoded` is:

    one_hot = F.one_hot(self._inference_prompt_index, num_prompts).float()
    one_hot = one_hot.unsqueeze(0).unsqueeze(0).expand(B, T, -1)   # [B,T,128]
    encoded = encoded.transpose(1, 2)                              # [B,T,1024]
    encoded = torch.cat([encoded, one_hot], dim=-1)                # [B,T,1152]
    encoded = self.prompt_kernel(encoded)                          # [B,T,1024]
    return encoded.transpose(1, 2)                                 # [B,1024,T]

The 24-layer conformer body is language-agnostic; only `prompt_kernel`
(a 2-layer MLP) is conditioned on language. There is no encoder kwarg
for the prompt in either offline or streaming paths.

## Checkpoint shapes (verified from `model_weights.ckpt`)

    prompt_kernel.0.weight  (2048, 1152)   # Linear(1024 + 128 -> 2048)
    prompt_kernel.0.bias    (2048,)
    prompt_kernel.2.weight  (1024, 2048)   # Linear(2048 -> 1024)
    prompt_kernel.2.bias    (1024,)

Concat order is `[encoded, one_hot]` — the 1024 encoder features live in
columns `[0:1024]` and the 128-d one-hot in columns `[1024:1152]`, which
is what the trained `prompt_kernel.0.weight` expects. Reversing the
order would silently break parity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

NUM_PROMPTS = 128  # from model_config.yaml: model_defaults.num_prompts
ENC_HIDDEN = 1024  # from model_config.yaml: model_defaults.enc_hidden


@dataclass
class PromptWrapperConfig:
    """Shared config for the prompt-aware streaming encoder wrapper."""

    num_prompts: int = NUM_PROMPTS
    encoder_d_model: int = ENC_HIDDEN


class PromptIdToOneHot(torch.nn.Module):
    """Convert int32 prompt_id [B] to one-hot [B, num_prompts] float32."""

    def __init__(self, num_prompts: int = NUM_PROMPTS) -> None:
        super().__init__()
        self.num_prompts = int(num_prompts)

    def forward(self, prompt_id: torch.Tensor) -> torch.Tensor:
        # prompt_id: [B] int32  ->  [B, num_prompts] float32
        return F.one_hot(prompt_id.to(torch.long), self.num_prompts).to(torch.float32)


class EncoderStreamingWithPostPrompt(torch.nn.Module):
    """Cache-aware streaming encoder with post-encoder language adapter.

    Forward graph (matches `conformer_stream_step` + `_apply_prompt_to_encoded`):

        features [B, 128, T_mel] + caches + prompt_id [B]
            └► encoder.cache_aware_stream_step(...)
                 └► encoded [B, 1024, T_enc] + new caches
            └► apply prompt_kernel on encoded (post-encoder, see module doc)
                 └► conditioned [B, 1024, T_enc]
            └► return (conditioned, enc_len, *new_caches)

    I/O contract identical to the English Nemotron encoder wrapper except
    for the extra `prompt_id` input. Cache layouts follow the existing
    Swift convention `[B, L, ...]`; transposes match the English wrapper.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        prompt_kernel: torch.nn.Module,
        num_prompts: int = NUM_PROMPTS,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.prompt_kernel = prompt_kernel
        self.to_onehot = PromptIdToOneHot(num_prompts)

    def _apply_prompt(
        self, encoded: torch.Tensor, prompt_id: torch.Tensor
    ) -> torch.Tensor:
        """Apply prompt_kernel on (B, D, T) encoder output.

        Mirrors `EncDecRNNTBPEModelWithPrompt._apply_prompt_to_encoded`
        exactly: transpose to (B, T, D), concat one-hot in the trailing
        128 channels, run the 2-layer MLP, transpose back.
        """
        # encoded: [B, D=1024, T_enc]
        b, _, t = encoded.shape
        one_hot = self.to_onehot(prompt_id)                # [B, 128]
        one_hot = one_hot.unsqueeze(1).expand(b, t, -1)    # [B, T_enc, 128]

        encoded_btd = encoded.transpose(1, 2)              # [B, T_enc, 1024]
        cat = torch.cat([encoded_btd, one_hot], dim=-1)    # [B, T_enc, 1152]
        conditioned = self.prompt_kernel(cat)              # [B, T_enc, 1024]
        return conditioned.transpose(1, 2)                 # [B, 1024, T_enc]

    def forward(
        self,
        features: torch.Tensor,
        length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
        prompt_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Caches stored as [B, L, ...] in Swift; NeMo wants [L, B, ...].
        cache_ch_t = cache_last_channel.transpose(0, 1)
        cache_t_t = cache_last_time.transpose(0, 1)
        cache_len_i64 = cache_last_channel_len.to(torch.int64)

        encoded, enc_len, cache_ch_n, cache_t_n, cache_len_n = (
            self.encoder.cache_aware_stream_step(
                processed_signal=features,
                processed_signal_length=length.to(torch.long),
                cache_last_channel=cache_ch_t,
                cache_last_time=cache_t_t,
                cache_last_channel_len=cache_len_i64,
                keep_all_outputs=True,
                drop_extra_pre_encoded=None,
                bypass_pre_encode=False,
            )
        )

        conditioned = self._apply_prompt(encoded, prompt_id)

        return (
            conditioned,
            enc_len.to(torch.int32),
            cache_ch_n.transpose(0, 1),
            cache_t_n.transpose(0, 1),
            cache_len_n.to(torch.int32),
        )
