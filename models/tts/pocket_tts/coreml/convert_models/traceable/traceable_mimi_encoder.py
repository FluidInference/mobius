"""Traceable Mimi audio encoder for CoreML conversion.

Wraps pocket-tts 2.0.0's audio-to-conditioning pipeline (TTSModel._encode_audio)
into a torch.jit.traceable module so it can be converted to a CoreML
mlpackage for voice cloning at runtime.

Pipeline (stateless one-shot encode):
    audio [1, 1, T] -> mimi.encoder           -> [1, 512, num_frames]
                    -> mimi.encoder_transformer -> [1, 512, num_frames]
                    -> mimi._to_framerate     -> [1,  32, num_frames]
                    -> transpose(-1, -2)      -> [1, num_frames, 32]
                    -> F.linear(speaker_proj) -> [1, num_frames, 1024]

For tracing we fix `T = 125 * frame_size = 240_000` samples (10s at 24kHz,
which is the standard PocketTTS voice prompt length). Callers that have
shorter/longer audio must pad/truncate to exactly 240_000 samples before
invoking the CoreML model.

Why we need this re-trace
-------------------------
The legacy `voice_cloning/mimi_encoder.mlmodelc` shipped under PocketTTS was
traced against a pocket-tts version older than the Apr 27 cond_step /
flowlm_step / flow_decoder mlpackages now deployed under `build/<lang>/`.
That older encoder maps audio to a conditioning latent space that disagrees
with the new cond_step weights, causing the flow LM to emit EOS within a
few steps (silent / garbled audio — FluidAudio issue #592). Re-tracing
against pocket-tts 2.0.0 (the version that produced the deployed v2 caches)
restores the correct latent space and unblocks the pure-CoreML voice
cloning path.

Tracing notes
-------------
The trace path is intentionally simpler than the decoder's:
1. **No state I/O** — `mimi.encode_to_latent(audio, model_state=None)` runs
   in stateless one-shot mode; per-layer streaming caches are initialized
   and discarded inside the call. Inputs: 1 (audio). Outputs: 1 (cond).
2. **`pad_for_conv1d` bypass** — upstream's pad helper goes through
   `get_extra_padding_for_conv1d` which is `@beartype`'d to return `int`.
   `torch.jit.trace` rewrites the return as a 0-d tensor; beartype then
   throws `BeartypeCallHintReturnViolation`. We replace
   `pocket_tts.models.mimi.pad_for_conv1d` with the identity (caller has
   already aligned the audio length to a multiple of `frame_size`).
3. **`init_state` beartype unwrap** — `StreamingConv1d.init_state` /
   `StreamingConvTranspose1d.init_state` are `@beartype`'d on
   `batch_size: int`; trace passes a 0-d tensor here too. Replace with the
   `__wrapped__` originals.
4. **In-place op patches** — reuse `traceable_mimi_decoder._patch_for_tracing`
   to swap `StreamingConv1d.forward`, `StreamingConvTranspose1d.forward`,
   and `StreamingMultiheadAttention.forward` with functional versions
   (`torch.jit.trace` can't capture `state[:] = ...` writes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fixed shapes for the traced model.
# 125 * 1920 = 240_000 samples = 10s at 24kHz, matching PocketTTS's
# `VOICE_PROMPT_LENGTH = 125` convention.
AUDIO_LENGTH_SAMPLES = 125 * 1920
NUM_FRAMES = 125
EMBEDDING_DIM = 1024


def _patch_beartype_unwrap_init_state():
    """Replace `@beartype`-decorated `init_state` methods with their __wrapped__.

    Required because torch.jit.trace passes a 0-d tensor for `batch_size`,
    which beartype rejects at runtime.
    """
    import pocket_tts.modules.conv as conv_mod

    for cls in (conv_mod.StreamingConv1d, conv_mod.StreamingConvTranspose1d):
        fn = getattr(cls, "init_state", None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(cls, "init_state", fn.__wrapped__)


def _patch_pad_for_conv1d_identity():
    """Replace `pad_for_conv1d` with identity on the mimi module path.

    `pad_for_conv1d` calls `get_extra_padding_for_conv1d` which is
    beartype'd to return `int`; trace converts the return to a 0-d tensor
    and the type check fails. Since our trace input length is already a
    multiple of `frame_size`, padding is a no-op.
    """
    import pocket_tts.models.mimi as mimi_mod

    mimi_mod.pad_for_conv1d = lambda x, k, s: x


def patch_mimi_for_encoder_tracing(mimi):
    """Apply all monkey-patches needed to trace `mimi.encode_to_latent`.

    Reuses the in-place op patches from `traceable_mimi_decoder` and adds
    the beartype unwraps + pad bypass specific to the encoder path.
    Idempotent — safe to call multiple times.
    """
    from traceable_mimi_decoder import _patch_for_tracing

    _patch_for_tracing(mimi)
    _patch_beartype_unwrap_init_state()
    _patch_pad_for_conv1d_identity()


class TraceableMimiEncoder(nn.Module):
    """Stateless audio-to-conditioning wrapper for CoreML tracing.

    Input:  audio        [1, 1, AUDIO_LENGTH_SAMPLES] float32
    Output: conditioning [1, NUM_FRAMES, EMBEDDING_DIM] float32

    The output matches `TTSModel._encode_audio(audio.unsqueeze(0))`
    bit-for-bit (within fp32 rounding) after `patch_mimi_for_encoder_tracing`.
    """

    def __init__(self, mimi_model, speaker_proj_weight: torch.Tensor):
        super().__init__()
        self.mimi = mimi_model
        self.register_buffer("speaker_proj_weight", speaker_proj_weight)
        patch_mimi_for_encoder_tracing(self.mimi)

    @classmethod
    def from_tts_model(cls, tts_model) -> "TraceableMimiEncoder":
        return cls(
            mimi_model=tts_model.mimi,
            speaker_proj_weight=tts_model.flow_lm.speaker_proj_weight,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Inline `mimi.encode_to_latent` minus `pad_for_conv1d` (handled by
        # the caller via fixed input length).
        emb = self.mimi.encoder(audio, model_state=None)
        (emb,) = self.mimi.encoder_transformer(emb, model_state=None)
        emb = self.mimi._to_framerate(emb)  # [1, 32, NUM_FRAMES]
        latents = emb.transpose(-1, -2).to(torch.float32)  # [1, NUM_FRAMES, 32]
        conditioning = F.linear(latents, self.speaker_proj_weight)  # [1, NUM_FRAMES, 1024]
        return conditioning


def test_traceable_mimi_encoder():
    import os
    import sys

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _script_dir)

    from pocket_tts import TTSModel

    print("Loading TTSModel...")
    model = TTSModel.load_model(language="english", lsd_decode_steps=8)
    model.eval()

    print("Creating traceable encoder...")
    wrap = TraceableMimiEncoder.from_tts_model(model)
    wrap.eval()

    audio = torch.zeros(1, 1, AUDIO_LENGTH_SAMPLES)
    with torch.no_grad():
        cond = wrap(audio)
    print(f"Forward output: shape={tuple(cond.shape)} dtype={cond.dtype}")
    assert tuple(cond.shape) == (1, NUM_FRAMES, EMBEDDING_DIM)

    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrap, (audio,), strict=False)
    print("Trace succeeded.")

    with torch.no_grad():
        traced_cond = traced(audio)
    print(f"Traced output: shape={tuple(traced_cond.shape)}")
    diff = (traced_cond - cond).abs().max().item()
    print(f"Max diff (traced vs eager): {diff:.2e}")


if __name__ == "__main__":
    test_traceable_mimi_encoder()
