"""Per-stage `nn.Module` wrappers for CoreML conversion.

Each wrapper takes a *reference* to the corresponding submodule from the
`model` dict that `run_inference.load_styletts2` produced — no copies, no
re-instantiations, no separate weights. The wrapper exists only to (a)
freeze the call signature to a tensor-only contract suitable for
`torch.jit.trace`, and (b) absorb any `.eval()`-time tweaks needed for
clean conversion (dropout-to-identity, weight_norm strip, etc.).

Conventions:

* All wrappers are `.eval()` and return a tuple of tensors (even if one
  output) — coremltools is happy with that.
* No in-place ops in wrapper code paths.
* `forward` accepts only torch.Tensors. Booleans and ints become tensor
  inputs where needed (CoreML doesn't trace Python scalars cleanly).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def _strip_dropout(module: nn.Module) -> None:
    """Replace any nn.Dropout / nn.Dropout1d/2d/3d with nn.Identity in-place.

    `.eval()` already disables dropout's RNG path, but tracing still
    captures the rate as an attribute and CoreML's converter occasionally
    chokes on it. Replacing with Identity makes the trace cleaner.
    """
    for name, child in list(module.named_children()):
        if isinstance(
            child,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
        ):
            setattr(module, name, nn.Identity())
        else:
            _strip_dropout(child)


def precompute_har_source(decoder: nn.Module, f0_curve: torch.Tensor) -> torch.Tensor:
    """Run HiFi-GAN's SineGen + SourceModuleHnNSF on Python side and
    return the harmonic source signal `har_source` shaped [B, 1, T_up].

    This is the input to `decoder.generator.noise_convs[*]` — the only
    use of the source signal inside the decoder. Computing it on CPU
    bypasses the problematic CoreML lowering of SineGen's
    `interpolate`/`cumsum`/`sin` chain (max|d|≈0.97, corr≈0 even with
    deterministic noise).

    Args:
        decoder: model.decoder (the StyleTTS2 HiFi-GAN decoder).
        f0_curve: [1, T_frames] f0 contour (same as decoder input #2).
    Returns:
        har_source: [1, 1, T_up] where T_up = T_frames * upsample_scale.
    """
    gen = decoder.generator
    src = gen.m_source
    sine_gen = src.l_sin_gen
    up = int(np.prod([10, 5, 3, 2]))  # = sine_gen.upsample_scale = 300

    # Replicate Generator.forward up through the first transpose:
    #   f0 = f0_upsamp(f0[:, None]).transpose(1, 2)
    #   har, _, _ = m_source(f0)
    #   har = har.transpose(1, 2)
    with torch.no_grad():
        f0_up = gen.f0_upsamp(f0_curve[:, None]).transpose(1, 2)  # [B, T_up, 1]
        # Inline SineGen.forward, deterministic (no torch.rand).
        fn = torch.multiply(
            f0_up,
            torch.FloatTensor([[list(range(1, sine_gen.harmonic_num + 2))]]).to(
                f0_up.device
            ),
        )
        # Inline _f02sine deterministically (rand_ini -> zeros). This
        # path lives in eager torch only — the conversion graph never
        # sees it.
        rad_values = (fn / sine_gen.sampling_rate) % 1
        rad_lo = torch.nn.functional.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=1 / up,
            mode="linear",
        ).transpose(1, 2)
        phase_lo = torch.cumsum(rad_lo, dim=1) * 2 * np.pi
        phase = torch.nn.functional.interpolate(
            (phase_lo * up).transpose(1, 2),
            scale_factor=up,
            mode="linear",
        ).transpose(1, 2)
        sines = torch.sin(phase) * sine_gen.sine_amp

        uv = (fn > sine_gen.voiced_threshold).type(torch.float32)
        sine_waves = sines * uv  # noise -> zeros, so additive term drops

        # SourceModuleHnNSF: l_linear + l_tanh -> sine_merge.
        sine_merge = src.l_tanh(src.l_linear(sine_waves))   # [B, T_up, 1]
        har_source = sine_merge.transpose(1, 2)             # [B, 1, T_up]
    return har_source


def _patch_generator_use_har(gen: nn.Module) -> None:
    """Replace `Generator.forward` so it accepts `har_source` directly
    instead of recomputing it from `f0`.

    The original forward does:
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source, noi_source, uv = self.m_source(f0)
        har_source = har_source.transpose(1, 2)
        ... rest of the generator (uses har_source, x, s, F0_curve) ...

    We replace the first three lines with `har_source = har_source` and
    skip `m_source` entirely. The rest of the generator (snake activation,
    noise_convs, ups, resblocks, conv_post, tanh) is unchanged.
    """
    if getattr(gen, "_har_patched", False):
        return

    def _forward(self, x, s, har_source, _f0_unused):
        # `har_source` is [B, 1, T_up], pre-computed on CPU.
        for i in range(self.num_upsamples):
            x = x + (1 / self.alphas[i]) * (torch.sin(self.alphas[i] * x) ** 2)
            x_source = self.noise_convs[i](har_source)
            x_source = self.noise_res[i](x_source, s)

            x = self.ups[i](x)
            x = x + x_source

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        x = x + (1 / self.alphas[i + 1]) * (torch.sin(self.alphas[i + 1] * x) ** 2)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    gen.forward = _forward.__get__(gen, type(gen))
    gen._har_patched = True


def _patch_attention_einsum(module: nn.Module) -> int:
    """Replace `AttentionBase.forward` einsum calls with explicit matmul.

    coremltools' einsum solver lowers the broadcasted-batch attention
    equations
        "... n d, ... m d -> ... n m"
        "... n m, ... m d -> ... n d"
    into a generic path that emits a `transpose(perm=...)` whose perm
    length doesn't match the input rank, and the converter dies with
        ValueError: perm should have the same length as rank(x): 5 != 4
    Direct matmul (`q @ k.transpose(-1,-2)`, `attn @ v`) is exactly
    equivalent for these equations and lowers cleanly.

    Patches every `AttentionBase` *instance* under `module`. Returns the
    count of patched instances. Idempotent (uses a sentinel attribute).
    """
    from einops import rearrange  # local import; only needed if patch fires

    count = 0
    for sub in module.modules():
        if type(sub).__name__ != "AttentionBase":
            continue
        if getattr(sub, "_einsum_patched", False):
            continue

        def _patched_forward(self, q, k, v):
            from einops import rearrange as _r
            # Split heads: [b, n, h*d] -> [b, h, n, d]
            q = _r(q, "b n (h d) -> b h n d", h=self.num_heads)
            k = _r(k, "b n (h d) -> b h n d", h=self.num_heads)
            v = _r(v, "b n (h d) -> b h n d", h=self.num_heads)
            # sim = q @ k^T  ->  [b, h, n, m]
            sim = torch.matmul(q, k.transpose(-1, -2))
            if self.use_rel_pos:
                sim = sim + self.rel_pos(*sim.shape[-2:])
            sim = sim * self.scale
            attn = sim.softmax(dim=-1)
            # out = attn @ v  ->  [b, h, n, d]
            out = torch.matmul(attn, v)
            out = _r(out, "b h n d -> b n (h d)")
            return self.to_out(out)

        sub.forward = _patched_forward.__get__(sub, type(sub))
        sub._einsum_patched = True
        count += 1
    return count


def _remove_weight_norm_recursive(module: nn.Module) -> int:
    """Strip torch.nn.utils.weight_norm parametrizations on every conv/linear.

    Returns count of layers stripped. Idempotent.
    """
    count = 0
    for sub in module.modules():
        # Old-style API (still in use by upstream StyleTTS2).
        try:
            torch.nn.utils.remove_weight_norm(sub)
            count += 1
        except (ValueError, AttributeError, RuntimeError):
            pass
    return count


# ---------------------------------------------------------------------------
# Stage 1 — text_encoder
# ---------------------------------------------------------------------------


class TextEncoderWrapper(nn.Module):
    """Stage 1: tokens -> t_en.

    Inputs:
        tokens         [1, T_text]  int64 (LongTensor)
        input_lengths  [1]          int64
        text_mask      [1, T_text]  bool
    Output:
        t_en           [1, 512, T_text]

    Trace-friendly reimplementation of `TextEncoder.forward`:
      * `pack_padded_sequence` / `pad_packed_sequence` removed (B=1, no
        padding to skip — the mask handles all suppression). Tracing the
        original packed path corrupts the LSTM `initial_h[0]` shape and
        produces a .mlpackage that fails to compile with
        `Dimension 0 of tensor parameter initial_h[0] has unexpected length`.
      * In-place `masked_fill_` replaced with out-of-place `masked_fill`.
      * The trailing zero-pad copy (`x_pad[:, :, :T] = x`) collapses to a
        plain pass-through since LSTM output already has the full T axis.
    """

    def __init__(self, text_encoder: nn.Module) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        _strip_dropout(self.text_encoder)
        # Detach the LSTM weights from any parametrizations / packed-seq
        # state (defensive — flatten_parameters is harmless on traced graphs).
        try:
            self.text_encoder.lstm.flatten_parameters()
        except Exception:  # noqa: BLE001
            pass
        self.eval()

    def forward(
        self,
        tokens: torch.Tensor,
        input_lengths: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        te = self.text_encoder
        x = te.embedding(tokens)            # [B, T, emb]
        x = x.transpose(1, 2)               # [B, emb, T]
        # text_mask flows in as bool from PyTorch / fp32 from CoreML I/O.
        # Express the mask multiplicatively so dtype is irrelevant: the
        # graph carries fp32 throughout.
        m = text_mask.to(x.dtype).unsqueeze(1)        # [B, 1, T]; 1.0 = masked
        keep = 1.0 - m
        x = x * keep
        for c in te.cnn:
            x = c(x)
            x = x * keep
        x = x.transpose(1, 2)               # [B, T, chn]
        # Skip pack/unpack — trace cleanly with the full sequence.
        x, _ = te.lstm(x)
        x = x.transpose(-1, -2)             # [B, chn, T]
        x = x * keep
        return x


# ---------------------------------------------------------------------------
# Stage 2 — bert + bert_encoder (combined)
# ---------------------------------------------------------------------------


class BertWrapper(nn.Module):
    """Stage 2: tokens -> (bert_dur, d_en).

    Inputs:
        tokens         [1, T_text]  int64
        attention_mask [1, T_text]  int32 (= (~text_mask).int())
    Outputs:
        bert_dur       [1, T_text, 768]
        d_en           [1, 512, T_text]   (= bert_encoder(bert_dur).transpose(-1,-2))
    """

    def __init__(self, bert: nn.Module, bert_encoder: nn.Module) -> None:
        super().__init__()
        self.bert = bert
        self.bert_encoder = bert_encoder
        _strip_dropout(self.bert)
        # Force eager attention for HF Albert; SDPA's mask helper
        # (`create_bidirectional_mask` -> `sdpa_mask`) does
        # `q_length.shape[0]` on a Python int during trace and raises
        # IndexError. Eager attention keeps everything tensor-typed.
        try:
            self.bert.config._attn_implementation = "eager"
            self.bert.config._attn_implementation_internal = "eager"
        except Exception:  # noqa: BLE001
            pass
        # Walk children so any cached attention layers are also marked.
        for sub in self.bert.modules():
            if hasattr(sub, "config"):
                try:
                    sub.config._attn_implementation = "eager"
                except Exception:  # noqa: BLE001
                    pass
        self.eval()

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Bypass `CustomAlbert.forward(*args, **kwargs)` (which TorchScript
        # can't trace through *args/**kwargs and HF's BaseModelOutput
        # dataclass) and call the underlying AlbertModel directly with
        # `return_dict=False`, taking element [0] = last_hidden_state.
        outputs = type(self.bert).__mro__[1].forward(
            self.bert,
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        bert_dur = outputs[0]
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        return bert_dur, d_en


# ---------------------------------------------------------------------------
# Stage 3 — ref_encoder (style + predictor encoders, output concatenated)
# ---------------------------------------------------------------------------


class RefEncoderWrapper(nn.Module):
    """Stage 3: mel -> ref_s.

    Mirrors `run_inference.compute_style` exactly. The Python helper does:

        mel  = preprocess(audio)                           # [1, 80, T_mel]
        x    = mel.to(device).unsqueeze(1)                 # [1, 1, 80, T_mel]
        ref_s = style_encoder(x)        # [1, 128]
        ref_p = predictor_encoder(x)    # [1, 128]
        return torch.cat([ref_s, ref_p], dim=1)            # [1, 256]

    We take `mel_4d` as the wrapper input so the unsqueeze is the
    caller's responsibility (CoreML doesn't need to re-do it).

    Input:
        mel_4d  [1, 1, 80, T_mel]
    Output:
        ref_s   [1, 256]
    """

    def __init__(self, style_encoder: nn.Module, predictor_encoder: nn.Module) -> None:
        super().__init__()
        self.style_encoder = style_encoder
        self.predictor_encoder = predictor_encoder
        _strip_dropout(self.style_encoder)
        _strip_dropout(self.predictor_encoder)
        self.eval()

    def forward(self, mel_4d: torch.Tensor) -> torch.Tensor:
        ref_s = self.style_encoder(mel_4d)
        ref_p = self.predictor_encoder(mel_4d)
        return torch.cat([ref_s, ref_p], dim=1)


# ---------------------------------------------------------------------------
# Stage 4 — diffusion UNet (one denoise step)
# ---------------------------------------------------------------------------


class DiffusionDenoiseStepWrapper(nn.Module):
    """Stage 4: one KDiffusion denoise step.

    Wraps `model.diffusion.diffusion.denoise_fn` for the
    `embedding_scale=1.0` / `embedding_mask_proba=0.0` path that
    `run_inference.py` uses (no classifier-free guidance, no random
    embedding masking). The ADPM2 schedule lives in Python and
    dispatches this graph 2× per step (once on `x`, once on `x_mid`),
    for a total of ~8 CoreML calls per 5-step sample.

    Inputs:
        x_noisy   [1, 1, 256]       current noisy style state
        sigma     [1]               scalar noise level for this step
        embedding [1, T_text, 768]  bert_dur conditioning
        features  [1, 256]          ref_s conditioning
    Output:
        x_denoised [1, 1, 256]

    We call `unet.run(...)` directly to skip `Transformer1d.forward`'s
    float-branching on `embedding_scale` / `embedding_mask_proba`
    (both Python-int comparisons that fold at trace time, but the
    wasted `fixed_embedding(...)` allocation in the else branch leaks
    constants into the trace).
    """

    def __init__(self, kdiffusion: nn.Module) -> None:
        super().__init__()
        self.kdiffusion = kdiffusion
        self.unet = kdiffusion.net
        _strip_dropout(self.kdiffusion)
        # Replace einsum-based attention with matmul (see helper docstring).
        self._attn_patched = _patch_attention_einsum(self.kdiffusion)
        self.eval()

    def forward(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        embedding: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        c_skip, c_out, c_in, c_noise = self.kdiffusion.get_scale_weights(sigma)
        x_pred = self.unet.run(
            c_in * x_noisy, c_noise, embedding=embedding, features=features
        )
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised


# ---------------------------------------------------------------------------
# Stage 5 — duration_predictor
# ---------------------------------------------------------------------------


def _duration_encoder_traceable(
    de: nn.Module, x: torch.Tensor, style: torch.Tensor, m: torch.Tensor
) -> torch.Tensor:
    """Trace-friendly inline of `DurationEncoder.forward` (B=1).

    The upstream forward does pack_padded_sequence / pad_packed_sequence
    around an internal LSTM, calls `text_lengths.cpu().numpy()` to drive
    the packing, in-place `masked_fill_` on the mask, and a separately
    allocated `x_pad = torch.zeros(...)` zero-pad copy. All of these are
    trace hostile (pack/unpack corrupts LSTM `initial_h` shape inference;
    `.cpu().numpy()` produces 0-d-tensor constants coremltools can't fold).

    For B=1, the mask alone fully suppresses the padded positions and
    the zero-pad copy is a no-op (T already matches `m.shape[-1]`). The
    output is identical to the eager path on the unpadded prefix.

    Layout: keep `[B, C, T]` throughout to avoid the [T, B, C]
    transpose dance.
    """
    _, _, T = x.shape
    keep = (1.0 - m.to(x.dtype)).unsqueeze(1)               # [B, 1, T]
    s_exp = style.unsqueeze(-1).expand(-1, style.shape[-1], T)  # [B, sty, T]
    x = torch.cat([x, s_exp], dim=1) * keep                 # [B, d+sty, T]

    for block in de.lstms:
        # Avoid `isinstance(..., AdaLayerNorm)` import (vendor module has
        # bad imports). Match by class name instead.
        if type(block).__name__ == "AdaLayerNorm":
            x = block(x.transpose(-1, -2), style).transpose(-1, -2)
            x = torch.cat([x, s_exp], dim=1) * keep
        else:  # nn.LSTM (bidirectional)
            x_t = x.transpose(-1, -2)                       # [B, T, C]
            try:
                block.flatten_parameters()
            except Exception:  # noqa: BLE001
                pass
            x_t, _ = block(x_t)                             # [B, T, hidden_full]
            x = x_t.transpose(-1, -2)                       # [B, hidden_full, T]

    return x.transpose(-1, -2)                              # [B, T, hidden_full]


class DurationPredictorWrapper(nn.Module):
    """Stage 5: (d_en, s, text_mask) -> (d, duration_logits).

    Inputs:
        d_en          [1, 512, T_text]
        s             [1, 128]
        text_mask     [1, T_text]  fp32 (1.0 = pad, 0.0 = keep)
    Outputs:
        d                [1, T_text, hidden]
        duration_logits  [1, T_text, max_dur]

    Downstream sigmoid + sum + round + clamp + alignment matrix run in
    Python (data-dependent shapes, can't go through CoreML).
    """

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.text_encoder = predictor.text_encoder
        self.lstm = predictor.lstm
        self.duration_proj = predictor.duration_proj
        _strip_dropout(self.text_encoder)
        _strip_dropout(self.lstm)
        _strip_dropout(self.duration_proj)
        self.eval()

    def forward(
        self,
        d_en: torch.Tensor,
        s: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d = _duration_encoder_traceable(self.text_encoder, d_en, s, text_mask)
        try:
            self.lstm.flatten_parameters()
        except Exception:  # noqa: BLE001
            pass
        x, _ = self.lstm(d)
        duration = self.duration_proj(x)
        return d, duration


# ---------------------------------------------------------------------------
# Stage 6 — f0n_predictor (a.k.a. F0Ntrain)
# ---------------------------------------------------------------------------


class F0NPredictorWrapper(nn.Module):
    """Stage 6: (en, s) -> (f0_pred, n_pred).

    Wraps `predictor.F0Ntrain`. Input `en` is already aligned (built by
    Python from `d.transpose(-1,-2) @ pred_aln_trg.unsqueeze(0)` with
    the hifigan asr-shift quirk applied).

    Inputs:
        en  [1, hidden, T_frames]
        s   [1, 128]
    Outputs:
        f0_pred  [1, T_frames]   (or [1, T_frames * upscale])
        n_pred   [1, T_frames]
    """

    def __init__(self, predictor: nn.Module) -> None:
        super().__init__()
        self.predictor = predictor
        _strip_dropout(self.predictor)
        self.eval()

    def forward(self, en: torch.Tensor, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predictor.F0Ntrain(en, s)


# ---------------------------------------------------------------------------
# Stage 7 — HiFi-GAN decoder
# ---------------------------------------------------------------------------


class DecoderWrapper(nn.Module):
    """Stage 7: (asr, f0_pred, n_pred, ref, har_source) -> waveform.

    `har_source` is the precomputed harmonic source signal — output of
    HiFi-GAN's SineGen + SourceModuleHnNSF, shaped [1, 1, T_up].
    Computing it on CPU side-steps a coremltools mis-conversion of the
    `interpolate`/`cumsum`/`sin` chain inside SineGen (corr collapsed to
    ~0 for the in-graph version even after replacing all RNG with zeros).

    The rest of the decoder (encode + decode AdaIN blocks + Generator's
    snake activations, ups, resblocks, conv_post, tanh) converts cleanly
    to fp32 mlprogram — verified at corr=0.999992 with SineGen stubbed.

    `weight_norm` parametrizations on every internal conv are stripped
    before the trace (idempotent).

    Inputs:
        asr         [1, hidden, T_frames]
        f0_pred     [1, T_frames]
        n_pred      [1, T_frames]
        ref         [1, 128]
        har_source  [1, 1, T_up]   (T_up = T_frames * 300)
    Output:
        wav         [1, 1, T_audio]   (24 kHz, T_audio = T_up)

    Use `precompute_har_source(model.decoder, f0_pred)` to build the
    extra input from a Python f0 contour.
    """

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        _strip_dropout(self.decoder)
        n = _remove_weight_norm_recursive(self.decoder)
        self._weight_norm_stripped = n
        # Replace Generator.forward with a version that accepts a
        # pre-computed `har_source` and skips f0_upsamp + m_source.
        _patch_generator_use_har(self.decoder.generator)
        self.eval()

    def forward(
        self,
        asr: torch.Tensor,
        f0_pred: torch.Tensor,
        n_pred: torch.Tensor,
        ref: torch.Tensor,
        har_source: torch.Tensor,
    ) -> torch.Tensor:
        # Inline `Decoder.forward` (the eval-mode branch) so we can pass
        # `har_source` through to the patched Generator.
        dec = self.decoder
        F0 = dec.F0_conv(f0_pred.unsqueeze(1))
        N = dec.N_conv(n_pred.unsqueeze(1))
        x = torch.cat([asr, F0, N], dim=1)
        ref_in = ref.squeeze(0).unsqueeze(0)
        x = dec.encode(x, ref_in)
        asr_res = dec.asr_res(asr)
        res = True
        for block in dec.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], dim=1)
            x = block(x, ref_in)
            if block.upsample_type != "none":
                res = False
        # Patched Generator.forward signature: (x, s, har_source, f0).
        return dec.generator(x, ref_in, har_source, f0_pred)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


STAGE_NAMES: Tuple[str, ...] = (
    "text_encoder",
    "bert",
    "ref_encoder",
    "diffusion_unet",
    "duration_predictor",
    "f0n_predictor",
    "decoder",
)


def build_wrapper(stage: str, model) -> nn.Module:
    """Construct the wrapper for the named stage from a loaded `model` dict."""
    if stage == "text_encoder":
        return TextEncoderWrapper(model.text_encoder)
    if stage == "bert":
        return BertWrapper(model.bert, model.bert_encoder)
    if stage == "ref_encoder":
        return RefEncoderWrapper(model.style_encoder, model.predictor_encoder)
    if stage == "diffusion_unet":
        return DiffusionDenoiseStepWrapper(model.diffusion.diffusion)
    if stage == "duration_predictor":
        return DurationPredictorWrapper(model.predictor)
    if stage == "f0n_predictor":
        return F0NPredictorWrapper(model.predictor)
    if stage == "decoder":
        return DecoderWrapper(model.decoder)
    raise ValueError(f"unknown stage: {stage!r} (valid: {STAGE_NAMES})")
