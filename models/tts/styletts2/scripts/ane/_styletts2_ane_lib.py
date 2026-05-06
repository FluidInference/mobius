"""Shared traceable wrappers for the StyleTTS2-ANE 7-graph re-cut.

This mirrors the laishere Kokoro-ANE conversion shape (Albert + PostAlbert +
Alignment + Prosody + Noise + Vocoder + Tail) but for StyleTTS2 — where the
diffusion sampler replaces Kokoro's single-shot prosody graph and the
HiFi-GAN decoder is iSTFT-free, so Kokoro's separate Tail collapses into the
Vocoder.

Result is 7 .mlpackage / .mlmodelc bundles (vs the existing 4-stage / 12-bucket
pipeline under `scripts/01_..04_` + `scripts/optimize/`):

  1. PLBert         — Albert (PLBERT) only.            [ANE, fp16+int8pal]
  2. PostBert       — TextEncoder + DurationEncoder    [ANE, fp16+int8pal]
                       + duration LSTM + duration_proj
                       + fixed_embedding for CFG.
  3. Alignment      — cumsum + broadcast (Kokoro-style) [ANE, fp16+int8pal]
                       en, asr from (pred_dur, d, t_en).
  4. DiffusionStep  — single ADPM2 denoise step.        [ANE, fp16+int8pal]
                       Sampler loop (5 steps × 2 calls
                       + 1 = 11) lives in Swift.
  5. Prosody        — F0Ntrain (en, s) → (F0, N).       [ANE, fp16+int8pal]
                       Single fixed shape — kills the
                       E5RT FlexibleShapeInfo bug that
                       pinned the legacy graph to CPU.
  6. Noise          — SineGen alone.                    [ALL,  fp32+int8pal]
                       Phase precision needs fp32; this
                       is the only fp32 graph (matches
                       Kokoro's Noise stage exactly).
  7. Vocoder        — HiFi-GAN body, no SineGen,        [ANE, fp16+int8pal]
                       cos-Snake patch from Kokoro.
                       (StyleTTS2 HiFi-GAN is iSTFT-free
                       so no separate Tail is needed.)

This module deliberately re-uses the heavy-lift from `_styletts2_lib.py`:
  - SineGen `_f02sine` constant-fold patch.
  - aten::multiply op shim.
  - AttentionBase einsum → matmul rewrite.
  - LibriTTSConfig + load_inference_modules.

We add three new bits not present in the legacy lib:
  - `_cos_resblock1_forward` (cos-Snake) — lifted verbatim from the laishere
    Kokoro convert-coreml.py:40-52. Replaces sin² with `(1 - cos(2αx))/2` for
    ANE friendliness.
  - `BiLstmUnrolled` helper — used by PostBert's DurationEncoder LSTM unroll
    (mirrors `CoreMLDurationEncoder` in the Kokoro script).
  - The 7 traceable wrappers themselves.

Usage:
    from _styletts2_ane_lib import (
        load_modules_for_ane,
        PLBertTraceable,
        PostBertTraceable,
        AlignmentTraceable,
        DiffusionStepTraceable,
        ProsodyTraceable,
        NoiseTraceable,
        VocoderTraceable,
    )

All wrappers assume B=1 and full-length inputs (no pack_padded_sequence). No
masking is used — the Swift host pads tokens to the token bucket and pads
acoustic frames to T_a_max.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use everything from the legacy lib so the two paths stay in lock-step
# whenever the upstream model conversion semantics change.
THIS_DIR = Path(__file__).resolve().parent
LEGACY_DIR = THIS_DIR.parent
if str(LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_DIR))

from _styletts2_lib import (  # noqa: E402  (sys.path mutated above)
    DEFAULT_CHECKPOINT,
    LibriTTSConfig,
    UPSAMPLE_SCALE,
    install_sinegen_v2_constfold_fix,
    load_inference_modules,
    register_coreml_op_shims,
)

# --- Bucketing constants (mirror Kokoro-ANE) ----------------------------------

MAX_T_TOK = 512   # PLBERT max position embeddings
MAX_T_A = 2000    # max acoustic frames (mel time after duration expansion)


# --- Cos-Snake patch (lifted from laishere/kokoro convert-coreml.py:40-52) ----
#
# The HiFi-GAN AdaINResBlock1 in the StyleTTS2 vendor is the same as Kokoro's
# (StyleTTS2 is its parent), so the same cos-identity rewrite applies:
#
#   sin²(αx) = (1 - cos(2αx)) / 2
#
# Snake's `x + sin²(αx)/α` → `x + (1 - cos(2αx))/(2α)` — purely composed of
# add/mul/cos which are ANE-native, vs `pow(sin(...), 2)` which forces a CPU
# fallback under ANECompiler.

def install_cos_snake_patch() -> None:
    """Patch `Modules.hifigan.AdaINResBlock1.forward` with the cos identity.

    Idempotent: stash the original on the class once.
    """
    from Modules.hifigan import AdaINResBlock1  # type: ignore

    if hasattr(AdaINResBlock1, "_forward_original"):
        return

    AdaINResBlock1._forward_original = AdaINResBlock1.forward  # type: ignore[attr-defined]

    def _cos_resblock1_forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(
            self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2
        ):
            xt = n1(x, s)
            cv = torch.cos(xt * (a1 * 2))
            xt = xt + (cv * (-0.5) + 0.5) * (1.0 / a1)
            xt = c1(xt)
            xt = n2(xt, s)
            cv = torch.cos(xt * (a2 * 2))
            xt = xt + (cv * (-0.5) + 0.5) * (1.0 / a2)
            xt = c2(xt)
            x = xt + x
        return x

    AdaINResBlock1.forward = _cos_resblock1_forward


# --- Module loader -----------------------------------------------------------

def load_modules_for_ane(checkpoint: Path = DEFAULT_CHECKPOINT, cfg: LibriTTSConfig | None = None):
    """Build inference modules and apply ANE-friendly patches.

    Always installs:
      - aten::multiply op shim (needed by SineGen).
      - cos-Snake patch on AdaINResBlock1.

    Does NOT install the SineGen const-fold yet — that's per-bucket and is
    applied by `06_export_noise.py` once it knows T_a.

    Returns (modules, cfg).
    """
    register_coreml_op_shims()
    modules, cfg = load_inference_modules(checkpoint=checkpoint, cfg=cfg)
    install_cos_snake_patch()
    return modules, cfg


# --- Stage 1: PLBert ---------------------------------------------------------

class PLBertTraceable(nn.Module):
    """Stage 1 — PLBERT only.

    Input:  tokens [1, T_tok] int64
    Output: bert_dur [1, T_tok, bert_dim]  (bert_dim = 768)

    `CustomAlbert.forward` (Utils/PLBERT/util.py) overrides AlbertModel.forward
    to return `last_hidden_state` directly as a Tensor.
    """

    def __init__(self, modules: dict):
        super().__init__()
        self.bert = modules["bert"]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.bert(tokens)


# --- Stage 2: PostBert (text + duration encoders + duration head) -----------

class _CoreMLFixedEmbedding(nn.Module):
    """CoreML-friendly replacement for `Modules.diffusion.modules.FixedEmbedding`.

    Upstream uses `torch.arange(length, ...)` where `length = x.shape[1]` is a
    symbolic dim under RangeDim — coremltools lowers the resulting embedding
    lookup + einops `repeat` into a `tile` whose reps factor depends on a
    symbolic shape, and that factor collapses to 0 (E5RT: "All values of reps
    must be at least 1"; output becomes `(1, 0, 0)`).

    This wrapper derives the 0..T-1 index range from `cumsum(ones)` over the
    input's T axis instead of `arange(T)`, which keeps the shape connected to
    the input tensor symbolically. The embedding lookup becomes a static
    `gather` op on a constant table, which RangeDim handles correctly.
    """

    def __init__(self, fixed_embedding: nn.Module):
        super().__init__()
        self.embedding = fixed_embedding.embedding  # nn.Embedding(max_length, features)
        self.max_length = fixed_embedding.max_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, T, C)
        # Generate indices 0..T-1 via cumsum of ones over T-axis. This avoids
        # `arange(symbolic)` while keeping T tied to the input shape.
        ones = torch.ones_like(x[:, :, 0])                      # (1, T)
        indices = torch.cumsum(ones, dim=-1).to(torch.long) - 1  # (1, T) values 0..T-1
        # Gather over the constant embedding table:
        fe = self.embedding(indices.squeeze(0))                  # (T, features)
        return fe.unsqueeze(0)                                   # (1, T, features)


class _TextEncoderUnrolled(nn.Module):
    """Drop-in `models.TextEncoder` without pack_padded_sequence and no masks."""

    def __init__(self, text_encoder: nn.Module):
        super().__init__()
        self.embedding = text_encoder.embedding
        self.cnn = text_encoder.cnn
        self.lstm = text_encoder.lstm

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)             # (1, T, ch)
        x = x.transpose(1, 2)                  # (1, ch, T)
        for c in self.cnn:
            x = c(x)
        x = x.transpose(1, 2)                  # (1, T, ch)
        x, _ = self.lstm(x)
        x = x.transpose(-1, -2)                # (1, ch, T)
        return x


class _DurationEncoderUnrolled(nn.Module):
    """`models.DurationEncoder` with the LSTM/AdaLayerNorm loop unrolled
    explicitly.  Same shape contract as the legacy `DurationEncoderTraceable`,
    but written so the BiLSTM is the only LSTM handed to the converter (no
    surrounding pack/pad calls).  This mirrors `CoreMLDurationEncoder` in
    the laishere Kokoro convert-coreml.py.
    """

    def __init__(self, duration_encoder: nn.Module):
        super().__init__()
        # Split lstms ModuleList into LSTMs and AdaLayerNorms preserving order
        # via parallel ModuleLists keyed by index.
        self.lstms = nn.ModuleList()
        self.norms = nn.ModuleList()
        for block in duration_encoder.lstms:
            if isinstance(block, nn.LSTM):
                self.lstms.append(block)
            else:
                self.norms.append(block)
        self.dropout = duration_encoder.dropout
        self.d_model = duration_encoder.d_model
        self.sty_dim = duration_encoder.sty_dim

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        # x:     (1, hidden_dim, T)
        # style: (1, style_dim)
        # Output: (1, T, hidden_dim + style_dim)
        #
        # CRITICAL: Avoid `torch.expand(seq_len, -1, -1)` here. coremltools
        # lowers it to `tile(x, reps=concat([T, var, var]) / [1,1,sty_dim])`
        # where the `-1` placeholders become free symbolic constants `var`
        # that the runtime evaluates to 0, producing a (T, 0, 0) tensor and
        # cat'ing it as a no-op (the empirical bug: `d` shape collapses from
        # 640 → 512 channels, dropping the style cat entirely).
        #
        # Instead build `s` via additive broadcast: `style + zeros_T` where
        # zeros_T is anchored to the input shape. This emits an `add` (with
        # broadcast) instead of a `tile`, which has no symbolic-reps trap.
        # x_chan: (1, hidden_dim, T) → derive (1, 1, T) anchor for broadcast.
        zeros_anchor = torch.zeros_like(x[:, :1, :])           # (1, 1, T)
        s_chan = style.unsqueeze(-1) + zeros_anchor            # (1, sty_dim, T) via broadcast
        x = torch.cat([x, s_chan], dim=1)                      # (1, h+s, T)

        for i in range(len(self.lstms)):
            x = x.transpose(-1, -2)                            # (1, T, ch)
            x, _ = self.lstms[i](x)
            x = F.dropout(x, p=self.dropout, training=False)
            x = x.transpose(-1, -2)                            # (1, ch, T)
            if i < len(self.norms):
                x = self.norms[i](x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s_chan], dim=1)              # (1, h+s, T)

        return x.transpose(-1, -2)                             # (1, T, h+s)


class PostBertTraceable(nn.Module):
    """Stage 2 — TextEncoder + DurationEncoder + duration LSTM + duration_proj
    + fixed_embedding (for CFG uncond branch).

    Inputs:
      bert_dur [1, T_tok, bert_dim]
      tokens   [1, T_tok] int64
      style    [1, style_dim*2]  — concatenated [acoustic | prosody] ref_s
                                   (yl4579 convention: style_encoder | predictor_encoder)

    Outputs:
      t_en             [1, hidden_dim, T_tok]
      d                [1, T_tok, hidden_dim + style_dim]
      pred_dur_log     [1, T_tok, max_dur]    (pre-sigmoid)
      fixed_embedding  [1, T_tok, bert_dim]   (for CFG uncond)

    Note: ProsodyPredictor uses style_dim*2 (acoustic + prosody concat). The
    DurationEncoder consumes `style[:, style_dim:]` (prosody half = second
    half under yl4579 convention) — same as the legacy text predictor.
    """

    def __init__(self, modules: dict, cfg: LibriTTSConfig):
        super().__init__()
        self.bert_encoder = modules["bert_encoder"]
        self.text_encoder = _TextEncoderUnrolled(modules["text_encoder"])

        predictor = modules["predictor"]
        self.duration_encoder = _DurationEncoderUnrolled(predictor.text_encoder)
        self.lstm = predictor.lstm
        self.duration_proj = predictor.duration_proj

        self.fixed_embedding = _CoreMLFixedEmbedding(modules["diffusion"].unet.fixed_embedding)
        self.style_dim = cfg.style_dim

    def forward(
        self,
        bert_dur: torch.Tensor,
        tokens: torch.Tensor,
        style: torch.Tensor,
    ):
        # yl4579 convention: ref_s = [acoustic | prosody]. Prosody half is
        # [:, style_dim:]; the first half is the acoustic style consumed only
        # by F0Ntrain and the decoder.
        s_pros = style[:, self.style_dim:]

        t_en = self.text_encoder(tokens)                      # (1, h, T)
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)  # (1, h, T)
        d = self.duration_encoder(d_en, s_pros)               # (1, T, h+s)
        x, _ = self.lstm(d)                                   # (1, T, h)
        pred_dur_log = self.duration_proj(x)                  # (1, T, max_dur)
        fixed_embedding = self.fixed_embedding(bert_dur)      # (1, T, bert_dim)
        return t_en, d, pred_dur_log, fixed_embedding


# --- Stage 3: Alignment (cumsum + broadcast → en, asr) ----------------------

class AlignmentTraceable(nn.Module):
    """Stage 3 — derive `en` and `asr` from (pred_dur, d, t_en).

    Replaces the Swift-side `repeat_interleave` expansion with a CoreML-friendly
    cumsum + broadcast, mirroring `CoreMLAlignmentStandalone` in Kokoro's
    convert-coreml.py:341-361. Keeps everything as static-rank ops so the
    converter doesn't need EnumeratedShapes.

    Inputs:
      pred_dur  [1, T_tok]            (already sigmoid+sum+round, integer-valued floats)
      d         [1, T_tok, h+s]
      t_en      [1, hidden_dim, T_tok]

    Outputs:
      en        [1, h+s, max_T_a]     (zero-padded after sum(pred_dur))
      asr       [1, h, max_T_a]       (zero-padded similarly)

    The host slices to the actual T_a (= sum(pred_dur)) before passing to
    Prosody / Noise / Vocoder.
    """

    def __init__(self, max_T_a: int = MAX_T_A):
        super().__init__()
        self.max_T_a = max_T_a

    def forward(
        self,
        pred_dur: torch.Tensor,
        d: torch.Tensor,
        t_en: torch.Tensor,
    ):
        # Promote to fp32 for the cumsum/comparison so we don't accumulate
        # rounding error across long phoneme sequences.
        pdur = pred_dur.float()
        cum = torch.cumsum(pdur, dim=-1)                          # (1, T_tok)
        starts = cum - pdur                                       # (1, T_tok)

        frames = torch.arange(self.max_T_a, device=d.device, dtype=torch.float32)
        frames = frames.unsqueeze(0).unsqueeze(0)                 # (1, 1, max_T_a)

        in_seg = (frames >= starts.unsqueeze(-1)) & (frames < cum.unsqueeze(-1))
        alignment = in_seg.float()                                # (1, T_tok, max_T_a)

        en = d.transpose(-1, -2).float() @ alignment              # (1, h+s, max_T_a)
        asr = t_en.float() @ alignment                            # (1, h,  max_T_a)
        return en, asr


# --- Stage 4: DiffusionStep --------------------------------------------------

class DiffusionStepTraceable(nn.Module):
    """Stage 4 — single ADPM2 / Karras denoise step (StyleTransformer1d).

    Identical math to the legacy `DiffusionDenoiseTraceable`, but the export
    script will give it fully static shapes (no EnumeratedShapes, no RangeDim
    on `embedding` / `attention_mask`) so the diffusion UNet attention block
    can stay ANE-resident.

    Inputs:
      x_noisy    [1, 1, style_dim*2]
      sigma      [1]
      embedding  [1, T_emb_max, bert_dim]   (fixed at max_position_embeddings)
      features   [1, style_dim*2]

    Output:
      denoised   [1, 1, style_dim*2]

    The 5-step ADPM2 sampler (11 invocations per utterance) lives in Swift —
    the existing `StyleTTS2Sampler` is model-agnostic and will be reused.
    """

    def __init__(self, modules: dict, sigma_data: float):
        super().__init__()
        self.transformer = modules["diffusion"].unet
        self.sigma_data = sigma_data

    def forward(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        embedding: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        s = sigma.view(-1, 1, 1)
        sd = self.sigma_data
        c_skip = (sd * sd) / (s * s + sd * sd)
        c_out = s * sd / torch.sqrt(s * s + sd * sd)
        c_in = 1.0 / torch.sqrt(s * s + sd * sd)
        c_noise = torch.log(sigma) * 0.25

        x_pred = self.transformer.run(
            c_in * x_noisy,
            c_noise,
            embedding=embedding,
            features=features,
        )
        return c_skip * x_noisy + c_out * x_pred


# --- Stage 5: Prosody (F0Ntrain) --------------------------------------------

class ProsodyTraceable(nn.Module):
    """Stage 5 — predictor.F0Ntrain.

    Input:
      en  [1, hidden_dim + style_dim, T_a]    (= d.transpose @ alignment)
      s   [1, style_dim]                      (prosody half of ref_s)

    Output:
      F0  [1, T_a*2]    (predictor.F0[0] has upsample=True → 2× time)
      N   [1, T_a*2]    (predictor.N[0]  has upsample=True → 2× time)

    Identical compute to the legacy `F0NEnergyTraceable` — the only change is
    the export script uses a single fixed `[1, 640, MAX_T_A]` shape instead of
    EnumeratedShapes so the E5RT FlexibleShapeInfo bug
    ("tensor_buffer has known strides while the model has FlexibleShapeInfo")
    no longer pins this stage to CPU.
    """

    def __init__(self, modules: dict):
        super().__init__()
        predictor = modules["predictor"]
        self.shared = predictor.shared
        self.F0 = predictor.F0
        self.N = predictor.N
        self.F0_proj = predictor.F0_proj
        self.N_proj = predictor.N_proj

    def forward(self, en: torch.Tensor, s: torch.Tensor):
        x, _ = self.shared(en.transpose(-1, -2))              # (1, T_a, h)

        F0 = x.transpose(-1, -2)                              # (1, h, T_a)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0).squeeze(1)                      # (1, T_a)

        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N).squeeze(1)                         # (1, T_a)

        return F0, N


# --- Stage 6: Noise (SineGen alone, fp32) -----------------------------------

class NoiseTraceable(nn.Module):
    """Stage 6 — SineGen alone, in fp32 (phase precision required).

    Wraps the upstream `Modules.hifigan.SineGen` with `_f02sine` rewritten to
    avoid `aten::remainder` and `F.interpolate` op-translation bugs (see the
    detailed comment block in `_styletts2_lib.install_sinegen_v2_constfold_fix`).

    Inputs:
      F0_curve  [1, T_a*2]                            (from Prosody — F0Ntrain upsamples by 2)
      noise     [1, T_a*2*UPSAMPLE_SCALE, harm+1]     (broadband Gaussian noise, host-generated)

    Output:
      sine_waves  [1, T_audio_chunk, harmonic_num + 1]   (voiced_harmonics + noise_amp * noise)
      uv          [1, T_audio_chunk, 1]

    `T_audio_chunk = T_a * 2 * UPSAMPLE_SCALE` where the leading `2` is the
    predictor's hop ratio (already baked into the input length) and
    `UPSAMPLE_SCALE` is the HiFi-GAN `f0_upsamp` factor (=300 for LibriTTS).
    The constant-fold lerp pre-bakes `fracs` for this T_audio_chunk; the
    export script must call `install_sinegen_v2_constfold_fix(t_mel=T_a)`
    before tracing this stage.

    Why the noise is a runtime input (not baked-in `randn`):
      * `torch.randn_like(...)` would be sampled once at trace time and frozen
        as a constant in the .mlpackage — every utterance would get the same
        noise pattern.
      * The previous workaround `noise = noise_amp * sin(fn * 100.0)` aliased
        above Nyquist and went to zero in unvoiced regions, producing the
        "raspy / weak fricatives" perceptual artifact.
      * Passing noise as a runtime tensor preserves both determinism (the
        host can seed) and broadband stochasticity (real white noise, not a
        phase-locked sine).

    Compute units = `.all` (CPU+GPU+ANE) but in practice runs CPU/GPU because
    cumsum on a long sequence in fp16 saturates phase. This matches Kokoro's
    fp32 Noise stage.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        # The decoder.generator owns the SineGen via m_source; for StyleTTS2
        # the module path is decoder.generator.m_source.l_sin_gen. We also
        # need f0_upsamp from the parent generator to mirror upstream
        # Generator.forward's `f0 = self.f0_upsamp(f0[:, None]).transpose(1,2)`.
        self.f0_upsamp = decoder.generator.f0_upsamp
        self.l_sin_gen = decoder.generator.m_source.l_sin_gen

    def forward(self, F0_curve: torch.Tensor, noise: torch.Tensor):
        # Mirror upstream `Generator.forward`:
        #   f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        #   har_source, _, uv = self.m_source(f0)
        # The patched SineGen.forward (`_forward_deterministic`) returns:
        #   - sine_waves_voiced: clean harmonics already gated by uv
        #   - uv:                voicing mask
        #   - noise_amp:         per-sample noise amplitude (handles voiced
        #                        breathiness vs unvoiced fricative levels)
        # We mix the host-supplied broadband `noise` here so the export is
        # deterministic but the synthesised excitation is real noise.
        f0 = self.f0_upsamp(F0_curve[:, None]).transpose(1, 2)  # (1, T_audio, 1)
        sine_waves_voiced, uv, noise_amp = self.l_sin_gen(f0)
        sine_waves = sine_waves_voiced + noise_amp * noise      # (1, T_audio, harm+1)
        return sine_waves, uv


# --- Stage 7: Vocoder (HiFi-GAN body, no SineGen) ---------------------------

class VocoderTraceable(nn.Module):
    """Stage 7 — HiFi-GAN body minus SineGen.

    Same compute graph as `_styletts2_lib.HifiGanDecoderTraceable` *except*
    the SineGen path is now consumed via the pre-computed sine_waves from
    Stage 6 (Noise). This lets the body stay fp16 + ANE while Noise stays
    fp32 — same split as Kokoro-ANE.

    Inputs:
      asr        [1, hidden_dim, T_a]              (alignment output, host-sliced)
      F0_curve   [1, T_a*2]    (raw F0 from Prosody — F0_conv stride=2 brings to T_a)
      N          [1, T_a*2]    (raw N  from Prosody — N_conv  stride=2 brings to T_a)
      s          [1, style_dim]                    (acoustic half of ref_s — ref_s[:, style_dim:])
      sine_waves [1, T_a*2*UPSAMPLE_SCALE, harm+1] (from Noise)

    Output:
      audio      [1, T_audio]    where T_audio = T_a * 2 * UPSAMPLE_SCALE

    Notes:
      - The cos-Snake patch is applied to AdaINResBlock1 by
        `install_cos_snake_patch` so this graph is composed of mul/add/cos/conv
        only.
      - We feed `sine_waves` straight into the rest of `m_source` (linear+tanh)
        to keep the noise injection identical to upstream eval mode.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.encode = decoder.encode
        self.decode = decoder.decode
        self.F0_conv = decoder.F0_conv
        self.N_conv = decoder.N_conv
        self.asr_res = decoder.asr_res
        self.generator = decoder.generator

    def forward(
        self,
        asr: torch.Tensor,
        F0_curve: torch.Tensor,
        N: torch.Tensor,
        s: torch.Tensor,
        sine_waves: torch.Tensor,
    ) -> torch.Tensor:
        # Replicate the eval path of `Modules.hifigan.Decoder.forward` up to
        # `self.generator(x, s, F0_curve)` — but inside the generator we splice
        # in the externally-computed sine_waves before they hit the harmonic
        # linear+tanh.

        F0 = self.F0_conv(F0_curve.unsqueeze(1))              # (1, 1, T_mel/2)
        Nc = self.N_conv(N.unsqueeze(1))                       # (1, 1, T_mel/2)

        x = torch.cat([asr, F0, Nc], dim=1)                    # (1, h+2, T_mel/2)
        x = self.encode(x, s)                                  # (1, 1024, T_mel/2)

        asr_res = self.asr_res(asr)

        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, F0, Nc], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False

        # Now we run the upstream `Generator.forward` path but with the
        # SineGen output replaced by our `sine_waves` argument. The upstream
        # Generator reads m_source(F0_upsampled) — we precompute that.
        return self._run_generator_with_external_sine(x, s, sine_waves)

    def _run_generator_with_external_sine(self, x, s, sine_waves):
        gen = self.generator
        # Mirror upstream `Modules.hifigan.Generator.forward` exactly. The only
        # difference is that SineGen has been factored out into Stage 6
        # (NoiseTraceable); the rest of `m_source` (linear + tanh) lives here
        # because it's tiny and ANE-friendly.
        #
        # Upstream m_source.forward:
        #     sine_merge = l_tanh(l_linear(sine_wavs))
        # Upstream Generator.forward:
        #     har_source, _, _ = m_source(f0_upsampled)
        #     har_source = har_source.transpose(1, 2)
        sine_merge = gen.m_source.l_tanh(gen.m_source.l_linear(sine_waves))
        har_source = sine_merge.transpose(1, 2)                # (1, harm+1, T_audio_chunk)

        # Snake activation `x + (1/a) * sin(a*x)**2` — kept in direct sin² form
        # rather than the cos double-angle identity. The cos form
        #     (1 - cos(2*a*x)) / 2
        # suffers catastrophic cancellation in fp16: when `a*x` is small,
        # `cos(2*a*x) ≈ 1` rounds to 1.0, the numerator collapses to 0, then
        # division by small α amplifies any residual error by orders of
        # magnitude. Empirically that path produced +23 dB RMS bias in the
        # exported mlpackage despite matching PyTorch wrappers at fp32.
        # `sin(a*x) * sin(a*x)` is bounded, monotone-stable in fp16, and
        # ANE-friendly (mul/add/sin only). The cos-Snake patch on
        # AdaINResBlock1 stays as-is for the resblock body.
        def _snake_sin(z, a):
            sv = torch.sin(z * a)
            return z + (sv * sv) / a

        # Walk the upsamples following upstream ordering exactly:
        #   snake → noise_conv(har_source) → noise_res → ups → x + x_source → resblocks
        for i in range(gen.num_upsamples):
            x = _snake_sin(x, gen.alphas[i])

            x_source = gen.noise_convs[i](har_source)
            x_source = gen.noise_res[i](x_source, s)

            x = gen.ups[i](x)
            x = x + x_source

            xs = None
            for j in range(gen.num_kernels):
                rb = gen.resblocks[i * gen.num_kernels + j]
                xs = rb(x, s) if xs is None else xs + rb(x, s)
            x = xs / gen.num_kernels

        # Final post-loop snake uses alphas[num_upsamples] (alphas has
        # num_upsamples+1 entries; the leading one is for the conv_pre input,
        # but upstream's loop uses indices 0..num_upsamples-1 inside the loop
        # and `alphas[i+1]` after the loop — same as `alphas[num_upsamples]`).
        x = _snake_sin(x, gen.alphas[gen.num_upsamples])
        x = gen.conv_post(x)
        x = torch.tanh(x)
        return x.squeeze(1)                                    # (1, T_audio)


# --- Helpers shared with export scripts -------------------------------------

def palettize_int8(mlpackage_path: Path) -> Path:
    """In-place int8 kmeans palettization (matches Kokoro-ANE preset).

    Returns the path of the palettized .mlpackage.
    """
    import coremltools as ct
    import coremltools.optimize.coreml as cto

    mlmodel = ct.models.MLModel(str(mlpackage_path))
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(mode="kmeans", nbits=8)
    )
    optimized = cto.palettize_weights(mlmodel, config)
    optimized.save(str(mlpackage_path))
    return mlpackage_path


__all__ = [
    "MAX_T_TOK",
    "MAX_T_A",
    "UPSAMPLE_SCALE",
    "LibriTTSConfig",
    "DEFAULT_CHECKPOINT",
    "load_modules_for_ane",
    "install_cos_snake_patch",
    "install_sinegen_v2_constfold_fix",
    "register_coreml_op_shims",
    "PLBertTraceable",
    "PostBertTraceable",
    "AlignmentTraceable",
    "DiffusionStepTraceable",
    "ProsodyTraceable",
    "NoiseTraceable",
    "VocoderTraceable",
    "palettize_int8",
]
