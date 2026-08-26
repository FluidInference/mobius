"""Convert per-language Mimi encoders to CoreML for voice cloning (issue #793).

Every pocket-tts language pack ships its OWN mimi weights (all 87 keys differ
across languages, including the 43 encoder-side keys) and its own
``flow_lm.speaker_proj_weight``. The original shared ``mimi_encoderv2`` was
traced from the default English model, so live-cloned conditioning for
non-English packs was built from the wrong codec's latents — the root cause of
the residual flaky/garbled non-English cloning that the #797 reprojection
could not fix (it corrected the projection, not the latents).

This script traces one encoder per language with that language's mimi encoder
weights AND its speaker projection baked in, producing conditioning directly
in the target language's space (no host-side reprojection needed).

The I/O contract matches the deployed root ``mimi_encoderv2.mlmodelc`` exactly:

    Input:  audio [1, 1, 240000] float32 (10 s @ 24 kHz, zero-padded, fixed)
    Output: conditioning [1, 125, 1024] float32

Usage:
    python convert_mimi_encoder_lang.py --language spanish_24l
    python convert_mimi_encoder_lang.py --all          # every non-English pack
"""
import argparse
import sys
from pathlib import Path

# Disable beartype before importing pocket_tts (interferes with JIT tracing:
# traced ints become 0-d tensors, tripping the runtime type checks).
# pocket_tts >= 2.x installs checks via beartype.claw import hooks, so the
# claw entry point must be neutralized before pocket_tts is first imported.
import beartype
import beartype.claw

beartype.claw.beartype_this_package = lambda *args, **kwargs: None
beartype.beartype = lambda func=None, **kwargs: func if func else (lambda f: f)

import math

import coremltools as ct
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent.absolute()
COREML_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from traceable_mimi_encoder import TraceableMimiEncoderSimple


def _install_trace_safe_rope() -> None:
    """Replace ``pocket_tts.modules.rope.apply_rope`` with a numerically
    identical version whose shape arithmetic uses Python ints.

    Under ``torch.jit.trace`` the upstream version's ``2 / D`` records an
    ``inverse`` op on an int32 shape tensor, which coremltools rejects
    ("Op (op_type: inverse) ... but got tensor[1,int32]"). This trace runs a
    fixed input shape, so baking the (static) head dim as a constant is exact.
    """
    from pocket_tts.modules import rope as rope_mod

    def apply_rope(q, k, offset=0, max_period=10_000):
        B, T, H, D = (int(v) for v in q.shape)
        Hk = int(k.shape[2])

        ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(max_period) * 2.0 / D))

        ts = torch.arange(T, device=q.device, dtype=torch.float32)
        ts = ts + offset
        ts = ts.view(-1, 1, 1)

        q = q.view(B, T, H, D // 2, 2)
        k = k.view(B, T, Hk, D // 2, 2)

        qr = q[..., 0].float()
        qi = q[..., 1].float()
        kr = k[..., 0].float()
        ki = k[..., 1].float()

        rotr = torch.cos(freqs * ts)
        roti = torch.sin(freqs * ts)
        qor = qr * rotr - qi * roti
        qoi = qr * roti + qi * rotr
        kor = kr * rotr - ki * roti
        koi = kr * roti + ki * rotr

        dtype = q.dtype
        qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
        ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
        return qo.view(B, T, H, D), ko.view(B, T, Hk, D)

    rope_mod.apply_rope = apply_rope


def _install_trace_safe_attention() -> None:
    """Replace ``StreamingMultiheadAttention.forward`` with a stateless
    static-shape equivalent for tracing.

    The upstream forward routes through the KV-cache backend even when
    ``model_state=None``; its shape bookkeeping records int-tensor ops that
    coremltools rejects (``aten::Int`` on non-scalar arrays). The encoder
    trace is always stateless and fixed-shape, so a plain causal
    (context-windowed) attention with Python-int shapes is numerically
    identical — it mirrors ``_LinearKVCacheBackend``'s ``state is None``
    branches exactly (rope offset 0, positions ``arange(T)``).

    Install AFTER computing any eager reference outputs: the patch applies
    class-wide, so reference computations made afterwards would silently use
    the patched forward too.
    """
    import torch.nn.functional as F
    from pocket_tts.modules.transformer import StreamingMultiheadAttention

    def forward(self, query, model_state=None):
        assert model_state is None, "trace-safe attention is stateless"
        T = int(query.shape[1])
        heads = self.num_heads
        d = self.dim_per_head

        projected = self.in_proj(query)
        packed = projected.view(1, T, 3, heads, d)
        q, k, v = torch.unbind(packed, dim=2)
        q, k = self.rope(q, k, offset=0)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        pos = torch.arange(T, device=query.device)
        delta = pos.view(-1, 1) - pos.view(1, -1)
        mask = delta >= 0
        if self.context is not None:
            mask = mask & (delta < self.context)
        x = F.scaled_dot_product_attention(q, k, v, mask.view(1, 1, T, T), dropout_p=0.0)
        x = x.transpose(1, 2).reshape(1, T, heads * d)
        return self.out_proj(x)

    StreamingMultiheadAttention.forward = forward


SAMPLE_RATE = 24_000
NUM_SAMPLES = 240_000  # fixed 10 s window, matches deployed contract
FRAME_SIZE = 1_920

# All packs deployed under v2.1/ on FluidInference/pocket-tts-coreml.
# English keeps the legacy root encoder (identical weights), listed here only
# so --language english can regenerate it for parity checks.
LANGUAGES = [
    "english",
    "german",
    "italian",
    "portuguese",
    "spanish",
    "french_24l",
    "german_24l",
    "italian_24l",
    "portuguese_24l",
    "spanish_24l",
]


def convert_language(language: str, out_dir: Path, verify_wav: Path | None) -> Path:
    from pocket_tts import TTSModel

    print(f"=== {language} ===")
    model = TTSModel.load_model(language=language)
    model.eval()

    # Compute the parity reference through the UNPATCHED reference pipeline
    # first — the trace-safety patches below apply class-wide and would
    # otherwise silently become their own reference.
    ref_cond = None
    padded = None
    if verify_wav is not None:
        from pocket_tts.models.tts_model import audio_read, convert_audio

        audio, sr = audio_read(str(verify_wav))
        audio = convert_audio(audio, sr, model.config.mimi.sample_rate, 1)
        n = min(audio.shape[-1], NUM_SAMPLES)
        padded = torch.zeros(1, 1, NUM_SAMPLES)
        padded[0, 0, :n] = audio.squeeze()[:n]
        with torch.no_grad():
            ref_cond = model._encode_audio(audio[..., :n].unsqueeze(0))

    _install_trace_safe_rope()
    _install_trace_safe_attention()

    traceable = TraceableMimiEncoderSimple.from_tts_model(model)
    traceable.eval()

    example_audio = torch.randn(1, 1, NUM_SAMPLES)
    with torch.no_grad():
        ref_out = traceable(example_audio)
    print(f"  trace output shape: {tuple(ref_out.shape)}")

    traced = torch.jit.trace(traceable, example_audio)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="audio", shape=(1, 1, NUM_SAMPLES), dtype=np.float32)],
        outputs=[ct.TensorType(name="conditioning", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
    )

    # v3 = per-language encoder generation (v2 is the shared English-mimi
    # root encoder). The version bump keeps the two cache-distinct.
    out_path = out_dir / language / "mimi_encoderv3.mlpackage"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))

    # Parity: CoreML vs the reference _encode_audio pipeline
    # (mimi.encode_to_latent -> F.linear(speaker_proj)) on real audio,
    # computed above before the trace-safety patches were installed.
    if ref_cond is not None:
        frames = ref_cond.shape[1]

        loaded = ct.models.MLModel(str(out_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        cm = loaded.predict({"audio": padded.numpy()})["conditioning"][:, :frames]
        diff = np.abs(cm - ref_cond.numpy())
        denom = np.abs(ref_cond.numpy()).mean()
        print(
            f"  parity vs _encode_audio ({frames} frames): "
            f"meanAbsDiff={diff.mean():.6f} maxAbsDiff={diff.max():.6f} refMeanAbs={denom:.6f}"
        )
        if diff.mean() > 0.005:
            raise RuntimeError(f"{language}: parity check failed (meanAbsDiff {diff.mean():.6f})")

    print(f"  saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=LANGUAGES)
    parser.add_argument("--all", action="store_true", help="Convert every non-English pack")
    parser.add_argument(
        "--output-dir", type=Path, default=COREML_DIR / "build_lang_encoders",
        help="Output root; per-language mlpackages land in <dir>/<language>/",
    )
    parser.add_argument(
        "--verify-wav", type=Path, default=None,
        help="Real speech clip for the CoreML-vs-PyTorch parity check",
    )
    args = parser.parse_args()

    if args.all:
        langs = [lang for lang in LANGUAGES if lang != "english"]
    elif args.language:
        langs = [args.language]
    else:
        parser.error("pass --language <lang> or --all")

    for lang in langs:
        convert_language(lang, args.output_dir, args.verify_wav)


if __name__ == "__main__":
    main()
