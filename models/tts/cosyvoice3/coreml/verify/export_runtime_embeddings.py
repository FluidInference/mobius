"""Export embedding tables as they appear AT RUNTIME inside CosyVoice3.model.llm.

This is a parity-critical variant of ``coreml/export-embeddings.py``.

Background
----------
``export-embeddings.py`` dumps weights directly from ``cosyvoice3_dl/llm.pt``.
For ``speech_embedding`` that is fine — it is a custom CosyVoice3 module and
stays in fp32 throughout load. For ``text_embedding`` (Qwen2
``model.embed_tokens.weight``) the HuggingFace transformers load path applies
a dtype round-trip; the runtime values after ``cv.model.llm.float()`` differ
from the raw ``.pt`` values by up to ~5e-4.

Because ``verify/export_swift_fixture.py`` records ``lm_input_embeds`` from
the live runtime (via ``build_frontend_inputs`` → ``embed_tokens(...)``),
Swift can only achieve bit-exact parity if it reads the same runtime tensors.
Use this script to regenerate ``embeddings-runtime-fp32.safetensors``
whenever the Python reference model changes.

Usage:
    uv run python verify/export_runtime_embeddings.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.numpy import save_file


HERE = Path(__file__).parent
ROOT = HERE.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "cosyvoice3_dl"))
    ap.add_argument(
        "--output", default=str(ROOT / "build" / "embeddings" / "embeddings-runtime-fp32.safetensors")
    )
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE / "CosyVoice"))
    sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import CosyVoice3  # noqa: E402

    print(f"[1/2] Loading CosyVoice3 from {args.model_dir}…")
    cv = CosyVoice3(args.model_dir, load_trt=False, load_vllm=False, fp16=False)
    cv.model.llm.float()
    llm = cv.model.llm

    text_w = llm.llm.model.model.embed_tokens.weight.detach().cpu().to(torch.float32).numpy()
    sp_w = llm.speech_embedding.weight.detach().cpu().to(torch.float32).numpy()
    print(f"      runtime text_embedding   : {text_w.shape} {text_w.dtype}")
    print(f"      runtime speech_embedding : {sp_w.shape} {sp_w.dtype}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[2/2] Saving → {out}")
    save_file(
        {"text_embedding": text_w, "speech_embedding": sp_w},
        str(out),
        metadata={
            "text_vocab_size": str(text_w.shape[0]),
            "speech_vocab": str(sp_w.shape[0]),
            "hidden_dim": str(text_w.shape[1]),
            "sos_id": "6561",
            "task_id": "6563",
            "endofprompt_id": "151646",
            "dtype": "fp32-runtime",
            "source": "CosyVoice3.model.llm.float() runtime weights",
        },
    )
    print(f"saved: {out}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
