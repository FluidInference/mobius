#!/usr/bin/env python3
"""
NeMo Nemotron Multilingual Streaming Reference Implementation

Streaming inference with nemotron-asr-streaming-multilingual-0.6b using 1.12s
chunks. Identical to the English reference except for the language prompt:

    - `target_lang` selects an integer `prompt_id` from `prompt_dictionary`.
    - The encoder receives the prompt every chunk (constant per utterance).
    - In "auto" mode (prompt 101) the model emits a leading `<xx-XX>` token;
      we report that as the detected language and strip it from the text.

There is no separate language-ID head — see ARCHITECTURE.md for the proof.
"""
import argparse
import re
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer


DEFAULT_MODEL_ID = "nvidia/nemotron-asr-streaming-multilingual-0.6b"
LANG_TAG_RE = re.compile(r"^<[A-Za-z]{2,4}-[A-Za-z]{2,4}>$")


def calc_drop_extra_pre_encoded(model, step_num: int, pad_and_drop_preencoded: bool) -> int:
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return model.encoder.streaming_cfg.drop_extra_pre_encoded


def _prompt_dictionary(model) -> dict:
    """Canonical location for the prompt dictionary on the loaded model.

    The dict is mirrored under several cfg subtrees but `model_defaults`
    is what `set_inference_prompt` reads (see EncDecRNNTBPEModelWithPrompt).
    """
    md = getattr(model.cfg, "model_defaults", None)
    if md is not None and "prompt_dictionary" in md:
        return dict(md.prompt_dictionary)
    return dict(getattr(model.cfg, "prompt_dictionary", {}) or {})


def resolve_target_lang(model, target_lang: str) -> str:
    """Resolve a user-supplied language code to one valid for the model.

    Returns the canonical key used in `prompt_dictionary` (e.g. "en-US"
    for inputs like "en"). Falls back to "auto" if the input cannot be
    mapped — `set_inference_prompt` raises if it's still not in the dict.
    """
    pd = _prompt_dictionary(model)
    if target_lang in pd:
        return target_lang
    if len(target_lang) == 2:
        for k in pd:
            if k.lower().startswith(target_lang.lower() + "-"):
                return k
    return "auto"


def resolve_prompt_id(model, target_lang: str) -> int:
    """Resolve a language string to its integer prompt id (for logging)."""
    pd = _prompt_dictionary(model)
    key = resolve_target_lang(model, target_lang)
    if key in pd:
        return int(pd[key])
    return int(pd.get("auto", 101))


def split_lang_tag(text: str) -> Tuple[Optional[str], str]:
    """Pull a leading <xx-XX> token off `text` if present."""
    if not text:
        return None, text
    m = re.match(r"^(<[A-Za-z]{2,4}-[A-Za-z]{2,4}>)\s*(.*)", text)
    if m:
        return m.group(1), m.group(2)
    return None, text


def transcribe_streaming(
    model,
    audio: np.ndarray,
    sr: int = 16000,
    target_lang: str = "auto",
    pad_and_drop_preencoded: bool = False,
) -> Tuple[Optional[str], str]:
    """
    Returns (detected_lang_tag, text_without_tag).

    `detected_lang_tag` is "<en-US>", "<zh-CN>" etc. — useful even when
    `target_lang` is forced, because the model still emits it as token #0.
    """
    if sr != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sr}")

    model.encoder.setup_streaming_params()

    # `conformer_stream_step` takes NO prompt kwarg — it reads
    # `self._inference_prompt_index` set by `set_inference_prompt`, then
    # `_apply_prompt_to_encoded` runs `prompt_kernel` on the (B, D, T)
    # encoder output per chunk.
    resolved_lang = resolve_target_lang(model, target_lang)
    model.set_inference_prompt(resolved_lang)

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        pad_and_drop_preencoded=pad_and_drop_preencoded,
    )
    streaming_buffer.reset_buffer()
    streaming_buffer.append_audio(audio)

    cache_last_channel, cache_last_time, cache_last_channel_len = \
        model.encoder.get_initial_cache_state(batch_size=1)

    previous_hypotheses = None
    pred_out_stream = None
    final_text = ""

    with torch.inference_mode():
        for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
            (
                pred_out_stream,
                transcribed_texts,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
                previous_hypotheses,
            ) = model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=streaming_buffer.is_buffer_empty(),
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=pred_out_stream,
                drop_extra_pre_encoded=calc_drop_extra_pre_encoded(
                    model, step_num, pad_and_drop_preencoded
                ),
                return_transcription=True,
            )

            if transcribed_texts and len(transcribed_texts) > 0:
                t = transcribed_texts[0]
                final_text = t.text if hasattr(t, "text") else str(t)

    return split_lang_tag(final_text)


def main():
    parser = argparse.ArgumentParser(description="NeMo Multilingual Reference")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (16 kHz mono).")
    parser.add_argument(
        "--target-lang",
        type=str,
        default="auto",
        help='Language code, e.g. "en-US", "zh-CN", or "auto" (default).',
    )
    parser.add_argument("--duration", type=float, default=None, help="Trim audio to N seconds.")
    parser.add_argument(
        "--nemo-path",
        type=str,
        default=None,
        help="Local .nemo path; if omitted, downloads via from_pretrained().",
    )
    parser.add_argument(
        "--pad-and-drop",
        action="store_true",
        help="Match CoreML/ONNX export behavior (pad_and_drop_preencoded=True).",
    )
    args = parser.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if args.duration:
        audio = audio[: int(args.duration * sr)]

    print("=" * 70)
    print("NEMOTRON MULTILINGUAL STREAMING (NeMo reference)")
    print("=" * 70)
    print(f"Audio:      {len(audio)/sr:.2f}s @ {sr}Hz")
    print(f"target_lang: {args.target_lang}")

    print("\nLoading model...")
    if args.nemo_path:
        model = nemo_asr.models.ASRModel.restore_from(args.nemo_path)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(DEFAULT_MODEL_ID)
    model.eval()

    pd = _prompt_dictionary(model)
    resolved = resolve_target_lang(model, args.target_lang)
    prompt_id = resolve_prompt_id(model, args.target_lang)
    print(f"resolved_lang: {resolved}")
    print(f"prompt_id:     {prompt_id} (auto={pd.get('auto', '?')})")

    print(f"\n[STREAMING] 1.12s chunks, pad_and_drop={args.pad_and_drop}")
    detected, text = transcribe_streaming(
        model,
        audio,
        sr=sr,
        target_lang=args.target_lang,
        pad_and_drop_preencoded=args.pad_and_drop,
    )
    print(f"  detected_lang: {detected}")
    print(f"  text:          {text}")


if __name__ == "__main__":
    main()
