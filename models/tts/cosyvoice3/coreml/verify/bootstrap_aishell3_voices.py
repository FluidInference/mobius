"""Bootstrap a handful of CosyVoice3 zero-shot voice bundles from AISHELL-3.

Pulls a curated mix of speakers from the ``shenyunhang/AISHELL-3`` HF mirror
(individual-file layout — no 19 GB tgz download), picks one clean utterance
per speaker, runs it through ``build_frontend_inputs`` to produce the three
tensors CosyVoice3 needs, and writes a ``<voice-id>.safetensors`` +
``<voice-id>.json`` bundle per speaker into ``build/voices/`` (same format
that :mod:`extract_voice_prompt` emits).

Usage:
    uv run python verify/bootstrap_aishell3_voices.py --num-voices 10

Voice IDs follow ``aishell3-zh-{spk}-{gender}`` so the AISHELL provenance is
explicit in the filename.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import torch
from huggingface_hub import hf_hub_download
from safetensors.numpy import save_file


HERE = Path(__file__).parent
ROOT = HERE.parent
AISHELL_REPO = "shenyunhang/AISHELL-3"

# Hand-picked so we get gender + accent variety while keeping quality tier B+.
# (spk_id, gender, accent, target_utterance_basename_hint)
# target hint picks a specific utterance index known to be clean/medium length;
# if the exact file isn't in the HF mirror we fall back to the first suitable
# utterance for that speaker in content.txt.
CURATED_SPEAKERS: list[tuple[str, str, str]] = [
    ("SSB0005", "female", "north"),
    ("SSB0009", "female", "south"),
    ("SSB0011", "female", "north"),
    ("SSB0012", "female", "south"),
    ("SSB0016", "male",   "north"),
    ("SSB0033", "male",   "south"),
    ("SSB0057", "female", "north"),
    ("SSB0080", "male",   "north"),
    ("SSB0112", "female", "south"),
    ("SSB0122", "male",   "south"),
]


def _prime_sys_path() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE / "CosyVoice"))
    sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))


class Utterance(NamedTuple):
    wav_name: str
    split: str  # "train" or "test"
    transcript: str
    char_count: int


def _parse_content_line(line: str) -> tuple[str, str, int]:
    """Return (wav_name, plain-chinese-transcript, n_chars)."""
    wav_name, body = line.rstrip("\n").split("\t", 1)
    tokens = body.split()
    # Alternating char / pinyin — chars are at even indices.
    chars = [tokens[i] for i in range(0, len(tokens), 2)]
    return wav_name, "".join(chars), len(chars)


def _load_content(split: str) -> dict[str, list[Utterance]]:
    """speaker_id -> list[Utterance] for that speaker in this split."""
    path = Path(hf_hub_download(AISHELL_REPO, f"{split}/content.txt",
                                repo_type="dataset"))
    per_speaker: dict[str, list[Utterance]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        wav_name, text, n_chars = _parse_content_line(line)
        spk = wav_name[:7]  # "SSBXXXX"
        per_speaker.setdefault(spk, []).append(
            Utterance(wav_name=wav_name, split=split, transcript=text,
                      char_count=n_chars))
    return per_speaker


def _pick_utterance(candidates: list[Utterance]) -> Utterance | None:
    """Pick a medium-length utterance (12-22 chars) — roughly 4-8 s @ 24 kHz."""
    sweet_spot = [u for u in candidates if 12 <= u.char_count <= 22]
    pool = sweet_spot or candidates
    if not pool:
        return None
    pool.sort(key=lambda u: abs(u.char_count - 17))
    return pool[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-voices", type=int, default=len(CURATED_SPEAKERS),
                    help="Upper bound on voices to emit (default all curated).")
    ap.add_argument("--output-dir", default=str(ROOT / "build" / "voices"))
    ap.add_argument("--model-dir",  default=str(ROOT / "cosyvoice3_dl"))
    ap.add_argument("--prompt-prefix",
                    default="You are a helpful assistant.<|endofprompt|>")
    args = ap.parse_args()

    _prime_sys_path()
    from src.text_frontend import build_frontend_inputs  # noqa: E402

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading AISHELL-3 content.txt (both splits)…")
    content_test  = _load_content("test")
    content_train = _load_content("train")

    picked: list[tuple[str, str, str, Utterance]] = []
    for spk, gender, _accent in CURATED_SPEAKERS[: args.num_voices]:
        # Prefer test set — smaller, less likely in anyone's training data.
        utterances = content_test.get(spk, []) + content_train.get(spk, [])
        utt = _pick_utterance(utterances)
        if utt is None:
            print(f"      skip {spk}: no utterances in either split.")
            continue
        picked.append((spk, gender, _accent, utt))

    if not picked:
        raise SystemExit("No usable speakers found — check AISHELL-3 HF mirror.")

    print(f"[2/3] Picked {len(picked)} speakers:")
    for spk, gender, accent, utt in picked:
        print(f"      {spk}  {gender:6s} {accent:6s}  "
              f"{utt.split:5s} chars={utt.char_count:2d} :: {utt.transcript}")

    summary_rows: list[dict] = []
    for i, (spk, gender, accent, utt) in enumerate(picked, start=1):
        voice_id = f"aishell3-zh-{spk}-{gender}"
        print(f"\n[3/3][{i}/{len(picked)}] extracting {voice_id}…")

        wav_path = Path(hf_hub_download(
            AISHELL_REPO,
            f"{utt.split}/wav/{spk}/{utt.wav_name}",
            repo_type="dataset",
        ))
        prompt_text = args.prompt_prefix + utt.transcript
        placeholder_tts = "你好。"

        fr = build_frontend_inputs(
            tts_text=placeholder_tts,
            prompt_text=prompt_text,
            prompt_wav=str(wav_path),
            model_dir=Path(args.model_dir),
        )
        n_speech = int(fr.llm_prompt_speech_ids.shape[1])
        mel_frames = int(fr.prompt_mel.shape[1])
        print(f"      N_speech={n_speech}  mel_frames={mel_frames}")

        tensors = {
            "llm_prompt_speech_ids":
                fr.llm_prompt_speech_ids.detach().to(torch.int32).cpu().numpy(),
            "prompt_mel":
                fr.prompt_mel.detach().to(torch.float32).cpu().numpy(),
            "spk_embedding":
                fr.spk_embedding.detach().to(torch.float32).cpu().numpy(),
        }
        st_path   = output_dir / f"{voice_id}.safetensors"
        json_path = output_dir / f"{voice_id}.json"
        save_file(
            tensors,
            str(st_path),
            metadata={
                "voice_id":   voice_id,
                "ref_wav":    utt.wav_name,
                "spk_id":     spk,
                "gender":     gender,
                "accent":     accent,
                "split":      utt.split,
                "n_speech":   str(n_speech),
                "mel_frames": str(mel_frames),
                "source": "AISHELL-3 (shenyunhang/AISHELL-3 HF mirror)",
            },
        )
        json_path.write_text(
            json.dumps({"prompt_text": prompt_text}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"      wrote {st_path.name}  ({st_path.stat().st_size/1024:.1f} KB)")

        summary_rows.append({
            "voice_id":      voice_id,
            "spk_id":        spk,
            "gender":        gender,
            "accent":        accent,
            "split":         utt.split,
            "ref_wav":       utt.wav_name,
            "transcript":    utt.transcript,
            "n_speech":      n_speech,
            "mel_frames":    mel_frames,
            "size_bytes":    st_path.stat().st_size,
        })

    (output_dir / "aishell3-bootstrap.json").write_text(
        json.dumps({"voices": summary_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone. {len(summary_rows)} voices written to {output_dir}")


if __name__ == "__main__":
    main()
