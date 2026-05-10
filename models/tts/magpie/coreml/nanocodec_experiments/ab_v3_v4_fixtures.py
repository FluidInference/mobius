"""mobius #60 Track 1 — generate matched-prompt v3 vs v4 fixture pairs.

For each prompt × {v3, v4}, drives the full Magpie TTS pipeline with the
same speaker, language, seed, and CFG settings — only the NanoCodec
backend swaps. v3 = production fp32 default; v4 = palette-quantized fp32
(Trial 10a, currently flagged DEAD-END for shipping but kept as a 4×
smaller artifact). The fixture pairs let the user A/B-listen and confirm
PERF.md's "acoustically transparent vs v3" claim with their ears.

Output: WAV pairs at
    nanocodec_experiments/results/ab_v3_v4/utt{NN}_{v3,v4}.wav

The Python `generate(...)` function from `generate_coreml.py` is reused
verbatim. Only the `NANOCODEC_PATH` env var changes between calls.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Allow running both as module (-m experiments…) and as direct script.
_HERE = Path(__file__).resolve().parent
_COREML_DIR = _HERE.parent
if str(_COREML_DIR) not in sys.path:
    sys.path.insert(0, str(_COREML_DIR))


# Per #60 acceptance criteria: 5+ pairs at mixed lengths, single speaker,
# deterministic seed. Speaker 0 = John (en) per `manifest.json`.
PROMPTS: List[Tuple[str, str, str]] = [
    # (utt_id, language, text)
    ("utt01", "en", "Hello there."),
    ("utt02", "en", "The quick brown fox jumps over the lazy dog."),
    ("utt03", "en",
     "In the quiet town of Millbrook, the postman delivered a single "
     "letter every Tuesday morning, and the children would wait at "
     "the corner to see whose family it was for."),
    ("utt04", "en",
     "She watched the snowfall blanket the rooftops, listening to the "
     "wind shift through the bare branches outside her window. Far below, "
     "a single car moved slowly along the empty street, its headlights "
     "cutting twin beams through the drifting flakes."),
    ("utt05", "en",
     "The expedition departed on March fourteenth, nineteen twenty-three, "
     "with twenty-eight men, four sled teams, and provisions for eighteen "
     "months. None of them had any idea that the unusually warm winds "
     "from the south would soon turn their carefully laid plans upside "
     "down within the first six weeks of travel."),
    # NOTE: longer prompts (>256 phoneme tokens) trip
    # `text_tokens_padded[:T_text]` in `generate_coreml.py` which caps at
    # `max_text_len=256`. The original utt06 (80-word "history of human
    # flight" passage) tokenized to 271 phonemes; dropped from the
    # fixture set. Magpie's production sentence-chunker handles long
    # inputs, but `generate_coreml.py` is single-pass-by-design.
]

# Local-build paths (the v3v4 worktree symlinks `build/` to the main
# checkout's `build/`, so both files exist).
_BUILD_DIR = _COREML_DIR / "build"
NANOCODEC_PATHS = {
    "v3": str(_BUILD_DIR / "nanocodec_decoder_v3.mlpackage"),
    "v4": str(_BUILD_DIR / "v4" / "nanocodec_decoder_v4.mlpackage"),
}

OUT_DIR = _HERE / "results" / "ab_v3_v4"


def _generate_one(text: str, language: str, output_path: Path,
                  speaker: int = 0, seed: int = 42,
                  use_cfg: bool = True, cfg_scale: float = 2.5) -> float:
    # Import inside the call so each invocation re-loads the CoreML
    # models with the *current* NANOCODEC_PATH env var — generate_coreml
    # reads it at MLModel-load time inside `generate(...)`.
    import generate_coreml
    t0 = time.time()
    generate_coreml.generate(
        text=text,
        speaker=speaker,
        language=language,
        output_path=str(output_path),
        seed=seed,
        use_cfg=use_cfg,
        cfg_scale=cfg_scale,
    )
    return time.time() - t0


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[fixtures] writing to {OUT_DIR}")
    for variant, path in NANOCODEC_PATHS.items():
        if not Path(path).exists():
            print(f"  [error] {variant} mlpackage missing: {path}",
                  file=sys.stderr)
            return 1

    summary = []
    for utt_id, lang, text in PROMPTS:
        for variant, nc_path in NANOCODEC_PATHS.items():
            os.environ["NANOCODEC_PATH"] = nc_path
            out = OUT_DIR / f"{utt_id}_{variant}.wav"
            print(f"[fixtures] {utt_id} / {variant} → {out.name}",
                  flush=True)
            elapsed = _generate_one(text, lang, out)
            print(f"  elapsed: {elapsed:.2f}s")
            summary.append((utt_id, variant, str(out), elapsed))

    print()
    print("[fixtures] summary:")
    for utt_id, variant, path, elapsed in summary:
        size_kb = Path(path).stat().st_size // 1024 if Path(path).exists() else -1
        print(f"  {utt_id} / {variant:>2}  {elapsed:6.2f}s  {size_kb:>6} KB  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
