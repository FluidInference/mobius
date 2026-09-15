# Kokoro English–Mandarin demo: 12 existing clips

**Local sampling/demo only. No new audio, inference, scoring, or training.**

Continuing in the training environment? Read the
[complete project handoff](../../project-handoff.md) first. It records settled
product decisions, local-only/missing artifacts, implementation gates, and the
inputs still to recover. This demo folder is evidence, not the full pipeline.

One voice: `zf_001`, existing ANE-zh checkpoint, speed 1, 24-kHz mono.
Total playback: **44.62 seconds**. The local WAVs are byte-for-byte copies
of the selected earlier outputs. All 12 prompts were authored for this project.

Open a WAV below or open [the playlist](playlist.m3u8) in a compatible local
player. Audio is **not committed**: Mobius ignores WAVs. Playback links below and the
playlist work only after the existing 12 WAVs are placed in a local `audio/`
folder, using their original filenames. Playback is manual; nothing autoplays.

## Suggested first listen

Start with **8 → 12 → 11** for the price/API/GitHub issues, then compare
**1 → 5 → 9** for the same window topic across English, Mandarin, and mixed
speech. Use 2 and 10 for the unresolved short-word cases.

This assembly inspected stored records and relevant code, not audio by ear.
“No ASR mismatch” below means text recovery only, not certified pronunciation,
naturalness, tones, or speaker identity.

## English

| Clip | Text | Duration | What to inspect |
| --- | --- | ---: | --- |
| [1. parallel_en_001](audio/parallel_en_001.wav) | Please close the window before you leave. | 3.08 s | Plain-English control; saved ASR matches the words. |
| [2. parallel_en_012](audio/parallel_en_012.wav) | The new version fixes a pronunciation problem. | 3.80 s | Saved ASR omits “a”; the vowel is present in the phoneme input. |
| [3. parallel_en_019](audio/parallel_en_019.wav) | Please check whether the microphone is connected. | 3.48 s | Technical-noun control; saved ASR matches the words. |
| [4. parallel_en_020](audio/parallel_en_020.wav) | Thank you for helping me finish the project. | 3.17 s | Sentence-rhythm control; saved ASR matches the words. |

## Mandarin

| Clip | Text | Duration | What to inspect |
| --- | --- | ---: | --- |
| [5. parallel_zh_001](audio/parallel_zh_001.wav) | 离开之前，请把窗户关上。 | 3.30 s | Mandarin control; same window topic as clips 1 and 9. |
| [6. challenge_zh_003](audio/challenge_zh_003.wav) | 你好，我很好，你最近过得好吗？ | 3.67 s | Tone-sandhi context; zero CER is not proof of correct tones. |
| [7. challenge_zh_008](audio/challenge_zh_008.wav) | 银行的行长走过人行道，来到旅行社。 | 4.42 s | Polyphone context; saved ASR recovers 行长/人行道/旅行社. |
| [8. challenge_zh_018](audio/challenge_zh_018.wav) | 这本书的价格是二十三元五角。 | 3.75 s | Price challenge; inspect 二 versus erhua merging (issue F1). |

## Mixed English–Mandarin

| Clip | Text | Duration | What to inspect |
| --- | --- | ---: | --- |
| [9. parallel_mixed_001](audio/parallel_mixed_001.wav) | 离开之前，please close the window。 | 3.20 s | Mixed control; auto-language ASR omits Mandarin (issue E1). |
| [10. parallel_mixed_002](audio/parallel_mixed_002.wav) | The meeting 明天早上开始。 | 3.05 s | Short English prefix becomes “Dmeeting” in ASR (issue U2). |
| [11. challenge_mixed_001](audio/challenge_mixed_001.wav) | 我正在用 GitHub train a small speech model，然后部署到 Mac 上。 | 5.28 s | GitHub input contains θ; technical-word frontend issue F3. |
| [12. challenge_mixed_002](audio/challenge_mixed_002.wav) | 这个 API 已经更新了，请 check the documentation。 | 4.42 s | API input is ˈæpi, not spelled letter names (issue F2). |

## Findings and evidence

Read [ISSUES.md](ISSUES.md) for the prioritized findings, evidence, and proposed
small experiments. No fixes or experiments have been executed by this demo
assembly. The clearest next work is frontend correctness, not fine-tuning.

[manifest.json](manifest.json) contains the selected texts, exact phonemes and
token IDs, WAV hashes, saved diagnostics, and per-clip evidence paths.
The `evidence/` folder preserves the original render and scorer JSONs.
The `provenance/` folder preserves redacted source run metadata, scorer
configurations, and a rendering-log excerpt; these describe the earlier exploratory run,
not a new demo run.

All selected WAVs have zero clipping fraction and no vocabulary-dropped
symbols in their saved records. That does not rule out pronunciation defects.
Predicted MOS is kept in the manifest/evidence as an uncalibrated diagnostic;
it is not used as a quality grade in this listening guide.

## Next action, not started

Address the `二`/erhua eligibility bug with focused text-only regression tests,
preserving real erhua behavior. Then consider exact-token API/GitHub
pronunciation overrides. A later listening check should touch only the
affected clips and retain the original WAVs for A/B comparison.

No full benchmark sweep, model download, remote GPU job, or training run is
part of this deliverable. Formal evaluation and training require a separately agreed environment,
data/split audit, protocol, and run budget. This local prototype is not a
ready-to-run CUDA training/evaluation stack. Private project plans remain
outside Git; this directory records exploratory findings only.

## Verify without running models

This subdirectory is self-contained and uses the Python standard library only.
No model conversion, download, ASR, synthesis, or training command is included.

```bash
uv sync --frozen
uv run python verify-demo.py
uv run python -m unittest discover -s tests -v
# Optional: check the already-generated WAVs, without decoding or scoring them.
uv run python verify-demo.py --audio-dir /absolute/path/to/existing/demo/audio
```

The default check validates all 12 bundled text/evidence records and their
scorer-config hashes. It explicitly reports that audio was not checked.
The optional check verifies the exact WAV SHA-256 values from the source run.
Neither mode certifies model quality. There are no third-party runtime
dependencies, and no model binaries are distributed.

The source run used a locally modified FluidAudio checkout. The exact
frontend source hashes are recorded under `signature.tts_source_files` in
[run metadata](provenance/run.json); the public source links are navigation
references, not a claim that the public branch exactly reproduces these WAVs.

Sources: [FluidAudio](https://github.com/FluidInference/FluidAudio),
[Kokoro v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh),
[MLX Whisper checkpoint](https://huggingface.co/mlx-community/whisper-large-v3-turbo),
[UTMOSv2](https://github.com/sarulab-speech/UTMOSv2).
All selected prompt text was authored for this project; no MiniMax prompt text,
private recordings, pretrained weights, or training data is redistributed.
