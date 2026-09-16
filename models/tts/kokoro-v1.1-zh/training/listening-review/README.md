# Listen: baseline vs trained Kokoro

These are **24 generated clips across all 12 development prompts** from the
selected update-500 comparison. They are labeled examples for your inspection,
not a completed blinded listening study. Both variants use the same text,
frontend, seed (1729), and speed (1.0), with their respective matching voice tables.

Click the WAV links below to preview or download each clip. For side-by-side
browser players, **download [listen.html](listen.html) and open it in a browser**;
it loads the clips from this PR branch. If you clone this directory, it can
also play the local WAVs offline. No player dependencies are required.

Listen at the same playback volume. Note incorrect words/tones, dropped
syllables, awkward pauses, language-switch pronunciation, voice changes, and
which version you prefer. A useful reply is: `prompt ID; baseline/trained/tie;
exact word or timestamp; what sounds wrong`.

| Prompt | Expected text | Baseline | Trained |
|---|---|---|---|
| `mixed-api` | 请打开 API，然后把结果发给我。 | [Baseline](audio/mixed-api-baseline.wav) | [Trained](audio/mixed-api-trained.wav) |
| `mixed-github` | 我把新的 PyTorch 模型上传到 GitHub 了。 | [Baseline](audio/mixed-github-baseline.wav) | [Trained](audio/mixed-github-trained.wav) |
| `mixed-meeting` | 今天下午三点有一个 meeting，请不要迟到。 | [Baseline](audio/mixed-meeting-baseline.wav) | [Trained](audio/mixed-meeting-trained.wav) |
| `mixed-coffee` | Could you bring 两杯咖啡 to the office? | [Baseline](audio/mixed-coffee-baseline.wav) | [Trained](audio/mixed-coffee-trained.wav) |
| `mixed-release` | The next release 支持中文和 English。 | [Baseline](audio/mixed-release-baseline.wav) | [Trained](audio/mixed-release-trained.wav) |
| `mixed-price` | 这个 package 的价格是二十三元。 | [Baseline](audio/mixed-price-baseline.wav) | [Trained](audio/mixed-price-trained.wav) |
| `zh-numerals` | 二加二等于四，十二加十一等于二十三。 | [Baseline](audio/zh-numerals-baseline.wav) | [Trained](audio/zh-numerals-trained.wav) |
| `zh-independent-er` | 儿童在公园里玩，一共有二十二个人。 | [Baseline](audio/zh-independent-er-baseline.wav) | [Trained](audio/zh-independent-er-trained.wav) |
| `zh-erhua` | 小孩儿在这儿等了一会儿。 | [Baseline](audio/zh-erhua-baseline.wav) | [Trained](audio/zh-erhua-trained.wav) |
| `en-api` | The API is ready. You can find the code on GitHub. | [Baseline](audio/en-api-baseline.wav) | [Trained](audio/en-api-trained.wav) |
| `en-numerals` | There are twenty three messages and twelve new files. | [Baseline](audio/en-numerals-baseline.wav) | [Trained](audio/en-numerals-trained.wav) |
| `en-question` | Would you like to hear the next sentence? | [Baseline](audio/en-question-baseline.wav) | [Trained](audio/en-question-trained.wav) |

## What these files contain

- Baseline: untouched Kokoro v1.1-zh with `zf_001`.
- Trained: the selected deterministic update-500 model and its matching voice.
- 24 kHz mono PCM16 WAVs, converted from the saved generated evaluation outputs.
  No audio was regenerated; no gain normalization, trimming, denoising, or
  resampling was applied. Float originals remain in the local evaluation run.
- [Manifest](manifest.json): exact text/IDs, model/source/output hashes, sample
  counts, and measured quantization error (at most one PCM16 step).
- No actual EMIME recordings or model weights are included in this review folder.

The mixed-control ASR result was inconclusive because the recognizer sometimes
translated instead of transcribing. These clips let you judge that speech directly.
See the [full training and evaluation report](../docs/training-2026-09-15.md)
for the development and held-out results and remaining qualification limits.
