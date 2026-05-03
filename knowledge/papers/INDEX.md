# Research Paper Index

Centralized reading list for the architectures, decoders, and training
techniques behind FluidAudio's shipped components. Each entry links the
canonical arXiv reference, the FluidAudio module(s) it underpins, and a
one-paragraph "what to learn from this paper".

This file is **lightweight** — it does not commit PDFs or full text
extractions. For deeper extracted entries (where they exist) see
`knowledge/audio/<Paper-Slug>/index.md`.

PDFs that motivated this index live locally in `~/Downloads/Papers/`;
canonical citations and arXiv links are below so anyone on the team can
fetch them without checking binaries into git.

---

## How to read this file

| Column | Meaning |
|---|---|
| `Paper` | Canonical title and link |
| `Year` | First arXiv submission |
| `FluidAudio component` | Where the idea lives in code |
| `Why we cite it` | The one design decision this paper unblocked |

---

## Foundations

| Paper | Year | FluidAudio component | Why we cite it |
|---|---|---|---|
| **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** — Vaswani et al. | 2017 | All transformer-backed models (ASR encoders, TTS LMs, diarization Sortformer) | The transformer block. Multi-head attention + position-wise FFN is the substrate every shipped model is built on. |
| **[Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)** — Graves | 2012 | `AsrManager` (Parakeet TDT), `StreamingAsrManager` | Defines the RNN-T loss and the joint-network output structure that TDT extends. |
| **Connectionist Temporal Classification** — Graves et al., ICML 2006 ([PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf)) | 2006 | CTC decoder paths (Parakeet hybrid CTC/RNNT, CTC-WS context biasing) | The blank-symbol alignment loss that lets us decode unsegmented audio without forced alignment. |

---

## ASR — Encoders

| Paper | Year | FluidAudio component | Why we cite it |
|---|---|---|---|
| **[Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)** — Gulati et al. | 2020 | Parakeet encoder family | Conformer block = MHSA + depth-wise conv + macaron FFN. The encoder shape every Parakeet variant inherits. |
| **[Fast Conformer with Linearly Scalable Attention](https://arxiv.org/abs/2305.05084)** — Rekesh et al. (NVIDIA) | 2023 | `AsrManager` (Parakeet TDT v3, Parakeet v2), `StreamingAsrManager` | 8× downsampling schema → 2.8× faster inference, scales to 1B params. The encoder we actually ship. Deep-dive: `knowledge/audio/Fast-Conformer-Efficient-Speech-Recognition/index.md`. |
| **[Stateful Conformer with Cache-Based Inference for Streaming ASR](https://arxiv.org/abs/2312.17279)** — Noroozi et al. (NVIDIA) | 2023 | `StreamingAsrManager` (Parakeet EOU / Nemotron) | How to run a non-autoregressive Conformer encoder autoregressively at inference: bounded look-ahead/past, activation cache, hybrid CTC+RNNT shared encoder. |

---

## ASR — Decoders & Decoding

| Paper | Year | FluidAudio component | Why we cite it |
|---|---|---|---|
| **[Token-and-Duration Transducer (TDT)](https://arxiv.org/abs/2304.06795)** — Xu et al. (NVIDIA / CMU) | 2023 | `AsrManager` TDT decoder | Joint network predicts `(token, duration)`. Frame-skipping at inference → 2.82× speedup vs RNN-T at equal accuracy. Deep-dive: `knowledge/audio/Token-Duration-Transducer-TDT/index.md`. |
| **[Fast Context-Biasing for CTC and Transducer ASR (CTC-WS)](https://arxiv.org/abs/2406.07096)** — Andrusenko et al. (NVIDIA) | 2024 | Parakeet custom-vocab path (CLI: `g2p-benchmark`, custom vocab) | Match CTC log-probs against a context graph → swap rare/new words into RNN-T greedy output without retraining. |
| **[Qwen3-ASR Technical Report](https://huggingface.co/collections/Qwen/qwen3-asr)** — Qwen Team | 2026 | `Qwen3AsrManager` | Qwen3-Omni-derived ASR for 52 languages, NAR forced-alignment. Foundation for the Qwen3 ASR pipeline (Whisper mel frontend + Qwen3 LM head). |

---

## Diarization — Segmentation

| Paper | Year | FluidAudio component | Why we cite it |
|---|---|---|---|
| **[pyannote.audio: Neural Building Blocks for Speaker Diarization](https://arxiv.org/abs/1911.01255)** — Bredin et al. | 2019 | `DiarizerManager` (online), `OfflineDiarizerManager` | The architectural template: VAD → speaker change → embedding → clustering → resegmentation. Our pipeline is a direct Core ML port of pyannote 3.1 / community-1. |
| **[End-to-end speaker segmentation for overlap-aware resegmentation](https://arxiv.org/abs/2104.04045)** — Bredin & Laurent | 2021 | `DiarizerManager` segmentation model | 5 s chunks at 16 ms resolution, multi-label PIT. The segmentation model we run frame-by-frame. |
| **[Powerset multi-class cross entropy loss for neural speaker diarization](https://arxiv.org/abs/2310.13025)** — Plaquet & Bredin | 2023 | Diarization segmentation loss formulation | Switch from multi-label to powerset multi-class (overlap pairs are dedicated classes). Eliminates the threshold hyperparameter and improves overlap handling — the loss the shipped pyannote 3.1 model is trained with. |

---

## Diarization — Embedding & Clustering

| Paper | Year | FluidAudio component | Why we cite it |
|---|---|---|---|
| **[WeSpeaker: A Research and Production Oriented Speaker Embedding Toolkit](https://arxiv.org/abs/2210.17016)** — Wang et al. | 2022 | Speaker embedding extractor used in offline diarization | Reference implementation for ResNet/ECAPA-TDNN style x-vectors and the embedding training recipes our extractor descends from. |
| **[Bayesian HMM clustering of x-vector sequences (VBx)](https://arxiv.org/abs/2012.14952)** — Landini et al. | 2020 | `OfflineDiarizerManager` clustering stage | The Bayesian-HMM clustering math we run on extracted x-vectors. SOTA on AMI/CALLHOME/DIHARD-II; the offline 17.7% DER on AMI traces back to this. |

---

## Diarization — End-to-End / Streaming

| Paper | Year | FluidAudio component | Why we cite it |
|---|---|---|---|
| **[Sortformer: Permutation-Resolved Speaker Supervision](https://arxiv.org/abs/2409.06656)** — Park et al. (NVIDIA) | 2024 | `sortformer` CLI command, `sortformer-benchmark` | Encoder-only diarization with Sort Loss + sinusoidal speaker labels — fixes the EEND permutation problem. |
| **[Streaming Sortformer: Speaker Cache-Based Online Diarization](https://arxiv.org/abs/2507.18446)** — Medennikov et al. (NVIDIA) | 2025 | Streaming diarization (future / experimental) | Arrival-Order Speaker Cache (AOSC) extends Sortformer to low-latency online use. |
| **[Property-Aware Multi-Speaker Data Simulation](https://arxiv.org/abs/2310.12371)** — Park et al. (NVIDIA) | 2023 | Diarization training-data context | The simulator for controllable silence/overlap statistics that's used to pretrain Sortformer-class models. |

---

## TTS — Vocoders

| Paper | Year | FluidAudio component | Why we cite it |
|---|---|---|---|
| **[iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder](https://arxiv.org/abs/2203.02395)** — Kaneko et al. (NTT) | 2022 | `KokoroSynthesizer` vocoder option | Replaces output upsampling layers with iSTFT — smaller, faster, comparable quality. The lineage Kokoro descends from. |
| **[Vocos: Closing the Gap Between Time-Domain and Fourier Vocoders](https://arxiv.org/abs/2306.00814)** — Siuzdak | 2023 | TTS vocoder reference | Predict Fourier coefficients directly. ~10× faster than HiFi-GAN at parity quality. Reference for any future Kokoro vocoder swap. |

---

## TTS — End-to-End Models

| Paper | Year | FluidAudio component | Why we cite it |
|---|---|---|---|
| **[StyleTTS 2: Human-Level TTS through Style Diffusion + Adversarial Training](https://arxiv.org/abs/2306.07691)** — Li et al. | 2023 | `KokoroSynthesizer` (Kokoro is StyleTTS-2 distilled) | Style diffusion + SLM-based discriminator. Kokoro is a distilled / open-weight derivative of StyleTTS 2; the architecture and training story are here. |
| **[Continuous Audio Language Models](https://arxiv.org/abs/2509.06926)** — Rouard, Orsini, Zeghidour, Roebel, Défossez (Kyutai) | 2025 | `PocketTtsSynthesizer` | The continuous audio LM family that PocketTTS derives from — flow-matching latents, Mimi codec, streaming text→speech. The Mimi split-decoder experiment in `models/tts/pocket_tts/coreml/TRIALS.md` Phase 6 is grounded here. |
| **[Qwen3-TTS Technical Report](https://huggingface.co/collections/Qwen/qwen3-tts)** — Qwen Team | 2026 | *(not yet integrated)* | Dual-track LM with two codecs (25 Hz semantic, 12.5 Hz multi-codebook). 97 ms first-packet streaming. Reference for any future Qwen-family TTS work. |

---

## Quick component → paper map

If you're working on **`<component>`**, these are the papers you should read first.

### `AsrManager` (Parakeet TDT)
1. Vaswani 2017 — Attention
2. Gulati 2020 — Conformer
3. Rekesh 2023 — Fast Conformer
4. Graves 2012 — RNN-T
5. Xu 2023 — TDT

### `StreamingAsrManager` (EOU / Nemotron)
1. Rekesh 2023 — Fast Conformer
2. Noroozi 2023 — Stateful Conformer (cache + bounded context)
3. Graves 2006 — CTC (for the hybrid CTC head)

### `Qwen3AsrManager`
1. Qwen Team 2026 — Qwen3-ASR Technical Report

### Custom-vocab / context biasing
1. Graves 2006 — CTC
2. Andrusenko 2024 — CTC-WS

### `DiarizerManager` (online, pyannote 3.1)
1. Bredin 2019 — pyannote.audio
2. Bredin & Laurent 2021 — End-to-end segmentation
3. Plaquet & Bredin 2023 — Powerset loss

### `OfflineDiarizerManager`
1. Bredin 2019 — pyannote.audio (pipeline shape)
2. Bredin & Laurent 2021 — Segmentation model
3. Wang 2022 — WeSpeaker (embedding training lineage)
4. Landini 2020 — VBx clustering

### Sortformer paths
1. Park 2024 — Sortformer
2. Medennikov 2025 — Streaming Sortformer
3. Park 2023 — Multi-Speaker Data Simulation

### `KokoroSynthesizer`
1. Li 2023 — StyleTTS 2 (architecture)
2. Kaneko 2022 — iSTFTNet (vocoder lineage)
3. Siuzdak 2023 — Vocos (vocoder reference for swaps)

### `PocketTtsSynthesizer`
1. Rouard et al. 2025 — Continuous Audio Language Models (Kyutai / Mimi)
2. `models/tts/pocket_tts/coreml/TRIALS.md` — Mobius conversion trial log
3. `models/tts/pocket_tts/coreml/CONVERSION.md` — Final conversion architecture

---

## Maintenance

When adding a paper:

1. Fetch the canonical citation (arXiv link, year, authors).
2. Add a row to the right topic table above with the FluidAudio component
   it touches and one sentence on **why we cite it** — not what the paper
   abstract says.
3. If the paper warrants a deep dive, add a folder under
   `knowledge/audio/<Paper-Slug>/index.md` with the existing front-matter
   format (`title`, `source_url`, `retrieved_at`) and link from this
   index.
4. Do **not** commit PDFs.

Notes on the existing `~/Downloads/Papers/` collection that seeded this
index:

- Several files are duplicates with confusing names. After de-dup by
  MD5, 22 unique papers remain (the directory has 27 files).
- `00_Parakeet_CTC.pdf` is an ACM landing-page snapshot, not the real
  CTC paper — citation above points at the canonical 2006 ICML paper.
- `00_Parakeet_Conformer.pdf` and `02_Parakeet_Nemotron.pdf` share an
  MD5 (both are the Conformer paper); the Nemotron filename is a
  mis-label.
- `01_Parakeet_FastConformer.pdf` ≡ `10_Parakeet_Fast_Conformer.pdf`,
  `06_Parakeet_TDT.pdf` ≡ `11_Parakeet_TDT_v2.pdf`, and
  `03_Diarization_Wespeaker.pdf` ≡ `04_diarization_WeSpeaker.pdf`.
- `16_TTS_PocketTTS.pdf` is the Kyutai *Continuous Audio Language
  Models* paper — PocketTTS is a derivative, not the paper title.
- `17_Parakeet_custom_vocab.pdf` is the CTC-WS context biasing paper.
