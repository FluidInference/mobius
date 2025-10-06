# StyleTTS2 & Kokoro CoreML Toolkit

This target bundles the experimental pipeline we use to export Kokoro TTS (StyleTTS2-derived) models to CoreML. It documents the linguistic and acoustic concepts that drive the system, the concrete architecture we implement in `v21.py`, and the practical workflow for converting and validating end-to-end voices.

## Core Concepts
- **Linguistic content** – the textual sequence we want to synthesize.
- **Prosody** – pitch, rhythm, loudness, and timing that deliver intent (e.g., questions vs. statements, excitement vs. disappointment).
- **Speaker style / acoustic signature** – a speaker's timbre, resonance, habitual pacing, and articulation patterns; in StyleTTS2 this is the *acoustic style* embedding.
- **Style diffusion** – the iterative denoising process that draws samples from the learned distribution of styles to sculpt realistic mel-spectrograms.
- **Duration / pitch / prosody predictors** – networks that estimate phoneme length, F0 contours, and expressive templates so the generator knows *how* to read the text.
- **G2P (grapheme-to-phoneme)** – converts orthographic text to phoneme IDs, handled here through `misaki.en.G2P` with ALBERT-style encoders downstream.
- **Reference audio** – speaker style vectors are extracted from voice packs such as `af_heart_all_voices.pkl` or `af_heart_voice.npy` and fed to the style encoder.

## StyleTTS2 Architecture (Training View)
- **Text encoder** – maps phoneme IDs into contextual embeddings.
- **Style encoder** – consumes reference mel-spectrograms and yields a speaker embedding.
- **Prosodic text encoder** – infers expressiveness from text alone.
- **Duration & pitch extractors** – derive ground-truth timing and intonation targets.
- **Diffusion denoiser** – UNet iterative refinement from Gaussian noise to clean mel-spectrograms.
- **Multi-period / multi-resolution discriminators (MPD/MRD)** – adversarial critics that pressure the generator toward humanlike quality.

## StyleTTS2 Architecture (Inference View)
1. **Text → phonemes** – G2P produces IPA-like tokens (e.g., `hello` → `[HH, AH0, L, OW1]`).
2. **Text encoder** – transformer / BERT stack emits contextual embeddings.
3. **Duration predictor** – CNN stack estimates log durations per phoneme and expands embeddings accordingly.
4. **Pitch predictor** – CNN regression head outputs per-frame F0.
5. **Style encoder** – aggregates reference mels into a 256-d acoustic style vector.
6. **Diffusion denoiser** – conditioned on text, F0, and style, iteratively denoises mel frames into speech.

### Prosody Examples
- "I can't believe it!" – rising pitch, faster tempo, louder delivery.
- "I can't believe it..." – falling pitch, slower tempo, softer energy.
- "I CAN'T believe it" – emphasis spike on the stressed word.
- Questions vs. statements – rising vs. falling terminal pitch.
- Anger – faster rate, clipped consonants, higher intensity.
- Sadness – slower tempo, narrow pitch range.
- Surprise – sudden pitch jumps and elongated vowels.
- Sarcasm – exaggerated contours, deliberate pacing on key tokens.

## Kokoro Pipeline Highlights
The Kokoro Python pipeline is realized through `KPipeline`, `KModel`, and the generator stack in `v21.py`. Key components:

- **ALBERT-style encoder** – Kokoro reuses parameters across transformer layers to produce 768-d phoneme embeddings.
- **`TextEncoderFixed`** – replaces the original text encoder to avoid `pack_padded_sequence` in traced TorchScript / CoreML and drives explicit LSTM state management.
- **`TextEncoderPredictorFixed`** – mirrors Kokoro's duration encoder with explicit masking and `AdaLayerNorm` handling so the CoreML predictor sees the same activations as PyTorch.
- **`SineGenDeterministic` & `SourceModuleHnNSFDeterministic`** – deterministic sine source that preserves prosody without random phase resets, smoothing unvoiced/voiced transitions.
- **`GeneratorDeterministic`** – keeps Kokoro's upsamplers, residual blocks, and STFT decoder while ensuring tensor-safe padding and stable harmonics injection.
- **`KokoroCompleteCoreML`** – end-to-end wrapper that consumes text IDs, reference style, random phase seeds, and attention masks to emit audio, audio length in samples, and the predicted duration grid in a single CoreML program.

## Conversion & Workflow
1. **Set up dependencies**
   ```bash
   uv sync
   ```
2. **Prepare example inputs** – load a reference voice (e.g., `af_heart`) and produce `input_ids`, `ref_s`, random phases, and attention masks via the pipeline helpers in `v21.py`.
3. **Trace the model** – `torch.jit.trace(KokoroCompleteCoreML, inputs)` captures graph-safe execution.
4. **Export to CoreML** – `coremltools.convert(..., convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16)` yields a `.mlpackage` ready for iOS 17+ and macOS 14+ with `.CpuOnly` tracing.
5. **Validate** – run sampling via `main.py` or an interactive notebook, compare durations against expected phoneme lengths, and listen for robotic artifacts.

## Issues & Observations
- **Robotic timbre** – despite fixing phoneme-level duration estimates, generated speech can sound synthetic; likely rooted in generator loss weighting or insufficient style diffusion steps.
- **Generator hot spots** – most audible artifacts originate in the `Generator` class where harmonic sources merge with learned upsamplers.
- **`SineGenDeterministic`** – getting harmonic phases aligned is critical; incorrect wrapping yields choppy prosody.
- **Alignment matrix** – `KokoroCompleteCoreML.create_alignment_matrix` maps phoneme durations to frames; off-by-one errors here destabilize downstream ASR features and length estimations.

## Model Variants & Runtimes
Different deployment recipes:
- **Complete model** – single CoreML package covering pregenerator + generator stages.
- **Pregenerator → Generator** – split to run text-side features separately (CPU+Neural Engine for pregenerator, generator where available).
- **Pregenerator → Generator → Decoder** – fully modular stack with discrete CoreML models for each phase.

## Testing Checklist
- Synthesize bundled sample audio (e.g., `yc_first_minute.wav`) and check durations predicted by `pred_dur` against measured lengths.
- Confirm `.CpuOnly` tracing works on iOS 17/macOS 14 targets.
- Document prerequisites like `git lfs install` before cloning large checkpoints.
- Capture key metrics (latency, MOS-like subjective ratings) and note whether additional post-filtering is required.

## References
- StyleTTS2 – [arXiv:2306.07691](https://arxiv.org/pdf/2306.07691)
- StyleTTS2 Deep Dive – <https://deepwiki.com/yl4579/StyleTTS2>
- Kokoro repository – <https://github.com/yl4579/StyleTTS2/blob/5cedc71c/README.md>

