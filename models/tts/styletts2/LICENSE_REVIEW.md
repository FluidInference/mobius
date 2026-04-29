# StyleTTS2 — License Review

Before redistributing any converted artifacts, resolve the items below.

## Code

- **Upstream repo**: [yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2) — **MIT**.
  Code-only changes (conversion scripts, traceable wrappers) are unencumbered.

## Pre-trained weights

- **`yl4579/StyleTTS2-LibriTTS`** and **`yl4579/StyleTTS2-LJSpeech`** ship with
  custom restrictions in the upstream README beyond MIT. Specifically:
  - Synthesized speech must disclose synthetic origin
  - Speaker-consent restrictions on voice cloning use

This is **not** a permissive ML-model license. Any redistribution (HuggingFace
upload, app bundle, public download URL) needs explicit confirmation that
these terms are preserved and surfaced to end users.

## Training data

- **LibriTTS** is CC-BY-4.0. The trained weights are a derivative work; the
  weight license is the author's call (see above).
- **LJSpeech** is public domain.

## Action items before any distribution

- [ ] Confirm the exact upstream weight license text (current README at the
      time of conversion).
- [ ] Confirm with counsel that converted CoreML packages can be redistributed
      under the same terms, with end-user disclosure.
- [ ] If the answer is "no", pivot to a permissively-licensed StyleTTS2
      derivative (e.g. Kokoro-class weights, Apache-2.0) before any public
      release.

## Default posture (this branch)

Converted `.mlpackage` files stay **on local disk**. Not committed to the repo,
not uploaded to HuggingFace, not bundled into FluidAudio's downloadable model
set. `coreml/*.mlpackage` is gitignored.
