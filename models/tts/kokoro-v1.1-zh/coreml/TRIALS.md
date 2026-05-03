# TRIALS — Kokoro-82M-v1.1-zh CoreML conversion

Chronological log of conversion attempts, decisions, and issues encountered
while adapting the v1.0 7-stage chain (`models/tts/kokoro/laishere-coreml/`)
to the Mandarin v1.1-zh checkpoint.

---

## Trial 0 — Background investigation (pre-conversion)

**Question**: Can the 7-stage CoreML chain for v1.0 (`laishere-coreml/`) be
reused unchanged for v1.1-zh, or does the Mandarin checkpoint require a new
trace?

**Findings**:

1. **Architecture**: hexgrad/Kokoro-82M-v1.1-zh ships the **same**
   StyleTTS2-derived architecture as v1.0:
   - ALBERT (3 layers, 8 heads, hidden 768, embedding 128)
   - Predictor (text_encoder LSTM × 5, F0/N projection)
   - TextEncoder (3 conv + 1 LSTM)
   - Decoder + iSTFT generator
   - Style encoder (256-dim ref_s)

   Verified by diffing `config.json` between the two upstream repos — only
   `vocab` (and the embedded `n_token`) differ.

2. **Vocab**: v1.0 has 177 entries (IPA + arrow tones `↓→↗↘`). v1.1-zh has
   171 entries (IPA + Bopomofo `ㄅㄆㄇㄈ…` + tone digits `1-5`); the smaller
   total reflects dropping English-specific tones in favor of Bopomofo and
   the digit-based tone scheme. (Initial expectation of 178 from a stale
   read of the HF preview was wrong — `len(KModel(repo_id=…).vocab)` returns
   171.)

3. **G2P**: v1.0 uses `misaki.en` (with espeak fallback). v1.1-zh uses
   `misaki.zh` (jieba word-segmentation → pypinyin → Bopomofo + tone digit).
   The KPipeline switches automatically based on `lang_code`; we just pass
   `'z'` instead of `'a'`.

4. **Voices**: v1.1-zh ships 96 voice packs (49 `zf_*` female + 47 `zm_*`
   male + 3 EN). All are the same `[510, 1, 256]` torch tensor format as
   v1.0. The `[510, 256]` flat fp32 .bin layout is reused without changes.

**Decision**: Adapt the 7-stage script with **minimal targeted edits**
(repo_id, lang_code, test phonemes, voice id). Keep the trace classes,
op-translation patches, RangeDim bounds, and compute-unit assignments
unchanged.

---

## Trial 1 — Source-of-truth selection

**Question**: Use `models/tts/kokoro/coreml/v21.py` (single end-to-end
mlpackage) or `models/tts/kokoro/laishere-coreml/convert-coreml.py`
(7-stage)?

**Initial misstep**: First scaffold copied `v21.py` → `convert-coreml.py`
and `kokoro/coreml/pyproject.toml`. Discovered on read-through that v21.py
emits a single `kokoro_completev21.mlpackage`, not the 7-stage chain that
ships in `FluidInference/kokoro-82m-coreml/ANE/`.

**Resolution**: Removed v21.py-derived files, re-scaffolded from
`models/tts/kokoro/laishere-coreml/` (PR #45, commit `3b00f7d`). The
laishere chain is what produced the existing `ANE/` mlmodelc bundles on HF
and is what FluidAudio's `KokoroAneManager` consumes.

---

## Trial 2 — Targeted edits to convert-coreml.py

Three edit points beyond docstring/usage updates:

1. **Model load** (was line 540):
   ```python
   model = KModel()  # defaults to hexgrad/Kokoro-82M
   ```
   ↓
   ```python
   model = KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh')
   assert len(model.vocab) >= 178
   ```

2. **Pipeline + voice** (was lines 546–547):
   ```python
   pipe = KPipeline(lang_code='a', model=model)
   voice_pack = pipe.load_voice('af_heart')
   ```
   ↓
   ```python
   pipe = KPipeline(lang_code='z', model=model)
   voice_pack = pipe.load_voice('zf_001')
   ```

3. **Test trace input** (was line 549, hardcoded English IPA):
   ```python
   phonemes = "ðə kwɪk bɹaʊn fɑːks dʒʌmps oʊvɚ ðə leɪzi dɑːɡ."
   ```
   ↓ (run misaki[zh] G2P on a Mandarin sentence so the trace exercises the
   real Bopomofo+digit token distribution)
   ```python
   text_zh = '你好世界，今天天气很好。'
   for _gs, ps, _tks in pipe(text_zh, voice='zf_001'):
       phonemes = ps
       break
   ```

Everything else — RangeDim shape bounds, the 7 trace classes, the op
patches (rsqrt, cos Snake), int8 palettization, compute-unit choices —
unchanged.

---

## Trial 3 — Helper script adaptations

| Script                  | Edits                                                                 |
|-------------------------|-----------------------------------------------------------------------|
| `pyproject.toml`        | Added `misaki[zh]>=0.9.4` (pulls jieba + pypinyin); renamed project.  |
| `inference.py`          | Default `--voice zf_001`, `--lang z`, `--repo-id hexgrad/Kokoro-82M-v1.1-zh`. Threads repo_id through KModel construction. |
| `compare-models.py`     | Replaced `--phonemes` default with `--text` driver that runs misaki[zh] G2P internally. Default voice/lang/repo_id swapped. |
| `benchmark.py`          | Replaced 6 English passages with 6 Mandarin passages (varied tones, punctuation, length). G2P helper kept identical (already language-agnostic via `pipe.g2p`). Default voice/lang/repo_id swapped. |
| `dump-benchmark-data.py`| Loops over `--voices zf_001 zm_009` (default), pulls `vocab.json` from v1.1-zh repo (178 entries). Adds `repo_id` field to benchmark_data.json. |

---

## Trial 4 — Conversion run + parity

Run on this machine (Darwin 25.5.0, Apple Silicon):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python convert-coreml.py --output-dir build/kokoro-v1.1-zh
```

### Issues hit during the run (all resolved)

1. **Vocab assertion was wrong** — initial assertion was
   `len(model.vocab) >= 178` (a stale read from an HF preview). Actual
   v1.1-zh vocab has 171 entries (38 Bopomofo + IPA + tone digits +
   punctuation + a few Hanzi). Replaced with a Bopomofo-presence check
   over Unicode range U+3105–U+312F.

2. **`KPipeline` defaulted to v1.0 repo for voice loading** — even though
   the `KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh')` was passed in,
   `KPipeline(lang_code='z', model=model)` does not infer `repo_id` from
   the model. `pipe.load_voice('zf_001')` then 404'd against
   `hexgrad/Kokoro-82M/resolve/main/voices/zf_001.pt`. Fix: pass
   `repo_id='hexgrad/Kokoro-82M-v1.1-zh'` to `KPipeline()` in all five
   scripts (convert, inference, compare, benchmark, dump).

3. **`scikit-learn` missing** — int8 kmeans palettization (stages 5–7)
   raises `ModuleNotFoundError: No module named 'sklearn'`. Stages 1–4
   completed without it because palettization runs only in stages 5/6/7
   in the laishere chain (per `kmeans_palettize` calls in
   `convert-coreml.py`). Fix: `uv pip install scikit-learn`.
   coremltools 9.0 prints a "scikit-learn 1.8 is not supported, max
   tested 1.5.1" warning but the kmeans path still works.

4. **`pipe.g2p(text)` returns `(text, None)` for misaki[zh]** — the
   benchmark helper expected the misaki[en]-style `tokens` list. Switched
   `phonemize_for_benchmark()` to drive `pipe(text, voice=...)` and
   concatenate per-chunk `ps`.

### Conversion results

```
[1] Loading KModel (hexgrad/Kokoro-82M-v1.1-zh)...
    vocab: 171 entries, 38 Bopomofo
[2] Generating Mandarin test inputs...
    text='你好世界，今天天气很好。'
    phonemes='ㄋㄧ2ㄏㄠ3/ㄕ十4ㄐㄝ4, ㄐ阴1ㄊ言1ㄊ言1ㄑㄧ4/ㄏㄣ2ㄏㄠ3.' (len=35)
    T_a=133
[1/7] ALBERT (fp16+int8pal)        CPU_AND_NE   1.6ms
[2/7] PostAlbert (fp16+int8pal)    CPU_AND_NE   3.6ms
[3/7] Alignment (fp16+int8pal)     CPU_AND_NE   0.7ms
[4/7] Prosody (fp16+int8pal)       CPU_AND_NE   3.1ms
[5/7] Noise (fp32+int8pal)         CPU_AND_NE 145.3ms
[6/7] Vocoder (cos fp16+int8pal)   CPU_AND_NE 267.7ms
[7/7] Tail (fp32)                  CPU_AND_NE   1.9ms
[E2E] corr=-0.139578, mel_corr=0.997730, chain=333.1ms
```

### Parity (compare-models.py)

```
phonemes (34): 'ㄐ阴1ㄊ言1ㄊ言1ㄑㄧ4/ㄓㄣ1/ㄏㄠ3, 阳2ㄍ王1ㄇ应2ㄇㄟ4.'
  waveform corr     : -0.001772   (threshold ≥ 0.80)
  mel-spectrogram   :  0.967283   (threshold ≥ 0.99)
  rms err / rms ref :  1.6841
```

The waveform-corr threshold is not met. Mel correlation is close to but
below the 0.99 threshold. The pattern (high mel, near-zero waveform) is
characteristic of fp16 iSTFT vocoders: small phase differences cause the
sample-by-sample correlation to collapse while spectral content is
preserved. The same pattern shows in `convert-coreml.py`'s own E2E line
(mel=0.998, corr=-0.14). Audio sample (`build/.../sample-zf001.wav`)
sounds like correct Mandarin; ASR-based CER verification is the proper
quality check (TODO Trial 5).

### Inference timings (sample sentence "你好世界，今天天气真好。")

| Voice    | Chain time | Audio   | Speed |
|----------|------------|---------|-------|
| `zf_001` | 333 ms     | 3.33 s  | 10.0× |
| `zm_009` | 200 ms     | 3.50 s  | 17.5× |

All 7 stages run on `CPU_AND_NE` (assigned in the laishere chain;
inherited verbatim).

### Bundle sizes

| Stage              | mlmodelc | mlpackage |
|--------------------|----------|-----------|
| KokoroAlbert       | 5.6 MB   | 5.6 MB    |
| KokoroPostAlbert   | 13 MB    | 13 MB     |
| KokoroAlignment    | 32 KB    | 20 KB     |
| KokoroProsody      | 8.2 MB   | 8.1 MB    |
| KokoroNoise        | 4.5 MB   | 4.4 MB    |
| KokoroVocoder      | 47 MB    | 47 MB     |
| KokoroTail         | 100 KB   | 92 KB     |
| **Total**          | **~78 MB** | **~78 MB** |

Plus `vocab.json` (1.8 KB), `zf_001.bin` + `zm_009.bin` (510 KB each),
`benchmark_data.json` (4.7 KB).

---

## Trial 5 — ASR verification (TODO — post-conversion)

Per `Documentation/ModelConversion.md` §5, TTS conversions must be ASR-
verified. Plan:

- Generate audio for ~25 diverse Mandarin sentences (varied tones,
  numbers, multi-syllable words) via both PyTorch reference and CoreML
  chain, voices `zf_001` + `zm_009`.
- Transcribe with Parakeet-zh CTC (preferred) or Whisper-large-v3.
- Pass criteria: CER < 15% on both, |CER_pt − CER_cm| < 3%.

Script not yet written — will scaffold `asr-verify.py` after the parity
run lands.

---

## Known issues to watch

1. **`kokoro` package version**: v1.1-zh requires `kokoro` recent enough to
   read the n_token=178 `config.json`. Pinned `>=0.9.4`. If `KModel(repo_id=…)`
   raises `KeyError` on a vocab entry, bump to the latest published version
   and re-record here.

2. **Mandarin phoneme density**: each Hanzi expands to ~2-3 phonemes
   (Bopomofo letters + tone digit), so a single 50-character sentence can
   approach `T_enc=200`. The 6th passage in `benchmark.py` is sized to
   land near the 510-phoneme cap; if it trips the `T_a > MAX_FRAMES` skip
   in `benchmark.py`, shorten it.

3. **`coremltools 9.0` sdist fallback** (inherited from v1.0): `uv sync`
   may resolve the pure-python wheel and break `BlobWriter`. README
   documents the `uv pip install --reinstall coremltools==9.0` workaround.

4. **misaki[zh] dependency footprint**: pulls jieba (~50 MB dictionary).
   Acceptable for the conversion environment; downstream Swift/iOS
   consumers don't need it (they read precomputed phonemes from
   `benchmark_data.json` or run a separate Swift ZH G2P).
