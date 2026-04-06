---
license: apache-2.0
language:
- ar
- de
- el
- en
- es
- fr
- it
- ja
- ko
- nl
- pl
- pt
- vi
- zh
pipeline_tag: automatic-speech-recognition
tags:
- audio
- hf-asr-leaderboard
- speech-recognition
- transcription
library_name: transformers
---
# Cohere Transcribe

Cohere Transcribe is an open source release of a 2B parameter dedicated audio-in, text-out automatic speech recognition (ASR) model. The model supports 14 languages.

Developed by: [Cohere](https://cohere.com) and [Cohere Labs](https://cohere.com/research). Point of Contact: [Cohere Labs](https://cohere.com/research).

<style>
    @scope {
        th, td {
            text-align: left;
            padding: 0.375rem 0.625rem;
            letter-spacing: 0;
            vertical-align: top;
            line-height: 133.3333%;
            border: 1px solid #e0e0e0;
        }
        ul {
            list-style-type: disc;
            margin: 0;
            padding-left: 1em;

            li {
                margin: 0.25rem 0 0;
                line-height: 133.3333%;
            }
        }
    }
</style>
<table>
    <tbody>
        <tr>
            <th>Name</th>
            <td><strong>cohere-transcribe-03-2026</strong></td>
        </tr>
        <tr>
            <th>Architecture</th>
            <td>conformer-based encoder-decoder</td>
        </tr>
        <tr>
            <th>Input</th>
            <td>audio waveform → log-Mel spectrogram. Audio is automatically resampled to 16kHz if necessary during preprocessing. Similarly, multi-channel (stereo) inputs are averaged to produce a single channel signal.</td>
        </tr>
        <tr>
            <th>Output</th>
            <td>transcribed text</td>
        </tr>
        <tr>
            <th>Model size</th>
            <td>2B</td>
        </tr>
        <tr>
            <th>Model</th>
            <td>a large Conformer encoder extracts acoustic representations, followed by a lightweight Transformer decoder for token generation</td>
        </tr>
        <tr>
            <th>Training objective</th>
            <td>supervised cross-entropy on output tokens; trained from scratch</td>
        </tr>
        <tr>
            <th>Languages</th>
            <td>
                Trained on 14 languages:
                <ul>
                    <li><strong>European:</strong> English, French, German, Italian, Spanish, Portuguese, Greek,
                        Dutch, Polish</li>
                    <li><strong>AIPAC:</strong> Chinese (Mandarin), Japanese, Korean, Vietnamese</li>
                    <li><strong>MENA:</strong> Arabic</li>
                </ul>
            </td>
        </tr>
        <tr>
            <th>License</th>
            <td>Apache 2.0</td>
        </tr>
    </tbody>
</table>

✨**Try the Cohere Transcribe** [**demo**](https://huggingface.co/spaces/CohereLabs/cohere-transcribe-03-2026)✨

## Usage

Cohere Transcribe is supported natively in `transformers`. This is the recommended way to use the model for 
offline inference. For online inference, see the vLLM integration example below.

```bash
pip install transformers>=5.4.0 torch huggingface_hub soundfile librosa sentencepiece protobuf
pip install datasets  # only needed for long-form and non-English examples
```

Testing was carried out with `torch==2.10.0` but it is expected to work with other versions.

### Quick Start 🤗

Transcribe any audio file in a few lines:

```python
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio
from huggingface_hub import hf_hub_download

processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
model = CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map="auto")

audio_file = hf_hub_download(
    repo_id="CohereLabs/cohere-transcribe-03-2026",
    filename="demo/voxpopuli_test_en_demo.wav",
)
audio = load_audio(audio_file, sampling_rate=16000)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt", language="en")
inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.decode(outputs, skip_special_tokens=True)
print(text)
```

<details>
<summary><b>Long-form transcription</b></summary>

For audio longer than the feature extractor's `max_audio_clip_s`, the feature extractor automatically splits the waveform into chunks.
The processor reassembles the per-chunk transcriptions using the returned `audio_chunk_index`.

This example transcribes a 55 minute earnings call:

```python
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from datasets import load_dataset
import time

processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
model = CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map="auto")

ds = load_dataset("distil-whisper/earnings22", "full", split="test", streaming=True)
sample = next(iter(ds))

audio_array = sample["audio"]["array"]
sr = sample["audio"]["sampling_rate"]
duration_s = len(audio_array) / sr
print(f"Audio duration: {duration_s / 60:.1f} minutes")

inputs = processor(audio=audio_array, sampling_rate=sr, return_tensors="pt", language="en")
audio_chunk_index = inputs.get("audio_chunk_index")
inputs.to(model.device, dtype=model.dtype)

start = time.time()
outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.decode(outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en")[0]
elapsed = time.time() - start
rtfx = duration_s / elapsed
print(f"Transcribed in {elapsed:.1f}s — RTFx: {rtfx:.1f}")
print(f"Transcription ({len(text.split())} words):")
print(text[:500] + "...")
```
</details>

<details>
<summary><b>Punctuation control</b></summary>

Pass `punctuation=False` to obtain lower-cased output without punctuation marks.

```python
inputs_pnc = processor(audio, sampling_rate=16000, return_tensors="pt", language="en", punctuation=True)
inputs_nopnc = processor(audio, sampling_rate=16000, return_tensors="pt", language="en", punctuation=False)
```

By default, punctuation is enabled.

</details>

<details>
<summary><b>Batched inference</b></summary>

Multiple audio files can be processed in a single call. When the batch mixes short-form and long-form audio, the
processor handles chunking and reassembly.

```python
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.audio_utils import load_audio

processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
model = CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map="auto")

audio_short = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
    sampling_rate=16000,
)
audio_long = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3",
    sampling_rate=16000,
)

inputs = processor([audio_short, audio_long], sampling_rate=16000, return_tensors="pt", language="en")
audio_chunk_index = inputs.get("audio_chunk_index")
inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, max_new_tokens=256)
text = processor.decode(
    outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="en"
)
print(text)
```
</details>

<details>
<summary><b>Non-English transcription</b></summary>

Specify the language code to transcribe in any of the 14 supported languages. This example transcribes Japanese audio from the FLEURS dataset:

```python
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
model = CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map="auto")

ds = load_dataset("google/fleurs", "ja_jp", split="test", streaming=True)
ds_iter = iter(ds)
samples = [next(ds_iter) for _ in range(3)]

for sample in samples:
    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", language="ja")
    inputs.to(model.device, dtype=model.dtype)

    outputs = model.generate(**inputs, max_new_tokens=256)
    text = processor.decode(outputs, skip_special_tokens=True)
    print(f"REF: {sample['transcription']}\nHYP: {text}\n")
```
</details>

### Broader dependency support with `trust_remote_code=True`

For a wider range of `torch` and `transformers` versions, run with `trust_remote_code=True`. 
You should expect greater stability via the transformers native path above. 
This option will be deprecated in the future.

<details>
<summary><b>Usage with <code>trust_remote_code=True</code></b></summary>

`trust_remote_code=True` inference exposes a single `model.transcribe()` method that automatically handles long-form audio chunking and exposes parameters to facilitate efficient inference. It is recommended that you let the transcribe method handle batching for you. This implementation is optimized for offline inference: for online inference, see the vLLM integration example below.

#### Installation

Recommended:

```bash
pip install "transformers>=4.56,<5.3,!=5.0.*,!=5.1.*" torch huggingface_hub soundfile librosa sentencepiece protobuf
pip install datasets  # only needed for examples 2 and 3
```

<details>
<summary><b>Installation with even broader <code>transformers</code> compatibility</b></summary>
<br>

For broader compatibility run the following install:
```bash
pip install "transformers>=4.52,!=5.0.*,!=5.1.*" torch huggingface_hub soundfile librosa sentencepiece protobuf
```

This will replace the efficient static-cache with a dynamic-cache fallback on some versions.

Transformers 5.0 and 5.1 have a weight-loading issue and are **not compatible**.
</details>

#### Example 1: Quick Start

Transcribe any audio file in a few lines. The model accepts file paths directly — no manual preprocessing required.

```python
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download

model_id = "CohereLabs/cohere-transcribe-03-2026"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()

audio_file = hf_hub_download(
    repo_id="CohereLabs/cohere-transcribe-03-2026",
    filename="demo/voxpopuli_test_en_demo.wav",
)

texts = model.transcribe(processor=processor, audio_files=[audio_file], language="en")
print(texts[0])
```

<details>
<summary><b>Example 2: Optimized Throughput</b></summary>

When audio is already in memory (streaming datasets, microphone input, etc.), pass numpy arrays directly instead of file paths. Enable `compile=True` to torch.compile the encoder for faster throughput, and `pipeline_detokenization=True` to overlap CPU detokenization with GPU inference.

> **Note:** `pipeline_detokenization=True` is not supported on Windows.

This example transcribes Japanese audio from the FLEURS dataset:

```python
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download


model_id = "CohereLabs/cohere-transcribe-03-2026"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()

from datasets import load_dataset

ds = load_dataset("google/fleurs", "ja_jp", split="test", streaming=True)
ds_iter = iter(ds)
samples = [next(ds_iter) for _ in range(3)]  # take 3 samples

audio_arrays = [s["audio"]["array"] for s in samples]
sample_rates = [s["audio"]["sampling_rate"] for s in samples]

# compile=True incurs a one-time warmup cost on the first call; subsequent calls are faster.
texts = model.transcribe(
    processor=processor,
    audio_arrays=audio_arrays,
    sample_rates=sample_rates,
    language="ja",
    compile=True,
    pipeline_detokenization=True,
    batch_size=16,
)
for ref, hyp in zip([s["transcription"] for s in samples], texts):
    print(f"REF: {ref}\nHYP: {hyp}\n")
```
</details>

<details>
<summary><b>Example 3: Long-Form Audio</b></summary>

Audio longer than 35 seconds is automatically split into overlapping chunks and reassembled. The API is identical — no special flags or configuration needed. This example transcribes a 55 minute earnings call. This will be slow if you haven't run `compile=True` in the previous example:

```python
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download


model_id = "CohereLabs/cohere-transcribe-03-2026"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True).to(device)
model.eval()

from datasets import load_dataset

ds = load_dataset("distil-whisper/earnings22", "full", split="test", streaming=True)
sample = next(iter(ds))

import time

audio_array = sample["audio"]["array"]
sr = sample["audio"]["sampling_rate"]
duration_s = len(audio_array) / sr
print(f"Audio duration: {duration_s / 60:.1f} minutes")

start = time.time()
texts = model.transcribe(
    processor=processor,
    audio_arrays=[audio_array],
    sample_rates=[sr],
    language="en",
    compile=True,
)
elapsed = time.time() - start
rtfx = duration_s / elapsed
print(f"Transcribed in {elapsed:.1f}s — RTFx: {rtfx:.1f}")
print(f"Transcription ({len(texts[0].split())} words):")
print(texts[0][:500] + "...")
```
</details>

<details>
<summary><b><code>transcribe()</code> API Reference</b></summary>


| Argument | Type | Default | Description |
|---|---|---|---|
| `processor` | `AutoProcessor` | required | Processor instance for this model |
| `language` | `str` | required | [ISO 639-1 language code](https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes). The model does not perform language detection, so this is always required |
| `audio_files` | `list[str]` | `None` | List of audio file paths. Mutually exclusive with `audio_arrays` |
| `audio_arrays` | `list[np.ndarray]` | `None` | List of 1-D numpy float arrays (raw waveforms). Requires `sample_rates` |
| `sample_rates` | `list[int]` | `None` | Sample rate for each entry in `audio_arrays` |
| `punctuation` | `bool` | `True` | Include punctuation in output |
| `batch_size` | `int` | from config | GPU batch size for inference |
| `compile` | `bool` | `False` | `torch.compile` encoder layers for faster throughput. First call incurs a one-time warmup cost |
| `pipeline_detokenization` | `bool` | `False` | Overlap CPU detokenization with GPU inference. Beneficial when more audio segments than `batch_size` are passed in a single call |



**Returns:** `list[str]` — one transcription string per input audio.

</details>

</details>

### vLLM Integration

For production serving we recommend running via vLLM following the instructions below.

<details>
<summary><b>Run cohere-transcribe-03-2026 via vLLM</b></summary>

First install vLLM (refer to [vLLM installation instructions](https://docs.vllm.ai/en/latest/getting_started/installation/)):

```bash
uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
uv pip install vllm[audio]
uv pip install librosa
```

Start vLLM server
```bash
vllm serve CohereLabs/cohere-transcribe-03-2026 --trust-remote-code
```

Send request
```bash
curl -v -X POST http://localhost:8000/v1/audio/transcriptions \
 -H "Authorization: Bearer $VLLM_API_KEY" \
-F "file=@$(realpath ${AUDIO_PATH})" \
-F "model=CohereLabs/cohere-transcribe-03-2026"
```
</details>

## Results

<details>
<summary><b>English ASR Leaderboard (as of 03.26.2026)</b></summary>

<style>
        table.simple {
        border-collapse: collapse;
        }

        table.simple th,
        table.simple td {
        text-align: left;
        padding: 0.375rem 0.625rem;
        min-width: 5em;
        vertical-align: top;
        line-height: 1.2;
        border-bottom: 1px solid rgba(127,127,127,0.35);
        }

        table.simple thead th {
        white-space: nowrap;
        }

        table.simple th:first-child {
        position: sticky;
        left: 0;
        background: inherit;
        z-index: 1;
        }

        table.simple .num {
        text-align: right;
        }

        table.simple .highlight-row > th,
        table.simple .highlight-row > td {
        background: rgba(127,127,127,0.12);
        }

        table.simple .highlight-cell {
        background: rgba(127,127,127,0.18);
        }
</style>

<div style="overflow-x: auto;">
    <table class="simple text-web3-14 font-body">
        <thead>
            <tr>
                <th>Model</th>
                <th class="num">Average WER</th>
                <th class="num">AMI</th>
                <th class="num">Earnings 22</th>
                <th class="num">Gigaspeech</th>
                <th class="num">LS clean</th>
                <th class="num">LS other</th>
                <th class="num">SPGISpeech</th>
                <th class="num">Tedlium</th>
                <th class="num">Voxpopuli</th>
            </tr>
        </thead>
        <tbody>
            <tr class="highlight-row">
                <th><strong style="white-space:nowrap">Cohere Transcribe</strong></th>
                <td class="num highlight-cell"><strong>5.42</strong></td>
                <td class="num"><strong>8.15</strong></td>
                <td class="num">10.84</td>
                <td class="num">9.33</td>
                <td class="num"><strong>1.25</strong></td>
                <td class="num"><strong>2.37</strong></td>
                <td class="num">3.08</td>
                <td class="num">2.49</td>
                <td class="num">5.87</td>
            </tr>
            <tr>
                <th style="font-weight:normal;">Zoom Scribe v1</th>
                <td class="num">5.47</td>
                <td class="num">10.03</td>
                <td class="num">9.53</td>
                <td class="num">9.61</td>
                <td class="num">1.63</td>
                <td class="num">2.81</td>
                <td class="num"><strong>1.59</strong></td>
                <td class="num">3.22</td>
                <td class="num"><strong>5.37</strong></td>
            </tr>
            <tr>
                <th style="font-weight:normal;">IBM Granite 4.0 1B Speech</th>
                <td class="num">5.52</td>
                <td class="num">8.44</td>
                <td class="num"><strong>8.48</strong></td>
                <td class="num">10.14</td>
                <td class="num">1.42</td>
                <td class="num">2.85</td>
                <td class="num">3.89</td>
                <td class="num">3.10</td>
                <td class="num">5.84</td>
            </tr>
            <tr>
                <th style="font-weight:normal;">NVIDIA Canary Qwen 2.5B</th>
                <td class="num">5.63</td>
                <td class="num">10.19</td>
                <td class="num">10.45</td>
                <td class="num">9.43</td>
                <td class="num">1.61</td>
                <td class="num">3.10</td>
                <td class="num">1.90</td>
                <td class="num">2.71</td>
                <td class="num">5.66</td>
            </tr>
            <tr>
                <th style="font-weight:normal;">Qwen3-ASR-1.7B</th>
                <td class="num">5.76</td>
                <td class="num">10.56</td>
                <td class="num">10.25</td>
                <td class="num"><strong>8.74</strong></td>
                <td class="num">1.63</td>
                <td class="num">3.40</td>
                <td class="num">2.84</td>
                <td class="num"><strong>2.28</strong></td>
                <td class="num">6.35</td>
            </tr>
            <tr>
                <th style="font-weight:normal;">ElevenLabs Scribe v2</th>
                <td class="num">5.83</td>
                <td class="num">11.86</td>
                <td class="num">9.43</td>
                <td class="num">9.11</td>
                <td class="num">1.54</td>
                <td class="num">2.83</td>
                <td class="num">2.68</td>
                <td class="num">2.37</td>
                <td class="num">6.80</td>
            </tr>
            <tr>
                <th style="font-weight:normal;">Kyutai STT 2.6B</th>
                <td class="num">6.40</td>
                <td class="num">12.17</td>
                <td class="num">10.99</td>
                <td class="num">9.81</td>
                <td class="num">1.70</td>
                <td class="num">4.32</td>
                <td class="num">2.03</td>
                <td class="num">3.35</td>
                <td class="num">6.79</td>
            </tr>
            <tr>
                <th style="font-weight:normal;">OpenAI Whisper Large v3</th>
                <td class="num">7.44</td>
                <td class="num">15.95</td>
                <td class="num">11.29</td>
                <td class="num">10.02</td>
                <td class="num">2.01</td>
                <td class="num">3.91</td>
                <td class="num">2.94</td>
                <td class="num">3.86</td>
                <td class="num">9.54</td>
            </tr>
            <tr>
                <th style="font-weight:normal;">Voxtral Mini 4B Realtime 2602</th>
                <td class="num">7.68</td>
                <td class="num">17.07</td>
                <td class="num">11.84</td>
                <td class="num">10.38</td>
                <td class="num">2.08</td>
                <td class="num">5.52</td>
                <td class="num">2.42</td>
                <td class="num">3.79</td>
                <td class="num">8.34</td>
            </tr>
        </tbody>
    </table>
</div>

Link to the live leaderboard: [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

</details>


#### Human-preference results
 
We observe similarly strong performance in human evaluations, where trained annotators assess transcription quality across 
real-world audio for accuracy, coherence and usability. 
The consistency between automated metrics and human judgments suggests that the model’s improvements translate 
beyond controlled benchmarks to practical transcription settings.

<img src="./assets/260326_TranscribeLaunch_Plot1.png" alt="Human-preference results" width="100%">

_Figure: Human preference evaluation of model transcripts. In a head-to-head comparison, 
annotators were asked to express preferences for generations which primarily preserved meaning - 
but also avoided hallucination, correctly identified named entities, 
and provided verbatim transcripts with appropriate formatting. 
A score of 50% or higher indicates that Cohere Transcribe was preferred on average in the comparison._

</details>
<details>
<summary><b>per-language WERs</b></summary>

<img src="./assets/HF_model_card_per-language-avg-plot.png" alt="per-language WERs" width="100%">

_Figure: per-language error rate averaged over FLEURS, Common Voice 17.0, MLS and Wenet tests sets (where relevant for a given language). CER for zh, ja, ko — WER otherwise_


</details>

## Resources

For more details and results:

* [Technical blog post](https://huggingface.co/blog/CohereLabs/cohere-transcribe-03-2026-release) contains WERs and other quality metrics.
* [Announcement blog post](https://cohere.com/blog/transcribe) for more information about the model.
* English, EU and long-form transcription WERs/RTFx are on the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

## Strengths and Limitations

Cohere Transcribe is a performant, dedicated ASR model intended for efficient speech transcription.

### Strengths

Cohere Transcribe demonstrates best-in-class transcription accuracy in 14 languages. As a dedicated speech recognition model, it is also efficient, benefitting from a real-time factor up to three times faster than that of other, dedicated ASR models in the same size range. The model was trained from scratch, and from the outset, we deliberately focused on maximizing transcription accuracy while keeping production readiness top-of-mind.


### Limitations

* **Single language.** The model performs best when remaining in-distribution of a single, pre-specified language amongst the 14 in the range it supports. It does not feature explicit, automatic language detection and exhibits inconsistent performance on code-switched audio.

* **Timestamps/Speaker diarization.** The model does not feature either of these.

* **Silence.** Like most AED speech models, Cohere Transcribe is eager to transcribe, even non-speech sounds. The model thus benefits from prepending a noise gate or VAD (voice activity detection) model in order to prevent low-volume, floor noise from turning into hallucinations.


## Ecosystem support 🚀

Cohere Transcribe is supported on the following libraries/platforms:

* [`transformers`](https://huggingface.co/docs/transformers/model_doc/cohere_asr) (see [Quick Start](#quick-start) above).
* [`vLLM`](https://github.com/vllm-project/vllm/pull/38120) (see [vLLM integration](#vllm-integration) above).
* [`mlx-audio`](https://github.com/Blaizzy/mlx-audio/pull/605) for Apple Silicon.
* Rust implementation: [`cohere_transcribe_rs`](https://github.com/second-state/cohere_transcribe_rs)
* In the browser ✨[**demo**](https://huggingface.co/spaces/CohereLabs/Cohere-Transcribe-WebGPU)✨ (via `transformers.js` and WebGPU)
* Chrome extension: [`cohere_transcribe_extension`](https://github.com/davila7/cohere_transcribe_extension)
* [Whisper Memos](https://whispermemos.com/kb/features/ai-models#cohere-transcribe) (iOS App).


If you have added support for the model somewhere not included above please raise an issue/PR!

If you find issues with any of these please raise an issue with the respective library.

## Model Card Contact
For errors or additional questions about details in this model card, contact [labs@cohere.com](mailto:labs@cohere.com) or raise an issue.


Terms of Use:
We hope that the release of this model will make community-based research efforts more accessible, by releasing the weights of a highly performant 2 billion parameter model to researchers all over the world. This model is governed by an Apache 2.0 license.
