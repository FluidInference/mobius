# CoreML Component I/O (Parakeet‑TDT‑0.6B‑v3)

This file documents the CoreML I/O contracts for each exported sub‑module. All shapes below target the fixed 15‑second window at 16 kHz (240,000 samples). Batch is fixed to 1 for on-device use, and we do not plan to support other window sizes in CoreML (re-export if the requirement changes).

## Conventions

- Dtypes: `float32` for real‑valued tensors, `int32` for lengths and token IDs.
- Batch: `B=1` everywhere.
- Vocab: `V=8192` BPE tokens; RNNT adds blank (+1). TDT adds 5 duration logits appended to the class axis.
- Subsampling: overall encoder time reduction ×8.

## PreprocessorWrapper

- Inputs
  - `audio_signal`: `[1, 240000]` float32, PCM range ~[-1, 1]
  - `length`: `[1]` int32, number of valid samples (<= 240000)
- Outputs
  - `mel`: `[1, 128, 1501]` float32 (center=True => frames = floor(240000/160)+1 = 1501)
  - `mel_length`: `[1]` int32 (=1501 for full 15 s)

## EncoderWrapper

- Inputs
  - `features`: `[1, 128, 1501]` float32 (mel)
  - `length`: `[1]` int32 (=1501)
- Outputs
  - `encoded`: `[1, 188, 1024]` float32 (wrapper has time‑major last: it transposes `[B,D,T] -> [B,T,D]`)
  - `encoded_length`: `[1]` int32 (`ceil(1501/8)=188`)

## DecoderWrapper (stateful LSTM, 2 layers)

- Inputs
  - `targets`: `[1, U]` int32 (token IDs in [0, V))
  - `target_lengths`: `[1]` int32 (=U)
  - `h_in`: `[2, 1, 640]` float32 (LSTM hidden, L=2 layers)
  - `c_in`: `[2, 1, 640]` float32 (LSTM cell,  L=2 layers)
- Outputs
  - `decoder_output`: `[1, U, 640]` float32
  - `h_out`: `[2, 1, 640]` float32
  - `c_out`: `[2, 1, 640]` float32

Notes

- For step‑wise greedy decoding, set `U=1` and feed back `h_out`, `c_out`.
- For batched token scoring, `U` can be >1 to evaluate multiple symbols per call.

## JointWrapper

- Inputs
  - `encoder_outputs`: `[1, 188, 1024]` float32 (time‑major)
  - `decoder_outputs`: `[1, U, 640]` float32
- Output
  - `logits`: `[1, 188, U, 8192 + 1 + 5]` float32

Notes

- Split the last dimension into `[token_logits: V+1]` and `[duration_logits: 5]` when post‑processing.
- `log_softmax` is disabled for export; apply on CPU if needed.

## MelEncoderWrapper (fused)

- Inputs
  - `audio_signal`: `[1, 240000]` float32
  - `audio_length`: `[1]` int32
- Outputs
  - `encoder`: `[1, 188, 1024]` float32
  - `encoder_length`: `[1]` int32

Notes

- Fuses preprocessor + encoder to avoid a mel round‑trip. Shapes match chaining `PreprocessorWrapper` then `EncoderWrapper` on the fixed 15 s window.

## JointDecisionWrapper (fused post‑processing)

- Inputs
  - `encoder`: `[1, 188, 1024]` float32
  - `decoder`: `[1, U, 640]` float32
- Outputs
  - `token_id`: `[1, 188, U]` int32 (argmax over token logits, includes blank)
  - `token_prob`: `[1, 188, U]` float32 (softmax probability of chosen token)
  - `duration`: `[1, 188, U]` int32 (argmax over 5 duration logits)

Notes

- Embeds the common host‑side steps: split joint logits, softmax over token logits, argmax over both heads, and gather chosen token probability. Replaces separate `runJoint` → `splitLogits` → `softmax/argmax` in Swift.

## Suggested CoreML tensor names

- Preprocessor: `audio_signal`, `audio_length` → `mel`, `mel_length`
- Encoder: `mel`, `mel_length` → `encoder`, `encoder_length`
- Decoder: `targets`, `target_length`, `h_in`, `c_in` → `decoder`, `h_out`, `c_out`
- Joint: `encoder`, `decoder` → `logits`
- MelEncoder (fused): `audio_signal`, `audio_length` → `encoder`, `encoder_length`
- JointDecision (fused): `encoder`, `decoder` → `token_id`, `token_prob`, `duration`

These names align with wrappers in `individual_components.py` and simplify downstream wiring.
