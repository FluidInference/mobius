# Parakeet‑TDT‑0.6B‑v3 Architecture Overview

## Model topology

- Streaming‑capable RNN‑Transducer (RNNT) ASR model trained at 16 kHz.
- Time‑Depth Transducer (TDT) extends RNNT to jointly predict token and duration buckets. In v3, durations = [0, 1, 2, 3, 4].
- Four major learnable blocks packaged in the `.nemo` checkpoint:
  1. `AudioToMelSpectrogramPreprocessor` – waveform → 128×T mel‑spectrogram.
  2. `ConformerEncoder` (FastConformer‑style) – 24 conformer blocks, `d_model=1024`, `n_heads=8`, depthwise CNN subsampling with total reduction ×8.
  3. `RNNTDecoder` – 2‑layer LSTM prediction network, hidden size 640, `blank_as_pad=True` for efficient batching and blank handling.
  4. `RNNTJoint` – linear projections from encoder (1024) and decoder (640) into joint space 640 with ReLU, then to class logits.

## Audio front‑end

- `sample_rate = 16000`, Hann window, 25 ms window, 10 ms hop (`n_fft=512`, `win_length=400`, `hop_length=160`, `center=True`).
- 128 mel filters (Slaney‑style), log‑magnitude power 2.0 and per‑feature normalization.
- Outputs `(mel, mel_length)` feed the encoder. For CoreML, both tensors must be surfaced to preserve valid‑lengths through subsampling.

## Encoder (FastConformer)

- Implemented as `nemo.collections.asr.modules.ConformerEncoder` configured per FastConformer.
- 24 blocks, `d_model=1024`, FFN expansion ×4, 8‑head relative‑position attention, conv kernel 9, batch‑norm.
- Dropouts: 0.1 general, 0.1 pre‑encoder, 0.0 embedding, 0.1 attention; stochastic depth disabled.
- Depthwise CNN subsampling yields overall time reduction ×8. Returns features `[B, D=1024, T_enc]` and lengths.

## Decoder (`RNNTDecoder`)

- Stateful LSTM prediction network (2 layers, hidden=640). See `nemo/collections/asr/modules/rnnt.py`.
- `blank_as_pad=True` adds a learnable pad/blank to the embedding; embedding returns zeros for blank, simplifying blank handling.
- Export toggles `_rnnt_export` and uses `.predict()` without prepending SOS so decoder outputs shape `[B, U, 640]`.
- Streaming state is a tuple `(h, c)`, each `[L=2, B, H=640]`. Utilities exist to batch and select states during beam search.

## Joint network & TDT

- `nemo.collections.asr.modules.RNNTJoint` with `joint_hidden=640`, activation=ReLU, dropout=0.2.
- Input shapes: encoder `[B, T_enc, 1024]`, decoder `[B, U, 640]` → output logits `[B, T_enc, U, V+1+E]`.
- Vocabulary `V=8192` (SentencePiece BPE). RNNT blank adds `+1`. TDT uses `num_extra_outputs=E=5`, appended in the last dimension of the joint logits. The TDT loss splits the tail 5 entries as duration logits for buckets `[0,1,2,3,4]`.
- Training often enables `fuse_loss_wer=true` with `fused_batch_size=4`; export/inference use just `joint.joint(...)`.

## Auxiliary heads

- No separate CTC decoder is attached in the public v3 checkpoint.
- Loss is TDT: `fastemit_lambda=0.0`, `durations=[0,1,2,3,4]`, `sigma=0.02`, `omega=0.1`.

## Tokenizer

- SentencePiece BPE (`tokenizer.model`) with 8192 subword units plus control symbols (e.g., punctuation/timestamp markers). RNNT adds blank internally.

## Notes for CoreML export

- Exportable subgraphs map 1:1 to NeMo modules: preprocessor, encoder, decoder, joint, and (optionally) a greedy decoder. In addition, we export two fused variants for on‑device efficiency: (1) `mel_encoder` which combines preprocessor+encoder, and (2) `joint_decision` which applies split/softmax/argmax to joint logits to output `token_id`, `token_prob`, and `duration`.
- Decoder state I/O: expose `h` and `c` separately as `[L=2, B, 640]` multi‑arrays to enable on‑device streaming.
- TDT duration logits are appended to the class dimension; downstream code must split `[V+1]` token logits from the last `5` duration logits.

## Fixed 15‑second window (CoreML constraint)

- Audio: `B=1`, `T_audio = 15s × 16kHz = 240,000` samples.
- Mel frames (center=True): `T_mel ≈ floor(T_audio / 160) + 1 = 1,500 + 1 = 1,501` → mel `[1, 128, 1501]`.
- Encoder length (×8 subsampling): `T_enc = ceil(1501 / 8) = 188` → encoder `[1, 1024, 188]` (wrapper transposes to `[1, 188, 1024]`).
- Decoder step budget: choose `U_max` to cap per‑window decoding (e.g., 256–384). Joint outputs then shape `[1, T_enc, U_max, 8192+1+5]`.
