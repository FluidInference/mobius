# TDT Decoding Notes (RNNT + Duration Buckets)

Parakeet‑TDT extends RNNT by appending 5 duration logits to the joint’s class axis. These describe coarse duration buckets associated with an emitted symbol. This note summarizes how to consume them alongside standard RNNT greedy decoding.

## Joint output

- Shape: `logits[B, T_enc, U, V+1+5]` where:
  - `V=8192` BPE vocab, `+1` RNNT blank, `+5` TDT duration buckets.
- Split last dimension:
  - `token_logits = logits[..., :V+1]`
  - `duration_logits = logits[..., V+1:]  # shape [..., 5]`

## Greedy decoding sketch

- Initialize decoder state `(h, c)` and `u=0`; iterate over encoder time `t=0..T_enc-1`.
- At each `(t,u)`:
  - Either use raw joint logits: retrieve `token_logits[t, u]` and pick `k = argmax(token_logits)`.
  - Or call the fused `JointDecisionWrapper` CoreML model to get `token_id[t,u]`, `token_prob[t,u]`, and `duration[t,u]` directly.
  - If `k` is blank: advance `t += 1` (RNNT time advance), keep `u`.
  - Else (emit token): append token `k` to hypothesis, update decoder state via `DecoderWrapper` with that token, increment `u += 1`.
  - Duration (optional): `d = argmax(duration_logits[t, u_prev])`. Map `d ∈ {0..4}` to bucket durations per training config. Use to assign sub‑frame timestamp estimates.

## Using duration buckets

- Bucket IDs 0..4 correspond to the configured `tdt_durations = [0,1,2,3,4]` used during training.
- A simple heuristic for timestamps per emitted token:
  - Keep track of current mel/encoder frame index.
  - When emitting a token at `(t,u)`, associate bucket `d = argmax(duration_logits[t,u])`.
  - Convert bucket to time by multiplying with the encoder frame stride (`8× hop_length / sr = 8×0.01 s = 80 ms` per encoder step) or use mel frame stride if preferred.
- For better quality, prefer server‑side timestamp decoding identical to NeMo’s implementation and use duration buckets as a hint when post‑processing on device.

## Notes

- NeMo trains the joint with `num_extra_outputs=5`. These logits are not part of the token softmax and should not affect argmax over `V+1`.
- The model card claims accurate word/segment timestamps; TDT buckets contribute to that, but exact use may vary per downstream aligner.
- If durations are not needed, you can ignore the last 5 logits entirely.
