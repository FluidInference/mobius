# Kokoro bilingual project continuation

For English–Mandarin work, read [project-handoff.md](project-handoff.md) before
planning changes, asking product-scope questions, or starting model execution.
It contains the shared requirements, artifact gaps, gate sequence, and runbook.

- Preserve settled scope: one female voice, English/Mandarin/code-switching,
  compact Kokoro v1.1-zh-compatible adaptation, FluidAudio/Core ML direction;
  no voice cloning or runtime multi-speaker expansion.
- Inspect the receiving environment's existing setup and approvals first. Ask
  only for genuinely missing inputs; do not make the user repeat settled goals.
- The source Mac is for demos/code preparation/lightweight tests, not official
  evaluation or training. This does not prohibit appropriately authorized work
  in the designated training/evaluation environment.
- Demo evidence is not an official baseline, training dataset, checkpoint-parity
  proof, or working trainer. Mark proposed/unimplemented components honestly.
- Fix and version frontend inputs before canonical training preprocessing.
  Require strict weight mapping, untouched-checkpoint reproduction, gradient
  coverage, and real-data micro-overfit evidence before a training pilot.
- Keep real recordings, heavy artifacts, secrets, and machine-specific paths
  out of Git. Do not create mock models or substitute synthetic audio for real
  training data. Real-model generated evaluation audio must be labeled as such.
- Do not infer authority to provision, spend, upload data, publish weights, or
  deploy from the existence of a runbook. Retain existing explicit approvals.
- Keep internal execution plans under ignored `.mobius/`; update the tracked
  handoff when shared requirements, readiness, or implementation status change.
- Preserve scoped instructions under `coreml/bilingual-demo/` when touching
  saved demo evidence. Validate documentation links and the offline checks;
  do not regenerate or rescore models for a documentation-only change.
