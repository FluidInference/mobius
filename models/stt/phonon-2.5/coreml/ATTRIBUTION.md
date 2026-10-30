# Sources and attribution

- Model: Fermion Research, **Phonon-2.5**, derived from NVIDIA
  `parakeet-tdt-0.6b-v3`. Weights are CC-BY-4.0; retain the upstream weight
  license, NOTICE and modification attribution when distributing converted weights.
  <https://huggingface.co/FermionResearch/Phonon-2.5>
- `fermion_container.py`: Fermion Research's Apache-2.0 reader, as recovered
  unmodified (apart from its attribution header) from möbius PR #111. Its source
  hash is recorded in the file. Phonon-2.5 publishes the same reader.
- Five-value palette conversion and initial exporter recipes: Alex-Wengg /
  FluidInference, möbius PR #111, commit
  `e35c19537b4e70d298cbd8436f731e1717b3dcc9`, Apache-2.0.
  <https://github.com/FluidInference/mobius/pull/111>
- NeMo wrappers and HF-to-NeMo name mapping: FluidInference's existing
  parakeet-tdt-v3 and parakeet-redux conversion tools at möbius
  `26f5fa8059d5ab3d34449adce6ff6ed86734e522`, Apache-2.0.
- This adaptation adds direct, bounded-memory loading of Phonon-2.5, explicit
  CLI inputs, reproducibility metadata, malformed-container tests, an independent
  fp32 reference and hardware parity checks. Material AI assistance: OpenAI Codex.

No model weights or dataset audio are committed in this directory.
