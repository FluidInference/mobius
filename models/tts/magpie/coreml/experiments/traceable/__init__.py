"""Experimental traceable wrappers for Magpie-TTS decoder variants.

These modules are dead-end probes (see ``coreml/PERF.md``); the
production traceable wrappers (``traceable_decoder_step``,
``traceable_decoder_prefill``, ``traceable_text_encoder``) live one
level up at ``coreml/traceable/``.

| Module                              | Verdict (per `coreml/PERF.md`)                        |
|-------------------------------------|-------------------------------------------------------|
| `traceable_decoder_step_n2.py`      | Trial 4a — N=2 unroll (DEAD-END; ANE rejects)         |
| `traceable_decoder_step_stateful.py`| STATEFUL MLState variant (DEAD-END; 2.2× regression)  |
"""
