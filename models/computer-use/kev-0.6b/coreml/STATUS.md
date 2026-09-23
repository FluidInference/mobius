# Kev 0.6B status

- Trained Qwen3 + Kev adapter/head merged and exported to FP16 Core ML L128/32.
- W8 compression validated; both variants selected the native answer on 4/4 local Choice, Noul, and Score fixtures.
- Standalone Core ML host uses the published tokenizer and pinned configuration without loading native Qwen/adapter weights. It passed offline end-to-end Choice, Noul, and Score smoke requests; the weight-free renderer matched native input arrays on all four fixtures.
- [Separate public HF repository](https://huggingface.co/FluidInference/kev-0.6b-coreml) contains both packages, tokenizer, source, and model card.
- Full Decision Index and 2048 evaluation are outstanding; public tracker score is not claimed as a Core ML result.
