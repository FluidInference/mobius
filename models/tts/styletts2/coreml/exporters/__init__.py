"""StyleTTS2 → CoreML exporter scripts.

Each module is a standalone CLI driver that builds one or more
`.mlpackage` artifacts under `coreml/packages/`.

| Module                          | Builds                                          |
|---------------------------------|-------------------------------------------------|
| `convert`                       | per-stage packages (text_encoder, bert, …)      |
| `fuse_diffusion_sampler`        | Trial 4: fused 5-step ADPM2 sampler             |
| `fuse_f0n_har_source`           | Trial 6: fused f0n_predictor + har_source       |
| `build_buckets`                 | Trial 11: per-bucket bert + fused sampler       |
| `trial10_decoder_upsample_fixed`| Trial 10: decoder_upsample fp32 fixed-shape probe |
| `trial10b_decoder_upsample_conv2d`| Trial 10b: decoder_upsample Conv1d→Conv2d rewrite |

Importing `coreml.exporters.convert` also installs MIL backend patches
needed by the fused/trial converters; the fused/trial scripts import
it for that side effect.
"""
