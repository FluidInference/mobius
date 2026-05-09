"""StyleTTS2 → CoreML production exporter scripts.

Each module is a standalone CLI driver that builds one or more
`.mlpackage` artifacts under `coreml/packages/`. These are the
shipping-pipeline exporters; diagnostic / dead-end scripts live in
`coreml/experiments/`.

| Module                          | Builds                                          |
|---------------------------------|-------------------------------------------------|
| `convert`                       | per-stage packages (text_encoder, bert, …)      |
| `fuse_diffusion_sampler`        | Trial 4: fused 5-step ADPM2 sampler             |
| `fuse_f0n_har_source`           | Trial 6: fused f0n_predictor + har_source       |
| `build_buckets`                 | Trial 11: per-bucket bert + fused sampler       |

Importing `coreml.exporters.convert` also installs MIL backend patches
needed by the fused converters; the fused scripts import it for that
side effect. The diagnostic scripts under `coreml.experiments` do the
same import for the same reason.
"""
