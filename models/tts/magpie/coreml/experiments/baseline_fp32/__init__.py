"""fp32 baseline + tail-fp16 probe harness — diagnostic-only.

Currently houses the Option 1 tail-fp16 probe (`tail_fp16_probe.py`),
which builds a custom MIL pass that casts only late-stage HiFi-GAN
ops to fp16 while keeping the early stages fp32. The hypothesis (the
one configuration Phase F.2 didn't cover) is that tail noise lacks
downstream layers to amplify it audibly.

Outputs land in `coreml/build/fp32/nanocodec_tail_fp16_v{1,2,3}.mlpackage`
and are not uploaded to HuggingFace.
"""
