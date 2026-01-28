"""Inspect CoreML model outputs."""
import coremltools as ct

model = ct.models.MLModel('mimi_decoder_v2.mlpackage')
spec = model.get_spec()

print("=== INPUTS ===")
for inp in spec.description.input:
    print(f"  {inp.name}: {inp.type}")

print("\n=== OUTPUTS ===")
for out in spec.description.output:
    shape = list(out.type.multiArrayType.shape) if out.type.HasField('multiArrayType') else 'unknown'
    print(f"  {out.name}: {shape}")
