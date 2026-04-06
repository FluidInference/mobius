#!/usr/bin/env python3
"""Export .mlmodelc compiled versions"""
import coremltools as ct
from pathlib import Path
import shutil

models = [
    ("build/cohere_encoder.mlpackage", "hf-upload/cohere_encoder.mlmodelc"),
    ("build/cohere_decoder_cached.mlpackage", "hf-upload/cohere_decoder_cached.mlmodelc"),
    ("build/cohere_decoder_optimized.mlpackage", "hf-upload/cohere_decoder_optimized.mlmodelc"),
    ("build/cohere_cross_kv_projector.mlpackage", "hf-upload/cohere_cross_kv_projector.mlmodelc"),
]

for mlpackage_path, mlmodelc_path in models:
    print(f"\nCompiling {mlpackage_path}...")
    
    # Load model (triggers compilation)
    model = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    
    # The compiled model cache location
    spec = model.get_spec()
    
    # Get compiled model from CoreML
    import subprocess
    import tempfile
    
    # Use xcrun to compile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_mlmodelc = Path(tmpdir) / Path(mlpackage_path).with_suffix('.mlmodelc').name
        
        cmd = [
            'xcrun', 'coremlcompiler', 'compile',
            mlpackage_path,
            tmpdir
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Find the generated .mlmodelc
            mlmodelc_files = list(Path(tmpdir).glob('*.mlmodelc'))
            if mlmodelc_files:
                src = mlmodelc_files[0]
                dst = Path(mlmodelc_path)
                print(f"  Copying {src.name} -> {dst}")
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"  ✓ Created {dst}")
            else:
                print(f"  ⚠️  No .mlmodelc found in {tmpdir}")
        else:
            print(f"  ❌ Compilation failed: {result.stderr}")

print("\n✓ Done")
