import sys, json, shutil
from pathlib import Path
import coremltools as ct
src = Path(sys.argv[1])   # coreml_2240ms dir (has *.mlpackage, encoder_int8.mlpackage, metadata, tokenizer)
dst = Path(sys.argv[2])   # output nemotron_coreml_2240ms dir
dst.mkdir(parents=True, exist_ok=True)
(dst / "encoder").mkdir(exist_ok=True)

def compile_to(pkg, out_mlmodelc):
    m = ct.models.MLModel(str(pkg), compute_units=ct.ComputeUnit.CPU_AND_NE)
    cpath = m.get_compiled_model_path()
    if out_mlmodelc.exists(): shutil.rmtree(out_mlmodelc)
    shutil.copytree(cpath, out_mlmodelc)
    print("compiled", out_mlmodelc.name)

compile_to(src / "preprocessor.mlpackage", dst / "preprocessor.mlmodelc")
compile_to(src / "decoder.mlpackage",      dst / "decoder.mlmodelc")
compile_to(src / "joint.mlpackage",        dst / "joint.mlmodelc")
compile_to(src / "encoder_int8.mlpackage", dst / "encoder" / "encoder_int8.mlmodelc")

# metadata + chunk_ms
md = json.loads((src / "metadata.json").read_text())
md["chunk_ms"] = md["chunk_mel_frames"] * 10
(dst / "metadata.json").write_text(json.dumps(md, indent=2))
shutil.copy(src / "tokenizer.json", dst / "tokenizer.json")
print("staged ->", dst)
