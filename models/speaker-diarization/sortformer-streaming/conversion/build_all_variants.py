"""
Rebuild all 6 Sortformer pipeline variants from current conversion code.
Fixes issue #726 (stale HF models carry an input==output BNNS alias built on torch 2.9.x).
Local only — no upload. Each variant: write config.py, run convert_to_coreml.py in a fresh
subprocess, compile the pipeline to its app-expected name, verify no alias + ANE load.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
# Env-overridable so the same driver builds fp16 and palettized sets:
#   PALETTIZE_NBITS=6 OUT_DIR=build_palettized_models python build_all_variants.py
PALETTIZE_NBITS = int(os.environ.get("PALETTIZE_NBITS", "0"))
OUT = os.path.join(HERE, os.environ.get("OUT_DIR", "build_fixed_models"))
os.makedirs(OUT, exist_ok=True)

CONFIG_TEMPLATE = '''class Config:
    chunk_len = {chunk_len}
    chunk_right_context = {chunk_right_context}
    chunk_left_context = {chunk_left_context}
    fifo_len = {fifo_len}
    spkcache_len = {spkcache_len}
    spkcache_update_period = {spkcache_update_period}

    # do not touch these
    subsampling_factor = 8
    sample_rate = 16000
    mel_window = 400
    mel_stride = 160
    frame_duration = 0.08

    chunk_frames = (chunk_len + chunk_right_context + chunk_left_context) * subsampling_factor
    coreml_audio_samples = (chunk_frames - 1) * mel_stride + mel_window
    preproc_feature_frames = chunk_len * subsampling_factor
    preproc_audio_hop = preproc_feature_frames * mel_stride
'''

CONFIGS = {
    "Default": dict(chunk_len=6, chunk_right_context=7, chunk_left_context=1,
                    fifo_len=40, spkcache_len=188, spkcache_update_period=31),
    "NvidiaLow": dict(chunk_len=6, chunk_right_context=7, chunk_left_context=1,
                      fifo_len=188, spkcache_len=188, spkcache_update_period=144),
    "NvidiaHigh": dict(chunk_len=340, chunk_right_context=40, chunk_left_context=1,
                       fifo_len=40, spkcache_len=188, spkcache_update_period=300),
    # Higher-throughput streaming: Default context, larger 25-frame chunk (~2s output
    # latency, ~4x RTFx of Default). Maps to Swift SortformerConfig.efficientV2_1.
    "Efficient": dict(chunk_len=25, chunk_right_context=7, chunk_left_context=1,
                      fifo_len=40, spkcache_len=188, spkcache_update_period=31),
}

# (config_key, model_version) -> final mlmodelc name expected by the app
VARIANTS = [
    ("Default", "v2.1", "Sortformer_v2.1"),
    ("Default", "v2", "Sortformer_v2"),
    ("NvidiaLow", "v2.1", "SortformerNvidiaLow_v2.1"),
    ("NvidiaLow", "v2", "SortformerNvidiaLow_v2"),
    ("NvidiaHigh", "v2.1", "SortformerNvidiaHigh_v2.1"),
    ("NvidiaHigh", "v2", "SortformerNvidiaHigh_v2"),
    ("Efficient", "v2.1", "SortformerEfficient_v2.1"),
]

MODEL_NAME = {
    "v2.1": "nvidia/diar_streaming_sortformer_4spk-v2.1",
    "v2": "nvidia/diar_streaming_sortformer_4spk-v2",
}


def write_config(cfg_key):
    with open(os.path.join(HERE, "config.py"), "w") as f:
        f.write(CONFIG_TEMPLATE.format(**CONFIGS[cfg_key]))


def verify(mlc):
    import coremltools as ct
    mil = open(os.path.join(mlc, "model1", "model.mil")).read().splitlines()[3]
    alias = "chunk_pre_encoder_embs_out" in mil
    m = ct.models.CompiledMLModel(mlc, ct.ComputeUnit.ALL)  # raises if it won't load on ANE
    return (not alias)


def main():
    results = []
    for cfg_key, ver, final_name in VARIANTS:
        print(f"\n{'='*70}\nBUILD {final_name}  (config={cfg_key}, model={ver})\n{'='*70}", flush=True)
        write_config(cfg_key)
        build_dir = os.path.join(HERE, f"build_v_{final_name}")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        cmd = [PY, "convert_to_coreml.py", "--model_name", MODEL_NAME[ver], "--output_dir", build_dir]
        if PALETTIZE_NBITS > 0:
            cmd += ["--palettize_head_nbits", str(PALETTIZE_NBITS)]
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        pkg = os.path.join(build_dir, "SortformerPipeline.mlpackage")
        if not os.path.exists(pkg):
            print(f"  FAILED: no pipeline produced.\n  stderr tail:\n{r.stderr[-1500:]}", flush=True)
            results.append((final_name, "BUILD_FAILED"))
            continue
        import coremltools as ct
        mlc = ct.utils.compile_model(pkg)
        dst = os.path.join(OUT, final_name + ".mlmodelc")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(mlc, dst)
        try:
            ok = verify(dst)
            status = "OK (no alias, ANE load)" if ok else "ALIAS STILL PRESENT"
        except Exception as e:
            status = f"VERIFY_FAILED: {type(e).__name__}: {str(e)[:120]}"
        sz = os.popen(f"du -sh '{dst}'").read().split()[0]
        print(f"  -> {dst}  [{sz}]  {status}", flush=True)
        results.append((final_name, status))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}", flush=True)
    for name, status in results:
        print(f"  {name:32s} {status}", flush=True)


if __name__ == "__main__":
    main()
