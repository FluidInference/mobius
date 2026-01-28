"""Measure RAM usage of CoreML TTS models.

Measures:
1. Baseline process memory
2. Each CoreML model individually (load + predict)
3. All CoreML models loaded together
4. Peak memory during generation loop
"""
import os
import sys
import resource
import numpy as np
import coremltools as ct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_rss_mb():
    """Get current RSS (Resident Set Size) in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def get_rss_bytes():
    """Get current RSS in bytes using /proc or ps."""
    import subprocess
    pid = os.getpid()
    result = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True)
    return int(result.stdout.strip()) * 1024  # ps reports in KB


def fmt_mb(bytes_val):
    return f"{bytes_val / (1024 * 1024):.1f} MB"


def measure_model_size_on_disk():
    """Measure CoreML model sizes on disk."""
    print("=" * 60)
    print("MODEL SIZES ON DISK")
    print("=" * 60)

    models = [
        ('flowlm_step.mlpackage', 'FlowLM Step'),
        ('flow_decoder_v2.mlpackage', 'Flow Decoder'),
        ('mimi_decoder_v2.mlpackage', 'Mimi Decoder'),
    ]

    total = 0
    for filename, label in models:
        path = os.path.join(SCRIPT_DIR, filename)
        if os.path.exists(path):
            size = sum(
                os.path.getsize(os.path.join(dirpath, f))
                for dirpath, _, filenames in os.walk(path)
                for f in filenames
            )
            total += size
            print(f"  {label:20s}: {fmt_mb(size):>10s}")
        else:
            print(f"  {label:20s}: NOT FOUND")

    print(f"  {'TOTAL':20s}: {fmt_mb(total):>10s}")
    print()


def measure_ram():
    """Measure RAM usage at each stage."""
    print("=" * 60)
    print("RAM USAGE MEASUREMENT")
    print("=" * 60)

    # Baseline
    mem_baseline = get_rss_bytes()
    print(f"\n  Baseline (Python + imports): {fmt_mb(mem_baseline)}")

    # Load FlowLM Step
    print("\n  Loading FlowLM Step model...")
    coreml_step = ct.models.MLModel(
        os.path.join(SCRIPT_DIR, 'flowlm_step.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    mem_after_step = get_rss_bytes()
    print(f"  After FlowLM Step load:     {fmt_mb(mem_after_step)} (+{fmt_mb(mem_after_step - mem_baseline)})")

    # Load Flow Decoder
    print("\n  Loading Flow Decoder model...")
    coreml_flow = ct.models.MLModel(
        os.path.join(SCRIPT_DIR, 'flow_decoder_v2.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    mem_after_flow = get_rss_bytes()
    print(f"  After Flow Decoder load:    {fmt_mb(mem_after_flow)} (+{fmt_mb(mem_after_flow - mem_after_step)})")

    # Load Mimi Decoder
    print("\n  Loading Mimi Decoder model...")
    coreml_mimi = ct.models.MLModel(
        os.path.join(SCRIPT_DIR, 'mimi_decoder_v2.mlpackage'),
        compute_units=ct.ComputeUnit.CPU_AND_GPU
    )
    mem_after_mimi = get_rss_bytes()
    print(f"  After Mimi Decoder load:    {fmt_mb(mem_after_mimi)} (+{fmt_mb(mem_after_mimi - mem_after_flow)})")

    total_models = mem_after_mimi - mem_baseline
    print(f"\n  All 3 CoreML models total:  {fmt_mb(total_models)}")

    # Warmup predictions to trigger any lazy allocation
    print("\n  Running warmup predictions...")

    # FlowLM step warmup - shapes: sequence [1,1,32], bos_emb [32], cache [2,1,200,16,64], position [1]
    dummy_seq = np.full((1, 1, 32), 0.0, dtype=np.float32)
    dummy_bos = np.zeros((32,), dtype=np.float32)
    dummy_caches = {}
    dummy_positions = {}
    for i in range(6):
        dummy_caches[f'cache{i}'] = np.zeros((2, 1, 200, 16, 64), dtype=np.float32)
        dummy_positions[f'position{i}'] = np.array([10.0], dtype=np.float32)

    step_inputs = {'sequence': dummy_seq, 'bos_emb': dummy_bos, **dummy_caches, **dummy_positions}
    step_out = coreml_step.predict(step_inputs)
    mem_after_step_warmup = get_rss_bytes()
    print(f"  After FlowLM Step warmup:   {fmt_mb(mem_after_step_warmup)} (+{fmt_mb(mem_after_step_warmup - mem_after_mimi)})")

    # Flow decoder warmup
    flow_inputs = {
        'transformer_out': np.zeros((1, 1024), dtype=np.float32),
        'latent': np.zeros((1, 32), dtype=np.float32),
        's': np.array([[0.0]], dtype=np.float32),
        't': np.array([[0.125]], dtype=np.float32),
    }
    coreml_flow.predict(flow_inputs)
    mem_after_flow_warmup = get_rss_bytes()
    print(f"  After Flow Decoder warmup:  {fmt_mb(mem_after_flow_warmup)} (+{fmt_mb(mem_after_flow_warmup - mem_after_step_warmup)})")

    # Mimi decoder warmup - need proper state shapes
    from traceable_decoder import TraceableMimiDecoder
    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()
    pytorch_decoder = TraceableMimiDecoder.from_mimi(model.mimi)
    mimi_state = pytorch_decoder.init_state(batch_size=1)

    mimi_inputs = {
        'latent': np.zeros((1, 512, 1), dtype=np.float32),
        'upsample_partial': mimi_state['upsample_partial'].numpy().astype(np.float32),
        'attn0_cache': mimi_state['attn0_cache'].numpy().astype(np.float32),
        'attn0_offset': np.array([0.0], dtype=np.float32),
        'attn0_end_offset': np.array([0.0], dtype=np.float32),
        'attn1_cache': mimi_state['attn1_cache'].numpy().astype(np.float32),
        'attn1_offset': np.array([0.0], dtype=np.float32),
        'attn1_end_offset': np.array([0.0], dtype=np.float32),
        'conv0_prev': mimi_state['conv0_prev'].numpy().astype(np.float32),
        'conv0_first': mimi_state['conv0_first'].numpy().astype(np.float32),
        'convtr0_partial': mimi_state['convtr0_partial'].numpy().astype(np.float32),
        'res0_conv0_prev': mimi_state['res0_conv0_prev'].numpy().astype(np.float32),
        'res0_conv0_first': mimi_state['res0_conv0_first'].numpy().astype(np.float32),
        'res0_conv1_prev': mimi_state['res0_conv1_prev'].numpy().astype(np.float32),
        'res0_conv1_first': mimi_state['res0_conv1_first'].numpy().astype(np.float32),
        'convtr1_partial': mimi_state['convtr1_partial'].numpy().astype(np.float32),
        'res1_conv0_prev': mimi_state['res1_conv0_prev'].numpy().astype(np.float32),
        'res1_conv0_first': mimi_state['res1_conv0_first'].numpy().astype(np.float32),
        'res1_conv1_prev': mimi_state['res1_conv1_prev'].numpy().astype(np.float32),
        'res1_conv1_first': mimi_state['res1_conv1_first'].numpy().astype(np.float32),
        'convtr2_partial': mimi_state['convtr2_partial'].numpy().astype(np.float32),
        'res2_conv0_prev': mimi_state['res2_conv0_prev'].numpy().astype(np.float32),
        'res2_conv0_first': mimi_state['res2_conv0_first'].numpy().astype(np.float32),
        'res2_conv1_prev': mimi_state['res2_conv1_prev'].numpy().astype(np.float32),
        'res2_conv1_first': mimi_state['res2_conv1_first'].numpy().astype(np.float32),
        'conv_final_prev': mimi_state['conv_final_prev'].numpy().astype(np.float32),
        'conv_final_first': mimi_state['conv_final_first'].numpy().astype(np.float32),
    }
    coreml_mimi.predict(mimi_inputs)
    mem_after_mimi_warmup = get_rss_bytes()
    print(f"  After Mimi Decoder warmup:  {fmt_mb(mem_after_mimi_warmup)} (+{fmt_mb(mem_after_mimi_warmup - mem_after_flow_warmup)})")

    # Delete PyTorch model to isolate CoreML memory
    del model, pytorch_decoder, mimi_state
    import gc
    gc.collect()

    mem_after_gc = get_rss_bytes()
    print(f"\n  After deleting PyTorch model + GC: {fmt_mb(mem_after_gc)}")

    # Run a few generation steps to measure peak
    print("\n  Running 10 generation steps...")
    for i in range(10):
        step_out = coreml_step.predict(step_inputs)
        transformer_out_flat = step_out['input'].reshape(1, 1024)

        latent = np.random.randn(1, 32).astype(np.float32) * 0.8367
        for lsd_step in range(8):
            s_np = np.array([[lsd_step / 8.0]], dtype=np.float32)
            t_np = np.array([[(lsd_step + 1) / 8.0]], dtype=np.float32)
            flow_out = coreml_flow.predict({
                'transformer_out': transformer_out_flat,
                'latent': latent,
                's': s_np,
                't': t_np,
            })
            velocity = list(flow_out.values())[0]
            latent = latent + velocity / 8.0

        quantized = np.zeros((1, 512, 1), dtype=np.float32)
        coreml_mimi.predict(mimi_inputs)

    mem_after_gen = get_rss_bytes()
    print(f"  After 10 gen steps:         {fmt_mb(mem_after_gen)} (+{fmt_mb(mem_after_gen - mem_after_gc)})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  CoreML models loaded:       {fmt_mb(total_models)}")
    print(f"  After warmup + generation:  {fmt_mb(mem_after_gen - mem_baseline)}")
    print(f"  Peak RSS (whole process):   {fmt_mb(mem_after_gen)}")

    # Also report ru_maxrss (peak RSS tracked by OS)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS, ru_maxrss is in bytes
    print(f"  OS-reported peak RSS:       {peak_rss / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    measure_model_size_on_disk()
    measure_ram()
