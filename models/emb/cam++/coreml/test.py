"""
Test script to validate CoreML CAM++ embeddings model against reference PyTorch implementation.
    1. Takes an audio file as input
    2. Converts it to 16kHz mono 16-bit WAV if needed
    3. Extracts Fbank features
    4. Runs features through both CoreML and reference PyTorch models
    5. Compares embeddings using cosine similarity
    6. Performs speed comparison between CoreML model & torch model
"""
import sys
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import coremltools as ct
from pathlib import Path
import warnings
from camplusplus import CAMPPlus

warnings.filterwarnings('ignore')

def extract_fbank_features(waveform, sample_rate=16000):
    # In Python here just for testing/demonstration.
    # For production deployment, see fbank_extractor C++ code in the Senko diarization pipeline:
    #   https://github.com/narcotic-sh/senko/tree/main/senko/fbank_extractor

    # Ensure waveform is 2D (1, num_samples)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    fbank = kaldi.fbank(
        waveform,
        sample_frequency=sample_rate,
        num_mel_bins=80,
        frame_length=25.0,  # 25ms
        frame_shift=10.0,   # 10ms
        dither=0.0,
        preemphasis_coefficient=0.97,
        remove_dc_offset=True,
        window_type='povey',
        round_to_power_of_two=True,
        blackman_coeff=0.42,
        snip_edges=True,
        low_freq=20,
        high_freq=0,  # 0 means Nyquist (8000 Hz for 16kHz sampling)
        use_energy=False,
        energy_floor=1.0,
        raw_energy=True,
        use_log_fbank=True,
        use_power=True
    )

    # Mean normalization
    fbank = fbank - fbank.mean(dim=0, keepdim=True)

    return fbank.numpy()

def load_and_convert_audio(audio_path):
    """Load audio file and convert to 16kHz mono if needed"""
    # Load audio with torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    return waveform, sample_rate

def generate_subsegments(audio_duration):
    segment_duration = 1.5
    shift = segment_duration / 2.5  # 0.6 seconds

    subsegments = []
    start = 0.0

    while start + segment_duration <= audio_duration:
        subsegments.append((start, start + segment_duration))
        start += shift

    # Add final segment if needed
    if start < audio_duration:
        end = min(audio_duration, start + segment_duration)
        start = max(0, end - segment_duration)
        subsegments.append((start, end))

    return subsegments

def main(audio_path):
    # Load and convert audio
    print(f"\n1. Loading audio from: {audio_path}")
    waveform, sample_rate = load_and_convert_audio(audio_path)
    duration = waveform.shape[1] / sample_rate
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Sample rate: {sample_rate} Hz")

    # Generate subsegments
    print("\n2. Generating subsegments...")
    subsegments = generate_subsegments(duration)
    print(f"   Generated {len(subsegments)} subsegments")

    # Limit to 16 subsegments for batch processing
    if len(subsegments) > 16:
        subsegments = subsegments[:16]
        print(f"   Using first 16 subsegments for testing")

    # Extract features for each subsegment
    print("\n3. Extracting Fbank features...")
    features = []

    for start_sec, end_sec in subsegments:
        # Extract subsegment
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)

        if start_sample >= waveform.shape[1]:
            # Empty subsegment
            features.append(np.zeros((150, 80), dtype=np.float32))
            continue

        segment_waveform = waveform[:, start_sample:end_sample]

        # Ensure minimum length (400 samples = 25ms at 16kHz)
        if segment_waveform.shape[1] < 400:
            padded = torch.zeros(1, 400)
            padded[:, :segment_waveform.shape[1]] = segment_waveform
            segment_waveform = padded

        # Extract features
        segment_features = extract_fbank_features(segment_waveform, sample_rate)

        # Pad or crop to 150 frames
        if segment_features.shape[0] < 150:
            padded = np.zeros((150, 80), dtype=np.float32)
            padded[:segment_features.shape[0]] = segment_features
            segment_features = padded
        elif segment_features.shape[0] > 150:
            # Center crop
            start = (segment_features.shape[0] - 150) // 2
            segment_features = segment_features[start:start + 150]

        features.append(segment_features)

    # Pad to batch size 16 if needed
    while len(features) < 16:
        features.append(np.zeros((150, 80), dtype=np.float32))

    features_batch = np.array(features[:16], dtype=np.float32)  # Shape: (16, 150, 80)
    print(f"   Feature shape: {features_batch.shape}")

    # Load CoreML model
    print("\n4. Loading CoreML model...")
    BATCH_SIZE = 16
    coreml_model_path = Path(f"models/camplusplus_batch{BATCH_SIZE}.mlpackage")
    coreml_model = ct.models.MLModel(str(coreml_model_path))
    print(f"   Loaded: {coreml_model_path}")

    # Run CoreML inference
    print("\n5. Running CoreML inference...")
    coreml_output = coreml_model.predict({'input_features': features_batch})
    coreml_embeddings = coreml_output['embeddings']
    print(f"   CoreML embeddings shape: {coreml_embeddings.shape}")

    # Load reference PyTorch model
    print("\n6. Loading reference PyTorch model...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"   Using device: {device}")

    reference_model = CAMPPlus(feat_dim=80, embedding_size=192)

    # Load weights
    weights_path = "models/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"
    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    reference_model.load_state_dict(state_dict)
    print(f"   Loaded weights from: {weights_path}")

    reference_model.eval()
    reference_model.to(device)

    # Run reference model inference
    print("\n7. Running reference PyTorch inference...")
    features_torch = torch.from_numpy(features_batch).to(device)

    with torch.no_grad():
        reference_embeddings = reference_model(features_torch).cpu().numpy()

    print(f"   Reference embeddings shape: {reference_embeddings.shape}")

    # Compare embeddings
    print("\n8. Comparing embeddings...")
    print("-" * 40)

    cosine_similarities = []
    l2_distances = []

    # Only compare actual subsegments (not padding)
    num_actual = min(len(subsegments), 16)

    for i in range(num_actual):
        # Normalize embeddings
        coreml_norm = coreml_embeddings[i] / (np.linalg.norm(coreml_embeddings[i]) + 1e-8)
        ref_norm = reference_embeddings[i] / (np.linalg.norm(reference_embeddings[i]) + 1e-8)

        # Cosine similarity
        cosine_sim = np.dot(coreml_norm, ref_norm)
        cosine_similarities.append(cosine_sim)

        # L2 distance
        l2_dist = np.linalg.norm(coreml_embeddings[i] - reference_embeddings[i])
        l2_distances.append(l2_dist)

        print(f"   Subsegment {i+1:2d}: Cosine Sim = {cosine_sim:.6f}, L2 Dist = {l2_dist:.6f}")

    print("-" * 40)

    # Summary statistics
    avg_cosine = np.mean(cosine_similarities)
    min_cosine = np.min(cosine_similarities)
    max_cosine = np.max(cosine_similarities)
    std_cosine = np.std(cosine_similarities)

    avg_l2 = np.mean(l2_distances)
    max_l2 = np.max(l2_distances)

    print("\n9. Summary Statistics:")
    print(f"   Cosine Similarity:")
    print(f"     Average: {avg_cosine:.6f}")
    print(f"     Min:     {min_cosine:.6f}")
    print(f"     Max:     {max_cosine:.6f}")
    print(f"     Std Dev: {std_cosine:.6f}")
    print(f"   L2 Distance:")
    print(f"     Average: {avg_l2:.6f}")
    print(f"     Max:     {max_l2:.6f}")

    # Validation result
    print("\n10. Validation Result:")
    if avg_cosine > 0.999:
        print("    ✅ Near-perfect match between CoreML and PyTorch models")
    elif avg_cosine > 0.99:
        print("    ✅ Very high similarity between models")
    elif avg_cosine > 0.95:
        print("    ⚠️ Good similarity but some differences detected")
    else:
        print("    ❌ FAIL: Significant differences between models")

    # Speed comparison
    print("\n11. Speed Comparison:")
    print("-" * 40)

    import time

    # Warm up both models
    print("   Warming up models...")
    for _ in range(5):
        _ = coreml_model.predict({'input_features': features_batch})
        with torch.no_grad():
            _ = reference_model(features_torch)

    # Benchmark CoreML
    print("   Benchmarking CoreML...")
    num_runs = 20
    coreml_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = coreml_model.predict({'input_features': features_batch})
        coreml_times.append(time.perf_counter() - start)

    # Benchmark PyTorch on MPS
    print(f"   Benchmarking PyTorch ({device})...")
    torch_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = reference_model(features_torch)
        if device.type == 'mps':
            torch.mps.synchronize()  # Ensure MPS operations complete
        torch_times.append(time.perf_counter() - start)

    # Calculate statistics
    coreml_avg = np.mean(coreml_times) * 1000  # Convert to ms
    coreml_std = np.std(coreml_times) * 1000
    torch_avg = np.mean(torch_times) * 1000
    torch_std = np.std(torch_times) * 1000

    speedup = torch_avg / coreml_avg

    print(f"\n   CoreML (Neural Engine):")
    print(f"     Average: {coreml_avg:.2f} ms ± {coreml_std:.2f} ms")
    print(f"   PyTorch ({device}):")
    print(f"     Average: {torch_avg:.2f} ms ± {torch_std:.2f} ms")
    print(f"\n   Speedup: {speedup:.2f}x" + (" faster" if speedup > 1 else " slower") + " with CoreML")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <audio_file>")
        print("Example: python test.py sample.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    if not Path(audio_file).exists():
        print(f"Error: Audio file '{audio_file}' not found")
        sys.exit(1)

    main(audio_file)