# CAM++ CoreML
CAM++ is a fast and efficient neural network for speaker verification that uses Context-Aware Masking to extract high-quality speaker embeddings. This repository contains a CoreML conversion of the CAM++ model for efficient inference on Apple devices.

## Model Details
- **Authors**: Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng, and Qian Chen
- **Organization**: Speech Lab, Alibaba Group
- **Paper**: [CAM++: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking](https://arxiv.org/abs/2303.00332)
- **Implementation**: [3D-Speaker Toolkit](https://github.com/modelscope/3D-Speaker/tree/main/speakerlab/models/campplus)
- **Weights**: [ModelScope Repo](https://modelscope.cn/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/)

CAM++ employs D-TDNN (Densely Connected Time Delay Neural Network) as its backbone and introduces a context-aware masking module to focus on the speaker of interest while suppressing unrelated noise. It achieves comparable performance to ECAPA-TDNN with significantly lower computational cost.

## CoreML Model I/O
#### Input Format: `(16, 150, 80)` tensor
- **16**: Batch size - processes 16 audio subsegments in parallel (each ~1.5 seconds)
- **150**: Number of frames - audio is divided into 25ms frames (400 samples @ 16kHz) with 10ms shift (160 samples @ 16kHz)
- **80**: Mel-filterbank features - log-transformed frequency bins (20-8000 Hz) extracted from each frame

#### Output Format: `(16, 192)` tensor
- **16**: Batch size
- **192**: Embedding dimension

## Usage
```bash
# Setup venv and install dependencies
uv sync

# Convert PyTorch model to CoreML
uv run convert.py

# Test CoreML model against reference implementation with real audio
uv run test.py audio/jfk.mp3
```