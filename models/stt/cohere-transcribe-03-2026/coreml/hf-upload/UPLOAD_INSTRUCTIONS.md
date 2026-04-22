# HuggingFace Upload Instructions

## Repository Ready for Upload

Directory: `cohere-transcribe-cache-external-coreml/`
Total size: **~7.3 GB**

## Files Included

```
cohere-transcribe-cache-external-coreml/
├── cohere_encoder.mlpackage           # 6.97 GB - Encoder model
├── cohere_decoder_cache_external.mlpackage  # 291 MB - Cache-external decoder
├── tokenizer.model                    # 481 KB - SentencePiece tokenizer
├── wer_results_cache_external.json    # 4 KB - WER test results
├── example.py                         # 5.8 KB - Example usage script
├── requirements.txt                   # 87 B - Python dependencies
├── .gitattributes                     # Git LFS configuration
└── README.md                          # 9.7 KB - Model card
```

## Upload Steps

### 1. Install HuggingFace CLI

```bash
pip install huggingface_hub[cli]
```

### 2. Login to HuggingFace

```bash
huggingface-cli login
```

Enter your HuggingFace access token when prompted.

### 3. Create Repository (if needed)

Option A: Via CLI
```bash
huggingface-cli repo create cohere-transcribe-cache-external-coreml --type model
```

Option B: Via Web
1. Go to https://huggingface.co/new
2. Repository name: `cohere-transcribe-cache-external-coreml`
3. Repository type: Model
4. License: cc-by-nc-4.0
5. Click "Create repository"

### 4. Clone Repository

```bash
git clone https://huggingface.co/FluidInference/cohere-transcribe-cache-external-coreml
cd cohere-transcribe-cache-external-coreml
```

### 5. Install Git LFS

```bash
git lfs install
```

### 6. Copy Files

```bash
# From this directory:
cp -r cohere-transcribe-cache-external-coreml/* /path/to/cloned/repo/
```

### 7. Track Large Files with LFS

```bash
cd /path/to/cloned/repo
git lfs track "*.mlpackage/**"
git lfs track "*.mlmodelc/**"
git lfs track "*.bin"
git lfs track "*.model"
```

### 8. Add and Commit

```bash
git add .
git commit -m "Initial upload: Cohere Transcribe Cache-External CoreML models

- Encoder: 6.97 GB (FP16)
- Decoder (cache-external): 291 MB
- Tokenizer: SentencePiece
- WER: 11.95% on LibriSpeech test-clean
- macOS 14+ / iOS 17+ compatible
- Correct EOS token (3, not 151643)
"
```

### 9. Push to HuggingFace

```bash
git push
```

Note: This will upload ~7.3 GB, may take some time depending on your connection.

## Alternative: Use huggingface_hub Python API

```python
from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="cohere-transcribe-cache-external-coreml",
    repo_id="FluidInference/cohere-transcribe-cache-external-coreml",
    repo_type="model",
    commit_message="Initial upload: Cache-external CoreML models"
)
```

## Post-Upload Checklist

- [ ] Verify all files uploaded correctly
- [ ] Check README.md renders properly on HuggingFace
- [ ] Test example.py with downloaded models
- [ ] Add model card tags if needed
- [ ] Link to original Cohere model
- [ ] Link to FluidAudio source code

## Model Card Preview

The README.md includes:
- ✅ Model description
- ✅ Architecture details
- ✅ Performance metrics (WER: 11.95%)
- ✅ Critical EOS token fix documentation
- ✅ Python usage example
- ✅ Swift usage reference
- ✅ Supported languages (14 total)
- ✅ Comparison with alternatives
- ✅ Citation
- ✅ License (CC-BY-NC-4.0)

## Key Features Highlighted

1. **Cache-External Pattern**: Parakeet-style external KV cache management
2. **Correct EOS Token**: Token 3 (not 151643) - critical fix documented
3. **macOS 14+ Compatible**: Works on older OS versions (vs stateful requiring 15+)
4. **Compilable to .mlmodelc**: For faster production loading
5. **O(n) Complexity**: Efficient decoding with cache
6. **Excellent WER**: 11.95% on LibriSpeech test-clean

## Repository URL

After upload, models will be available at:
```
https://huggingface.co/FluidInference/cohere-transcribe-cache-external-coreml
```

## Notes

- Large files use Git LFS automatically
- .gitattributes is configured for proper LFS tracking
- README.md will render as the model card on HuggingFace
- wer_results_cache_external.json provides detailed per-sample results
- example.py is a complete working example users can run immediately
