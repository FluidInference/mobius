#!/usr/bin/env python3
"""Investigate Chinese transcription issue."""
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import load_dataset

model_id = "CohereLabs/cohere-transcribe-03-2026"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    trust_remote_code=True
)
model.eval()

print("=" * 80)
print("CHINESE TRANSCRIPTION INVESTIGATION")
print("=" * 80)

# Load Chinese FLEURS samples
dataset = load_dataset("google/fleurs", "cmn_hans_cn", split="test", streaming=False)
samples = list(dataset)[:3]

for i, sample in enumerate(samples):
    audio = sample["audio"]["array"]
    reference = sample["transcription"]
    
    print(f"\n{'=' * 80}")
    print(f"Sample {i+1}")
    print(f"{'=' * 80}")
    print(f"Reference: {reference}")
    
    # Generate
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    batch_size = inputs["input_features"].shape[0]
    decoder_input_ids = torch.full(
        (batch_size, 1),
        model.generation_config.decoder_start_token_id,
        dtype=torch.long
    )
    
    with torch.no_grad():
        outputs = model.generate(
            input_features=inputs["input_features"],
            length=inputs.get("length"),
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=128,
        )
        hypothesis = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print(f"Generated: {hypothesis}")
    print(f"Match: {reference == hypothesis}")
    
    # Character comparison
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    print(f"\nReference length: {len(ref_chars)} chars")
    print(f"Generated length: {len(hyp_chars)} chars")
    
    # Show first 20 characters
    print(f"\nFirst 20 chars comparison:")
    print(f"Ref: {ref_chars[:20]}")
    print(f"Hyp: {hyp_chars[:20]}")

print("\n" + "=" * 80)
