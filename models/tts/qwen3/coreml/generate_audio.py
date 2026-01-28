# Generate audio WAV output using CoreML pipeline
# CoreML prefill + CoreML decode + PyTorch code_predictor + PyTorch decoder
import torch
import torch.nn.functional as F
import numpy as np
import coremltools as ct
import soundfile as sf
from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
from convert_lm_prefill_v9 import MAX_TEXT_LENGTH
from convert_lm_decode_v4 import compute_decode_inputs
import warnings
import time
warnings.filterwarnings('ignore')

TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]
SAMPLE_RATE = 24000
MAX_CODEC_TOKENS = 125  # ~10 seconds at 12Hz

# Sampling parameters - match official defaults exactly
TEMPERATURE = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.05  # Official default

def sample_token(logits, suppress_mask, past_tokens=None, temperature=TEMPERATURE, top_k=TOP_K, rep_penalty=REPETITION_PENALTY):
    """Sample next token with temperature, top-k, and repetition penalty."""
    logits = logits.copy()
    logits[0, suppress_mask] = -float('inf')

    # Apply repetition penalty
    if past_tokens and len(past_tokens) > 0:
        for token in set(past_tokens[-20:]):  # Look at last 20 tokens
            if 0 <= token < logits.shape[1]:
                if logits[0, token] > 0:
                    logits[0, token] /= rep_penalty
                else:
                    logits[0, token] *= rep_penalty

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    logits_tensor = torch.from_numpy(logits)
    top_k_values, top_k_indices = torch.topk(logits_tensor, top_k, dim=-1)

    # Softmax over top-k
    probs = F.softmax(top_k_values, dim=-1)

    # Sample
    idx = torch.multinomial(probs, num_samples=1)
    token = top_k_indices[0, idx[0, 0]].item()

    return token

text = "Hello world, this is a test of the text to speech system."

print("=" * 60)
print("Audio Generation with CoreML Pipeline")
print("=" * 60)

print("\n1. Loading models...")
t0 = time.time()
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = tts_model.model.talker
config = talker.config
tokenizer = tts_model.processor.tokenizer
print(f"   PyTorch model: {time.time() - t0:.1f}s")

t0 = time.time()
prefill_coreml = ct.models.MLModel("qwen3_tts_lm_prefill_v9.mlpackage")
decode_coreml = ct.models.MLModel("qwen3_tts_lm_decode_v4.mlpackage")
# Use PyTorch decoder via speech_tokenizer (CoreML decoder has issues)
speech_tokenizer = tts_model.model.speech_tokenizer
print(f"   CoreML models: {time.time() - t0:.1f}s")

# Prepare inputs
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)
actual_len = text_len + 11  # role_prefix(3) + text + think(4) + lang(1) + bos(1) + pad(1) + eos(1)

print(f"\n2. Input text: '{text}'")
print(f"   Text tokens: {text_len}, Total sequence length: {actual_len}")

# Prepare embeddings
with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

speaker_embed = torch.from_numpy(np.load("speaker_embedding_official.npy").reshape(1, 1024)).float()

# === Prefill ===
print("\n3. Running Prefill (CoreML)...")
t0 = time.time()

role_ids = np.array([ROLE_PREFIX], dtype=np.int32)
text_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
text_ids[0, :text_len] = np.array(text_ids_list, dtype=np.int32)
text_length = np.array([text_len], dtype=np.int32)

prefill_out = prefill_coreml.predict({
    'role_ids': role_ids,
    'text_ids': text_ids,
    'text_length': text_length,
    'tts_bos_embed': tts_bos_embed.numpy().astype(np.float32),
    'tts_pad_embed': tts_pad_embed.numpy().astype(np.float32),
    'tts_eos_embed': tts_eos_embed.numpy().astype(np.float32),
    'speaker_embed': speaker_embed.numpy().astype(np.float32),
})

logits = prefill_out['logits']
kv_cache = prefill_out['kv_cache'][:, :, :, :actual_len, :]
past_hidden = prefill_out['past_hidden']

prefill_time = time.time() - t0
print(f"   Prefill time: {prefill_time * 1000:.1f}ms")

# Convert to torch for code_predictor
kv_cache = torch.from_numpy(kv_cache)
past_hidden = torch.from_numpy(past_hidden)

# Get first token (with sampling)
suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

first_token = sample_token(logits, suppress_mask, past_tokens=[])

# === Decode Loop ===
print("\n4. Running Decode Loop (CoreML + PyTorch code_predictor)...")
t0 = time.time()

codebook0_tokens = [first_token]
all_codebooks = []  # Will store [16, seq_len] for decoder
pos = actual_len

# First token's code_predictor output
token_id = torch.tensor([[first_token]], dtype=torch.long)
with torch.no_grad():
    # Get codebook 0 embedding
    last_id_hidden = talker.model.codec_embedding(token_id)

    # Run code_predictor
    predictor_input = torch.cat([past_hidden, last_id_hidden], dim=1)
    predictor_result = talker.code_predictor.generate(
        inputs_embeds=predictor_input,
        max_new_tokens=config.num_code_groups - 1,  # 15
        do_sample=True,  # Required for proper audio
        temperature=0.9,
        top_k=50,
        return_dict_in_generate=True,
    )

    # Store all 16 codebooks for first token
    first_codes = [first_token]
    for i in range(config.num_code_groups - 1):
        first_codes.append(predictor_result.sequences[0, i].item())
    all_codebooks.append(first_codes)

# Decode loop
for step in range(MAX_CODEC_TOKENS - 1):
    token_id = torch.tensor([[codebook0_tokens[-1]]], dtype=torch.long)

    # Compute inputs_embeds using PyTorch code_predictor
    with torch.no_grad():
        inputs_embeds = compute_decode_inputs(talker, token_id, past_hidden, tts_pad_embed)

    # Run CoreML decode
    out = decode_coreml.predict({
        'inputs_embeds': inputs_embeds.numpy().astype(np.float32),
        'kv_cache': kv_cache.numpy().astype(np.float32),
        'position': np.array([pos], dtype=np.int32),
    })

    logits = out['logits']
    kv_cache = torch.from_numpy(out['new_kv_cache'])
    past_hidden = torch.from_numpy(out['past_hidden'])

    # Get next token (with sampling + repetition penalty)
    next_token = sample_token(logits, suppress_mask, past_tokens=codebook0_tokens)

    if next_token == config.codec_eos_token_id:
        print(f"   EOS at step {step + 1}")
        break

    codebook0_tokens.append(next_token)
    pos += 1

    # Generate codebooks 1-15 for this token
    with torch.no_grad():
        next_token_id = torch.tensor([[next_token]], dtype=torch.long)
        last_id_hidden = talker.model.codec_embedding(next_token_id)

        predictor_input = torch.cat([past_hidden, last_id_hidden], dim=1)
        predictor_result = talker.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=config.num_code_groups - 1,
            do_sample=True,  # Required for proper audio
            temperature=0.9,
            top_k=50,
            return_dict_in_generate=True,
        )

        token_codes = [next_token]
        for i in range(config.num_code_groups - 1):
            token_codes.append(predictor_result.sequences[0, i].item())
        all_codebooks.append(token_codes)

    if (step + 1) % 20 == 0:
        print(f"   Step {step + 1}: token={next_token}")

decode_time = time.time() - t0
num_tokens = len(codebook0_tokens)
print(f"   Generated {num_tokens} tokens in {decode_time:.2f}s ({num_tokens/decode_time:.1f} tok/s)")

# === Decode to Audio ===
print("\n5. Decoding to Audio (PyTorch speech_tokenizer)...")
t0 = time.time()

# Stack codebooks: [num_tokens, 16]
codes_tensor = torch.tensor(all_codebooks, dtype=torch.int64)  # [num_tokens, 16]
print(f"   Codes shape: {codes_tensor.shape}")

# Decode using speech_tokenizer
with torch.no_grad():
    wavs, sr = speech_tokenizer.decode([{'audio_codes': codes_tensor}])

audio_out = wavs[0]
decoder_time = time.time() - t0
print(f"   Decoder time: {decoder_time * 1000:.1f}ms")
print(f"   Audio shape: {audio_out.shape}")
print(f"   Audio min/max: {audio_out.min():.4f}, {audio_out.max():.4f}")

duration = len(audio_out) / sr
print(f"   Audio duration: {duration:.2f}s")

# Save
output_path = "output_coreml.wav"
sf.write(output_path, audio_out, sr)
print(f"\n6. Saved: {output_path}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Text: '{text}'")
print(f"Tokens generated: {num_tokens}")
print(f"Audio duration: {duration:.2f}s")
print(f"Prefill: {prefill_time * 1000:.1f}ms")
print(f"Decode: {decode_time:.2f}s ({num_tokens/decode_time:.1f} tok/s)")
print(f"Vocoder: {decoder_time * 1000:.1f}ms")
print(f"Total: {prefill_time + decode_time + decoder_time:.2f}s")
print(f"\nOutput: {output_path}")
