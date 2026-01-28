# Generate audio with best-of-n sampling for improved prosody
# Generate N samples, pick the one with highest prosody score
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
import librosa
warnings.filterwarnings('ignore')

TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]
SAMPLE_RATE = 24000
MAX_CODEC_TOKENS = 125

# Sampling parameters
TEMPERATURE = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.05

# Best-of-N parameter
N_CANDIDATES = 5

def compute_prosody_score(audio, sr=24000):
    """Compute prosody score (higher = more expressive)."""
    if len(audio) < sr * 0.5:  # Too short
        return 0.0

    # Energy variation (dB)
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    energy_std = np.std(rms_db)

    # Pitch variation
    f0, voiced_flag, _ = librosa.pyin(audio, fmin=50, fmax=500, sr=sr)
    f0_voiced = f0[~np.isnan(f0)]
    if len(f0_voiced) > 10:
        pitch_cv = np.std(f0_voiced) / (np.mean(f0_voiced) + 1e-10) * 100
    else:
        pitch_cv = 0.0

    # Combined score (weighted average)
    score = energy_std * 0.5 + pitch_cv * 0.5
    return score

def sample_token(logits, suppress_mask, past_tokens=None, temperature=TEMPERATURE, top_k=TOP_K, rep_penalty=REPETITION_PENALTY):
    """Sample next token with temperature, top-k, and repetition penalty."""
    logits = logits.copy()
    logits[0, suppress_mask] = -float('inf')

    # Apply repetition penalty
    if past_tokens and len(past_tokens) > 0:
        for token in set(past_tokens[-20:]):
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

def generate_audio_candidate(prefill_coreml, decode_coreml, talker, config, prefill_inputs,
                            tts_pad_embed, suppress_mask, actual_len, speech_tokenizer, seed=None):
    """Generate one audio candidate."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Run prefill
    prefill_out = prefill_coreml.predict(prefill_inputs)
    logits = prefill_out['logits']
    kv_cache = prefill_out['kv_cache'][:, :, :, :actual_len, :]
    past_hidden = prefill_out['past_hidden']

    kv_cache = torch.from_numpy(kv_cache)
    past_hidden = torch.from_numpy(past_hidden)

    # Get first token
    first_token = sample_token(logits, suppress_mask, past_tokens=[])

    # Decode loop
    codebook0_tokens = [first_token]
    all_codebooks = []
    pos = actual_len

    # First token's code_predictor output
    token_id = torch.tensor([[first_token]], dtype=torch.long)
    with torch.no_grad():
        last_id_hidden = talker.model.codec_embedding(token_id)
        predictor_input = torch.cat([past_hidden, last_id_hidden], dim=1)
        predictor_result = talker.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=config.num_code_groups - 1,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            return_dict_in_generate=True,
        )
        first_codes = [first_token]
        for i in range(config.num_code_groups - 1):
            first_codes.append(predictor_result.sequences[0, i].item())
        all_codebooks.append(first_codes)

    # Decode loop
    for step in range(MAX_CODEC_TOKENS - 1):
        token_id = torch.tensor([[codebook0_tokens[-1]]], dtype=torch.long)

        with torch.no_grad():
            inputs_embeds = compute_decode_inputs(talker, token_id, past_hidden, tts_pad_embed)

        out = decode_coreml.predict({
            'inputs_embeds': inputs_embeds.numpy().astype(np.float32),
            'kv_cache': kv_cache.numpy().astype(np.float32),
            'position': np.array([pos], dtype=np.int32),
        })

        logits = out['logits']
        kv_cache = torch.from_numpy(out['new_kv_cache'])
        past_hidden = torch.from_numpy(out['past_hidden'])

        next_token = sample_token(logits, suppress_mask, past_tokens=codebook0_tokens)

        if next_token == config.codec_eos_token_id:
            break

        codebook0_tokens.append(next_token)
        pos += 1

        # Early stopping if too repetitive (low diversity)
        if len(codebook0_tokens) >= 20:
            unique_ratio = len(set(codebook0_tokens)) / len(codebook0_tokens)
            if unique_ratio < 0.15:  # Less than 15% unique tokens
                break  # Abort this candidate

        with torch.no_grad():
            next_token_id = torch.tensor([[next_token]], dtype=torch.long)
            last_id_hidden = talker.model.codec_embedding(next_token_id)
            predictor_input = torch.cat([past_hidden, last_id_hidden], dim=1)
            predictor_result = talker.code_predictor.generate(
                inputs_embeds=predictor_input,
                max_new_tokens=config.num_code_groups - 1,
                do_sample=True,
                temperature=0.9,
                top_k=50,
                return_dict_in_generate=True,
            )
            token_codes = [next_token]
            for i in range(config.num_code_groups - 1):
                token_codes.append(predictor_result.sequences[0, i].item())
            all_codebooks.append(token_codes)

    # Decode to audio
    codes_tensor = torch.tensor(all_codebooks, dtype=torch.int64)
    with torch.no_grad():
        wavs, sr = speech_tokenizer.decode([{'audio_codes': codes_tensor}])

    audio = wavs[0]
    return audio, codebook0_tokens

text = "Hello world, this is a test of the text to speech system."

print("=" * 60)
print(f"Best-of-{N_CANDIDATES} Audio Generation")
print("=" * 60)

print("\n1. Loading models...")
t0 = time.time()
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = tts_model.model.talker
config = talker.config
tokenizer = tts_model.processor.tokenizer
speech_tokenizer = tts_model.model.speech_tokenizer
print(f"   PyTorch model: {time.time() - t0:.1f}s")

t0 = time.time()
prefill_coreml = ct.models.MLModel("qwen3_tts_lm_prefill_v9.mlpackage")
decode_coreml = ct.models.MLModel("qwen3_tts_lm_decode_v4.mlpackage")
print(f"   CoreML models: {time.time() - t0:.1f}s")

# Prepare inputs
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)
actual_len = text_len + 11

print(f"\n2. Input text: '{text}'")

# Prepare embeddings
with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

speaker_embed = torch.from_numpy(np.load("speaker_embedding_official.npy").reshape(1, 1024)).float()

# Prepare prefill inputs
role_ids = np.array([ROLE_PREFIX], dtype=np.int32)
text_ids = np.zeros((1, MAX_TEXT_LENGTH), dtype=np.int32)
text_ids[0, :text_len] = np.array(text_ids_list, dtype=np.int32)
text_length = np.array([text_len], dtype=np.int32)

prefill_inputs = {
    'role_ids': role_ids,
    'text_ids': text_ids,
    'text_length': text_length,
    'tts_bos_embed': tts_bos_embed.numpy().astype(np.float32),
    'tts_pad_embed': tts_pad_embed.numpy().astype(np.float32),
    'tts_eos_embed': tts_eos_embed.numpy().astype(np.float32),
    'speaker_embed': speaker_embed.numpy().astype(np.float32),
}

suppress_mask = np.zeros(config.vocab_size, dtype=bool)
suppress_mask[2048:] = True
suppress_mask[config.codec_eos_token_id] = False

# Generate N candidates
print(f"\n3. Generating {N_CANDIDATES} candidates...")
candidates = []

for i in range(N_CANDIDATES):
    seed = 42 + i * 1000
    t0 = time.time()
    audio, tokens = generate_audio_candidate(
        prefill_coreml, decode_coreml, talker, config, prefill_inputs,
        tts_pad_embed, suppress_mask, actual_len, speech_tokenizer, seed=seed
    )
    gen_time = time.time() - t0

    prosody = compute_prosody_score(audio, SAMPLE_RATE)
    candidates.append({
        'audio': audio,
        'tokens': tokens,
        'prosody': prosody,
        'seed': seed,
        'time': gen_time,
    })

    unique_tokens = len(set(tokens))
    print(f"   Candidate {i+1}: prosody={prosody:.1f}, tokens={len(tokens)}, unique={unique_tokens}, time={gen_time:.1f}s")

# Select best
best = max(candidates, key=lambda x: x['prosody'])
print(f"\n4. Best candidate: seed={best['seed']}, prosody={best['prosody']:.1f}")

# Save best
output_path = "output_coreml_best.wav"
sf.write(output_path, best['audio'], SAMPLE_RATE)
print(f"   Saved: {output_path}")

# Also save all candidates for comparison
for i, c in enumerate(candidates):
    sf.write(f"output_candidate_{i+1}.wav", c['audio'], SAMPLE_RATE)

# Compare with official
print("\n5. Generating official reference...")
speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024)
voice_clone_prompt = {
    'ref_spk_embedding': [torch.from_numpy(speaker_embed_np.squeeze(0))],
    'x_vector_only_mode': [True],
    'icl_mode': [False],
    'ref_code': None,
}

input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]

# Generate multiple official samples for fair comparison
print("   Generating 5 official samples for comparison...")
official_prosodys = []
for i in range(5):
    torch.manual_seed(42 + i * 1000)
    with torch.no_grad():
        result = tts_model.model.generate(
            input_ids=[full_input_ids],
            languages=['english'],
            voice_clone_prompt=voice_clone_prompt,
            non_streaming_mode=True,
            max_new_tokens=125,
            do_sample=True,
            temperature=0.9,
            top_k=50,
        )

    codes = result[0][0]
    tokenizer_model = Qwen3TTSTokenizer.from_pretrained("./tokenizer_12hz", device_map="cpu")
    codes_for_decoder = codes.T.unsqueeze(0)
    with torch.no_grad():
        audio = tokenizer_model.model.decoder(codes_for_decoder)
    audio_np = audio[0, 0].numpy()
    prosody = compute_prosody_score(audio_np, SAMPLE_RATE)
    official_prosodys.append(prosody)
    print(f"   Official {i+1}: prosody={prosody:.1f}")

official_best = max(official_prosodys)
official_avg = np.mean(official_prosodys)

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Text: '{text}'")
print(f"\nCoreML Best-of-{N_CANDIDATES}:")
print(f"  Best prosody: {best['prosody']:.1f}")
print(f"  All prosodys: {[round(c['prosody'], 1) for c in candidates]}")
print(f"\nOfficial (5 samples):")
print(f"  Best prosody: {official_best:.1f}")
print(f"  Avg prosody: {official_avg:.1f}")
print(f"  All prosodys: {[round(p, 1) for p in official_prosodys]}")
print(f"\nCoreML best vs Official best: {best['prosody']/official_best*100:.1f}%")
print(f"CoreML best vs Official avg: {best['prosody']/official_avg*100:.1f}%")
print(f"\nOutput: {output_path}")
