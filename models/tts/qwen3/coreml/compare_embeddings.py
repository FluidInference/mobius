# Compare embeddings between V9 wrapper and official model
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
processor = tts_model.processor
talker = tts_model.model.talker

# Special token IDs
TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]

text = "Hello world, this is a test of the text to speech system."
tokenizer = processor.tokenizer
text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)

# Load speaker embedding
speaker_embed = np.load("speaker_embedding_official.npy").reshape(1, 1024)
speaker_embed_t = torch.from_numpy(speaker_embed).to(torch.float32)

print(f"\nText: '{text}'")
print(f"Text tokens: {text_ids_list} ({text_len} total)")

# Build full input for official model
input_text = tts_model._build_assistant_text(text)
full_input_ids = tts_model._tokenize_texts([input_text])[0]
print(f"\nFull input IDs: {full_input_ids.tolist()}")

# Check what methods are available
print(f"\n=== Checking embedding methods ===")
print(f"talker has get_text_embeddings: {hasattr(talker, 'get_text_embeddings')}")
print(f"talker has get_input_embeddings: {hasattr(talker, 'get_input_embeddings')}")
print(f"talker.model has text_embedding: {hasattr(talker.model, 'text_embedding')}")

# Compare text embeddings
print(f"\n=== Comparing text embeddings ===")
with torch.no_grad():
    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)

    # Method 1: V9 uses talker.model.text_embedding
    embed1 = talker.model.text_embedding(role_ids)
    embed1_proj = talker.text_projection(embed1)
    print(f"V9 method (model.text_embedding): shape={embed1.shape}")
    print(f"V9 projected: {embed1_proj[0, 0, :5].tolist()}")

    # Method 2: Official uses talker.get_text_embeddings()
    text_embedding_fn = talker.get_text_embeddings()
    embed2 = text_embedding_fn(role_ids)
    embed2_proj = talker.text_projection(embed2)
    print(f"Official method (get_text_embeddings): shape={embed2.shape}")
    print(f"Official projected: {embed2_proj[0, 0, :5].tolist()}")

    # Compare
    diff = (embed1 - embed2).abs().max().item()
    print(f"Difference: {diff}")
    if diff < 1e-6:
        print("SAME!")
    else:
        print("DIFFERENT!")

# Compare codec embeddings
print(f"\n=== Comparing codec embeddings ===")
with torch.no_grad():
    # V9 uses talker.model.codec_embedding
    codec_pad_id = talker.config.codec_pad_id
    pad_ids = torch.tensor([[codec_pad_id]], dtype=torch.long)

    embed1 = talker.model.codec_embedding(pad_ids)
    print(f"V9 method (model.codec_embedding): {embed1[0, 0, :5].tolist()}")

    # Official uses talker.get_input_embeddings()
    input_embedding_fn = talker.get_input_embeddings()
    embed2 = input_embedding_fn(pad_ids)
    print(f"Official method (get_input_embeddings): {embed2[0, 0, :5].tolist()}")

    diff = (embed1 - embed2).abs().max().item()
    print(f"Difference: {diff}")
    if diff < 1e-6:
        print("SAME!")
    else:
        print("DIFFERENT!")

# Now let's trace the full prefill manually
print(f"\n=== Building official prefill sequence manually ===")
with torch.no_grad():
    config = talker.config

    # TTS embeddings
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]], dtype=torch.long)
    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]],
                device=talker.device,
                dtype=full_input_ids.dtype,
            )
        )
    ).chunk(3, dim=1)

    print(f"TTS BOS embed: {tts_bos_embed[0, 0, :5].tolist()}")
    print(f"TTS EOS embed: {tts_eos_embed[0, 0, :5].tolist()}")
    print(f"TTS PAD embed: {tts_pad_embed[0, 0, :5].tolist()}")

    # Role prefix embedding
    role_embed = talker.text_projection(
        talker.get_text_embeddings()(full_input_ids[:3].unsqueeze(0))
    )
    print(f"Role prefix embed (official): {role_embed[0, 0, :5].tolist()}")

    # Build codec think tokens
    language_id = config.codec_language_id["english"]
    codec_prefill_list = [[
        config.codec_think_id,
        config.codec_think_bos_id,
        language_id,
        config.codec_think_eos_id,
    ]]
    codec_input_emebdding_0 = talker.get_input_embeddings()(
        torch.tensor(codec_prefill_list, device=talker.device, dtype=full_input_ids.dtype)
    )
    print(f"Codec think embed[0]: {codec_input_emebdding_0[0, 0, :5].tolist()}")

    # Build codec pad/bos tokens
    codec_input_emebdding_1 = talker.get_input_embeddings()(
        torch.tensor([[config.codec_pad_id, config.codec_bos_id]], device=talker.device, dtype=full_input_ids.dtype)
    )
    print(f"Codec pad embed: {codec_input_emebdding_1[0, 0, :5].tolist()}")
    print(f"Codec bos embed: {codec_input_emebdding_1[0, 1, :5].tolist()}")

    # Full codec embedding with speaker
    codec_input_emebdding = torch.cat([
        codec_input_emebdding_0,  # 4 tokens
        speaker_embed_t.view(1, 1, -1),  # 1 token (speaker)
        codec_input_emebdding_1  # 2 tokens (pad, bos)
    ], dim=1)
    print(f"Full codec embedding shape: {codec_input_emebdding.shape}")
    print(f"Codec positions: [think, bos, lang, eos, speaker, pad, bos]")

    # Build _talker_input_embed (positions 3-8)
    # tts_pad_embed * (7-2) = 5 positions, then tts_bos_embed = 1 position
    # Total 6 TTS positions, added to codec[:, :-1] (6 codec positions)
    _talker_input_embed = torch.cat((
        tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),  # 5 tts_pad
        tts_bos_embed,  # 1 tts_bos
    ), dim=1) + codec_input_emebdding[:, :-1]  # 6 codec
    print(f"_talker_input_embed shape: {_talker_input_embed.shape}")
    print(f"Position 0 (tts_pad + think): {_talker_input_embed[0, 0, :5].tolist()}")
    print(f"Position 4 (tts_pad + speaker): {_talker_input_embed[0, 4, :5].tolist()}")
    print(f"Position 5 (tts_bos + pad): {_talker_input_embed[0, 5, :5].tolist()}")

    # Full prefill embed before text
    talker_input_embed = torch.cat((role_embed, _talker_input_embed), dim=1)
    print(f"\nPre-text talker_input_embed shape: {talker_input_embed.shape}")
    print(f"Position 0 (role): {talker_input_embed[0, 0, :5].tolist()}")
    print(f"Position 3 (tts_pad + think): {talker_input_embed[0, 3, :5].tolist()}")
    print(f"Position 7 (tts_pad + speaker): {talker_input_embed[0, 7, :5].tolist()}")
    print(f"Position 8 (tts_bos + pad): {talker_input_embed[0, 8, :5].tolist()}")

    # Add first text token (for streaming) then remove for non-streaming
    talker_input_embed = torch.cat([
        talker_input_embed,
        talker.text_projection(talker.get_text_embeddings()(full_input_ids[3:4].unsqueeze(0))) + codec_input_emebdding[:, -1:]
    ], dim=1)
    print(f"After first text: shape={talker_input_embed.shape}")

    # Remove for non-streaming
    talker_input_embed = talker_input_embed[:, :-1]
    print(f"After removal: shape={talker_input_embed.shape}")

    # Add full text + eos + final bos
    text_part = full_input_ids[3:-5]  # Text tokens
    print(f"Text part: {text_part.tolist()} ({len(text_part)} tokens)")

    text_embed = talker.text_projection(
        talker.get_text_embeddings()(text_part.unsqueeze(0))
    )
    text_with_eos = torch.cat([text_embed, tts_eos_embed], dim=1)

    # codec_pad for all text + eos positions
    codec_pad_for_text = talker.get_input_embeddings()(
        torch.tensor([[config.codec_pad_id] * (text_part.shape[0] + 1)], dtype=full_input_ids.dtype)
    )

    # Final bos position
    final_bos = tts_pad_embed + talker.get_input_embeddings()(
        torch.tensor([[config.codec_bos_id]], dtype=full_input_ids.dtype)
    )

    talker_input_embed = torch.cat([
        talker_input_embed,
        text_with_eos + codec_pad_for_text,
        final_bos
    ], dim=1)

    print(f"\nFinal talker_input_embed shape: {talker_input_embed.shape}")
    print(f"Expected: {3 + 6 + text_len + 2} = {3 + 6 + text_len + 2}")

    # Print position structure
    print(f"\nPosition structure:")
    print(f"  0-2: role prefix")
    print(f"  3-8: think + speaker + pad")
    print(f"  9-{9+text_len-1}: text")
    print(f"  {9+text_len}: eos")
    print(f"  {9+text_len+1}: final bos")
