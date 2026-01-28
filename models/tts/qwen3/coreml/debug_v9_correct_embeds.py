# Build inputs_embeds exactly like V9 and compare hidden states
import torch
import numpy as np
from qwen_tts import Qwen3TTSModel
from convert_lm_prefill_v9 import TracablePrefillV9, MAX_TEXT_LENGTH
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)

TTS_BOS_TOKEN_ID = 151672
TTS_PAD_TOKEN_ID = 151671
TTS_EOS_TOKEN_ID = 151673
ROLE_PREFIX = [151644, 77091, 198]

text = "Hello world, this is a test."

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = tts_model.model.talker
config = talker.config
tokenizer = tts_model.processor.tokenizer

text_ids_list = tokenizer.encode(text, add_special_tokens=False)
text_len = len(text_ids_list)
actual_len = text_len + 11  # 8 + 11 = 19

with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

speaker_embed = torch.from_numpy(np.load("speaker_embedding_official.npy").reshape(1, 1024)).float()

# Build inputs_embeds EXACTLY like V9
with torch.no_grad():
    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
    text_ids_tensor = torch.tensor([text_ids_list], dtype=torch.long)

    # Role embedding
    role_embed = talker.text_projection(talker.model.text_embedding(role_ids))

    # Text embedding
    text_embed = talker.text_projection(talker.model.text_embedding(text_ids_tensor))

    # Get codec IDs from config
    codec_think_id = config.codec_think_id
    codec_think_bos_id = config.codec_think_bos_id
    english_language_id = config.codec_language_id["english"]  # English language ID
    codec_think_eos_id = config.codec_think_eos_id
    codec_pad_id = config.codec_pad_id
    codec_bos_id = config.codec_bos_id

    print(f"codec_think_id: {codec_think_id}")
    print(f"codec_think_bos_id: {codec_think_bos_id}")
    print(f"english_language_id: {english_language_id}")
    print(f"codec_think_eos_id: {codec_think_eos_id}")
    print(f"codec_pad_id: {codec_pad_id}")
    print(f"codec_bos_id: {codec_bos_id}")

    # Position 3-6: tts_pad + codec_think_embeds
    codec_think_ids = torch.tensor([[codec_think_id, codec_think_bos_id, english_language_id, codec_think_eos_id]])
    codec_think_embeds = talker.model.codec_embedding(codec_think_ids)

    # Position 8: tts_bos + codec_pad (NOT codec_bos!)
    codec_pad_embed = talker.model.codec_embedding(torch.tensor([[codec_pad_id]]))

    # Position 10+text_len: tts_pad + codec_bos
    codec_bos_embed = talker.model.codec_embedding(torch.tensor([[codec_bos_id]]))

    # Build the sequence (actual_len = text_len + 11 = 19)
    inputs_embeds = torch.zeros(1, actual_len, 1024)

    # Position 0-2: Role
    inputs_embeds[:, 0:3, :] = role_embed

    # Position 3-6: tts_pad + codec_think
    inputs_embeds[:, 3:7, :] = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds

    # Position 7: tts_pad + speaker
    inputs_embeds[:, 7:8, :] = tts_pad_embed + speaker_embed.unsqueeze(1)

    # Position 8: tts_bos + codec_pad
    inputs_embeds[:, 8:9, :] = tts_bos_embed + codec_pad_embed

    # Position 9 to 9+text_len-1: text[i] + codec_pad
    for i in range(text_len):
        inputs_embeds[:, 9+i:10+i, :] = text_embed[:, i:i+1, :] + codec_pad_embed

    # Position 9+text_len: tts_eos + codec_pad
    inputs_embeds[:, 9+text_len:10+text_len, :] = tts_eos_embed + codec_pad_embed

    # Position 10+text_len: tts_pad + codec_bos
    inputs_embeds[:, 10+text_len:11+text_len, :] = tts_pad_embed + codec_bos_embed

print(f"\nManual inputs_embeds shape: {inputs_embeds.shape}")

# Run through talker.model directly with correct position_ids
position_ids = torch.arange(actual_len).view(1, -1).expand(1, -1)
position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

with torch.no_grad():
    outputs = talker.model(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        use_cache=True,
    )
    direct_hidden = outputs.last_hidden_state
    direct_last_hidden = direct_hidden[:, -1:, :]

print(f"Direct last hidden: mean={direct_last_hidden.mean().item():.6f}, std={direct_last_hidden.std().item():.6f}")

# Run V9 prefill
prefill_wrapper = TracablePrefillV9(talker)
prefill_wrapper.eval()

role_ids_input = torch.tensor([ROLE_PREFIX], dtype=torch.long)
text_ids_input = torch.zeros((1, MAX_TEXT_LENGTH), dtype=torch.long)
text_ids_input[0, :text_len] = torch.tensor(text_ids_list)
text_length = torch.tensor([text_len], dtype=torch.long)

with torch.no_grad():
    v9_logits, v9_kv_cache, v9_past_hidden = prefill_wrapper(
        role_ids_input, text_ids_input, text_length,
        tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed
    )

print(f"V9 past_hidden: mean={v9_past_hidden.mean().item():.6f}, std={v9_past_hidden.std().item():.6f}")

# Compare
diff = (direct_last_hidden - v9_past_hidden).abs().max().item()
print(f"\nMax diff between direct and V9: {diff:.6f}")

if diff < 0.001:
    print("MATCH! inputs_embeds are constructed correctly in V9.")
else:
    print("MISMATCH! V9 constructs inputs_embeds differently.")

    # Let's compare position by position
    print("\n=== Position-by-position comparison ===")
    for pos in range(actual_len):
        manual_pos = inputs_embeds[0, pos, :]
        # We need to extract V9's inputs from its hidden_states but before transformer
        # Actually, we can't directly get V9's inputs_embeds from output
        pass

    # Instead, let's check if V9's hidden_states_0 matches inputs_embeds
    # They should be identical since hidden_states_0 = inputs_embeds before any layer

# Let me trace V9 to get the actual inputs_embeds it uses
print("\n=== Tracing V9 to extract inputs_embeds ===")

# Modify wrapper to expose inputs
class DebugV9(TracablePrefillV9):
    def get_inputs_embeds(self, role_ids, text_ids, text_length, tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed):
        batch_size = role_ids.shape[0]
        device = role_ids.device
        text_len_val = text_length[0].item()

        hidden_states = torch.zeros(batch_size, actual_len, self.hidden_size, device=device, dtype=tts_bos_embed.dtype)

        # Position 0-2: Role
        role_embed = self.text_projection(self.text_embedding(role_ids))
        hidden_states[:, 0:3, :] = role_embed

        # Position 3-6: tts_pad + codec_think
        codec_think_ids = torch.tensor([
            [self.codec_think_id, self.codec_think_bos_id, self.english_language_id, self.codec_think_eos_id]
        ], dtype=torch.long, device=device)
        codec_think_embeds = self.codec_embedding(codec_think_ids)
        hidden_states[:, 3:7, :] = tts_pad_embed.expand(-1, 4, -1) + codec_think_embeds

        # Position 7: tts_pad + speaker
        hidden_states[:, 7:8, :] = tts_pad_embed + speaker_embed.unsqueeze(1)

        # Position 8: tts_bos + codec_pad
        codec_pad_embed = self.codec_embedding(
            torch.tensor([[self.codec_pad_id]], dtype=torch.long, device=device)
        )
        hidden_states[:, 8:9, :] = tts_bos_embed + codec_pad_embed

        # Position 9 to 9+text_len-1: text + codec_pad
        text_embed = self.text_projection(self.text_embedding(text_ids[:, :text_len_val]))
        for i in range(text_len_val):
            hidden_states[:, 9+i:10+i, :] = text_embed[:, i:i+1, :] + codec_pad_embed

        # Position 9+text_len: eos + codec_pad
        hidden_states[:, 9+text_len_val:10+text_len_val, :] = tts_eos_embed + codec_pad_embed

        # Position 10+text_len: tts_pad + codec_bos
        codec_bos_embed = self.codec_embedding(
            torch.tensor([[self.codec_bos_id]], dtype=torch.long, device=device)
        )
        hidden_states[:, 10+text_len_val:11+text_len_val, :] = tts_pad_embed + codec_bos_embed

        return hidden_states[:, :actual_len, :]

debug_v9 = DebugV9(talker)
debug_v9.eval()

with torch.no_grad():
    v9_inputs_embeds = debug_v9.get_inputs_embeds(
        role_ids_input, text_ids_input, text_length,
        tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed
    )

print(f"V9 inputs_embeds shape: {v9_inputs_embeds.shape}")

# Compare inputs_embeds
ie_diff = (inputs_embeds - v9_inputs_embeds).abs().max().item()
print(f"Max diff between manual and V9 inputs_embeds: {ie_diff:.6f}")

if ie_diff < 0.001:
    print("inputs_embeds MATCH!")
else:
    print("inputs_embeds MISMATCH!")
    # Find where
    for pos in range(actual_len):
        pos_diff = (inputs_embeds[0, pos, :] - v9_inputs_embeds[0, pos, :]).abs().max().item()
        if pos_diff > 0.001:
            print(f"  Position {pos}: max diff = {pos_diff:.6f}")
