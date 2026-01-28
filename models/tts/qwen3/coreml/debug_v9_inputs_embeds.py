# Debug V9's inputs_embeds construction
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
actual_len = text_len + 11

with torch.no_grad():
    tts_ids = torch.tensor([[TTS_BOS_TOKEN_ID, TTS_PAD_TOKEN_ID, TTS_EOS_TOKEN_ID]])
    tts_embed = talker.text_projection(talker.model.text_embedding(tts_ids))
    tts_bos_embed = tts_embed[:, 0:1, :]
    tts_pad_embed = tts_embed[:, 1:2, :]
    tts_eos_embed = tts_embed[:, 2:3, :]

speaker_embed = torch.from_numpy(np.load("speaker_embedding_official.npy").reshape(1, 1024)).float()

# Build inputs_embeds manually (matching V9 logic)
# Let me trace through V9's forward more carefully
print("\n=== Reading V9 prefill forward ===")
# V9 builds:
# 1. role_embed = text_projection(text_embedding(role_ids))  - 3 tokens
# 2. text_embed = text_projection(text_embedding(text_ids[:text_len])) - text_len tokens
# 3. codec_pad_embed = codec_embedding(codec_pad_id) - 1 token repeated
# 4. codec_bos_embed = codec_embedding(codec_bos_id) - 1 token
# 5. Combines them

# Let me check the V9 forward method to see exact structure
with torch.no_grad():
    role_ids = torch.tensor([ROLE_PREFIX], dtype=torch.long)
    text_ids_tensor = torch.tensor([text_ids_list], dtype=torch.long)

    role_embed = talker.text_projection(talker.model.text_embedding(role_ids))
    text_embed = talker.text_projection(talker.model.text_embedding(text_ids_tensor))

    codec_pad_id = config.codec_pad_id
    codec_bos_id = config.codec_bos_id
    codec_pad_embed = talker.model.codec_embedding(torch.tensor([[codec_pad_id]]))
    codec_bos_embed = talker.model.codec_embedding(torch.tensor([[codec_bos_id]]))
    speaker_codec_embed = speaker_embed.view(1, 1, -1)

# Read V9's actual forward to understand the construction
prefill_wrapper = TracablePrefillV9(talker)

# Modify V9 to expose inputs_embeds
class DebugPrefillV9(TracablePrefillV9):
    def forward_debug(self, role_ids, text_ids, text_length,
                      tts_bos_embed, tts_pad_embed, tts_eos_embed, speaker_embed):
        """Same as forward but returns inputs_embeds."""
        batch_size = role_ids.shape[0]
        text_len = text_length[0].item()

        # Role embedding
        role_embed = self.text_projection(self.text_embedding(role_ids))

        # Text embedding (only up to text_length)
        text_embed = self.text_projection(self.text_embedding(text_ids[:, :text_len]))

        # Codec embeddings
        codec_pad_embed = self.codec_embedding(
            torch.tensor([[self.config.codec_pad_id]], device=role_ids.device)
        )
        codec_bos_embed = self.codec_embedding(
            torch.tensor([[self.config.codec_bos_id]], device=role_ids.device)
        )
        speaker_codec_embed = speaker_embed.view(batch_size, 1, -1)

        # Build sequence:
        # [role] + [pad*4 + pad_codec]*4 + [pad + speaker_codec] + [bos + bos_codec]
        # + [text + pad_codec]*text_len + [eos + pad_codec] + [pad + bos_codec]

        seq_parts = []

        # Role (3 tokens)
        seq_parts.append(role_embed)

        # Pad*4 + pad_codec
        for _ in range(4):
            seq_parts.append(tts_pad_embed + codec_pad_embed)

        # Speaker + pad_codec (wait, should be speaker_codec not pad_codec?)
        # Let me check V9 actual code
        seq_parts.append(tts_pad_embed + speaker_codec_embed)

        # BOS + bos_codec
        seq_parts.append(tts_bos_embed + codec_bos_embed)

        # Text + pad_codec
        for i in range(text_len):
            seq_parts.append(text_embed[:, i:i+1, :] + codec_pad_embed)

        # EOS + pad_codec
        seq_parts.append(tts_eos_embed + codec_pad_embed)

        # Final: pad + bos_codec
        seq_parts.append(tts_pad_embed + codec_bos_embed)

        inputs_embeds = torch.cat(seq_parts, dim=1)
        return inputs_embeds

# Actually, let me just read the V9 code directly
from convert_lm_prefill_v9 import TracablePrefillV9

import inspect
v9_forward_source = inspect.getsource(TracablePrefillV9.forward)
print("V9 forward code (partial):")
# Find the inputs_embeds construction part
lines = v9_forward_source.split('\n')
for i, line in enumerate(lines):
    if 'inputs_embeds' in line or 'seq_parts' in line or 'codec_' in line or 'tts_' in line:
        print(f"{i}: {line}")
