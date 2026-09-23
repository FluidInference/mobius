"""Export GatedDeltaNet / FullAttention vs transformers' Qwen3.5 modules (shared random weights)."""

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

from qwen35_export import DecoderLayer, GatedDeltaNet, TextConfig, rope_cos_sin

CFG = {
    "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2,
    "layer_types": ["linear_attention", "full_attention"], "rms_norm_eps": 1e-6,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4, "vocab_size": 16,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                        "mrope_section": [3, 3, 2], "mrope_interleaved": True},
}


@pytest.mark.parametrize("layer_idx", [0, 1])
@pytest.mark.parametrize("chunk", [16, 64])
def test_layer_matches_hf(layer_idx, chunk):
    torch.manual_seed(layer_idx)
    hf_cfg = Qwen3_5TextConfig(**CFG)
    hf_cfg._attn_implementation = "eager"
    ref = Qwen3_5DecoderLayer(hf_cfg, layer_idx).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.05)
        if layer_idx == 0:
            ref.linear_attn.A_log.uniform_(-1, 2.5)  # strong decays: stresses cumulative-decay precision
    L, n_real = 128, 101
    ours = DecoderLayer(TextConfig(CFG), layer_idx, L, chunk).eval()
    ours.load_state_dict(ref.state_dict(), strict=False)
    x = torch.randn(1, L, CFG["hidden_size"])
    pos = torch.arange(L)[None].expand(3, L)
    rot = Qwen3_5TextRotaryEmbedding(hf_cfg)
    cos_hf, sin_hf = rot(x, pos[:, None, :])
    mask = torch.full((L, L), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        want = ref(x, position_embeddings=(cos_hf, sin_hf), attention_mask=mask if layer_idx == 1 else None)
        cos, sin = rope_cos_sin(TextConfig(CFG), pos)
        got = ours(x, cos, sin, torch.full((L, L), -1e4).triu(1)[None, None])
        # right padding must not affect real positions
        xp = x.clone()
        xp[:, n_real:] = torch.randn_like(xp[:, n_real:])
        got_pad = ours(xp, cos, sin, torch.full((L, L), -1e4).triu(1)[None, None])
    assert torch.allclose(got, want, atol=2e-4, rtol=1e-3), (got - want).abs().max()
    assert torch.allclose(got_pad[:, :n_real], got[:, :n_real], atol=1e-5)


@pytest.mark.parametrize("chunk", [16, 64])
def test_unit_lower_inverse_adversarial(chunk):
    net = GatedDeltaNet(TextConfig(CFG), chunk, chunk)
    t = torch.eye(chunk) + torch.ones(chunk, chunk).tril(-1)  # power-series worst case
    t = torch.stack([t, torch.eye(chunk) + 0.9 * torch.rand(chunk, chunk).tril(-1)])
    got = net.unit_lower_inverse(t)
    want = torch.linalg.inv(t)
    assert torch.allclose(got, want, atol=1e-4)


def test_vision_tower_matches_hf():
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from qwen35_vision import VisionTower, host_inputs, patches_from_pixel_values

    vcfg = json.loads(Path(hf_hub_download("Qwen/Qwen3.5-4B", "config.json")).read_text())["vision_config"]
    vcfg = {**vcfg, "depth": 2}
    torch.manual_seed(0)
    hf_cfg = Qwen3_5VisionConfig(**vcfg)
    hf_cfg._attn_implementation = "eager"
    ref = Qwen3_5VisionModel(hf_cfg).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0, 0.02)
    gh, gw, budget = 6, 8, 64
    pixel_values = torch.rand(gh * gw, 3, 1, 16, 16).expand(-1, -1, 2, -1, -1).reshape(gh * gw, -1) * 2 - 1
    ours = VisionTower(vcfg, budget).eval()
    table = ours.load_merged(ref.state_dict())
    with torch.no_grad():
        want = ref(pixel_values, grid_thw=torch.tensor([[1, gh, gw]])).pooler_output
        got = ours(patches_from_pixel_values(pixel_values, budget), *host_inputs(ours.cfg, table, gh, gw, budget))
    assert torch.allclose(got[: gh * gw // 4], want, atol=1e-4), (got[: gh * gw // 4] - want).abs().max()
