from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from torch_inference.ls_eend_runtime import LSEENDInferenceEngine


@dataclass(frozen=True)
class CoreMLStateLayout:
    input_dim: int
    full_output_dim: int
    real_output_dim: int
    encoder_layers: int
    decoder_layers: int
    encoder_dim: int
    num_heads: int
    key_dim: int
    head_dim: int
    encoder_conv_cache_len: int
    top_buffer_len: int
    conv_delay: int
    max_nspks: int


def build_state_layout(engine: LSEENDInferenceEngine) -> CoreMLStateLayout:
    model = engine.model
    params = engine.config["model"]["params"]
    n_units = int(params["n_units"])
    n_heads = int(params["n_heads"])
    max_nspks = int(engine.decode_max_nspks)
    encoder_conv_cache_len = int(params["conv_kernel_size"]) - 1
    top_buffer_len = 2 * int(params["conv_delay"]) + 1
    return CoreMLStateLayout(
        input_dim=(2 * engine.config["data"]["context_recp"] + 1) * engine.config["data"]["feat"]["n_mels"],
        full_output_dim=max_nspks,
        real_output_dim=max(0, max_nspks - 2),
        encoder_layers=int(params["enc_n_layers"]),
        decoder_layers=int(params["dec_n_layers"]),
        encoder_dim=n_units,
        num_heads=n_heads,
        key_dim=n_units // n_heads,
        head_dim=n_units // n_heads,
        encoder_conv_cache_len=encoder_conv_cache_len,
        top_buffer_len=top_buffer_len,
        conv_delay=int(params["conv_delay"]),
        max_nspks=max_nspks,
    )


def initial_state_tensors(layout: CoreMLStateLayout, dtype: np.dtype = np.float32) -> dict[str, np.ndarray]:
    return {
        "enc_ret_kv": np.zeros(
            (layout.encoder_layers, 1, layout.num_heads, layout.key_dim, layout.head_dim),
            dtype=dtype,
        ),
        "enc_ret_scale": np.zeros((layout.encoder_layers, 1, layout.num_heads), dtype=dtype),
        "enc_conv_cache": np.zeros(
            (layout.encoder_layers, 1, layout.encoder_conv_cache_len, layout.encoder_dim),
            dtype=dtype,
        ),
        "dec_ret_kv": np.zeros(
            (layout.decoder_layers, layout.max_nspks, layout.num_heads, layout.key_dim, layout.head_dim),
            dtype=dtype,
        ),
        "dec_ret_scale": np.zeros(
            (layout.decoder_layers, layout.max_nspks, layout.num_heads),
            dtype=dtype,
        ),
        "top_buffer": np.zeros((1, layout.top_buffer_len, layout.encoder_dim), dtype=dtype),
    }


def _as_rank3_scalar(value: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return value.to(device=device, dtype=dtype).reshape(1, 1, 1)


def _safe_l2_normalize(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 1e-12 underflows to zero in fp16 CoreML execution and can produce NaNs
    # during warmup frames when an embedding or attractor vector is exactly zero.
    return x / torch.norm(x, dim=dim, keepdim=True).clamp_min(1e-4)


class CoreMLOnlineStepModule(torch.nn.Module):
    """Single online LS-EEND step with explicit state tensors for CoreML export."""

    def __init__(self, model: torch.nn.Module, layout: CoreMLStateLayout) -> None:
        super().__init__()
        self.model = model
        self.layout = layout
        self.encoder_decay = torch.exp(
            self.model.enc.encoder.layers[0].sequential[1].module.ret_pos.decay
        ).float()
        self.decoder_decay = torch.exp(
            self.model.dec.attractor_decoder.layers[0].ret_pos1.decay
        ).float()

    def forward(
        self,
        frame: torch.Tensor,
        enc_ret_kv: torch.Tensor,
        enc_ret_scale: torch.Tensor,
        enc_conv_cache: torch.Tensor,
        dec_ret_kv: torch.Tensor,
        dec_ret_scale: torch.Tensor,
        top_buffer: torch.Tensor,
        ingest: torch.Tensor,
        decode: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = frame.dtype
        device = frame.device
        ingest_scalar = _as_rank3_scalar(ingest, dtype, device)
        decode_scalar = _as_rank3_scalar(decode, dtype, device)
        ingest_vec = ingest.to(device=device, dtype=dtype).reshape(1, 1)
        decode_vec = decode.to(device=device, dtype=dtype).reshape(1, 1)

        x = self.model.enc.encoder.input_projection(frame)
        x = self.model.enc.encoder.layer_norm(x)

        new_enc_ret_kv = []
        new_enc_ret_scale = []
        new_enc_conv_cache = []

        for layer_index, layer in enumerate(self.model.enc.encoder.layers):
            old_kv = enc_ret_kv[layer_index]
            old_scale = enc_ret_scale[layer_index]
            old_conv = enc_conv_cache[layer_index]
            x, candidate_kv, candidate_scale, candidate_conv = self._encoder_layer_step(
                layer=layer,
                x=x,
                old_kv=old_kv,
                old_scale=old_scale,
                old_conv_cache=old_conv,
            )
            blended_kv = old_kv + (candidate_kv - old_kv) * ingest_scalar.unsqueeze(-1)
            blended_scale = old_scale + (candidate_scale - old_scale) * ingest_vec
            blended_conv = old_conv + (candidate_conv - old_conv) * ingest_scalar
            new_enc_ret_kv.append(blended_kv)
            new_enc_ret_scale.append(blended_scale)
            new_enc_conv_cache.append(blended_conv)

        appended_encoder_frame = x * ingest_scalar
        top_buffer = torch.cat([top_buffer[:, 1:, :], appended_encoder_frame], dim=1)

        emb = F.conv1d(
            top_buffer.transpose(1, 2),
            self.model.cnn.weight,
            self.model.cnn.bias,
        ).transpose(1, 2)
        emb = _safe_l2_normalize(emb, dim=-1)

        logits, candidate_dec_ret_kv, candidate_dec_ret_scale = self._decoder_step(
            emb=emb,
            dec_ret_kv=dec_ret_kv,
            dec_ret_scale=dec_ret_scale,
        )

        new_dec_ret_kv = dec_ret_kv + (candidate_dec_ret_kv - dec_ret_kv) * decode_scalar.unsqueeze(-1)
        new_dec_ret_scale = dec_ret_scale + (candidate_dec_ret_scale - dec_ret_scale) * decode_vec.unsqueeze(-1)

        logits = logits * decode_scalar

        return (
            logits,
            torch.stack(new_enc_ret_kv, dim=0),
            torch.stack(new_enc_ret_scale, dim=0),
            torch.stack(new_enc_conv_cache, dim=0),
            new_dec_ret_kv,
            new_dec_ret_scale,
            top_buffer,
        )

    def _encoder_layer_step(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        old_kv: torch.Tensor,
        old_scale: torch.Tensor,
        old_conv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ff1 = layer.sequential[0]
        attn = layer.sequential[1].module
        conv = layer.sequential[2].module
        ff2 = layer.sequential[3]
        final_norm = layer.sequential[4]

        x = ff1.module(x) * ff1.module_factor + x * ff1.input_factor
        attn_input = attn.layer_norm(x)
        attn_output, candidate_kv, candidate_scale = self._retention_recurrent(
            retention_module=attn.self_attn,
            x=attn_input,
            old_kv=old_kv,
            old_scale=old_scale,
            decay=self.encoder_decay,
        )
        x = x + attn.dropout(attn_output)
        conv_output, candidate_conv = self._conformer_conv_step(conv, x, old_conv_cache)
        x = x + conv_output
        x = ff2.module(x) * ff2.module_factor + x * ff2.input_factor
        return final_norm(x), candidate_kv, candidate_scale, candidate_conv

    def _conformer_conv_step(
        self,
        conv_module: torch.nn.Module,
        x: torch.Tensor,
        old_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modules = conv_module.sequential

        current = modules[0](x)
        current = modules[1](current)
        current = modules[2](current)
        current = modules[3](current)

        cache = old_cache.transpose(1, 2)
        depthwise_window = torch.cat([cache, current], dim=2)
        depthwise_conv = modules[4].conv
        depthwise = F.conv1d(
            depthwise_window,
            depthwise_conv.weight,
            depthwise_conv.bias,
            stride=depthwise_conv.stride,
            padding=0,
            dilation=depthwise_conv.dilation,
            groups=depthwise_conv.groups,
        )
        candidate_cache = depthwise_window[:, :, -self.layout.encoder_conv_cache_len :].transpose(1, 2)

        depthwise = modules[5](depthwise)
        depthwise = modules[6](depthwise)
        depthwise = modules[7](depthwise)
        depthwise = modules[8](depthwise)
        return depthwise.transpose(1, 2), candidate_cache

    def _decoder_step(
        self,
        emb: torch.Tensor,
        dec_ret_kv: torch.Tensor,
        dec_ret_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_enc = self.model.dec.pos_enc(emb, self.layout.max_nspks)
        repeated_emb = emb.unsqueeze(dim=2).repeat(1, 1, self.layout.max_nspks, 1)
        attractors = self.model.dec.convert(torch.cat([repeated_emb, pos_enc], dim=-1))

        new_dec_ret_kv = []
        new_dec_ret_scale = []
        for layer_index, layer in enumerate(self.model.dec.attractor_decoder.layers):
            attractors, candidate_kv, candidate_scale = self._fusion_layer_step(
                layer=layer,
                src=attractors,
                old_kv=dec_ret_kv[layer_index],
                old_scale=dec_ret_scale[layer_index],
            )
            new_dec_ret_kv.append(candidate_kv)
            new_dec_ret_scale.append(candidate_scale)

        if self.model.dec.attractor_decoder.norm is not None:
            attractors = self.model.dec.attractor_decoder.norm(attractors)
        attractors = _safe_l2_normalize(attractors, dim=-1)
        logits = torch.matmul(emb.unsqueeze(dim=-2), attractors.transpose(-1, -2)).squeeze(dim=-2)
        return logits, torch.stack(new_dec_ret_kv, dim=0), torch.stack(new_dec_ret_scale, dim=0)

    def _fusion_layer_step(
        self,
        layer: torch.nn.Module,
        src: torch.Tensor,
        old_kv: torch.Tensor,
        old_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, speaker_count, feat_dim = src.shape
        x = src.transpose(1, 2).reshape(batch_size * speaker_count, time_steps, feat_dim)

        if layer.norm_first:
            time_input = layer.norm11(x)
            time_output, candidate_kv, candidate_scale = self._retention_recurrent(
                retention_module=layer.self_attn1,
                x=time_input,
                old_kv=old_kv,
                old_scale=old_scale,
                decay=self.decoder_decay,
            )
            x = x + layer.dropout11(time_output)
        else:
            time_output, candidate_kv, candidate_scale = self._retention_recurrent(
                retention_module=layer.self_attn1,
                x=x,
                old_kv=old_kv,
                old_scale=old_scale,
                decay=self.decoder_decay,
            )
            x = layer.norm11(x + layer.dropout11(time_output))

        x = x.reshape(batch_size, speaker_count, time_steps, feat_dim).transpose(1, 2)
        x = x.reshape(batch_size * time_steps, speaker_count, feat_dim)

        if layer.norm_first:
            x = x + self._speaker_attention(layer.self_attn2, layer.norm21(x))
            x = x + layer._ff_block(layer.norm22(x))
        else:
            x = layer.norm21(x + self._speaker_attention(layer.self_attn2, x))
            x = layer.norm22(x + layer._ff_block(x))

        return x.reshape(batch_size, time_steps, speaker_count, feat_dim), candidate_kv, candidate_scale

    def _retention_recurrent(
        self,
        retention_module: torch.nn.Module,
        x: torch.Tensor,
        old_kv: torch.Tensor,
        old_scale: torch.Tensor,
        decay: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, target_length, _ = x.shape
        q = retention_module.q_proj(x)
        k = retention_module.k_proj(x)
        v = retention_module.v_proj(x)
        g = retention_module.g_proj(x)

        k = k * retention_module.scaling
        q = q.view(batch_size, target_length, retention_module.num_heads, retention_module.key_dim).transpose(1, 2)
        k = k.view(batch_size, target_length, retention_module.num_heads, retention_module.key_dim).transpose(1, 2)
        v = v.view(batch_size, retention_module.num_heads, retention_module.head_dim, 1)

        qr = q
        kr = k
        kv = kr * v

        decay = decay.to(device=x.device, dtype=x.dtype).reshape(1, retention_module.num_heads)
        candidate_scale = old_scale * decay + 1.0
        blend = (old_scale.sqrt() * decay / candidate_scale.sqrt()).unsqueeze(-1).unsqueeze(-1)
        candidate_kv = old_kv * blend + kv / candidate_scale.sqrt().unsqueeze(-1).unsqueeze(-1)

        output = torch.sum(qr * candidate_kv, dim=3)
        output = retention_module.group_norm(output).reshape(
            batch_size, target_length, retention_module.head_dim * retention_module.num_heads
        )
        output = retention_module.gate_fn(g) * output
        output = retention_module.out_proj(output)
        return output, candidate_kv, candidate_scale

    def _speaker_attention(self, attention: torch.nn.MultiheadAttention, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // attention.num_heads
        q_weight, k_weight, v_weight = attention.in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = attention.in_proj_bias.chunk(3, dim=0)

        q = F.linear(x, q_weight, q_bias)
        k = F.linear(x, k_weight, k_bias)
        v = F.linear(x, v_weight, v_bias)

        q = q.view(batch_size, seq_len, attention.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, attention.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, attention.num_heads, head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return F.linear(out, attention.out_proj.weight, attention.out_proj.bias)


def load_coreml_step_module(
    checkpoint_path: Path,
    config_path: Path,
    device: str = "cpu",
) -> tuple[CoreMLOnlineStepModule, CoreMLStateLayout, LSEENDInferenceEngine]:
    engine = LSEENDInferenceEngine(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )
    engine.model = engine.model.float().to(torch.device(device))
    engine.model.eval()
    layout = build_state_layout(engine)
    module = CoreMLOnlineStepModule(engine.model, layout).to(torch.device(device)).eval()
    return module, layout, engine
