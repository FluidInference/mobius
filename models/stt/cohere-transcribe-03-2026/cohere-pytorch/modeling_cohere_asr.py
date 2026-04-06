import atexit
import logging
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache, EncoderDecoderCache, StaticCache
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from .configuration_cohere_asr import NO_SPACE_LANGS, CohereAsrConfig, _dynamo_disable

logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)


class CohereAsrPreTrainedModel(PreTrainedModel):
    config_class = CohereAsrConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = False
    _no_split_modules = ["ConformerLayer", "TransformerDecoderLayer"]
    _supports_cache_class = True
    _supports_static_cache = True

    @property
    def all_tied_weights_keys(self):
        return {}

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# --- Encoder Components (Conformer) ---


class MaskedConvSequential(nn.Sequential):
    def forward(self, x, lengths):
        # x: (batch, channels, time, features)
        current_lengths = lengths.clone().float()
        mask = self._create_mask(x, current_lengths.long())
        for layer in self:
            x = self.apply_channel_mask(x, mask)
            x = layer(x)
            if hasattr(layer, "stride") and layer.stride != (1, 1):
                current_lengths = self.calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], layer.padding
                )
                mask = self._create_mask(x, current_lengths.long())
        x = self.apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _create_mask(self, tensor, lengths):
        batch_size, _, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(batch_size, time) < lengths.unsqueeze(1)
        return time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)

    def apply_channel_mask(self, tensor, mask):
        batch_size, channels, time, features = tensor.shape
        expanded_mask = mask.unsqueeze(1).expand(batch_size, channels, time, features)
        return tensor * expanded_mask

    def calculate_conv_output_size(
        self,
        input_size: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: tuple[int, int],
    ):
        return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


class ConvSubsampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_in = int(config["feat_in"])
        conv_channels = int(config["subsampling_conv_channels"])
        self._conv_channels = conv_channels
        feat_out = int(config["feat_out"])
        if feat_out <= 0:
            feat_out = int(config["d_model"])
        subsampling_factor = int(config["subsampling_factor"])

        self.conv = MaskedConvSequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1, groups=conv_channels),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2, padding=1, groups=conv_channels),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.out = nn.Linear(conv_channels * (feat_in // subsampling_factor), feat_out)

    def _check_input_shape(self, x):
        max_size_32bit = 2_147_483_647
        B, C, T, F = x.shape
        out_T = (T + 2 - 3) // 2 + 1
        out_F = (F + 2 - 3) // 2 + 1
        projected = B * self._conv_channels * out_T * out_F

        if projected > max_size_32bit:
            valid_batch_size = max_size_32bit // (self._conv_channels * out_T * out_F)
            raise RuntimeError(
                f"Batch too large for first conv: projected output numel={projected}, "
                f"input shape={(B, C, T, F)}. Reduce batch size to {valid_batch_size} or lower. "
                "You can try commenting out this code but depending on your pytorch version you may get an error like: \n"
                "'RuntimeError: Expected canUse32BitIndexMath(input) && canUse32BitIndexMath(output) to be true, but got false.'"
            )

    @_dynamo_disable
    def _needs_conv_split(self, x: torch.Tensor) -> bool:
        """Check if input would exceed PyTorch's 2^31 int32 CUDA indexing limit
        after the first Conv2d (stride=2) expands channels to conv_channels."""
        B, C, T, F = x.shape
        out_T = (T + 2 - 3) // 2 + 1
        out_F = (F + 2 - 3) // 2 + 1
        projected = B * self._conv_channels * out_T * out_F
        return projected > 2_147_483_647

    @_dynamo_disable
    def _conv_split_by_batch(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split input along batch dim, run conv on each chunk, then concatenate.

        This is to work around the PyTorch/CUDA int32 indexing limit (https://github.com/pytorch/pytorch/issues/80020).
        """
        b = x.size(0)
        _, _, t, f = x.shape
        out_t = (t + 2 - 3) // 2 + 1
        out_f = (f + 2 - 3) // 2 + 1
        per_sample_projected = self._conv_channels * out_t * out_f
        max_size_32bit = 2_147_483_647
        max_batch_for_first_conv = max_size_32bit // per_sample_projected
        safe_batch = min(b, max_batch_for_first_conv)
        # Prefer power-of-two chunk sizes for better kernel utilization while
        # still respecting the first-conv int32 indexing limit.
        chunk_size = 1 << max(0, safe_batch.bit_length() - 1)
        parts = []
        for chunk, ln in zip(
            torch.split(x, chunk_size, 0),
            torch.split(lengths, chunk_size, 0),
        ):
            self._check_input_shape(chunk)
            parts.append(self.conv(chunk, ln))
        return (
            torch.cat([p[0] for p in parts], dim=0),
            torch.cat([p[1] for p in parts], dim=0),
        )

    def forward(self, x, lengths):
        # x: (B, feat_in, T) -> (B, 1, T, feat_in)
        x = x.transpose(1, 2).unsqueeze(1)

        if self._needs_conv_split(x):
            x, lengths = self._conv_split_by_batch(x, lengths)
        else:
            self._check_input_shape(x)
            x, lengths = self.conv(x, lengths)

        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.out(x)
        return x, lengths


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def _create_pe(self, positions: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.d_model, device=positions.device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=positions.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return pe.unsqueeze(0).to(dtype)

    @_dynamo_disable
    def _materialize_pe(self, length: int, device: torch.device, dtype: torch.dtype):
        needed_size = 2 * length - 1
        if hasattr(self, "pe") and self.pe.size(1) >= needed_size:
            if self.pe.device != device:
                self.pe = self.pe.to(device=device)
            if self.pe.dtype != dtype:
                self.pe = self.pe.to(dtype=dtype)
            return
        effective_length = max(length, self.max_len)
        positions = torch.arange(
            effective_length - 1, -effective_length, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        pe = self._create_pe(positions=positions, dtype=dtype)
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        self._materialize_pe(length=x.size(1), device=x.device, dtype=x.dtype)
        # center_pos would be the index of position 0
        # negative positions would be used for right and
        # positive for left tokens
        # for input of length L, 2*L-1 positions are needed,
        # positions from (L-1) to -(L-1)
        input_len = x.size(1)
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]

        return x, pos_emb


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ConformerConvolution(nn.Module):
    def __init__(self, d_model, kernel_size):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, groups=d_model, padding=(kernel_size - 1) // 2
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x, pad_mask=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__()
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(dropout_rate)
        self.scaling = self.d_k**-0.5
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))

    def rel_shift(self, x):
        """Compute relative positional encoding.
        Args:
            x (torch.Tensor): (batch, nheads, time, 2*time-1)
        """
        b, h, qlen, pos_len = x.size()  # (b, h, t1, t2)
        # need to add a column of zeros on the left side of
        # last dimension to perform the relative shifting
        x = torch.nn.functional.pad(x, pad=(1, 0))  # (b, h, t1, t2+1)
        x = x.view(b, h, -1, qlen)  # (b, h, t2+1, t1)
        # need to drop the first row
        x = x[:, :, 1:].view(b, h, qlen, pos_len)  # (b, h, t1, t2)
        return x

    def forward(self, x, pos_emb, mask=None):
        batch_size = x.size(0)
        q = self.linear_q(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # pos_emb might be shared across batch
        if pos_emb.size(0) == 1 and batch_size > 1:
            pos_emb = pos_emb.expand(batch_size, -1, -1)
        p = self.linear_pos(pos_emb).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        q_with_u = q + self.pos_bias_u.unsqueeze(0).unsqueeze(2)
        q_with_v = q + self.pos_bias_v.unsqueeze(0).unsqueeze(2)
        matrix_ac = torch.matmul(q_with_u, k.transpose(-1, -2))
        matrix_bd = torch.matmul(q_with_v, p.transpose(-1, -2))
        matrix_bd = self.rel_shift(matrix_bd)

        # drops extra elements in the matrix_bd to match the matrix_ac's size
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) * self.scaling

        if mask is not None:
            expanded_mask = mask.unsqueeze(1)
            scores = scores.masked_fill(expanded_mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        if mask is not None:
            attn = attn.masked_fill(expanded_mask, 0.0)
        x = torch.matmul(self.dropout(attn), v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linear_out(x)


class ConformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, conv_kernel_size, dropout):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff, dropout)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(n_heads, d_model, dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model, conv_kernel_size)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff, dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, mask=None, pad_mask=None):
        residual = x
        x = self.norm_feed_forward1(x)
        x = residual + 0.5 * self.dropout(self.feed_forward1(x))

        residual = x
        x = self.norm_self_att(x)
        x = residual + self.dropout(self.self_attn(x, pos_emb, mask))

        residual = x
        x = self.norm_conv(x)
        x = residual + self.dropout(self.conv(x, pad_mask=pad_mask))

        residual = x
        x = self.norm_feed_forward2(x)
        x = residual + 0.5 * self.dropout(self.feed_forward2(x))

        return self.norm_out(x)


class ConformerEncoder(nn.Module):
    """
    Fast Conformer encoder.

    Follows [Fast Conformer with Linearly Scalable Attention for Efficient Speech
    Recognition](https://arxiv.org/abs/2305.05084).
    """

    main_input_name = "input_features"

    def __init__(self, config):
        super().__init__()
        enc_config = config.encoder
        self.d_model = enc_config["d_model"]
        d_ff = self.d_model * enc_config["ff_expansion_factor"]
        n_heads = enc_config["n_heads"]
        conv_kernel_size = enc_config["conv_kernel_size"]
        dropout = enc_config["dropout"]
        n_layers = enc_config["n_layers"]
        pos_emb_max_len = enc_config["pos_emb_max_len"]

        self.pre_encode = ConvSubsampling(enc_config)
        self.pos_enc = RelPositionalEncoding(self.d_model, pos_emb_max_len)

        self.layers = nn.ModuleList(
            [ConformerLayer(self.d_model, d_ff, n_heads, conv_kernel_size, dropout) for _ in range(n_layers)]
        )

    def _create_masks(self, padding_length, max_audio_length, device):
        att_mask = torch.ones(1, max_audio_length, max_audio_length, dtype=torch.bool, device=device)
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)
        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2))
        att_mask = torch.logical_and(att_mask.to(pad_mask_for_att_mask.device), pad_mask_for_att_mask)
        att_mask = ~att_mask
        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def forward(
        self,
        input_features=None,
        length=None,
        return_dict: bool = False,
        **kwargs,
    ):
        if input_features is None:
            raise ValueError("Expected `input_features` for encoder forward.")
        if length is None:
            length = torch.full(
                (input_features.shape[0],),
                input_features.shape[-1],
                device=input_features.device,
                dtype=torch.long,
            )
        conv_dtype = self.pre_encode.conv[0].weight.dtype
        if input_features.dtype != conv_dtype:
            input_features = input_features.to(dtype=conv_dtype)
        x, length = self.pre_encode(input_features, length)
        length = length.to(torch.int64)
        max_audio_length = x.size(1)
        x, pos_emb = self.pos_enc(x)
        pad_mask, att_mask = self._create_masks(
            padding_length=length,
            max_audio_length=max_audio_length,
            device=x.device,
        )
        for i, layer in enumerate(self.layers):
            x = layer(x, pos_emb, mask=att_mask, pad_mask=pad_mask)
        if return_dict:
            return BaseModelOutput(last_hidden_state=x)
        return x, length


# --- Decoder Components ---


class FixedPositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_sequence_length=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length

        pos_enc = torch.zeros(max_sequence_length, hidden_size)
        position = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        coef = -math.log(10000.0) / hidden_size
        div_term = torch.exp(coef * torch.arange(0.0, hidden_size, 2))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc.div_(math.sqrt(hidden_size))
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, position_ids):
        return torch.index_select(self.pos_enc, 0, position_ids.reshape(-1)).reshape(*position_ids.shape, -1)


class DecoderAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, layer_idx):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.query_net = nn.Linear(hidden_size, hidden_size)
        self.key_net = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Linear(hidden_size, hidden_size)
        self.out_projection = nn.Linear(hidden_size, hidden_size)

    def _reshape(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states,
        context_states=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        is_cross_attention=False,
        kv_seq_len=None,
    ):
        query = self._reshape(self.query_net(hidden_states))
        source = hidden_states if context_states is None else context_states
        cache_layer = None
        is_cross_cache_updated = False
        if past_key_values is not None and isinstance(past_key_values, EncoderDecoderCache):
            is_cross_cache_updated = past_key_values.is_updated.get(self.layer_idx, False)
            if is_cross_attention:
                cache_layer = past_key_values.cross_attention_cache
            else:
                cache_layer = past_key_values.self_attention_cache
        elif past_key_values is not None and isinstance(past_key_values, DynamicCache):
            cache_layer = past_key_values

        if is_cross_attention and cache_layer is not None and is_cross_cache_updated:
            key, value = _get_cache_kv(cache_layer, self.layer_idx)
        else:
            key = self._reshape(self.key_net(source))
            value = self._reshape(self.value_net(source))
            if cache_layer is not None:
                cache_kwargs = None
                if not is_cross_attention and cache_position is not None:
                    cache_kwargs = {"cache_position": cache_position}
                key, value = cache_layer.update(key, value, self.layer_idx, cache_kwargs=cache_kwargs)
                if not is_cross_attention and kv_seq_len is not None:
                    key = key[:, :, :kv_seq_len]
                    value = value[:, :, :kv_seq_len]
                if is_cross_attention:
                    past_key_values.is_updated[self.layer_idx] = True

        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, scale=self.scale
        )
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(hidden_states.shape[0], hidden_states.shape[1], self.hidden_size)
        )
        return self.out_projection(attn_output)


class DecoderFeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_act="relu"):
        super().__init__()
        self.dense_in = nn.Linear(hidden_size, inner_size)
        hidden_act = str(hidden_act).lower().replace("swish", "silu")
        if hidden_act not in ACT2FN:
            raise ValueError(f"Unsupported decoder hidden_act: {hidden_act}")
        self.activation = ACT2FN[hidden_act]
        self.dense_out = nn.Linear(inner_size, hidden_size)

    def forward(self, x):
        return self.dense_out(self.activation(self.dense_in(x)))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, inner_size, num_heads, layer_idx, hidden_act="relu"):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.first_sub_layer = DecoderAttention(hidden_size, num_heads, layer_idx=layer_idx)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.second_sub_layer = DecoderAttention(hidden_size, num_heads, layer_idx=layer_idx)
        self.layer_norm_3 = nn.LayerNorm(hidden_size)
        self.third_sub_layer = DecoderFeedForward(hidden_size, inner_size, hidden_act=hidden_act)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        self_attention_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        cache_position=None,
        kv_seq_len=None,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm_1(hidden_states)
        self_out = self.first_sub_layer(
            hidden_states,
            context_states=None,
            attention_mask=self_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            is_cross_attention=False,
            kv_seq_len=kv_seq_len,
        )
        hidden_states = residual + self_out

        residual = hidden_states
        hidden_states = self.layer_norm_2(hidden_states)
        cross_out = self.second_sub_layer(
            hidden_states,
            context_states=encoder_hidden_states,
            attention_mask=cross_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            is_cross_attention=True,
        )
        hidden_states = residual + cross_out

        residual = hidden_states
        hidden_states = self.layer_norm_3(hidden_states)
        hidden_states = residual + self.third_sub_layer(hidden_states)
        return hidden_states


class TransformerDecoderEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_sequence_length, padding_idx=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.position_embedding = FixedPositionalEncoding(hidden_size, max_sequence_length)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, positions):
        return self.layer_norm(self.token_embedding(input_ids) + self.position_embedding(positions))


class TransformerDecoderCore(nn.Module):
    def __init__(self, hidden_size, inner_size, num_heads, num_layers, hidden_act="relu"):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(hidden_size, inner_size, num_heads, layer_idx=i, hidden_act=hidden_act)
                for i in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        self_attention_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        cache_position=None,
        kv_seq_len=None,
    ):
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                kv_seq_len=kv_seq_len,
            )
        return self.final_layer_norm(hidden_states), past_key_values


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        dec_config = config.transf_decoder["config_dict"]
        hidden_size = dec_config["hidden_size"]
        self._embedding = TransformerDecoderEmbedding(
            vocab_size=config.head["num_classes"],
            hidden_size=hidden_size,
            max_sequence_length=dec_config["max_sequence_length"],
            padding_idx=2,
        )
        self._decoder = TransformerDecoderCore(
            hidden_size=hidden_size,
            inner_size=dec_config["inner_size"],
            num_heads=dec_config["num_attention_heads"],
            num_layers=dec_config["num_layers"],
            hidden_act=dec_config.get("hidden_act", "relu"),
        )

    def forward(
        self,
        input_ids,
        positions,
        encoder_hidden_states=None,
        self_attention_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        cache_position=None,
        kv_seq_len=None,
    ):
        hidden_states = self._embedding(input_ids, positions)
        return self._decoder(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            kv_seq_len=kv_seq_len,
        )


# --- Top-level Model ---


class CohereAsrModel(CohereAsrPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = ConformerEncoder(config)
        self.transf_decoder = TransformerDecoderWrapper(config)
        self.decoder_hidden_size = config.transf_decoder["config_dict"]["hidden_size"]

        if self.encoder.d_model != self.decoder_hidden_size:
            self.encoder_decoder_proj = nn.Linear(self.encoder.d_model, self.decoder_hidden_size)
        else:
            self.encoder_decoder_proj = None

    def forward(
        self,
        input_ids,
        positions,
        input_features,
        length,
        attention_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
    ):
        encoder_hidden_states, _ = self.encoder(input_features, length)
        if self.encoder_decoder_proj is not None:
            encoder_hidden_states = self.encoder_decoder_proj(encoder_hidden_states)

        return self.transf_decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            past_key_values=past_key_values,
        )


class TokenClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_classes, log_softmax=False):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.layer0 = nn.Linear(hidden_size, num_classes)
        self.use_log_softmax = log_softmax

    def forward(self, hidden_states):
        logits = self.mlp.layer0(hidden_states)
        if self.use_log_softmax:
            return torch.log_softmax(logits, dim=-1)
        return logits


class CohereAsrForConditionalGeneration(CohereAsrPreTrainedModel):
    """Encoder-decoder Cohere ASR model with generation and transcription helpers."""

    _keys_to_ignore_on_load_unexpected = [
        "preprocessor.featurizer.window",
        "preprocessor.featurizer.fb",
    ]

    def _supports_default_dynamic_cache(self):
        return True

    def __init__(self, config):
        super().__init__(config)
        self.encoder = ConformerEncoder(config)
        self.transf_decoder = TransformerDecoderWrapper(config)
        self.decoder_hidden_size = config.transf_decoder["config_dict"]["hidden_size"]
        if self.encoder.d_model != self.decoder_hidden_size:
            self.encoder_decoder_proj = nn.Linear(self.encoder.d_model, self.decoder_hidden_size)
        else:
            self.encoder_decoder_proj = None
        self.log_softmax = TokenClassifierHead(
            hidden_size=config.head["hidden_size"],
            num_classes=config.head["num_classes"],
            log_softmax=bool(config.head.get("log_softmax", False)),
        )
        # Tie token classifier head weights to decoder token embeddings.
        self.log_softmax.mlp.layer0.weight = self.transf_decoder._embedding.token_embedding.weight
        self._decode_pool = None
        self._decode_pool_spm_model_file = None

    def _infer_encoder_lengths_from_raw(self, raw_length: torch.Tensor) -> torch.Tensor:
        lengths = raw_length.to(dtype=torch.long)
        for layer in self.encoder.pre_encode.conv:
            if isinstance(layer, nn.Conv2d):
                if layer.stride[0] > 1:
                    lengths = (lengths + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
        return torch.clamp(lengths, min=1)

    def forward(
        self,
        input_ids=None,
        positions=None,
        input_features=None,
        length=None,
        attention_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        cache_position=None,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        **kwargs,
    ):
        if input_ids is None and decoder_input_ids is not None:
            input_ids = decoder_input_ids
        if input_ids is None:
            raise ValueError("Expected `input_ids` or `decoder_input_ids`.")
        if positions is None:
            positions = (
                torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            )

        encoder_lengths = None
        if encoder_outputs is not None:
            if hasattr(encoder_outputs, "last_hidden_state"):
                encoder_hidden_states = encoder_outputs.last_hidden_state
            else:
                encoder_hidden_states = encoder_outputs
            if self.encoder_decoder_proj is not None:
                encoder_hidden_states = self.encoder_decoder_proj(encoder_hidden_states)
        else:
            encoder_hidden_states, encoder_lengths = self.encoder(input_features, length)
            if self.encoder_decoder_proj is not None:
                encoder_hidden_states = self.encoder_decoder_proj(encoder_hidden_states)

        # Wrap encoder_hidden_states in BaseModelOutput for return_dict compatibility if needed
        if encoder_outputs is None:
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        dtype = encoder_hidden_states.dtype
        batch_size, tgt_len = input_ids.shape
        past_len = _get_cache_seq_length(past_key_values)
        total_kv_len = past_len + tgt_len
        static_max_cache_len = _get_static_cache_len(past_key_values)
        if static_max_cache_len is not None and cache_position is None:
            raise ValueError(
                "cache_position is required when using StaticCache. "
                "Ensure generate() or the caller passes cache_position."
            )

        query_positions = torch.arange(past_len, past_len + tgt_len, device=input_ids.device)[:, None]
        key_positions = torch.arange(total_kv_len, device=input_ids.device)[None, :]
        causal_bool = key_positions > query_positions
        self_attention_mask = torch.zeros((batch_size, 1, tgt_len, total_kv_len), device=input_ids.device, dtype=dtype)
        self_attention_mask.masked_fill_(causal_bool[None, None, :, :], float("-inf"))

        effective_decoder_mask = decoder_attention_mask if decoder_attention_mask is not None else attention_mask
        if effective_decoder_mask is not None:
            effective_decoder_mask = _align_decoder_attention_mask(effective_decoder_mask, total_kv_len=total_kv_len)
            key_padding = (1.0 - effective_decoder_mask[:, None, None, :].to(dtype=dtype)) * -1e9
            self_attention_mask = self_attention_mask + key_padding

        effective_cross_attention_mask = cross_attention_mask
        if effective_cross_attention_mask is None:
            if encoder_lengths is None and length is not None:
                encoder_lengths = self._infer_encoder_lengths_from_raw(length)
            if encoder_lengths is not None:
                src_len = encoder_hidden_states.shape[1]
                enc_positions = torch.arange(src_len, device=encoder_hidden_states.device)[None, :]
                valid = enc_positions < encoder_lengths.to(device=encoder_hidden_states.device)[:, None]
                effective_cross_attention_mask = (1.0 - valid[:, None, None, :].to(dtype=dtype)) * -1e9

        kv_seq_len = total_kv_len if static_max_cache_len is not None else None

        outputs, updated_cache = self.transf_decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=effective_cross_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            kv_seq_len=kv_seq_len,
        )

        logits = self.log_softmax(outputs)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.head["num_classes"]), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=updated_cache,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.transf_decoder

    def generate(self, input_features=None, input_ids=None, length=None, attention_mask=None, **kwargs):
        # If input_ids is provided, use it as decoder_input_ids
        # This matches the multimodal encoder-decoder expectation where the prompt is the decoder start
        decoder_input_ids = kwargs.pop("decoder_input_ids", None)
        if input_ids is not None and decoder_input_ids is None:
            decoder_input_ids = input_ids
            # We must provide some input_ids to super().generate to avoid validation errors,
            # but for encoder-decoder it usually expects encoder input_ids.
            # Here input_features is the encoder input.
            input_ids = None

        decoder_attention_mask = kwargs.pop("decoder_attention_mask", None)
        if decoder_input_ids is not None and decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(
                decoder_input_ids, dtype=torch.long, device=decoder_input_ids.device
            )

        generation_kwargs = dict(kwargs)
        generation_kwargs["input_features"] = input_features
        generation_kwargs["length"] = length
        generation_kwargs["decoder_input_ids"] = decoder_input_ids
        generation_kwargs["decoder_attention_mask"] = decoder_attention_mask

        decoder_start_token_id = getattr(self.config, "decoder_start_token_id", None)
        eos_token_id = getattr(self.config, "eos_token_id", None)
        pad_token_id = getattr(self.config, "pad_token_id", None)
        if decoder_start_token_id is not None:
            generation_kwargs["bos_token_id"] = decoder_start_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if input_ids is not None:
            generation_kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask
        if "cache_implementation" not in generation_kwargs:
            generation_kwargs["cache_implementation"] = "static"

        # Fall back to dynamic cache when static cache is incompatible:
        # - transformers 4.52-4.55: _supports_static_cache gate + StaticCache
        #   reads config.hidden_size which our nested config doesn't expose.
        # - transformers >= 5.3: StaticCache.update() API changed (cache_position
        #   shape must match key_states, breaking our usage).
        if generation_kwargs.get("cache_implementation") == "static":
            _skip_static = hasattr(PreTrainedModel, "_supports_static_cache")
            if not _skip_static:
                import transformers

                _v = tuple(int(x) for x in transformers.__version__.split(".")[:2])
                _skip_static = _v >= (5, 3)
            if _skip_static:
                generation_kwargs.pop("cache_implementation", None)

        # We disable_compile for generate() because when passing "cache_implementation"="static"
        # transformers will auto-compile the forward pass setting dynamic=False.
        # We need dynamic=True to avoid excessive recompilation. Note that this doesn't
        # control whether we compile the encoder layers which is set according to
        # the transcribe(...,compile=True) flag.
        generation_kwargs["disable_compile"] = True

        return super().generate(**generation_kwargs)

    def _setup_compile(self, processor=None):
        if getattr(self, "_compiled", False):
            return
        if not hasattr(torch, "compile"):
            self._compiled = True
            return

        # Dynamo guards on submodule identity per layer, so each ConformerLayer
        # causes a recompilation. Raise the limit so no layers fall back to eager.
        needed = len(self.encoder.layers) + 4
        if torch._dynamo.config.cache_size_limit < needed:
            torch._dynamo.config.cache_size_limit = needed

        for layer in self.encoder.layers:
            layer.forward = torch.compile(layer.forward, dynamic=True)

        if (
            processor is not None
            and hasattr(processor, "feature_extractor")
            and hasattr(processor.feature_extractor, "filterbank")
        ):
            filterbank = processor.feature_extractor.filterbank
            filterbank.forward = torch.compile(filterbank.forward)

        self._compiled = True

    def _validate_transcribe_language(self, language: str) -> None:
        supported_languages = set(getattr(self.config, "supported_languages", []))
        if language not in supported_languages:
            supported_joined = ", ".join(sorted(supported_languages))
            raise ValueError(f"Unsupported language '{language}'. Supported languages: {supported_joined}.")

    def build_prompt(self, language: str, punctuation: bool = True) -> str:
        """Build the decoder prompt prefix for language and punctuation settings."""
        pnc_token = "<|pnc|>" if punctuation else "<|nopnc|>"
        task_token = "<|noitn|>"
        return (
            "<|startofcontext|><|startoftranscript|><|emo:undefined|>"
            f"<|{language}|><|{language}|>{pnc_token}{task_token}<|notimestamp|><|nodiarize|>"
        )

    def _load_and_resample_audio(
        self,
        target_sample_rate: int,
        audio_file: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None,
    ) -> tuple[np.ndarray, int]:
        if (audio_file is None) == (audio_array is None):
            raise ValueError("Exactly one of audio_file or audio_array must be provided.")

        if audio_file is not None:
            audio, loaded_sample_rate = sf.read(audio_file)
            arr = np.asarray(audio, dtype=np.float32)
            sample_rate_int = int(loaded_sample_rate)
        else:
            if sample_rate is None:
                raise ValueError("sample_rate is required when audio_array is provided.")
            arr = np.asarray(audio_array, dtype=np.float32)
            sample_rate_int = int(sample_rate)

        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if arr.ndim != 1:
            raise ValueError(f"Expected mono waveform (1D), got shape={arr.shape}")

        if sample_rate_int != target_sample_rate:
            arr = librosa.resample(
                arr,
                orig_sr=sample_rate_int,
                target_sr=target_sample_rate,
            ).astype(np.float32, copy=False)
            sample_rate_int = target_sample_rate

        return arr, sample_rate_int

    def _prepare_segments(
        self,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        max_audio_clip_s: float,
        overlap_chunk_second: float,
        min_energy_window_samples: int,
    ) -> tuple[list[np.ndarray], list[int], list[tuple[int, Optional[int]]]]:
        segment_waveforms: list[np.ndarray] = []
        segment_sample_rates: list[int] = []
        segment_meta: list[tuple[int, Optional[int]]] = []
        fast_path_threshold_s = max(0.0, max_audio_clip_s - overlap_chunk_second)

        for sample_idx, (waveform, sample_rate) in enumerate(zip(waveforms, sample_rates)):
            duration_s = float(waveform.shape[0]) / float(sample_rate)
            if duration_s <= fast_path_threshold_s:
                segment_waveforms.append(waveform)
                segment_sample_rates.append(sample_rate)
                segment_meta.append((sample_idx, None))
                continue

            chunks = split_audio_chunks_energy(
                waveform=waveform,
                sample_rate=sample_rate,
                max_audio_clip_s=max_audio_clip_s,
                overlap_chunk_second=overlap_chunk_second,
                min_energy_window_samples=min_energy_window_samples,
            )
            for chunk_idx, chunk in enumerate(chunks):
                segment_waveforms.append(chunk)
                segment_sample_rates.append(sample_rate)
                segment_meta.append((sample_idx, chunk_idx))

        return segment_waveforms, segment_sample_rates, segment_meta

    def transcribe(
        self,
        processor,
        language: str,
        audio_files: Optional[list[str]] = None,
        audio_arrays: Optional[list[np.ndarray]] = None,
        sample_rates: Optional[list[int]] = None,
        punctuation: bool = True,
        batch_size: Optional[int] = None,
        compile: bool = False,
        pipeline_detokenization: bool = False,
    ) -> list[str]:
        """Transcribe one or more audio inputs into text.

        Audio longer than ``max_audio_clip_s`` (default 35 s) is automatically split into overlapping
        chunks and reassembled.

        Args:
            processor: ``AutoProcessor`` instance for this model.
            language: ISO 639-1 language code. The model does not perform language detection, so this
                is required. Supported: en, fr, de, es, it, pt, nl, pl, el, ar, ja, zh, vi, ko.
            audio_files: List of audio file paths. Mutually exclusive with *audio_arrays*.
            audio_arrays: List of 1-D numpy float arrays (raw waveforms). Requires *sample_rates*.
            sample_rates: Sample rate for each entry in *audio_arrays*.
            punctuation: Include punctuation in output (default ``True``).
            batch_size: GPU batch size. Defaults to ``config.batch_size``.
            compile: ``torch.compile`` encoder layers on first call for faster throughput (default
                ``False``). The first call incurs a one-time warmup cost; subsequent calls are faster.
            pipeline_detokenization: Overlap CPU detokenization with GPU inference using a background
                process (default ``False``). Beneficial when more audio segments than *batch_size* are
                passed in a single call, so that detokenization of one batch overlaps with inference on
                the next.

        Returns:
            List of transcription strings, one per input audio.
        """
        if (audio_files is None) == (audio_arrays is None):
            raise ValueError("Provide exactly one of audio_files or audio_arrays.")
        if audio_arrays is not None and sample_rates is None:
            raise ValueError("sample_rates is required when audio_arrays is provided.")
        if audio_arrays is not None and len(audio_arrays) != len(sample_rates):
            raise ValueError(
                f"audio_arrays and sample_rates must have same length, got {len(audio_arrays)} and {len(sample_rates)}."
            )

        if compile:
            self._setup_compile(processor=processor)

        total_inputs = len(audio_files) if audio_files is not None else len(audio_arrays)
        if total_inputs == 0:
            return []
        if pipeline_detokenization:
            self._ensure_decode_pool(processor=processor)

        self._validate_transcribe_language(language)
        prompt_text = self.build_prompt(language=language, punctuation=punctuation)

        effective_batch_size = int(batch_size) if batch_size is not None else int(self.config.batch_size)
        max_audio_clip_s = float(self.config.max_audio_clip_s)
        overlap_chunk_second = float(self.config.overlap_chunk_second)
        min_energy_window_samples = int(self.config.min_energy_window_samples)
        target_sample_rate = int(self.config.sample_rate)

        waveforms: list[np.ndarray] = []
        normalized_sample_rates: list[int] = []
        if audio_files is not None:
            for audio_file in audio_files:
                waveform, waveform_sr = self._load_and_resample_audio(
                    audio_file=audio_file, target_sample_rate=target_sample_rate
                )
                waveforms.append(waveform)
                normalized_sample_rates.append(waveform_sr)
        else:
            for audio, sample_rate in zip(audio_arrays, sample_rates):
                waveform, waveform_sr = self._load_and_resample_audio(
                    audio_array=audio, sample_rate=sample_rate, target_sample_rate=target_sample_rate
                )
                waveforms.append(waveform)
                normalized_sample_rates.append(waveform_sr)

        segment_waveforms, segment_sample_rates, segment_meta = self._prepare_segments(
            waveforms=waveforms,
            sample_rates=normalized_sample_rates,
            max_audio_clip_s=max_audio_clip_s,
            overlap_chunk_second=overlap_chunk_second,
            min_energy_window_samples=min_energy_window_samples,
        )
        segment_texts = self._transcribe_waveforms_batched(
            processor=processor,
            waveforms=segment_waveforms,
            sample_rates=segment_sample_rates,
            prompt_text=prompt_text,
            batch_size=effective_batch_size,
            max_new_tokens=256,
            pipeline_detokenization=pipeline_detokenization,
        )

        outputs = [""] * total_inputs
        chunked_outputs: dict[int, list[tuple[int, str]]] = {}
        for (sample_idx, chunk_idx), text in zip(segment_meta, segment_texts):
            if chunk_idx is None:
                outputs[sample_idx] = text
                continue
            if sample_idx not in chunked_outputs:
                chunked_outputs[sample_idx] = []
            chunked_outputs[sample_idx].append((chunk_idx, text))

        for sample_idx, chunk_items in chunked_outputs.items():
            chunk_items.sort(key=lambda item: item[0])
            outputs[sample_idx] = join_chunk_texts(
                [text for _, text in chunk_items], separator=get_chunk_separator(language)
            )

        return outputs

    def _transcribe_waveforms_batched(
        self,
        processor,
        waveforms: list[np.ndarray],
        sample_rates: list[int],
        prompt_text: str,
        batch_size: int,
        max_new_tokens: int,
        pipeline_detokenization: bool = False,
    ) -> list[str]:
        if not waveforms:
            return []

        transcriptions = [""] * len(waveforms)
        tokenizer = processor.tokenizer
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        ordered_indices = sorted(range(len(waveforms)), key=lambda idx: waveforms[idx].shape[0], reverse=True)
        previous_batch_decode_job = None
        previous_batch_indices: Optional[list[int]] = None

        for batch_order_indices in _batched_indices(len(ordered_indices), batch_size):
            batch_indices = [ordered_indices[i] for i in batch_order_indices]
            batch_waves = [waveforms[i] for i in batch_indices]
            batch_srs = [sample_rates[i] for i in batch_indices]
            if not all(sr == batch_srs[0] for sr in batch_srs):
                raise ValueError("Batched waveforms require a shared sampling rate.")
            prompts = [prompt_text] * len(batch_waves)
            inputs = processor(audio=batch_waves, text=prompts, sampling_rate=batch_srs[0], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "input_ids" in inputs and "decoder_input_ids" not in inputs:
                inputs["decoder_input_ids"] = inputs.pop("input_ids")
            if "decoder_input_ids" in inputs and "decoder_attention_mask" not in inputs:
                if pad_token_id is None:
                    inputs["decoder_attention_mask"] = torch.ones(
                        inputs["decoder_input_ids"].shape,
                        dtype=torch.long,
                        device=inputs["decoder_input_ids"].device,
                    )
                else:
                    inputs["decoder_attention_mask"] = inputs["decoder_input_ids"].ne(pad_token_id).long()

            with torch.inference_mode():
                generated_ids = self.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    decoder_start_token_id=int(inputs["decoder_input_ids"][0, 0].item()),
                    use_cache=True,
                )

            if "decoder_attention_mask" in inputs:
                prompt_lens = inputs["decoder_attention_mask"].sum(dim=1)
            elif "decoder_input_ids" in inputs:
                if pad_token_id is None:
                    prompt_lens = torch.full(
                        (inputs["decoder_input_ids"].shape[0],),
                        inputs["decoder_input_ids"].shape[1],
                        dtype=torch.long,
                        device=inputs["decoder_input_ids"].device,
                    )
                else:
                    prompt_lens = inputs["decoder_input_ids"].ne(pad_token_id).sum(dim=1)
            elif "attention_mask" in inputs:
                prompt_lens = inputs["attention_mask"].sum(dim=1)
            else:
                if pad_token_id is None:
                    prompt_lens = torch.full(
                        (inputs["input_ids"].shape[0],),
                        inputs["input_ids"].shape[1],
                        dtype=torch.long,
                        device=inputs["input_ids"].device,
                    )
                else:
                    prompt_lens = inputs["input_ids"].ne(pad_token_id).sum(dim=1)

            generated_ids = generated_ids.cpu().tolist()
            prompt_lens = prompt_lens.cpu().tolist()

            decoder_input_ids = None
            if "decoder_input_ids" in inputs:
                decoder_input_ids = inputs["decoder_input_ids"].cpu().tolist()

            trimmed_token_ids = []
            for row_idx, prompt_len in enumerate(prompt_lens):
                token_ids = generated_ids[row_idx]
                prompt_ids = decoder_input_ids[row_idx][:prompt_len]
                starts_with_prompt = (
                    prompt_len > 0 and len(token_ids) >= prompt_len and token_ids[:prompt_len] == prompt_ids
                )
                if starts_with_prompt:
                    token_ids = token_ids[prompt_len:]

                if eos_token_id is not None:
                    try:
                        token_ids = token_ids[: token_ids.index(eos_token_id)]
                    except ValueError:
                        pass

                trimmed_token_ids.append(token_ids)

            if pipeline_detokenization:
                # We use python multiprocessing to decode the tokens in a separate process so that, for all but
                # the final batch, CPU decoding can take place concurrently with GPU inference. This is only
                # necessary because we aren't using a fast rust tokenizer. The current tokenizer is slow and
                # steals the GIL if it is run in the main thread.
                if previous_batch_decode_job is not None and previous_batch_indices is not None:
                    ready_texts = previous_batch_decode_job.result()
                    for row_idx, text in enumerate(ready_texts):
                        transcriptions[previous_batch_indices[row_idx]] = text.strip()

                previous_batch_decode_job = self._decode_pool.submit(decode_worker_fn, trimmed_token_ids, True)
                previous_batch_indices = batch_indices
            else:
                texts = tokenizer.batch_decode(trimmed_token_ids, skip_special_tokens=True)
                for row_idx, text in enumerate(texts):
                    transcriptions[batch_indices[row_idx]] = text.strip()

        if previous_batch_decode_job is not None and previous_batch_indices is not None:
            ready_texts = previous_batch_decode_job.result()
            for row_idx, text in enumerate(ready_texts):
                transcriptions[previous_batch_indices[row_idx]] = text.strip()

        return transcriptions

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        cache_position=None,
        next_sequence_length=None,
        **kwargs,
    ):
        if next_sequence_length is not None:
            input_ids = input_ids[:, -next_sequence_length:]
        else:
            past_length = _get_cache_seq_length(past_key_values)
            if past_length > 0:
                input_ids = input_ids[:, -1:]

        if cache_position is not None:
            position_ids = cache_position[-input_ids.shape[1] :].unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            past_length = _get_cache_seq_length(past_key_values)
            position_ids = torch.arange(past_length, past_length + input_ids.shape[1], device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        return {
            "input_ids": input_ids,
            "positions": position_ids,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "input_features": kwargs.get("input_features"),
            "encoder_outputs": kwargs.get("encoder_outputs"),
            "length": kwargs.get("length"),
            "attention_mask": attention_mask,
            "cross_attention_mask": kwargs.get("cross_attention_mask"),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": kwargs.get("use_cache"),
        }

    def _ensure_decode_pool(self, processor):
        """
        Creates a single worker process for decoding tokens in a separate process.
        """
        tokenizer = processor.tokenizer
        if tokenizer is None:
            raise ValueError("processor.tokenizer is required for decode worker initialization.")

        spm_model_file = tokenizer.spm_model_file
        if not spm_model_file:
            raise ValueError("Tokenizer must expose spm_model_file for decode worker initialization.")

        if self._decode_pool is not None and self._decode_pool_spm_model_file == spm_model_file:
            return
        if self._decode_pool is not None:
            self._shutdown_decode_pool()

        tokenizer_init_kwargs = {
            "spm_model_file": spm_model_file,
            "bos_token": tokenizer.bos_token,
            "eos_token": tokenizer.eos_token,
            "unk_token": tokenizer.unk_token,
            "pad_token": tokenizer.pad_token,
            "additional_special_tokens": list(tokenizer.additional_special_tokens),
            "split_special_tokens": bool(getattr(tokenizer, "split_special_tokens", False)),
            "add_prefix_space": bool(getattr(tokenizer, "add_prefix_space", False)),
            "sp_model_kwargs": dict(getattr(tokenizer, "sp_model_kwargs", {}) or {}),
        }
        self._decode_pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=mp.get_context("fork"),
            initializer=decode_worker_init,
            initargs=(tokenizer_init_kwargs,),
        )
        self._decode_pool_spm_model_file = spm_model_file
        atexit.register(self._shutdown_decode_pool)

    def _shutdown_decode_pool(self):
        if self._decode_pool is None:
            return
        self._decode_pool.shutdown(wait=True)
        self._decode_pool = None
        self._decode_pool_spm_model_file = None


def _batched_indices(total: int, batch_size: int) -> list[list[int]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    return [list(range(i, min(i + batch_size, total))) for i in range(0, total, batch_size)]


DECODE_WORKER_TOKENIZER = None


def decode_worker_init(tokenizer_init_kwargs: dict):
    from .tokenization_cohere_asr import CohereAsrTokenizer

    global DECODE_WORKER_TOKENIZER
    DECODE_WORKER_TOKENIZER = CohereAsrTokenizer(**tokenizer_init_kwargs)


def decode_worker_fn(trimmed_token_ids: list[list[int]], skip_special_tokens: bool) -> list[str]:
    if DECODE_WORKER_TOKENIZER is None:
        raise RuntimeError("Decode worker tokenizer was not initialized.")
    return DECODE_WORKER_TOKENIZER.batch_decode(trimmed_token_ids, skip_special_tokens=skip_special_tokens)


def _align_decoder_attention_mask(decoder_attention_mask: torch.Tensor, total_kv_len: int) -> torch.Tensor:
    current_len = int(decoder_attention_mask.shape[-1])
    if current_len < total_kv_len:
        # Decoder masks are prefix-aligned and should grow toward the right as
        # autoregressive generation appends tokens.
        pad = torch.ones(
            (decoder_attention_mask.shape[0], total_kv_len - current_len),
            device=decoder_attention_mask.device,
            dtype=decoder_attention_mask.dtype,
        )
        return torch.cat([decoder_attention_mask, pad], dim=-1)
    if current_len > total_kv_len:
        return decoder_attention_mask[:, -total_kv_len:]
    return decoder_attention_mask


def _get_cache_seq_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, tuple) and past_key_values:
        return int(past_key_values[0][0][0].shape[-2])
    return 0


def _get_static_cache_len(past_key_values) -> Optional[int]:
    """Return self-attention max_cache_len for StaticCache, otherwise None."""
    cache = past_key_values
    if isinstance(cache, EncoderDecoderCache):
        cache = cache.self_attention_cache
    if isinstance(cache, StaticCache) and cache.layers:
        return cache.layers[0].max_cache_len
    return None


def _get_cache_kv(cache_layer, layer_idx: int):
    if hasattr(cache_layer, "layers"):
        if layer_idx < len(cache_layer.layers):
            layer = cache_layer.layers[layer_idx]
            return layer.keys, layer.values
        return None, None

    key_cache = getattr(cache_layer, "key_cache", None)
    value_cache = getattr(cache_layer, "value_cache", None)
    if key_cache is not None and value_cache is not None and layer_idx < len(key_cache):
        return key_cache[layer_idx], value_cache[layer_idx]

    return None, None


# --- Automatic chunking helper functions ---


def split_audio_chunks_energy(
    waveform: np.ndarray,
    sample_rate: int,
    max_audio_clip_s: float,
    overlap_chunk_second: float,
    min_energy_window_samples: int,
) -> list[np.ndarray]:
    """
    Split audio waveform into chunks based on energy-based boundaries.
    """
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono waveform (1D), got shape={waveform.shape}")
    chunk_size = max(1, int(round(max_audio_clip_s * sample_rate)))
    # NeMo parity: overlap_chunk_second in energy_split mode is the split-search
    # context near the chunk boundary, not literal waveform overlap between chunks.
    boundary_context_size = max(1, int(round(overlap_chunk_second * sample_rate)))
    total_samples = waveform.shape[0]
    if total_samples <= chunk_size:
        return [waveform.copy()]

    chunks_meta: list[tuple[int, int]] = []
    idx = 0
    while idx < total_samples:
        if idx + chunk_size >= total_samples:
            chunks_meta.append((idx, total_samples))
            break

        search_start = max(idx, idx + chunk_size - boundary_context_size)
        search_end = min(idx + chunk_size, total_samples)
        if search_end <= search_start:
            split_point = idx + chunk_size
        else:
            split_point = _find_split_point_energy(
                waveform,
                start_idx=search_start,
                end_idx=search_end,
                min_energy_window_samples=min_energy_window_samples,
            )
        split_point = max(idx + 1, min(split_point, total_samples))
        chunks_meta.append((idx, split_point))
        idx = split_point

    return [waveform[start:end].copy() for start, end in chunks_meta if end > start]


def _find_split_point_energy(
    waveform: np.ndarray, start_idx: int, end_idx: int, min_energy_window_samples: int
) -> int:
    segment = waveform[start_idx:end_idx]
    if segment.shape[0] <= min_energy_window_samples:
        return (start_idx + end_idx) // 2

    min_energy = float("inf")
    quietest_idx = start_idx
    upper = segment.shape[0] - min_energy_window_samples
    for i in range(0, upper, min_energy_window_samples):
        window = segment[i : i + min_energy_window_samples]
        energy = float(np.sqrt(np.mean(window * window)))
        if energy < min_energy:
            min_energy = energy
            quietest_idx = start_idx + i
    return quietest_idx


def join_chunk_texts(texts: list[str], separator: str = " ") -> str:
    parts = [piece.strip() for piece in texts if piece and piece.strip()]
    if not parts:
        return ""
    return separator.join(parts)


def get_chunk_separator(language: str) -> str:
    return "" if language in NO_SPACE_LANGS else " "
