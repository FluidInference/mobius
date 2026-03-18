from __future__ import annotations
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')

import math
from dataclasses import dataclass
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F

import feature as ls_feature
from nnet.model.onl_conformer_retention_enc_1dcnn_tfm_retention_enc_linear_non_autoreg_pos_enc_l2norm_emb_loss_mask import (
    OnlineConformerRetentionDADiarization,
)


@dataclass
class StreamingUpdate:
    start_frame: int
    logits: np.ndarray
    probabilities: np.ndarray
    preview_start_frame: int
    preview_logits: np.ndarray
    preview_probabilities: np.ndarray
    frame_hz: float
    duration_seconds: float
    total_emitted_frames: int


class StreamingFeatureExtractor:
    """Incremental feature extraction that matches the repo's offline path."""

    def __init__(self, config: dict) -> None:
        feat_config = config["data"]["feat"]
        self.sample_rate = int(feat_config["sample_rate"])
        self.frame_size = int(feat_config["win_length"])
        self.n_fft = 1 << (self.frame_size - 1).bit_length()
        self.hop_length = int(feat_config["hop_length"])
        self.n_mels = int(feat_config["n_mels"])
        self.context_size = int(config["data"]["context_recp"])
        self.subsampling = int(config["data"]["subsampling"])
        self.transform_type = str(config["data"]["feat_type"])
        self.left_pad = self.n_fft // 2
        self.model_input_dim = (2 * self.context_size + 1) * self.n_mels

        self.audio_buffer = np.zeros(0, dtype=np.float32)
        self.audio_start_sample = 0
        self.total_samples = 0

        self.next_stft_frame = 0
        self.next_model_frame = 0

        self.base_feature_start = 0
        self.base_feature_buffer = np.zeros((0, self.n_mels), dtype=np.float32)
        self.cumulative_feature_sum = np.zeros(self.n_mels, dtype=np.float64)

        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
        )

    def push_audio(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return np.zeros((0, self.model_input_dim), dtype=np.float32)
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk.astype(np.float32, copy=False)], axis=0)
        self.total_samples += len(chunk)
        self._append_stft_frames(self._stable_stft_frame_count(), allow_right_pad=False)
        return self._emit_model_frames(final=False)

    def finalize(self) -> np.ndarray:
        usable_samples = self._usable_sample_count(self.total_samples)
        total_stft_frames = self._offline_stft_frame_count(usable_samples)
        self._append_stft_frames(total_stft_frames, allow_right_pad=True, effective_total_samples=usable_samples)
        return self._emit_model_frames(final=True, total_stft_frames=total_stft_frames)

    def _usable_sample_count(self, sample_count: int) -> int:
        block_size = self.hop_length * self.subsampling
        return (sample_count // block_size) * block_size

    def _stable_stft_frame_count(self) -> int:
        if self.total_samples <= self.left_pad:
            return 0
        return max(0, (self.total_samples - self.left_pad) // self.hop_length + 1)

    def _offline_stft_frame_count(self, usable_samples: int) -> int:
        if usable_samples <= 0:
            return 0
        return max(0, usable_samples // self.hop_length - 1)

    def _model_frame_count(self, total_stft_frames: int) -> int:
        if total_stft_frames <= 0:
            return 0
        return (total_stft_frames + self.subsampling - 1) // self.subsampling

    def _append_stft_frames(
        self,
        target_frame_count: int,
        allow_right_pad: bool,
        effective_total_samples: int | None = None,
    ) -> None:
        if target_frame_count <= self.next_stft_frame:
            return
        frame_start = self.next_stft_frame
        frame_stop = target_frame_count
        segment = self._stft_segment(
            frame_start=frame_start,
            frame_stop=frame_stop,
            allow_right_pad=allow_right_pad,
            effective_total_samples=effective_total_samples,
        )
        stft = librosa.stft(
            segment,
            n_fft=self.n_fft,
            win_length=self.frame_size,
            hop_length=self.hop_length,
            center=False,
        ).T
        expected_frames = frame_stop - frame_start
        if stft.shape[0] < expected_frames:
            raise RuntimeError(
                f"Streaming STFT underflow: expected {expected_frames} frames, got {stft.shape[0]}"
            )
        stft = stft[:expected_frames]
        transformed = self._transform_batch(stft, frame_start)
        if self.base_feature_buffer.size == 0:
            self.base_feature_buffer = transformed
        else:
            self.base_feature_buffer = np.concatenate([self.base_feature_buffer, transformed], axis=0)
        self.next_stft_frame = frame_stop
        self._drop_consumed_audio()

    def _stft_segment(
        self,
        frame_start: int,
        frame_stop: int,
        allow_right_pad: bool,
        effective_total_samples: int | None,
    ) -> np.ndarray:
        if frame_stop <= frame_start:
            return np.zeros(0, dtype=np.float32)
        total_samples = self.total_samples if effective_total_samples is None else effective_total_samples
        global_start = frame_start * self.hop_length - self.left_pad
        global_stop = (frame_stop - 1) * self.hop_length - self.left_pad + self.n_fft

        prefix = np.zeros(max(0, -global_start), dtype=np.float32)
        suffix = np.zeros(max(0, global_stop - total_samples), dtype=np.float32) if allow_right_pad else np.zeros(0, dtype=np.float32)

        raw_start = max(0, global_start)
        raw_stop = min(total_samples, global_stop)
        if raw_start < self.audio_start_sample:
            raise RuntimeError(
                f"Audio buffer underflow: need sample {raw_start}, buffer starts at {self.audio_start_sample}"
            )
        local_start = raw_start - self.audio_start_sample
        local_stop = raw_stop - self.audio_start_sample
        core = self.audio_buffer[local_start:local_stop]
        if prefix.size == 0 and suffix.size == 0:
            return core
        return np.concatenate([prefix, core, suffix], axis=0)

    def _transform_batch(self, stft: np.ndarray, frame_start: int) -> np.ndarray:
        if self.transform_type != "logmel23_cummn":
            return ls_feature.transform(stft, self.transform_type)
        magnitude = np.abs(stft)
        mel = np.dot(magnitude ** 2, self.mel_basis.T)
        logmel = np.log10(np.maximum(mel, 1e-10))
        counts = np.arange(frame_start + 1, frame_start + 1 + len(logmel), dtype=np.float64)[:, None]
        cumsum = np.cumsum(logmel, axis=0, dtype=np.float64) + self.cumulative_feature_sum[None, :]
        cummean = cumsum / counts
        self.cumulative_feature_sum = cumsum[-1]
        return (logmel - cummean).astype(np.float32, copy=False)

    def _emit_model_frames(self, final: bool, total_stft_frames: int | None = None) -> np.ndarray:
        outputs: list[np.ndarray] = []
        latest_frame = self.next_stft_frame - 1
        total_model_frames = self._model_frame_count(total_stft_frames or 0) if final else None
        while True:
            center_index = self.next_model_frame * self.subsampling
            if final:
                if total_model_frames is None or self.next_model_frame >= total_model_frames:
                    break
                max_index = (total_stft_frames or 0) - 1
            else:
                if center_index + self.context_size > latest_frame:
                    break
                max_index = latest_frame
            outputs.append(self._splice_frame(center_index, max_index))
            self.next_model_frame += 1
            self._drop_consumed_base_features()
        if not outputs:
            return np.zeros((0, self.model_input_dim), dtype=np.float32)
        return np.stack(outputs, axis=0).astype(np.float32, copy=False)

    def _splice_frame(self, center_index: int, max_index: int) -> np.ndarray:
        pieces = []
        for frame_index in range(center_index - self.context_size, center_index + self.context_size + 1):
            if frame_index < 0 or frame_index > max_index:
                pieces.append(np.zeros(self.n_mels, dtype=np.float32))
                continue
            local_index = frame_index - self.base_feature_start
            if local_index < 0 or local_index >= self.base_feature_buffer.shape[0]:
                raise RuntimeError(
                    f"Feature buffer underflow: need frame {frame_index}, buffer covers "
                    f"[{self.base_feature_start}, {self.base_feature_start + self.base_feature_buffer.shape[0] - 1}]"
                )
            pieces.append(self.base_feature_buffer[local_index])
        return np.concatenate(pieces, axis=0)

    def _drop_consumed_audio(self) -> None:
        keep_from = max(0, self.next_stft_frame * self.hop_length - self.left_pad)
        drop = keep_from - self.audio_start_sample
        if drop <= 0:
            return
        self.audio_buffer = self.audio_buffer[drop:]
        self.audio_start_sample += drop

    def _drop_consumed_base_features(self) -> None:
        keep_from = max(0, self.next_model_frame * self.subsampling - self.context_size)
        drop = keep_from - self.base_feature_start
        if drop <= 0:
            return
        self.base_feature_buffer = self.base_feature_buffer[drop:]
        self.base_feature_start += drop


class StatefulLSEENDRunner:
    """Frame-wise LS-EEND forward pass with recurrent retention state."""

    def __init__(self, model: OnlineConformerRetentionDADiarization, max_nspks: int) -> None:
        self.model = model
        self.max_nspks = int(max_nspks)
        self.device = next(model.parameters()).device
        self.encoder_states = [
            {"retention": {}, "time_index": 0, "conv_buffer": None}
            for _ in self.model.enc.encoder.layers
        ]
        self.decoder_states = [
            {"retention": {}, "time_index": 0}
            for _ in self.model.dec.attractor_decoder.layers
        ]
        self.encoder_history: list[torch.Tensor] = []
        self.encoder_history_start = 0
        self.total_encoder_frames = 0
        self.next_output_frame = 0
        encoder_dim = int(self.model.cnn.in_channels)
        self.zero_encoder_frame = torch.zeros((1, 1, encoder_dim), device=self.device)

    def push_features(self, features: np.ndarray) -> torch.Tensor:
        if features.size == 0:
            return self.zero_encoder_frame.new_zeros((0, self.max_nspks))
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for frame in features:
                x = torch.from_numpy(frame).to(self.device).unsqueeze(0).unsqueeze(0)
                encoder_frame = self._encoder_step(x)
                self.encoder_history.append(encoder_frame)
                self.total_encoder_frames += 1
                ready_index = self.total_encoder_frames - self.model.delay - 1
                if ready_index >= self.next_output_frame:
                    delayed_emb = self._delayed_embedding(self.next_output_frame, allow_right_pad=False)
                    outputs.append(self._decoder_step(delayed_emb))
                    self.next_output_frame += 1
                    self._drop_consumed_encoder_history()
        if not outputs:
            return self.zero_encoder_frame.new_zeros((0, self.max_nspks))
        return torch.cat(outputs, dim=0)

    def finalize(self) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            while self.next_output_frame < self.total_encoder_frames:
                delayed_emb = self._delayed_embedding(self.next_output_frame, allow_right_pad=True)
                outputs.append(self._decoder_step(delayed_emb))
                self.next_output_frame += 1
                self._drop_consumed_encoder_history()
        if not outputs:
            return self.zero_encoder_frame.new_zeros((0, self.max_nspks))
        return torch.cat(outputs, dim=0)

    def peek_preview(self) -> torch.Tensor:
        if self.next_output_frame >= self.total_encoder_frames:
            return self.zero_encoder_frame.new_zeros((0, self.max_nspks))
        preview_states = [self._clone_decoder_state(state) for state in self.decoder_states]
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for frame_index in range(self.next_output_frame, self.total_encoder_frames):
                delayed_emb = self._delayed_embedding(frame_index, allow_right_pad=True)
                outputs.append(self._decoder_step(delayed_emb, preview_states))
        if not outputs:
            return self.zero_encoder_frame.new_zeros((0, self.max_nspks))
        return torch.cat(outputs, dim=0)

    def _encoder_step(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.enc.encoder.input_projection(x)
        x = self.model.enc.encoder.layer_norm(x)
        for layer, state in zip(self.model.enc.encoder.layers, self.encoder_states):
            x = self._conformer_block_step(layer, x, state)
        return x

    def _conformer_block_step(self, block: torch.nn.Module, x: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
        ff1 = block.sequential[0]
        attn = block.sequential[1].module
        conv = block.sequential[2].module
        ff2 = block.sequential[3]
        final_norm = block.sequential[4]

        x = ff1.module(x) * ff1.module_factor + x * ff1.input_factor

        attn_input = attn.layer_norm(x)
        rel_pos = attn.ret_pos(slen=state["time_index"] + 1, activate_recurrent=True)
        attn_output = attn.self_attn(attn_input, rel_pos=rel_pos, incremental_state=state["retention"])
        state["time_index"] += 1
        x = x + attn.dropout(attn_output)

        conv_output, conv_buffer = self._causal_conv_step(conv, x, state["conv_buffer"])
        state["conv_buffer"] = conv_buffer
        x = x + conv_output

        x = ff2.module(x) * ff2.module_factor + x * ff2.input_factor
        return final_norm(x)

    def _causal_conv_step(
        self,
        conv_module: torch.nn.Module,
        x: torch.Tensor,
        cached_inputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_size = int(conv_module.sequential[4].conv.kernel_size[0])
        window = x if cached_inputs is None else torch.cat([cached_inputs, x], dim=1)
        output = conv_module(window)[:, -1:, :]
        keep = max(0, kernel_size - 1)
        next_cache = window[:, -keep:, :].detach() if keep > 0 else window[:, :0, :].detach()
        return output, next_cache

    def _delayed_embedding(self, center_index: int, allow_right_pad: bool) -> torch.Tensor:
        delay = int(self.model.delay)
        window: list[torch.Tensor] = []
        for frame_index in range(center_index - delay, center_index + delay + 1):
            if frame_index < 0 or frame_index >= self.total_encoder_frames:
                if frame_index >= self.total_encoder_frames and not allow_right_pad:
                    raise RuntimeError("Requested non-final delayed embedding without enough future context.")
                window.append(self.zero_encoder_frame)
                continue
            local_index = frame_index - self.encoder_history_start
            if local_index < 0 or local_index >= len(self.encoder_history):
                raise RuntimeError(
                    f"Encoder history underflow: need frame {frame_index}, history covers "
                    f"[{self.encoder_history_start}, {self.encoder_history_start + len(self.encoder_history) - 1}]"
                )
            window.append(self.encoder_history[local_index])
        stacked = torch.cat(window, dim=1)
        convolved = F.conv1d(
            stacked.transpose(1, 2),
            self.model.cnn.weight,
            self.model.cnn.bias,
        ).transpose(1, 2)
        return convolved / torch.norm(convolved, dim=-1, keepdim=True).clamp_min(1e-12)

    def _decoder_step(self, emb: torch.Tensor, decoder_states: list[dict[str, Any]] | None = None) -> torch.Tensor:
        if decoder_states is None:
            decoder_states = self.decoder_states
        pos_enc = self.model.dec.pos_enc(emb, self.max_nspks)
        repeated_emb = emb.unsqueeze(dim=2).repeat(1, 1, self.max_nspks, 1)
        attractors = self.model.dec.convert(torch.cat([repeated_emb, pos_enc], dim=-1))
        for layer, state in zip(self.model.dec.attractor_decoder.layers, decoder_states):
            attractors = self._fusion_layer_step(layer, attractors, state)
        if self.model.dec.attractor_decoder.norm is not None:
            attractors = self.model.dec.attractor_decoder.norm(attractors)
        attractors = attractors / torch.norm(attractors, dim=-1, keepdim=True).clamp_min(1e-12)
        logits = torch.matmul(emb.unsqueeze(dim=-2), attractors.transpose(-1, -2)).squeeze(dim=-2)
        return logits.squeeze(0)

    def _fusion_layer_step(
        self,
        layer: torch.nn.Module,
        src: torch.Tensor,
        state: dict[str, Any],
    ) -> torch.Tensor:
        batch_size, time_steps, speaker_count, feat_dim = src.shape
        x = src.transpose(1, 2).reshape(batch_size * speaker_count, time_steps, feat_dim)

        if layer.norm_first:
            time_input = layer.norm11(x)
            time_output = self._retention_step(layer.self_attn1, layer.ret_pos1, time_input, state)
            x = x + layer.dropout11(time_output)
        else:
            time_output = self._retention_step(layer.self_attn1, layer.ret_pos1, x, state)
            x = layer.norm11(x + layer.dropout11(time_output))

        x = x.reshape(batch_size, speaker_count, time_steps, feat_dim).transpose(1, 2)
        x = x.reshape(batch_size * time_steps, speaker_count, feat_dim)

        if layer.norm_first:
            x = x + layer._sa_block2(layer.norm21(x), None, None)
            x = x + layer._ff_block(layer.norm22(x))
        else:
            x = layer.norm21(x + layer._sa_block2(x, None, None))
            x = layer.norm22(x + layer._ff_block(x))

        return x.reshape(batch_size, time_steps, speaker_count, feat_dim)

    def _retention_step(
        self,
        retention_module: torch.nn.Module,
        rel_pos_module: torch.nn.Module,
        x: torch.Tensor,
        state: dict[str, Any],
    ) -> torch.Tensor:
        rel_pos = rel_pos_module(slen=state["time_index"] + 1, activate_recurrent=True)
        state["time_index"] += 1
        return retention_module(x, rel_pos=rel_pos, incremental_state=state["retention"])

    def _drop_consumed_encoder_history(self) -> None:
        keep_from = max(0, self.next_output_frame - int(self.model.delay))
        drop = keep_from - self.encoder_history_start
        if drop <= 0:
            return
        self.encoder_history = self.encoder_history[drop:]
        self.encoder_history_start += drop

    def _clone_decoder_state(self, state: dict[str, Any]) -> dict[str, Any]:
        retention = {
            key: value.clone()
            for key, value in state["retention"].items()
        }
        return {
            "retention": retention,
            "time_index": int(state["time_index"]),
        }
