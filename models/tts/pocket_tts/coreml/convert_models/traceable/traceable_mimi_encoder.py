"""Traceable Mimi Encoder for CoreML conversion.

Converts audio waveform to voice conditioning embeddings for PocketTTS.

Pipeline: audio [1, 1, T] → SEANet encoder → transformer → downsample → speaker_proj → conditioning [1, F, 1024]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonStreamingConv1d(nn.Module):
    """Non-streaming Conv1d wrapper for tracing.

    Replaces StreamingConv1d by pre-computing all padding.
    Works with batch_size=1 for voice cloning.
    """

    def __init__(self, streaming_conv):
        super().__init__()
        self.conv = streaming_conv.conv
        self.pad_mode = streaming_conv.pad_mode
        self._stride = streaming_conv._stride
        self._kernel_size = streaming_conv._kernel_size
        self._effective_kernel_size = streaming_conv._effective_kernel_size

    def forward(self, x):
        # Compute padding needed (same as init_state would provide)
        kernel = self._effective_kernel_size
        stride = self._stride
        pad_left = kernel - stride

        # Handle replicate padding
        if pad_left > 0:
            if self.pad_mode == "replicate":
                # Replicate first value for padding
                init = x[..., :1].expand(-1, -1, pad_left)
                x = torch.cat([init, x], dim=-1)
            else:
                # Zero padding
                x = F.pad(x, (pad_left, 0))

        return self.conv(x)


class NonStreamingResnetBlock(nn.Module):
    """Non-streaming ResnetBlock wrapper."""

    def __init__(self, resnet_block):
        super().__init__()
        self.block = nn.ModuleList()
        for layer in resnet_block.block:
            if hasattr(layer, 'conv'):  # StreamingConv1d
                self.block.append(NonStreamingConv1d(layer))
            else:
                self.block.append(layer)

    def forward(self, x):
        v = x
        for layer in self.block:
            v = layer(v)
        return x + v


class NonStreamingEncoder(nn.Module):
    """Non-streaming SEANet encoder wrapper."""

    def __init__(self, encoder):
        super().__init__()
        self.model = nn.ModuleList()
        for layer in encoder.model:
            if hasattr(layer, 'conv'):  # StreamingConv1d
                self.model.append(NonStreamingConv1d(layer))
            elif hasattr(layer, 'block'):  # SEANetResnetBlock
                self.model.append(NonStreamingResnetBlock(layer))
            else:
                self.model.append(layer)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class NonStreamingTransformerLayer(nn.Module):
    """Non-streaming transformer layer wrapper."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        # The transformer layer expects model_state but we can pass None
        # and it should work for non-streaming
        return self.layer(x, model_state=None)


class NonStreamingTransformer(nn.Module):
    """Non-streaming transformer wrapper."""

    def __init__(self, transformer):
        super().__init__()
        self.layers = nn.ModuleList([
            NonStreamingTransformerLayer(layer)
            for layer in transformer.layers
        ])
        self.input_norm = transformer.input_norm if hasattr(transformer, 'input_norm') else None
        self.output_norm = transformer.output_norm if hasattr(transformer, 'output_norm') else None

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C] for transformer
        x = x.transpose(-1, -2)

        if self.input_norm is not None:
            x = self.input_norm(x)

        for layer in self.layers:
            x = layer(x)

        if self.output_norm is not None:
            x = self.output_norm(x)

        # Back to [B, C, T]
        x = x.transpose(-1, -2)
        return x


class NonStreamingDownsample(nn.Module):
    """Non-streaming downsample wrapper.

    ConvDownsample1d uses StreamingConv1d internally, so we need to
    wrap the inner conv with our non-streaming version.
    """

    def __init__(self, downsample):
        super().__init__()
        # ConvDownsample1d has a .conv attribute which is StreamingConv1d
        if hasattr(downsample, 'conv'):
            self.conv = NonStreamingConv1d(downsample.conv)
        else:
            self.conv = downsample

    def forward(self, x):
        return self.conv(x)


class TraceableMimiEncoderSimple(nn.Module):
    """Simplified traceable encoder that works with JIT tracing.

    Wraps the Mimi encoder components in non-streaming wrappers
    that handle padding statically rather than dynamically.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_transformer: nn.Module,
        downsample: nn.Module,
        speaker_proj_weight: torch.Tensor,
    ):
        super().__init__()
        self.encoder = NonStreamingEncoder(encoder)
        self.encoder_transformer = encoder_transformer  # ProjectedTransformer handles model_state=None
        self.downsample = NonStreamingDownsample(downsample)
        self.speaker_proj_weight = nn.Parameter(speaker_proj_weight)

    @classmethod
    def from_tts_model(cls, tts_model) -> "TraceableMimiEncoderSimple":
        """Create from a loaded TTSModel instance."""
        mimi = tts_model.mimi
        speaker_proj = tts_model.flow_lm.speaker_proj_weight.data

        return cls(
            encoder=mimi.encoder,
            encoder_transformer=mimi.encoder_transformer,
            downsample=mimi.downsample,
            speaker_proj_weight=speaker_proj,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode pre-padded audio to voice conditioning.

        Args:
            audio: Input waveform [1, 1, T] at 24kHz, pre-padded to frame boundary

        Returns:
            Voice conditioning [1, num_frames, 1024]
        """
        # SEANet encoder (non-streaming wrapper)
        emb = self.encoder(audio)

        # Encoder transformer (ProjectedTransformer)
        # It handles transposition internally: [B, C, T] -> [B, T, C] -> transformer -> [B, C, T]
        emb_list = self.encoder_transformer(emb, model_state=None)
        emb = emb_list[0]  # ProjectedTransformer returns a list

        # Downsample to target frame rate (non-streaming wrapper)
        emb = self.downsample(emb)

        # emb: [1, 512, num_frames] -> [1, num_frames, 512]
        latents = emb.transpose(-1, -2).to(torch.float32)

        # Speaker projection: [1, num_frames, 512] @ [512, 1024] -> [1, num_frames, 1024]
        conditioning = F.linear(latents, self.speaker_proj_weight)

        return conditioning


# Keep the original class for reference
class TraceableMimiEncoder(nn.Module):
    """Original traceable wrapper (may not work with JIT tracing due to beartype)."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_transformer: nn.Module,
        downsample: nn.Module,
        speaker_proj_weight: torch.Tensor,
        frame_size: int = 1920,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_transformer = encoder_transformer
        self.downsample = downsample
        self.speaker_proj_weight = nn.Parameter(speaker_proj_weight)
        self.frame_size = frame_size

    @classmethod
    def from_tts_model(cls, tts_model) -> "TraceableMimiEncoder":
        """Create from a loaded TTSModel instance."""
        mimi = tts_model.mimi
        speaker_proj = tts_model.flow_lm.speaker_proj_weight.data

        return cls(
            encoder=mimi.encoder,
            encoder_transformer=mimi.encoder_transformer,
            downsample=mimi.downsample,
            speaker_proj_weight=speaker_proj,
            frame_size=mimi.frame_size,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to voice conditioning."""
        # Pad to multiple of frame_size
        _, _, length = audio.shape
        pad_length = (self.frame_size - (length % self.frame_size)) % self.frame_size
        if pad_length > 0:
            audio = F.pad(audio, (0, pad_length))

        # SEANet encoder
        emb = self.encoder(audio, model_state=None)

        # Encoder transformer
        emb, = self.encoder_transformer(emb, model_state=None)

        # Downsample
        emb = self.downsample(emb, model_state=None)

        # emb: [1, 512, num_frames] -> [1, num_frames, 512]
        latents = emb.transpose(-1, -2).to(torch.float32)

        # Speaker projection
        conditioning = F.linear(latents, self.speaker_proj_weight)

        return conditioning
