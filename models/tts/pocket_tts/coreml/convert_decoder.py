"""Convert PocketTTS Mimi decoder to CoreML-compatible format.

This script loads the original PocketTTS model and creates a traceable
wrapper with explicit state inputs/outputs.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, List, Dict
import sys
import os

# Add pocket_tts to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coreml_modules import (
    CoreMLStreamingConv1d,
    CoreMLStreamingConvTranspose1d,
    CoreMLStreamingAttention
)


class CoreMLMimiDecoderWrapper(nn.Module):
    """Traceable wrapper for Mimi decoder with explicit state I/O.

    This wraps the decoder path: upsample -> decoder_transformer -> decoder (SEANet)

    State layout (all float32 for CoreML):
    - upsample_partial: [B, 512, 16]
    - decoder_conv_0_prev: [B, 512, 6], decoder_conv_0_first: [B]
    - decoder_convtr_2_partial: [B, 256, 6]
    - decoder_conv_3_1_prev: [B, 256, 2], decoder_conv_3_1_first: [B]
    - decoder_conv_3_3_prev: [B, 128, 0], decoder_conv_3_3_first: [B]
    - ... (etc for all layers)
    - attn_0_cache: [2, B, 8, 256, 64], attn_0_offset: [B], attn_0_end_offset: [B]
    - attn_1_cache: [2, B, 8, 256, 64], attn_1_offset: [B], attn_1_end_offset: [B]
    """

    def __init__(self):
        super().__init__()

        # We'll populate these from the original model
        self.dimension = 512
        self.num_heads = 8
        self.capacity = 256
        self.context = 256

        # Upsample (ConvTrUpsample1d with stride 8)
        self.upsample_convtr = None  # CoreMLStreamingConvTranspose1d

        # Transformer (2 layers)
        self.transformer_layers = nn.ModuleList()

        # SEANet decoder layers - stored in order
        self.decoder_layers = nn.ModuleList()  # Mix of conv, convtr, ELU
        self.decoder_layer_types = []  # Track type for each layer

    @classmethod
    def from_original(cls, mimi_model) -> "CoreMLMimiDecoderWrapper":
        """Create wrapper from original Mimi model, copying weights."""
        wrapper = cls()

        # Copy upsample
        orig_upsample = mimi_model.upsample.convtr  # StreamingConvTranspose1d
        wrapper.upsample_convtr = CoreMLStreamingConvTranspose1d(orig_upsample.convtr)

        # Copy transformer layers
        orig_transformer = mimi_model.decoder_transformer.transformer
        for orig_layer in orig_transformer.layers:
            # Create CoreML-compatible layer
            attn = CoreMLStreamingAttention(
                embed_dim=wrapper.dimension,
                num_heads=wrapper.num_heads,
                capacity=wrapper.capacity,
                context=wrapper.context,
                rope_max_period=orig_transformer.max_period
            )
            # Copy attention weights
            attn.in_proj.weight.data.copy_(orig_layer.self_attn.in_proj.weight.data)
            attn.out_proj.weight.data.copy_(orig_layer.self_attn.out_proj.weight.data)

            # Create full layer dict with norms and FFN
            layer_dict = nn.ModuleDict({
                'self_attn': attn,
                'norm1': orig_layer.norm1,
                'norm2': orig_layer.norm2,
                'linear1': orig_layer.linear1,
                'linear2': orig_layer.linear2,
                'layer_scale_1': orig_layer.layer_scale_1,
                'layer_scale_2': orig_layer.layer_scale_2,
            })
            wrapper.transformer_layers.append(layer_dict)

        # Copy SEANet decoder layers
        for i, layer in enumerate(mimi_model.decoder.model):
            if hasattr(layer, 'convtr'):  # StreamingConvTranspose1d
                wrapper.decoder_layers.append(
                    CoreMLStreamingConvTranspose1d(layer.convtr)
                )
                wrapper.decoder_layer_types.append('convtr')
            elif hasattr(layer, 'conv'):  # StreamingConv1d
                wrapper.decoder_layers.append(
                    CoreMLStreamingConv1d(layer.conv, getattr(layer, 'pad_mode', 'constant'))
                )
                wrapper.decoder_layer_types.append('conv')
            elif hasattr(layer, 'block'):  # SEANetResnetBlock
                # Unroll the residual block
                wrapper.decoder_layer_types.append('residual_start')
                wrapper.decoder_layers.append(nn.Identity())  # Placeholder

                for j, sublayer in enumerate(layer.block):
                    if hasattr(sublayer, 'conv'):  # StreamingConv1d inside block
                        wrapper.decoder_layers.append(
                            CoreMLStreamingConv1d(sublayer.conv, getattr(sublayer, 'pad_mode', 'constant'))
                        )
                        wrapper.decoder_layer_types.append('conv')
                    else:  # ELU
                        wrapper.decoder_layers.append(sublayer)
                        wrapper.decoder_layer_types.append('elu')

                wrapper.decoder_layer_types.append('residual_end')
                wrapper.decoder_layers.append(nn.Identity())  # Placeholder
            else:  # ELU or other
                wrapper.decoder_layers.append(layer)
                wrapper.decoder_layer_types.append('elu')

        return wrapper

    def get_num_conv_states(self) -> int:
        """Count number of conv layers needing state."""
        return sum(1 for t in self.decoder_layer_types if t == 'conv')

    def get_num_convtr_states(self) -> int:
        """Count number of convtr layers needing state (including upsample)."""
        return 1 + sum(1 for t in self.decoder_layer_types if t == 'convtr')

    def init_state(self, batch_size: int = 1) -> Dict[str, torch.Tensor]:
        """Initialize all state tensors."""
        state = {}

        # Upsample state
        state['upsample_partial'] = self.upsample_convtr.init_state(batch_size)

        # Transformer states
        for i, layer in enumerate(self.transformer_layers):
            cache, offset, end_offset = layer['self_attn'].init_state(batch_size)
            state[f'attn_{i}_cache'] = cache
            state[f'attn_{i}_offset'] = offset.float()  # Convert to float for CoreML
            state[f'attn_{i}_end_offset'] = end_offset.float()

        # Decoder conv/convtr states
        conv_idx = 0
        convtr_idx = 0
        for i, (layer, layer_type) in enumerate(zip(self.decoder_layers, self.decoder_layer_types)):
            if layer_type == 'conv':
                prev, first = layer.init_state(batch_size)
                state[f'decoder_conv_{conv_idx}_prev'] = prev
                state[f'decoder_conv_{conv_idx}_first'] = first
                conv_idx += 1
            elif layer_type == 'convtr':
                state[f'decoder_convtr_{convtr_idx}_partial'] = layer.init_state(batch_size)
                convtr_idx += 1

        return state


def test_wrapper_creation():
    """Test creating wrapper from original model."""
    from pocket_tts.models.mimi import MimiModel
    from pocket_tts.modules.seanet import SEANetEncoder, SEANetDecoder
    from pocket_tts.modules.dummy_quantizer import DummyQuantizer
    from pocket_tts.modules.mimi_transformer import ProjectedTransformer

    # Create minimal Mimi model for testing
    # (In practice, load from checkpoint)
    print("Creating test Mimi model...")

    encoder = SEANetEncoder(channels=1, dimension=512)
    decoder = SEANetDecoder(channels=1, dimension=512)
    quantizer = DummyQuantizer()

    encoder_transformer = ProjectedTransformer(
        input_dimension=512,
        output_dimensions=(512,),
        d_model=512,
        num_heads=8,
        num_layers=2,
        layer_scale=0.01,
        context=256,
        max_period=10000.0,
        dim_feedforward=2048
    )
    decoder_transformer = ProjectedTransformer(
        input_dimension=512,
        output_dimensions=(512,),
        d_model=512,
        num_heads=8,
        num_layers=2,
        layer_scale=0.01,
        context=256,
        max_period=10000.0,
        dim_feedforward=2048
    )

    mimi = MimiModel(
        encoder=encoder,
        decoder=decoder,
        quantizer=quantizer,
        frame_rate=12.5,
        encoder_frame_rate=100.0,
        sample_rate=24000,
        channels=1,
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer
    )

    print("Creating CoreML wrapper...")
    wrapper = CoreMLMimiDecoderWrapper.from_original(mimi)

    print(f"Number of conv states: {wrapper.get_num_conv_states()}")
    print(f"Number of convtr states: {wrapper.get_num_convtr_states()}")
    print(f"Number of transformer layers: {len(wrapper.transformer_layers)}")

    # Initialize state
    state = wrapper.init_state(batch_size=1)
    print(f"Total state tensors: {len(state)}")
    for k, v in state.items():
        print(f"  {k}: {v.shape}")

    return wrapper, state


if __name__ == "__main__":
    test_wrapper_creation()
