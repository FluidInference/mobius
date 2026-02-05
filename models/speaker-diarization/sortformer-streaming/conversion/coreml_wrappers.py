import torch
from torch import nn


def fixed_concat_and_pad(embs, lengths, max_total_len=188+188+6):
    """
    ANE-safe concat and pad that avoids zero-length slices.
    
    Uses gather with arithmetic-computed indices to pack valid frames efficiently.
    
    Args:
        embs: List of 3 tensors [spkcache, fifo, chunk], each (B, seq_len, D)
        lengths: List of 3 length tensors, each (1,) or scalar
                 First two may be 0, third is always > 0
        max_total_len: Output sequence length (padded with zeros)
    
    Returns:
        output: (B, max_total_len, D) with valid frames packed at the start
        total_length: sum of lengths
    """
    B, _, D = embs[0].shape
    device = embs[0].device
    
    # Fixed sizes (known at trace time, becomes constants in graph)
    size0, size1, size2 = embs[0].shape[1], embs[1].shape[1], embs[2].shape[1]
    total_input_size = size0 + size1 + size2
    
    # Concatenate all embeddings at full size (no zero-length slices!)
    full_concat = torch.cat(embs, dim=1)  # (B, total_input_size, D)
    
    # Get lengths (reshape to scalar for efficient broadcast)
    len0 = lengths[0].reshape(())
    len1 = lengths[1].reshape(())
    len2 = lengths[2].reshape(())
    total_length = len0 + len1 + len2
    
    # Output positions: [0, 1, 2, ..., max_total_len-1]
    out_pos = torch.arange(max_total_len, device=device, dtype=torch.long)
    
    # Compute gather indices using arithmetic (more efficient than multiple where())
    # 
    # For output position p:
    #   seg0 (p < len0):           index = p
    #   seg1 (len0 <= p < len0+len1): index = (p - len0) + size0 = p + (size0 - len0)
    #   seg2 (len0+len1 <= p < total): index = (p - len0 - len1) + size0 + size1
    #                                        = p + (size0 + size1 - len0 - len1)
    #
    # This simplifies to: index = p + offset, where offset depends on segment.
    # offset_seg0 = 0
    # offset_seg1 = size0 - len0
    # offset_seg2 = size0 + size1 - len0 - len1 = offset_seg1 + (size1 - len1)
    #
    # Using segment indicators (0 or 1):
    #   offset = in_seg1_or_2 * (size0 - len0) + in_seg2 * (size1 - len1)
    
    cumsum0 = len0
    cumsum1 = len0 + len1
    
    # Segment indicators (bool -> long for arithmetic)
    in_seg1_or_2 = (out_pos >= cumsum0).long()  # 1 if in seg1 or seg2
    in_seg2 = (out_pos >= cumsum1).long()       # 1 if in seg2
    
    # Compute offset and gather index
    offset = in_seg1_or_2 * (size0 - len0) + in_seg2 * (size1 - len1)
    gather_idx = (out_pos + offset).clamp(0, total_input_size - 1)
    
    # Expand for gather: (B, max_total_len, D)
    gather_idx = gather_idx.unsqueeze(0).unsqueeze(-1).expand(B, max_total_len, D)
    
    # Gather and mask padding
    output = torch.gather(full_concat, dim=1, index=gather_idx)
    output = output * (out_pos < total_length).float().unsqueeze(0).unsqueeze(-1)
    
    return output, total_length


class PreprocessorWrapper(nn.Module):
    """
    Wraps the NeMo preprocessor (FilterbankFeaturesTA) for CoreML export.
    We need to ensure it takes (audio, length) and returns (features, length).
    """

    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor

    def forward(self, audio_signal, length):
        # NeMo preprocessor returns (features, length)
        # features shape: [B, D, T]
        return self.preprocessor(input_signal=audio_signal, length=length)


class SortformerHeadWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pre_encoder_embs, pre_encoder_lengths, chunk_pre_encoder_embs, chunk_pre_encoder_lengths):
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.model.frontend_encoder(
            processed_signal=pre_encoder_embs,
            processed_signal_length=pre_encoder_lengths,
            bypass_pre_encode=True,
        )

        # forward pass for inference
        spkcache_fifo_chunk_preds = self.model.forward_infer(
            spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths
        )
        # Explicit identity ops to create distinct output tensors (required for macOS 26+)
        # macOS 26 BNNS compiler rejects models where input and output tensors share the same name
        chunk_pre_encoder_embs_out = chunk_pre_encoder_embs + 0.0  # identity via add
        chunk_pre_encoder_lengths_out = chunk_pre_encoder_lengths + 0  # identity via add
        return spkcache_fifo_chunk_preds, chunk_pre_encoder_embs_out, chunk_pre_encoder_lengths_out


class SortformerCoreMLWrapper(nn.Module):
    """
    Wraps the entire Sortformer pipeline (Encoder + Streaming Logic for Export)
    The 'forward_for_export' method in the model is the target.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pre_encoder = PreEncoderWrapper(model)

    def forward(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        (spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths,
         chunk_pre_encode_embs, chunk_pre_encode_lengths) = self.pre_encoder(
            chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths
        )

        # encode the concatenated embeddings
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.model.frontend_encoder(
            processed_signal=spkcache_fifo_chunk_pre_encode_embs,
            processed_signal_length=spkcache_fifo_chunk_pre_encode_lengths,
            bypass_pre_encode=True,
        )

        # forward pass for inference
        spkcache_fifo_chunk_preds = self.model.forward_infer(
            spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths
        )
        return spkcache_fifo_chunk_preds, chunk_pre_encode_embs, chunk_pre_encode_lengths


class PreEncoderWrapper(nn.Module):
    """
    Wraps the entire Sortformer pipeline (Encoder + Streaming Logic for Export)
    The 'forward_for_export' method in the model is the target.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        modules = model.sortformer_modules
        chunk_length = modules.chunk_left_context + modules.chunk_len + modules.chunk_right_context
        self.pre_encoder_length = modules.spkcache_len + modules.fifo_len + chunk_length

    def forward(self, *args):
        if len(args) == 6:
            return self.forward_concat(*args)
        else:
            return self.forward_pre_encode(*args)

    def forward_concat(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.model.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
        chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)
        spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths = fixed_concat_and_pad(
            [spkcache, fifo, chunk_pre_encode_embs],
            [spkcache_lengths, fifo_lengths, chunk_pre_encode_lengths],
            self.pre_encoder_length
        )
        return (spkcache_fifo_chunk_pre_encode_embs, spkcache_fifo_chunk_pre_encode_lengths,
                chunk_pre_encode_embs, chunk_pre_encode_lengths)

    def forward_pre_encode(self, chunk, chunk_lengths):
        chunk_pre_encode_embs, chunk_pre_encode_lengths = self.model.encoder.pre_encode(x=chunk, lengths=chunk_lengths)
        chunk_pre_encode_lengths = chunk_pre_encode_lengths.to(torch.int64)

        return chunk_pre_encode_embs, chunk_pre_encode_lengths


class ConformerEncoderWrapper(nn.Module):
    """
    Wraps the entire Sortformer pipeline (Encoder + Streaming Logic for Export)
    The 'forward_for_export' method in the model is the target.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pre_encode_embs, pre_encode_lengths):
        spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths = self.model.frontend_encoder(
            processed_signal=pre_encode_embs,
            processed_signal_length=pre_encode_lengths,
            bypass_pre_encode=True,
        )
        return spkcache_fifo_chunk_fc_encoder_embs, spkcache_fifo_chunk_fc_encoder_lengths


class SortformerEncoderWrapper(nn.Module):
    """
    Wraps the entire Sortformer pipeline (Encoder + Streaming Logic for Export)
    The 'forward_for_export' method in the model is the target.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, encoder_embs, encoder_lengths):
        spkcache_fifo_chunk_preds = self.model.forward_infer(
            encoder_embs, encoder_lengths
        )
        return spkcache_fifo_chunk_preds
