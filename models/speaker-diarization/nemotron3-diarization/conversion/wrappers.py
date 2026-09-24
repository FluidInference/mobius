"""CoreML export wrapper for Nemotron 3 Diarization (streaming Sortformer, 8 spk).

Mirrors ``SortformerEncLabelModel.forward_for_export`` with two deviations:

- ``concat_and_pad`` (in-place ``index_copy_`` scatter, not convertible) is replaced by
  the gather-based ``fixed_concat_and_pad`` proven ANE-safe in the v2 Sortformer export.
- A fourth output exposes the 10 ms high-resolution predictions so the host can emit
  fine-grained timestamps; the 80 ms output (their avg-pool downsample) is kept for
  speaker-cache / FIFO state updates.

Requires ``export_patches.apply_patches`` to have replaced FlexAttention first.
"""

import torch
from torch import nn

from config import UPSAMPLE_FACTOR


def fixed_concat_and_pad(embs, lengths, max_total_len):
    """ANE-safe concat+pad of [spkcache, fifo, chunk] via gather with arithmetic indices.

    Valid frames of each segment are packed contiguously at the start of the output;
    the remainder is zeroed. Ported unchanged from the v2 Sortformer conversion.
    """
    B, _, D = embs[0].shape
    device = embs[0].device

    size0, size1, size2 = embs[0].shape[1], embs[1].shape[1], embs[2].shape[1]
    total_input_size = size0 + size1 + size2

    full_concat = torch.cat(embs, dim=1)  # (B, total_input_size, D)

    len0 = lengths[0].reshape(())
    len1 = lengths[1].reshape(())
    len2 = lengths[2].reshape(())
    total_length = len0 + len1 + len2

    out_pos = torch.arange(max_total_len, device=device, dtype=torch.long)

    cumsum0 = len0
    cumsum1 = len0 + len1
    in_seg1_or_2 = (out_pos >= cumsum0).long()
    in_seg2 = (out_pos >= cumsum1).long()

    offset = in_seg1_or_2 * (size0 - len0) + in_seg2 * (size1 - len1)
    gather_idx = (out_pos + offset).clamp(0, total_input_size - 1)
    gather_idx = gather_idx.unsqueeze(0).unsqueeze(-1).expand(B, max_total_len, D)

    output = torch.gather(full_concat, dim=1, index=gather_idx)
    output = output * (out_pos < total_length).float().unsqueeze(0).unsqueeze(-1)

    return output, total_length


def matmul_concat_and_pad(embs, lengths, max_total_len):
    """``fixed_concat_and_pad`` with the gather replaced by a one-hot selection matmul.

    Same packing, but no ``gather_along_axis`` (int16 indices), which M3-generation ANEs
    can't run (FluidAudio #951). Exact in fp16: each output row sums one selected frame
    and zeros.
    """
    size0, size1, size2 = embs[0].shape[1], embs[1].shape[1], embs[2].shape[1]
    total_input_size = size0 + size1 + size2

    full_concat = torch.cat(embs, dim=1)  # (B, total_input_size, D)

    len0 = lengths[0].reshape(())
    len1 = lengths[1].reshape(())
    len2 = lengths[2].reshape(())
    total_length = len0 + len1 + len2

    out_pos = torch.arange(max_total_len, dtype=torch.long)
    in_seg1_or_2 = (out_pos >= len0).long()
    in_seg2 = (out_pos >= len0 + len1).long()
    offset = in_seg1_or_2 * (size0 - len0) + in_seg2 * (size1 - len1)
    src_idx = (out_pos + offset).clamp(0, total_input_size - 1)
    valid = (out_pos < total_length).float()

    src_pos = torch.arange(total_input_size, dtype=torch.long)
    select = (src_idx.unsqueeze(-1) == src_pos.unsqueeze(0)).float() * valid.unsqueeze(-1)
    output = torch.matmul(select.unsqueeze(0), full_concat)  # (B, max_total_len, D)
    return output, total_length


class Nemotron3ExportWrapper(nn.Module):
    """chunk mel + spkcache/fifo state -> speaker preds (80 ms + 10 ms) + chunk embs."""

    def __init__(self, model, packed_len: int, gather_free: bool = False):
        super().__init__()
        self.model = model
        self.packed_len = packed_len
        self.concat_and_pad = matmul_concat_and_pad if gather_free else fixed_concat_and_pad

    def forward(self, chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths):
        sm = self.model.sortformer_modules

        # FeatureStacking pre-encode, inlined with static shapes (its forward computes
        # t_new as a traced tensor, which coremltools rejects). Export mel frame counts
        # are always a multiple of the stacking factor, so no padding branch is needed.
        B, T_mel, C = chunk.shape
        stack = self.model.encoder.pre_encode
        chunk_embs = stack.proj(chunk.reshape(B, T_mel // stack.subsampling_factor, C * stack.subsampling_factor))
        # ceil(len / factor) via explicit integer floor_divide — `//` traces into a
        # float divide whose fp16 lowering misrounds lengths > 2048 (3047 -> 3048).
        chunk_enc_lengths = torch.floor_divide(
            chunk_lengths.to(torch.int32) + (stack.subsampling_factor - 1), stack.subsampling_factor
        ).to(torch.int64)

        packed, packed_length = self.concat_and_pad(
            [spkcache, fifo, chunk_embs],
            [spkcache_lengths.to(torch.int64), fifo_lengths.to(torch.int64), chunk_enc_lengths],
            self.packed_len,
        )

        emb_seq, emb_seq_length = self.model.frontend_encoder(
            processed_signal=packed,
            processed_signal_length=packed_length.reshape(1),
            bypass_pre_encode=True,
        )

        # forward_infer, inlined (transformer_encoder is None for this checkpoint).
        T = emb_seq.shape[1]
        encoder_mask = sm.length_to_mask(emb_seq_length.reshape(1), T)  # (B, T)
        hidden = sm.upsample_hidden(emb_seq)  # (B, T*8, H)
        # repeat_interleave(8, dim=1) via expand+reshape (trace-friendly).
        output_mask = (
            encoder_mask.unsqueeze(-1)
            .expand(-1, T, UPSAMPLE_FACTOR)
            .reshape(-1, T * UPSAMPLE_FACTOR)
        )
        preds_hires = sm.forward_speaker_sigmoids(hidden) * output_mask.unsqueeze(-1).float()
        preds = sm.downsample_preds(preds_hires, UPSAMPLE_FACTOR)

        return preds, chunk_embs, chunk_enc_lengths, preds_hires
