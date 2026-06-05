#!/usr/bin/env python3
"""Shard the Nemotron streaming encoder into N CoreML models for ANE.

The 24-layer FastConformer encoder is too big for the iOS ANE working set
(1.4GB / 130s load as a single model). Splitting the layer stack into N shards
lets each shard fit + load fast while keeping ANE-speed inference. The shared
tensors (pos_emb, att_mask, pad_mask) computed in the preamble are passed
between shards; per-layer caches are sliced by layer index.

  Head shard : mel + cache[0:k] -> hidden, pos_emb, att_mask, pad_mask, cache_out[0:k]
  Body/Tail  : hidden + pos_emb + att_mask + pad_mask + cache[a:b]
               -> hidden(+out_proj on tail) + cache_out[a:b]
"""
from __future__ import annotations
import torch


class HeadShard(torch.nn.Module):
    """Preamble (pre_encode, pos_enc, masks) + layers[0:n_layers]."""

    def __init__(self, enc, n_layers):
        super().__init__()
        self.enc = enc
        self.n = n_layers

    def forward(self, features, length, cache_ch, cache_t, cache_len_in):
        enc = self.enc
        # cache passed as [B, n, ...] -> [n, B, ...]
        cache_ch = cache_ch.transpose(0, 1)
        cache_t = cache_t.transpose(0, 1)
        clcl = cache_len_in.to(torch.int64)

        audio_signal = torch.transpose(features, 1, 2)
        audio_signal, length = enc.pre_encode(x=audio_signal, lengths=length.to(torch.int64))
        if enc.streaming_cfg.drop_extra_pre_encoded > 0:
            audio_signal = audio_signal[:, enc.streaming_cfg.drop_extra_pre_encoded:, :]
            length = (length - enc.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        max_audio_length = audio_signal.size(1)
        cache_len = enc.streaming_cfg.last_channel_cache_size
        cache_keep_size = max_audio_length - enc.streaming_cfg.cache_drop_size
        max_audio_length = max_audio_length + cache_len
        padding_length = length + cache_len
        offset = torch.neg(clcl) + cache_len

        audio_signal, pos_emb = enc.pos_enc(x=audio_signal, cache_len=cache_len)
        pad_mask, att_mask = enc._create_masks(
            att_context_size=enc.att_context_size, padding_length=padding_length,
            max_audio_length=max_audio_length, offset=offset, device=audio_signal.device)
        pad_mask = pad_mask[:, cache_len:]
        if att_mask is not None:
            att_mask = att_mask[:, cache_len:]

        ch_next, t_next = [], []
        for lth in range(self.n):
            out = enc.layers[lth](
                x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask,
                cache_last_channel=cache_ch[lth], cache_last_time=cache_t[lth])
            audio_signal, cc, ct = out
            ch_next.append(cc); t_next.append(ct)
        cache_len_out = torch.clamp(clcl + cache_keep_size, max=cache_len).to(torch.int32)
        # masks/pos_emb passed to next shard; stack our cache slice back to [B, n, ...]
        return (audio_signal, pos_emb, att_mask, pad_mask,
                torch.stack(ch_next, 0).transpose(0, 1), torch.stack(t_next, 0).transpose(0, 1),
                cache_len_out)


class BodyShard(torch.nn.Module):
    """layers[start:end]; optional out_proj + final transpose on the tail."""

    def __init__(self, enc, start, end, is_tail):
        super().__init__()
        self.enc = enc
        self.start, self.end = start, end
        self.is_tail = is_tail

    def forward(self, audio_signal, pos_emb, att_mask, pad_mask, cache_ch, cache_t):
        enc = self.enc
        cache_ch = cache_ch.transpose(0, 1)
        cache_t = cache_t.transpose(0, 1)
        ch_next, t_next = [], []
        for i, lth in enumerate(range(self.start, self.end)):
            out = enc.layers[lth](
                x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask,
                cache_last_channel=cache_ch[i], cache_last_time=cache_t[i])
            audio_signal, cc, ct = out
            ch_next.append(cc); t_next.append(ct)
        ch_out = torch.stack(ch_next, 0).transpose(0, 1)
        t_out = torch.stack(t_next, 0).transpose(0, 1)
        if self.is_tail:
            if enc.out_proj is not None:
                audio_signal = enc.out_proj(audio_signal)
            encoded = torch.transpose(audio_signal, 1, 2)
            return encoded, ch_out, t_out
        return audio_signal, ch_out, t_out


def parity_check():
    """Compose head+body shards and compare against the full encoder forward."""
    import sys
    sys.path.insert(0, ".")
    import nemo.collections.asr as nemo_asr
    from individual_components import EncoderStreamingWrapper
    m = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu").eval()
    enc = m.encoder
    enc.setup_streaming_params()
    L = len(enc.layers)
    print(f"layers={L}, out_proj={enc.out_proj is not None}, reduction_position={enc.reduction_position}")

    cc, ct, cl = enc.get_initial_cache_state(batch_size=1, device="cpu")
    cl = cl.to(torch.int32)
    cc_b, ct_b = cc.transpose(0, 1), ct.transpose(0, 1)  # [B, L, ...]
    n_mel = int(m.cfg.preprocessor.features)
    tmf = 233
    mel = torch.randn(1, n_mel, tmf)
    mlen = torch.tensor([tmf], dtype=torch.int32)

    # reference: full streaming wrapper
    ref = EncoderStreamingWrapper(enc.eval())
    with torch.no_grad():
        r_enc, r_len, r_cc, r_ct, r_cl = ref(mel, mlen, cc_b, ct_b, cl)

    # sharded composition: 4 shards of 6
    splits = [0, 6, 12, 18, 24]
    head = HeadShard(enc, splits[1]).eval()
    bodies = [BodyShard(enc, splits[i], splits[i+1], is_tail=(i == 3)).eval() for i in range(1, 4)]
    with torch.no_grad():
        a, pos, att, pad, h_cc, h_ct, cl_out = head(
            mel, mlen, cc_b[:, 0:6], ct_b[:, 0:6], cl)
        cc_parts, ct_parts = [h_cc], [h_ct]
        for bi, b in enumerate(bodies):
            s, e = splits[bi+1], splits[bi+2]
            a, b_cc, b_ct = b(a, pos, att, pad, cc_b[:, s:e], ct_b[:, s:e])
            cc_parts.append(b_cc); ct_parts.append(b_ct)
        s_enc = a
        s_cc = torch.cat(cc_parts, dim=1)
        s_ct = torch.cat(ct_parts, dim=1)

    d = lambda x, y: (x - y).abs().max().item()
    # EncoderStreamingWrapper already returns caches as [B, L, ...]; compare directly.
    print(f"encoded   max|Δ| = {d(r_enc, s_enc):.6e}")
    print(f"cache_ch  max|Δ| = {d(r_cc, s_cc):.6e}")
    print(f"cache_t   max|Δ| = {d(r_ct, s_ct):.6e}")
    print(f"cache_len ref={r_cl.flatten().tolist()} shard={cl_out.flatten().tolist()}")


if __name__ == "__main__":
    parity_check()


class FrontEndShard(torch.nn.Module):
    """Preamble only: pre_encode + pos_enc + masks. No conformer layers.
    Outputs the intermediates the conformer shard needs, plus cache_len_out."""
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, features, length, cache_len_in):
        enc = self.enc
        clcl = cache_len_in.to(torch.int64)
        audio_signal = torch.transpose(features, 1, 2)
        audio_signal, length = enc.pre_encode(x=audio_signal, lengths=length.to(torch.int64))
        if enc.streaming_cfg.drop_extra_pre_encoded > 0:
            audio_signal = audio_signal[:, enc.streaming_cfg.drop_extra_pre_encoded:, :]
            length = (length - enc.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)
        max_audio_length = audio_signal.size(1)
        cache_len = enc.streaming_cfg.last_channel_cache_size
        cache_keep_size = max_audio_length - enc.streaming_cfg.cache_drop_size
        max_audio_length = max_audio_length + cache_len
        padding_length = length + cache_len
        offset = torch.neg(clcl) + cache_len
        audio_signal, pos_emb = enc.pos_enc(x=audio_signal, cache_len=cache_len)
        pad_mask, att_mask = enc._create_masks(
            att_context_size=enc.att_context_size, padding_length=padding_length,
            max_audio_length=max_audio_length, offset=offset, device=audio_signal.device)
        pad_mask = pad_mask[:, cache_len:]
        if att_mask is not None:
            att_mask = att_mask[:, cache_len:]
        cache_len_out = torch.clamp(clcl + cache_keep_size, max=cache_len).to(torch.int32)
        return audio_signal, pos_emb, att_mask, pad_mask, cache_len_out
