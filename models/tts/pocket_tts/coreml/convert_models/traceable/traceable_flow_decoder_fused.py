"""Traceable Flow Decoder with the full LSD Euler loop fused into the graph.

The non-fused `TraceableFlowDecoder` traces a SINGLE flow step and returns a
velocity; the N-step Euler integration runs in the Swift host, which calls
`flow_decoder.predict()` N times per audio frame (N=8 → 336 dispatches for a
42-frame utterance). Each call pays MLModel dispatch + the fp32↔fp16 IO cast
twice, and the kernel (1024+32 → 32 MLP) is far too small to amortize ANE
residency, so the partitioner keeps spilling it to CPU.

This wrapper unrolls all N Euler steps inside `forward`, so the host makes ONE
`predict()` per frame (42 dispatches instead of 336). `transformer_out` is the
per-frame conditioning and is CONSTANT across the N steps — feeding it once and
looping internally removes 7/8 of the redundant boundary traffic. The `s`/`t`
time endpoints become trace-time constants (i/N, (i+1)/N), which lets the MIL
optimizer fold the AdaLN time-embedding for each step into constants.

This is the same fusion pattern PR #66 used for the Nemotron decoder+joint
("B1 fusion", +15% throughput, output-identical). The math is bit-identical to
the host-side loop: z_{i+1} = z_i + flow_net(c, i/N, (i+1)/N, z_i) * (1/N).
"""
import torch
import torch.nn as nn


class TraceableFlowDecoderFused(nn.Module):
    """Flow decoder that runs the entire N-step LSD Euler integration internally.

    For LSD with N steps, starting from noise z_0:
        for i in [0, 1, ..., N-1]:
            s = i / N
            t = (i + 1) / N
            velocity = flow_net(transformer_out, s, t, z_i)
            z_{i+1} = z_i + velocity * (1 / N)
    returns z_N.
    """

    def __init__(self, flow_net, num_steps: int = 8, ldim: int = 32):
        super().__init__()
        self.flow_net = flow_net
        self.num_steps = num_steps
        self.ldim = ldim

    @classmethod
    def from_flowlm(cls, flow_lm, num_steps: int = 8) -> "TraceableFlowDecoderFused":
        return cls(flow_lm.flow_net, num_steps=num_steps, ldim=flow_lm.ldim)

    def forward(
        self,
        transformer_out: torch.Tensor,  # [B, 1024] per-frame conditioning (constant across steps)
        latent_init: torch.Tensor,      # [B, 32] initial noise z_0 (host-provided for seed control)
    ) -> torch.Tensor:
        """Run all N Euler steps and return the final latent z_N [B, 32]."""
        dt = 1.0 / self.num_steps
        latent = latent_init
        for i in range(self.num_steps):
            # Trace-time constants: each step's (s, t) is a pure Python-float
            # constant tensor of shape [1, 1] — identical to what the verified
            # single-step path passes as inputs. Building these from
            # `transformer_out.shape[0]` (a traced dynamic size) instead would
            # emit an aten::Int that coremltools can't const-fold
            # ("only 0-dimensional arrays can be converted to Python scalars").
            # Batch is fixed at 1, and [1,1] broadcasts against [1,1024]/[1,32].
            s = torch.tensor([[i * dt]], dtype=transformer_out.dtype)
            t = torch.tensor([[(i + 1) * dt]], dtype=transformer_out.dtype)
            velocity = self.flow_net(transformer_out, s, t, latent)
            latent = latent + velocity * dt
        return latent


def test_traceable_flow_decoder_fused():
    import sys
    import os
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    sys.path.insert(0, _project_dir)

    from pocket_tts import TTSModel
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    # Reference: host-side loop using the single-step decoder.
    from traceable_flow_decoder import TraceableFlowDecoder
    single = TraceableFlowDecoder.from_flowlm(model.flow_lm)
    single.eval()
    fused = TraceableFlowDecoderFused.from_flowlm(model.flow_lm, num_steps=8)
    fused.eval()

    transformer_out = torch.randn(1, 1024)
    z0 = torch.randn(1, 32)

    num_steps = 8
    dt = 1.0 / num_steps
    latent = z0.clone()
    with torch.no_grad():
        for step in range(num_steps):
            s = torch.tensor([[step * dt]])
            t = torch.tensor([[(step + 1) * dt]])
            velocity = single(transformer_out, latent, s, t)
            latent = latent + velocity * dt
        ref = latent
        got = fused(transformer_out, z0.clone())

    max_diff = (ref - got).abs().max().item()
    print(f"fused vs host-loop max abs diff: {max_diff:.3e}")
    assert max_diff < 1e-5, f"fused decoder diverged from host loop: {max_diff}"
    print("Done! (bit-identical to host-side Euler loop)")


if __name__ == "__main__":
    test_traceable_flow_decoder_fused()
