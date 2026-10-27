# Adapted from FluidInference/mobius v3 individual_components.py (26f5fa80), Apache-2.0.
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import coremltools as ct
import torch

@dataclass
class ExportSettings:
    output_dir: Path
    compute_units: ct.ComputeUnit
    deployment_target: Optional[ct.target.iOS17]
    compute_precision: Optional[ct.precision]
    max_audio_seconds: float
    max_symbol_steps: int

class PreprocessorWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mel, mel_length = self.module(input_signal=audio_signal, length=length.to(dtype=torch.long))
        return mel, mel_length

class EncoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, features: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, encoded_lengths = self.module(audio_signal=features, length=length.to(dtype=torch.long))
        return encoded, encoded_lengths

class DecoderWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(
        self,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = [h_in, c_in]
        decoder_output, _, new_state = self.module(
            targets=targets.to(dtype=torch.long),
            target_length=target_lengths.to(dtype=torch.long),
            states=state,
        )
        return decoder_output, new_state[0], new_state[1]

class JointWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, encoder_outputs: torch.Tensor, decoder_outputs: torch.Tensor) -> torch.Tensor:
        # Input: encoder_outputs [B, D, T], decoder_outputs [B, D, U]
        # Transpose to match what projection layers expect
        encoder_outputs = encoder_outputs.transpose(1, 2)  # [B, T, D]
        decoder_outputs = decoder_outputs.transpose(1, 2)  # [B, U, D]

        # Apply projections
        enc_proj = self.module.enc(encoder_outputs)        # [B, T, 640]
        dec_proj = self.module.pred(decoder_outputs)       # [B, U, 640]

        # Explicit broadcasting along T and U to avoid converter ambiguity
        x = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)  # [B, T, U, 640]
        x = self.module.joint_net[0](x)                   # ReLU
        x = self.module.joint_net[1](x)                   # Dropout (no-op in eval)
        out = self.module.joint_net[2](x)                 # Linear -> logits [B, T, U, 8198]
        return out

class JointDecisionSingleStep(torch.nn.Module):
    """Single-step variant for streaming: encoder_step [1, 1024, 1] -> [1,1,1].

    Inputs:
      - encoder_step: [B=1, D=1024, T=1]
      - decoder_step: [B=1, D=640, U=1]

    Returns:
      - token_id: [1, 1, 1] int32
      - token_prob: [1, 1, 1] float32
      - duration: [1, 1, 1] int32
    """
    def __init__(self, joint: JointWrapper, vocab_size: int, num_extra: int, top_k: int = 64) -> None:
        super().__init__()
        self.joint = joint
        self.vocab_with_blank = int(vocab_size) + 1
        self.num_extra = int(num_extra)
        # Emit top-K candidates to enable host-side re-ranking with contextual biasing
        self.top_k = int(top_k)

    def forward(self, encoder_step: torch.Tensor, decoder_step: torch.Tensor):
        # Reuse JointWrapper which expects [B, D, T] and [B, D, U]
        logits = self.joint(encoder_step, decoder_step)  # [1, 1, 1, V+extra]
        token_logits = logits[..., : self.vocab_with_blank]
        duration_logits = logits[..., -self.num_extra :]

        token_ids = torch.argmax(token_logits, dim=-1, keepdim=False).to(dtype=torch.int32)
        token_probs_all = torch.softmax(token_logits, dim=-1)
        token_prob = torch.gather(
            token_probs_all, dim=-1, index=token_ids.long().unsqueeze(-1)
        ).squeeze(-1)
        duration = torch.argmax(duration_logits, dim=-1, keepdim=False).to(dtype=torch.int32)

        # Also expose top-K candidates for host-side re-ranking.
        # Shapes preserved as [1, 1, 1, K] to match CoreML broadcasting expectations.
        # Note: topk expects last dimension; original shape is [1, 1, 1, V].
        topk_logits, topk_ids_long = torch.topk(token_logits, k=min(self.top_k, token_logits.shape[-1]), dim=-1)
        topk_ids = topk_ids_long.to(dtype=torch.int32)
        return token_ids, token_prob, duration, topk_ids, topk_logits

def _coreml_convert(
    traced: torch.jit.ScriptModule,
    inputs,
    outputs,
    settings: ExportSettings,
    compute_units_override: Optional[ct.ComputeUnit] = None,
) -> ct.models.MLModel:
    cu = compute_units_override if compute_units_override is not None else settings.compute_units
    kwargs = {
        "convert_to": "mlprogram",
        "inputs": inputs,
        "outputs": outputs,
        "compute_units": cu,
    }
    print("Converting:", traced.__class__.__name__)
    print("Conversion kwargs:", kwargs)
    if settings.deployment_target is not None:
        kwargs["minimum_deployment_target"] = settings.deployment_target
    if settings.compute_precision is not None:
        kwargs["compute_precision"] = settings.compute_precision
    return ct.convert(traced, **kwargs)
