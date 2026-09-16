"""Compatible generator assembly and explicit, fail-closed checkpoint loading.

Assembly follows hexgrad/kokoro model.py (Apache-2.0); the pinned upstream
modules remain the implementation of the generator layers. The inference
orchestration below is separate from KModel.forward_with_tokens.
"""

import math
import torch
from torch import nn
from kokoro.istftnet import AdaIN1d, Decoder
from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
from transformers import AlbertConfig


class CheckpointError(ValueError):
    def __init__(self, report):
        super().__init__("Checkpoint mapping failed; inspect checkpoint-map.json")
        self.report = report


def map_checkpoint(expected: dict, checkpoint: dict, identity_constants=None) -> tuple[dict, dict]:
    """Only explicitly documented DDP/legacy weight_norm renames are permitted."""
    mapped, entries, errors = {}, [], []
    for group, state in checkpoint.items():
        if not isinstance(state, dict):
            errors.append({"key": group, "reason": "expected module state dictionary"})
            continue
        for source, tensor in state.items():
            key = source.removeprefix("module.")
            if key.endswith(".weight_g"):
                key = key[:-9] + ".parametrizations.weight.original0"
            elif key.endswith(".weight_v"):
                key = key[:-9] + ".parametrizations.weight.original1"
            target = f"{group}.{key}"
            entry = {"source": f"{group}.{source}", "target": target, "renamed": source != key}
            if target in mapped:
                errors.append({**entry, "reason": "mapping collision"})
                continue
            if target not in expected:
                errors.append({**entry, "reason": "unexpected key"})
                continue
            if not isinstance(tensor, torch.Tensor):
                errors.append({**entry, "reason": "not a tensor"})
                continue
            entry.update(shape=list(tensor.shape), dtype=str(tensor.dtype), elements=tensor.numel())
            if tensor.shape != expected[target].shape or tensor.dtype != expected[target].dtype:
                errors.append(
                    {
                        **entry,
                        "reason": "shape or dtype mismatch",
                        "expected_shape": list(expected[target].shape),
                        "expected_dtype": str(expected[target].dtype),
                    }
                )
                continue
            if not torch.isfinite(tensor).all():
                errors.append({**entry, "reason": "nonfinite tensor"})
                continue
            mapped[target] = tensor
            entries.append(entry)
    initialized = []
    for key, constant in (identity_constants or {}).items():
        if key not in mapped:
            if key not in expected or constant.shape != expected[key].shape:
                raise ValueError("Invalid identity-constant allowlist")
            mapped[key] = constant
            initialized.append(
                {
                    "target": key,
                    "shape": list(constant.shape),
                    "value": constant.flatten()[0].item(),
                    "reason": "upstream AdaIN1d export workaround; frozen identity affine",
                }
            )
    missing = sorted(set(expected) - set(mapped))
    report = {
        "passed": not errors and not missing,
        "matched_checkpoint_keys": len(entries),
        "loaded_state_keys": len(mapped),
        "renamed_keys": sum(e["renamed"] for e in entries),
        "missing": missing,
        "errors": errors,
        "entries": entries,
        "identity_constants": initialized,
    }
    if not report["passed"]:
        raise CheckpointError(report)
    return mapped, report


def validate_tokens(ids, n_token: int, context_length: int) -> None:
    if not 3 <= len(ids) <= context_length:
        raise ValueError("Expected 1..510 content tokens with BOS/EOS; no silent truncation")
    if ids[0] != 0 or ids[-1] != 0:
        raise ValueError("Expected zero BOS/EOS tokens")
    if any(type(i) is not int or not 0 <= i < n_token for i in ids):
        raise ValueError("Invalid token ID")
    if 0 in ids[1:-1]:
        raise ValueError("Unexpected special token inside content")


def select_style(voices: torch.Tensor, content_length: int) -> tuple[torch.Tensor, int]:
    if voices.shape != (510, 1, 256) or not torch.isfinite(voices).all():
        raise ValueError("Expected a finite [510, 1, 256] upstream voice table")
    if not 1 <= content_length <= 510:
        raise ValueError("Voice length must be 1..510 content IDs (BOS/EOS excluded)")
    row = content_length - 1
    return voices[row], row


class CompatibleGenerator(nn.Module):
    """Pinned upstream layers with independently assembled inference orchestration."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.bert = CustomAlbert(AlbertConfig(vocab_size=config["n_token"], **config["plbert"]))
        self.bert_encoder = nn.Linear(self.bert.config.hidden_size, config["hidden_dim"])
        self.predictor = ProsodyPredictor(
            style_dim=config["style_dim"],
            d_hid=config["hidden_dim"],
            nlayers=config["n_layer"],
            max_dur=config["max_dur"],
            dropout=config["dropout"],
        )
        self.text_encoder = TextEncoder(
            channels=config["hidden_dim"],
            kernel_size=config["text_encoder_kernel_size"],
            depth=config["n_layer"],
            n_symbols=config["n_token"],
        )
        self.decoder = Decoder(
            dim_in=config["hidden_dim"],
            style_dim=config["style_dim"],
            dim_out=config["n_mels"],
            disable_complex=False,
            **config["istftnet"],
        )
        # Pinned upstream AdaIN1d adds affine=True solely for ONNX shape export.
        # The release contains no such affine weights. Preserve exact identity
        # behavior and keep these non-learned constants out of future optimizers.
        self.identity_constants = {}
        for name, module in self.named_modules():
            if isinstance(module, AdaIN1d):
                self.identity_constants[f"{name}.norm.weight"] = torch.ones_like(module.norm.weight)
                self.identity_constants[f"{name}.norm.bias"] = torch.zeros_like(module.norm.bias)
                module.norm.requires_grad_(False)

    def load_checkpoint(self, checkpoint: dict) -> dict:
        mapped, report = map_checkpoint(self.state_dict(), checkpoint, self.identity_constants)
        self.load_state_dict(mapped, strict=True)
        report["parameter_count"] = sum(p.numel() for p in self.parameters())
        report["parameter_groups"] = {
            name: sum(p.numel() for p in child.parameters())
            for name, child in self.named_children()
        }
        return report

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, style: torch.Tensor, speed: float = 1.0):
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype != torch.long:
            raise ValueError("Parity inference accepts one int64 sequence at a time")
        validate_tokens(
            ids[0].tolist(), self.config["n_token"], self.bert.config.max_position_embeddings
        )
        if style.shape != (1, 256) or not torch.isfinite(style).all():
            raise ValueError("Expected finite [1, 256] style")
        if not math.isfinite(speed) or not 0.25 <= speed <= 4:
            raise ValueError("Speed must be finite and within 0.25..4")
        lengths = torch.tensor([ids.shape[1]], device=ids.device, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        encoded = self.bert(ids, attention_mask=(~mask).int())
        projected = self.bert_encoder(encoded).transpose(1, 2)
        prosody = self.predictor.text_encoder(projected, style[:, 128:], lengths, mask)
        recurrent, _ = self.predictor.lstm(prosody)
        logits = self.predictor.duration_proj(recurrent)
        durations = (logits.sigmoid().sum(-1) / speed).round().clamp(min=1).long()[0]
        if int(durations.sum()) > 4000:
            raise ValueError("Generated duration exceeds the 4000-frame inference limit; split text or increase speed")
        # An explicit one-hot alignment is inference-only. Training needs real targets.
        token_at_frame = torch.repeat_interleave(
            torch.arange(ids.shape[1], device=ids.device), durations
        )
        alignment = torch.nn.functional.one_hot(token_at_frame, ids.shape[1]).T[None].float()
        features = prosody.transpose(1, 2) @ alignment
        f0, noise = self.predictor.F0Ntrain(features, style[:, 128:])
        text = self.text_encoder(ids, lengths, mask)
        audio = self.decoder(text @ alignment, f0, noise, style[:, :128]).squeeze()
        return audio, durations
