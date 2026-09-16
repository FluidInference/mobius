"""Supervised acoustic path: real alignment targets bypass integer-duration gradients."""

import torch
from torch import nn
import torch.nn.functional as F

from .features import MelFeatures
from .model import CompatibleGenerator, select_style, validate_tokens


class DeterministicLeftReflection(nn.Module):
    """Exact ReflectionPad1d((1, 0)) with a deterministic slice gradient."""

    def forward(self, value):
        if value.shape[-1] < 2:
            raise ValueError("Reflection requires at least two frames")
        return torch.cat((value[..., 1:2], value), dim=-1)


class AdaptationModel(nn.Module):
    def __init__(self, generator: CompatibleGenerator, voices: torch.Tensor, train_style=False):
        super().__init__()
        self.generator = generator
        if generator.decoder.generator.reflection_pad.padding != (1, 0):
            raise ValueError("Unexpected upstream decoder reflection padding")
        generator.decoder.generator.reflection_pad = DeterministicLeftReflection()
        self.register_buffer("voices", voices, persistent=False)
        self.style_offset = nn.Parameter(torch.zeros(1, 256), requires_grad=train_style)
        generator.requires_grad_(False)
        for name in ["bert_encoder", "predictor", "text_encoder"]:
            getattr(generator, name).requires_grad_(True)
        for name, parameter in generator.named_parameters():
            if name in generator.identity_constants:
                parameter.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.generator.bert.eval()
        self.generator.decoder.eval()
        return self

    def forward(self, batch: dict, crop_start=0, crop_frames=None):
        ids, durations = batch["ids"], batch["durations"]
        if ids.shape[0] != 1 or ids.shape != durations.shape:
            raise ValueError("Initial trainer uses batch=1 with explicit accumulation")
        validate_tokens(ids[0].tolist(), self.generator.config["n_token"], 512)
        if durations.dtype != torch.long or durations.min() < 1 or durations.max() > 50:
            raise ValueError("Invalid supervised durations")
        total_frames = int(durations.sum())
        if batch["audio"].shape != (1, total_frames * 600):
            raise ValueError("Waveform and duration sum do not share the 600-sample grid")
        if batch["f0"].numel() != total_frames * 2 or batch["energy"].numel() != total_frames * 2:
            raise ValueError("Pitch/energy targets do not share the 300-sample grid")
        crop_frames = total_frames if crop_frames is None else min(crop_frames, total_frames)
        if crop_start < 0 or crop_start + crop_frames > total_frames:
            raise ValueError("Invalid crop boundary")
        lengths = torch.tensor([ids.shape[1]], device=ids.device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        style, _ = select_style(self.voices, ids.shape[1] - 2)
        style = style + self.style_offset
        with torch.no_grad():
            representation = self.generator.bert(ids, attention_mask=(~mask).int())
        projected = self.generator.bert_encoder(representation).transpose(1, 2)
        prosody = self.generator.predictor.text_encoder(projected, style[:, 128:], lengths, mask)
        recurrent, _ = self.generator.predictor.lstm(prosody)
        duration_logits = self.generator.predictor.duration_proj(recurrent)
        frame_tokens = torch.repeat_interleave(torch.arange(ids.shape[1], device=ids.device), durations[0])
        alignment = F.one_hot(frame_tokens, ids.shape[1]).T[None].float()
        aligned_prosody = prosody.transpose(1, 2) @ alignment
        f0, energy = self.generator.predictor.F0Ntrain(aligned_prosody, style[:, 128:])
        text = self.generator.text_encoder(ids, lengths, mask) @ alignment
        sl = slice(crop_start, crop_start + crop_frames)
        sl2 = slice(crop_start * 2, (crop_start + crop_frames) * 2)
        audio = self.generator.decoder(text[..., sl], f0[..., sl2], energy[..., sl2], style[:, :128]).reshape(1, -1)
        target = batch["audio"][..., crop_start * 600:(crop_start + crop_frames) * 600]
        if audio.shape != target.shape:
            raise ValueError("Decoder crop length differs from real audio target")
        return {"audio": audio, "target_audio": target, "duration_logits": duration_logits,
                "f0": f0[..., sl2], "target_f0": batch["f0"].reshape(1, -1)[..., sl2],
                "energy": energy[..., sl2], "target_energy": batch["energy"].reshape(1, -1)[..., sl2],
                "duration_targets": durations, "style_offset": self.style_offset}


class SupervisedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = MelFeatures()

    def forward(self, output):
        logits, target_duration = output["duration_logits"], output["duration_targets"]
        survival = torch.arange(logits.shape[-1], device=logits.device)[None, None] < target_duration[..., None]
        duration_ce = F.binary_cross_entropy_with_logits(logits, survival.float())
        duration = F.smooth_l1_loss(torch.log1p(logits.sigmoid().sum(-1)), torch.log1p(target_duration.float()))
        predicted_mel = self.mel.normalized(output["audio"])
        target_mel = self.mel.normalized(output["target_audio"])
        mel = F.l1_loss(predicted_mel, target_mel)
        voiced = output["target_f0"] >= 50
        f0 = F.smooth_l1_loss(output["f0"][voiced] / 100, output["target_f0"][voiced] / 100) if voiced.any() else logits.sum() * 0
        unvoiced = F.smooth_l1_loss(output["f0"][~voiced] / 100, output["target_f0"][~voiced] / 100) if (~voiced).any() else logits.sum() * 0
        energy = F.smooth_l1_loss(output["energy"], output["target_energy"])
        style = output["style_offset"].square().mean()
        total = 5 * mel + duration_ce + duration + f0 + 0.1 * unvoiced + energy + 0.01 * style
        return {"total": total, "mel": mel, "duration_ce": duration_ce, "duration": duration,
                "f0": f0, "unvoiced": unvoiced, "energy": energy, "style": style}
