"""Streaming inference helper that stitches Parakeet CoreML components using the RNNT greedy loop from Nemo."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import coremltools as ct
import librosa
import numpy as np
import torch

from parakeet_components import CoreMLModelBundle

LOGGER = logging.getLogger("parakeet_streaming")


class BatchedHyps:
    """Minimal port of Nemo's batched hypothesis buffer."""

    def __init__(
        self,
        batch_size: int,
        init_length: int,
        device: torch.device,
        float_dtype: torch.dtype,
    ) -> None:
        if init_length <= 0:
            raise ValueError("init_length must be > 0")
        self._max_length = init_length
        self.current_lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
        self.transcript = torch.zeros((batch_size, self._max_length), device=device, dtype=torch.long)
        self.timestamps = torch.zeros((batch_size, self._max_length), device=device, dtype=torch.long)
        self.scores = torch.zeros(batch_size, device=device, dtype=float_dtype)
        self.last_timestamp = torch.full((batch_size,), -1, device=device, dtype=torch.long)
        self.last_timestamp_lasts = torch.zeros(batch_size, device=device, dtype=torch.long)
        self._batch_indices = torch.arange(batch_size, device=device)
        self._ones = torch.ones_like(self._batch_indices)

    def add_results(
        self,
        active_mask: torch.Tensor,
        labels: torch.Tensor,
        time_indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        self.scores = torch.where(active_mask, self.scores + scores, self.scores)
        self.transcript[self._batch_indices, self.current_lengths] = labels
        self.timestamps[self._batch_indices, self.current_lengths] = time_indices
        torch.where(
            torch.logical_and(active_mask, self.last_timestamp == time_indices),
            self.last_timestamp_lasts + 1,
            self.last_timestamp_lasts,
            out=self.last_timestamp_lasts,
        )
        torch.where(
            torch.logical_and(active_mask, self.last_timestamp != time_indices),
            self._ones,
            self.last_timestamp_lasts,
            out=self.last_timestamp_lasts,
        )
        torch.where(active_mask, time_indices, self.last_timestamp, out=self.last_timestamp)
        self.current_lengths += active_mask


class CoreMLStreamingDecoder:
    """Use exported decoder and joint CoreML models with Nemo's greedy RNNT loop."""

    def __init__(
        self,
        decoder_model: ct.models.MLModel,
        joint_model: ct.models.MLModel,
        *,
        vocab_size: int,
        blank_id: int,
        num_layers: int,
        hidden_size: int,
        durations: Sequence[int] = (0, 1, 2, 3, 4),
        max_symbols: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.decoder_model = decoder_model
        self.joint_model = joint_model
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.durations = torch.tensor(durations, dtype=torch.long, device=device)
        self.max_symbols = max_symbols
        self.device = device

    def _predict_decoder(self, labels: torch.Tensor, h_in: np.ndarray, c_in: np.ndarray) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        outputs = self.decoder_model.predict(
            {
                "targets": np.array([labels.cpu().numpy()], dtype=np.int32),
                "target_lengths": np.array([labels.numel()], dtype=np.int32),
                "h_in": h_in,
                "c_in": c_in,
            }
        )
        decoder_output = torch.from_numpy(outputs["decoder_output"]).to(self.device)
        return decoder_output, outputs["h_out"], outputs["c_out"]

    def _predict_joint(self, encoder_frame: torch.Tensor, decoder_output: torch.Tensor) -> torch.Tensor:
        outputs = self.joint_model.predict(
            {
                "encoder_outputs": encoder_frame.unsqueeze(1).cpu().numpy().astype(np.float32),
                "decoder_outputs": decoder_output.cpu().numpy().astype(np.float32),
            }
        )
        logits = torch.from_numpy(outputs["logits"]).to(self.device)
        return logits.squeeze(1).squeeze(1)

    def decode(self, encoder_output: torch.Tensor, encoder_lengths: torch.Tensor) -> List[List[int]]:
        batch_size, max_time, _ = encoder_output.shape
        encoder_output = encoder_output.to(self.device)
        encoder_lengths = encoder_lengths.to(self.device)

        float_dtype = encoder_output.dtype
        batch_indices = torch.arange(batch_size, device=self.device)
        labels = torch.full((batch_size,), fill_value=self.blank_id, device=self.device, dtype=torch.long)
        time_indices = torch.zeros_like(labels)
        safe_time_indices = torch.zeros_like(labels)
        time_indices_current = torch.zeros_like(labels)
        last_timesteps = encoder_lengths - 1
        active_mask = encoder_lengths > 0
        advance_mask = torch.empty_like(active_mask)
        active_mask_prev = torch.empty_like(active_mask)
        became_inactive = torch.empty_like(active_mask)

        hyps = BatchedHyps(
            batch_size=batch_size,
            init_length=max_time * self.max_symbols if self.max_symbols else max_time,
            device=self.device,
            float_dtype=float_dtype,
        )

        h_in = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)
        c_in = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)

        while active_mask.any():
            active_mask_prev.copy_(active_mask)
            decoder_output, h_in, c_in = self._predict_decoder(labels, h_in, c_in)
            logits = self._predict_joint(encoder_output[batch_indices, safe_time_indices], decoder_output)

            scores, labels = logits[:, : self.vocab_size].max(dim=-1)
            duration_indices = logits[:, self.vocab_size : self.vocab_size + len(self.durations)].argmax(dim=-1)
            durations = self.durations[duration_indices]

            blank_mask = labels == self.blank_id
            durations.masked_fill_(torch.logical_and(durations == 0, blank_mask), 1)
            time_indices_current.copy_(time_indices)
            time_indices += durations
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            torch.less(time_indices, encoder_lengths, out=active_mask)
            torch.logical_and(active_mask, blank_mask, out=advance_mask)

            while advance_mask.any():
                torch.where(advance_mask, time_indices, time_indices_current, out=time_indices_current)
                logits = self._predict_joint(encoder_output[batch_indices, safe_time_indices], decoder_output)
                more_scores, more_labels = logits[:, : self.vocab_size].max(dim=-1)
                labels = torch.where(advance_mask, more_labels, labels)
                scores = torch.where(advance_mask, more_scores, scores)
                duration_indices = logits[:, self.vocab_size : self.vocab_size + len(self.durations)].argmax(dim=-1)
                durations = self.durations[duration_indices]
                blank_mask = labels == self.blank_id
                durations.masked_fill_(torch.logical_and(durations == 0, blank_mask), 1)
                torch.where(advance_mask, time_indices + durations, time_indices, out=time_indices)
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                torch.less(time_indices, encoder_lengths, out=active_mask)
                torch.logical_and(active_mask, blank_mask, out=advance_mask)

            torch.ne(active_mask, active_mask_prev, out=became_inactive)
            hyps.add_results(active_mask, labels, time_indices_current, scores)

            if self.max_symbols is not None:
                force_blank = torch.logical_and(
                    active_mask,
                    torch.logical_and(
                        torch.logical_and(labels != self.blank_id, hyps.last_timestamp_lasts >= self.max_symbols),
                        hyps.last_timestamp == time_indices,
                    ),
                )
                time_indices += force_blank
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                torch.less(time_indices, encoder_lengths, out=active_mask)

        results: List[List[int]] = []
        for hyp in hyps.transcript:
            tokens = [int(token) for token in hyp.tolist() if 0 < token < self.vocab_size]
            results.append(tokens)
        return results


class StreamingTranscriber:
    def __init__(
        self,
        bundle: CoreMLModelBundle,
        *,
        blank_id: Optional[int] = None,
        num_layers: int = 2,
        hidden_size: int = 640,
        durations: Sequence[int] = (0, 1, 2, 3, 4),
    ) -> None:
        self.preprocessor = ct.models.MLModel(str(bundle.preprocessor), compute_units=ct.ComputeUnit.CPU_ONLY)
        self.encoder = ct.models.MLModel(str(bundle.encoder), compute_units=ct.ComputeUnit.CPU_ONLY)
        self.decoder = ct.models.MLModel(str(bundle.decoder), compute_units=ct.ComputeUnit.CPU_ONLY)
        self.joint = ct.models.MLModel(str(bundle.joint), compute_units=ct.ComputeUnit.CPU_ONLY)
        self.tokenizer = self._load_tokenizer(bundle.tokenizer)

        vocab_size = max(self.tokenizer.keys()) + 1
        if blank_id is None:
            blank_id = vocab_size - 1
        self.decoder_helper = CoreMLStreamingDecoder(
            self.decoder,
            self.joint,
            vocab_size=vocab_size,
            blank_id=blank_id,
            num_layers=num_layers,
            hidden_size=hidden_size,
            durations=durations,
        )
        self.blank_id = blank_id

    @staticmethod
    def _load_tokenizer(tokenizer_path: Optional[Path]) -> dict[int, str]:
        if tokenizer_path is None:
            raise ValueError("Tokenizer JSON is required")
        with Path(tokenizer_path).open() as f:
            data = json.load(f)
        return {int(k): v for k, v in data.items()}

    def _tokens_to_text(self, tokens: Iterable[int]) -> str:
        pieces: List[str] = []
        for token in tokens:
            piece = self.tokenizer.get(token)
            if piece is None:
                continue
            if piece.startswith("▁"):
                if pieces:
                    pieces.append(" ")
                pieces.append(piece[1:])
            else:
                pieces.append(piece)
        return "".join(pieces).strip()

    def _preprocess(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        audio_2d = audio.reshape(1, -1).astype(np.float32)
        length = np.array([audio_2d.shape[-1]], dtype=np.int32)
        outputs = self.preprocessor.predict({
            "audio_signal": audio_2d,
            "audio_length": length,
        })
        return outputs["melspectrogram"], int(outputs["melspectrogram_length"][0])

    def _encode(self, mel: np.ndarray, mel_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder.predict({
            "melspectrogram": mel.astype(np.float32),
            "melspectrogram_length": np.array([mel_length], dtype=np.int32),
        })
        encoder_output = outputs["encoder_output"]
        if encoder_output.ndim == 3:
            encoder_output = np.transpose(encoder_output, (0, 2, 1))
        length = torch.tensor(outputs["encoder_output_length"], dtype=torch.long)
        return torch.from_numpy(encoder_output.astype(np.float32)), length

    def transcribe(self, audio_path: Path) -> str:
        audio, _ = librosa.load(str(audio_path), sr=16000)
        mel, mel_length = self._preprocess(audio)
        encoder_output, encoder_length = self._encode(mel, mel_length)
        token_ids = self.decoder_helper.decode(encoder_output, encoder_length)[0]
        return self._tokens_to_text(token_ids)

    def transcribe_many(self, audio_paths: Sequence[Path]) -> List[str]:
        results: List[str] = []
        for path in audio_paths:
            LOGGER.info("Transcribing %s", path)
            results.append(self.transcribe(path))
        return results


def _resolve_bundle(args: argparse.Namespace) -> CoreMLModelBundle:
    base = Path(args.model_dir) if args.model_dir else None
    if base is None and not all([args.preprocessor, args.encoder, args.decoder, args.joint, args.tokenizer]):
        raise ValueError("Either --model-dir or explicit model paths are required")
    return CoreMLModelBundle(
        preprocessor=Path(args.preprocessor) if args.preprocessor else base / "Melspectrogram.mlpackage",
        encoder=Path(args.encoder) if args.encoder else base / "ParakeetEncoder.mlpackage",
        decoder=Path(args.decoder) if args.decoder else base / "ParakeetDecoder.mlpackage",
        joint=Path(args.joint) if args.joint else base / "RNNTJoint.mlpackage",
        tokenizer=Path(args.tokenizer) if args.tokenizer else base / "tokenizer.json",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming RNNT inference with CoreML components")
    parser.add_argument("--model-dir", type=Path, help="Directory containing exported CoreML models")
    parser.add_argument("--preprocessor", type=Path, help="Path to the preprocessor .mlpackage")
    parser.add_argument("--encoder", type=Path, help="Path to the encoder .mlpackage")
    parser.add_argument("--decoder", type=Path, help="Path to the decoder .mlpackage")
    parser.add_argument("--joint", type=Path, help="Path to the joint .mlpackage")
    parser.add_argument("--tokenizer", type=Path, help="Path to tokenizer JSON")
    parser.add_argument("audio", nargs="+", help="Audio files to transcribe")
    parser.add_argument("--blank-id", type=int, help="Blank token id")
    parser.add_argument("--num-layers", type=int, default=2, help="Prediction network layer count")
    parser.add_argument("--hidden-size", type=int, default=640, help="Prediction network hidden size")
    parser.add_argument("--durations", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="RNNT duration bucket values")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase log verbosity")
    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING - (10 * verbosity)
    logging.basicConfig(level=max(logging.DEBUG, level), format="[%(levelname)s] %(message)s")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    try:
        bundle = _resolve_bundle(args)
        transcriber = StreamingTranscriber(
            bundle,
            blank_id=args.blank_id,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            durations=tuple(args.durations),
        )
        transcripts = transcriber.transcribe_many([Path(p) for p in args.audio])
        for path, text in zip(args.audio, transcripts):
            print(f"{path}: {text}")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()
