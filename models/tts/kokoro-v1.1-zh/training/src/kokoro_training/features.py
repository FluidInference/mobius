"""Explicit native-24k waveform and 300/600-sample feature grids."""

import numpy as np
import torch
from torch import nn
import librosa

SAMPLE_RATE = 24000
HOP = 300
ALIGNMENT_HOP = 600


class MelFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        # Match StyleTTS2's published MelSpectrogram defaults, including its
        # 16k filter-bank parameter applied to native 24k audio. Do not silently
        # change the pretrained JDC/N-target convention to another mel scale.
        bank = librosa.filters.mel(sr=16000, n_fft=2048, n_mels=80,
                                  fmin=0, fmax=8000, htk=True, norm=None)
        self.register_buffer("bank", torch.tensor(bank))
        self.register_buffer("window", torch.hann_window(1200))

    def forward(self, audio: torch.Tensor):
        # Equivalent to STFT center=True/reflect, with deterministic slice/flip
        # gradients instead of CUDA reflection_pad1d's atomic backward.
        if audio.shape[-1] <= 1024:
            raise ValueError("Mel analysis requires more than 1024 waveform samples")
        padded = torch.cat((audio[..., 1:1025].flip(-1), audio,
                            audio[..., -1025:-1].flip(-1)), dim=-1)
        spec = torch.stft(padded, n_fft=2048, hop_length=300, win_length=1200,
                          window=self.window, center=False, return_complex=True).abs().square()
        mel = self.bank @ spec
        return mel

    def normalized(self, audio: torch.Tensor):
        return (torch.log(self(audio) + 1e-5) + 4) / 4


def alignment_features(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=audio, sr=24000, n_fft=2048,
                                         hop_length=600, win_length=1200, n_mels=80)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)[1:]
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-5)
    return mfcc[:, :len(audio) // ALIGNMENT_HOP]


def dtw_durations(teacher: np.ndarray, real: np.ndarray, durations: np.ndarray):
    """Transfer teacher token boundaries through a real-recording DTW path.

    This is approximate acoustic alignment, not independent recognition. The
    real waveform remains the sole reconstruction target; teacher audio is
    transient alignment scaffolding, never inserted into the recording corpus.
    """
    source = alignment_features(teacher)
    target = alignment_features(real)
    if target.shape[1] < len(durations):
        raise ValueError("Fewer real alignment frames than text tokens")
    from scipy.spatial.distance import cdist
    cost = cdist(source.T, target.T, metric="cosine")
    accumulated, reverse_path = librosa.sequence.dtw(C=cost, global_constraints=True, band_rad=0.25)
    path = reverse_path[::-1]
    if not np.isfinite(accumulated[-1, -1]):
        raise ValueError("No finite monotone alignment path")
    # Every source frame receives at least one target correspondence.
    mapped = np.array([np.mean(path[path[:, 0] == i, 1]) for i in range(source.shape[1])])
    raw = np.interp(np.cumsum(durations)[:-1] - 0.5, np.arange(len(mapped)), mapped) + 0.5
    boundaries = np.zeros(len(durations) + 1, dtype=np.int64)
    boundaries[-1] = target.shape[1]
    for i, value in enumerate(raw, 1):
        boundaries[i] = np.clip(round(value), boundaries[i - 1] + 1,
                                target.shape[1] - (len(durations) - i))
    result = np.diff(boundaries)
    if result.min() < 1 or result.max() > 50 or result.sum() != target.shape[1]:
        raise ValueError("Transferred durations violate model frame/duration bounds")
    return result, {"method": "teacher-mfcc-dtw-v1", "mean_path_cost": float(cost[path[:, 0], path[:, 1]].mean()),
                    "path_steps": len(path), "teacher_frames": source.shape[1],
                    "real_frames": target.shape[1], "duration_ratio": target.shape[1] / source.shape[1],
                    "max_token_duration": int(result.max())}
