#!/usr/bin/env python3
"""Pure Python implementation of Cohere Transcribe mel spectrogram preprocessing.

This matches the exact preprocessing used by the Cohere model, without requiring
the transformers library's feature extractor.
"""

import numpy as np


class CohereMelSpectrogram:
    """Mel spectrogram preprocessor matching Cohere Transcribe's exact parameters."""

    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=160,
        n_mels=128,
        fmin=0.0,
        fmax=8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        # Create mel filterbank
        self.mel_filters = self._create_mel_filterbank()

    def _create_mel_filterbank(self):
        """Create mel filterbank matrix."""
        # Convert Hz to Mel
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel scale
        mel_min = hz_to_mel(self.fmin)
        mel_max = hz_to_mel(self.fmax)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert to FFT bin numbers
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        # Create filterbank
        fbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for m in range(1, self.n_mels + 1):
            f_left = bin_points[m - 1]
            f_center = bin_points[m]
            f_right = bin_points[m + 1]

            # Left slope
            for k in range(f_left, f_center):
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)

            # Right slope
            for k in range(f_center, f_right):
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

        return fbank

    def __call__(self, audio):
        """
        Compute mel spectrogram from audio.

        Args:
            audio: 1D numpy array of audio samples (float32, range roughly -1 to 1)

        Returns:
            mel: (1, n_mels, n_frames) numpy array
        """
        # Ensure float32
        audio = audio.astype(np.float32)

        # Add padding to match transformers behavior
        n_samples = len(audio)
        n_frames = 1 + (n_samples - self.n_fft) // self.hop_length

        # Compute STFT
        stft = self._stft(audio)

        # Compute power spectrogram
        power = np.abs(stft) ** 2

        # Apply mel filterbank
        mel = np.dot(self.mel_filters, power)

        # Log mel spectrogram (matching transformers)
        mel = np.log10(np.maximum(mel, 1e-10))

        # Add batch dimension
        mel = mel[np.newaxis, :, :]

        return mel

    def _stft(self, audio):
        """Compute Short-Time Fourier Transform."""
        # Pad audio
        pad_length = self.n_fft // 2
        audio_padded = np.pad(audio, (pad_length, pad_length), mode="reflect")

        # Hann window
        window = np.hanning(self.n_fft)

        # Calculate number of frames
        n_frames = 1 + (len(audio_padded) - self.n_fft) // self.hop_length

        # Initialize STFT matrix
        stft = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=np.complex64)

        # Compute STFT
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio_padded[start : start + self.n_fft]
            windowed = frame * window
            fft = np.fft.rfft(windowed)
            stft[:, i] = fft

        return stft
