import logging
import math
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file
from torch import nn
from transformers import AutoFeatureExtractor, AutoTokenizer, BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import ProcessorMixin

from .configuration_cohere_asr import _dynamo_disable

logger = logging.getLogger(__name__)

DITHER_CONSTANT = 1e-5


class FilterbankFeatures(nn.Module):
    """Filterbank features extraction module.

    Follows NeMo's FilterbankFeatures implementation.
    """

    window: torch.Tensor
    fb: torch.Tensor

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=DITHER_CONSTANT,
        pad_to=16,
        max_duration=30,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,
        stft_conv=False,
        device="cpu",
    ):
        super().__init__()
        if stft_conv or stft_exact_pad:
            logger.warning(
                "torch_stft compatibility flags are deprecated; " "forcing behavior to default torch.stft path."
            )
        if exact_pad and n_window_stride % 2 == 1:
            raise NotImplementedError(f"{self} received exact_pad=True with odd hop length ({n_window_stride}).")

        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError("n_window_size and n_window_stride must be positive ints.")

        self.log_zero_guard_value = log_zero_guard_value
        self.sample_rate = sample_rate
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None
        self.exact_pad = exact_pad
        self.max_duration = max_duration

        torch_windows = {
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
            "blackman": torch.blackman_window,
            "bartlett": torch.bartlett_window,
            "none": None,
        }
        window_fn = torch_windows.get(window)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2
        self.pad_min_duration = 0.0
        self.pad_direction = "both"
        self.pad_value = pad_value
        self.mag_power = mag_power
        self.nb_augmentation_prob = nb_augmentation_prob

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq, norm=mel_norm
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad

        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError("log_zero_guard_type must be 'add' or 'clamp'.")
        self.log_zero_guard_type = log_zero_guard_type

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng

        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * self.n_fft)

        if self.window is None:
            raise RuntimeError("Expected a window tensor for STFT feature extraction.")
        if self.fb is None:
            raise RuntimeError("Expected mel filterbank weights for feature extraction.")
        self.window = self.window.to(dtype=torch.bfloat16)
        self.fb = self.fb.to(dtype=torch.bfloat16)
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(0)

    @_dynamo_disable
    def _apply_dither(self, x, seq_len_time):
        """Apply deterministic per-sample dither outside torch.compile.

        Each sample is seeded by its valid waveform length so that dither noise
        is batch-composition invariant (a sample's features depend only on its
        own content, not on what else is in the batch).
        """
        if self.dither <= 0:
            return x
        for i in range(x.shape[0]):
            valid_samples = min(int(seq_len_time[i].item()), x.shape[1])
            if valid_samples <= 0:
                continue
            self.generator.manual_seed(valid_samples)
            noise = torch.randn(
                (valid_samples,),
                dtype=x.dtype,
                device=x.device,
                generator=self.generator,
            )
            x[i, :valid_samples] += self.dither * noise
        return x

    @_dynamo_disable
    def stft(self, x):
        with torch.amp.autocast(x.device.type, enabled=False):
            return torch.view_as_real(
                torch.stft(
                    x,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    center=not self.exact_pad,
                    window=self.window.to(dtype=torch.float, device=x.device),
                    return_complex=True,
                    pad_mode="constant",
                )
            )

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            if self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            raise ValueError("log_zero_guard_value must be number, 'tiny', or 'eps' when str.")
        return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
        return seq_len.to(dtype=torch.long)

    def splice_frames(self, x, frame_splicing):
        seq = [x]
        for n in range(1, frame_splicing):
            seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
        return torch.cat(seq, dim=1)

    def normalize_batch(self, x, seq_len, normalize_type):
        if normalize_type != "per_feature":
            raise ValueError("Only per_feature normalization is supported.")
        batch_size = x.shape[0]
        max_time = x.shape[2]
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        valid_mask = time_steps < seq_len.unsqueeze(1)
        x_mean_num = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
        x_mean_den = valid_mask.sum(axis=1)
        x_mean = x_mean_num / x_mean_den.unsqueeze(1)
        x_std = torch.sqrt(
            torch.sum(
                torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2,
                axis=2,
            )
            / (x_mean_den.unsqueeze(1) - 1.0)
        )
        x_std = x_std.masked_fill(x_std.isnan(), 0.0)
        x_std += DITHER_CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std

    def forward(self, x, seq_len, linear_spec=False):
        if x.shape[1] < self.sample_rate * self.pad_min_duration:
            pad_amount = int(self.sample_rate * self.pad_min_duration) - x.shape[1]
            if self.pad_direction == "right":
                x = F.pad(x, (0, pad_amount), value=self.pad_value)
            elif self.pad_direction == "left":
                x = F.pad(x, (pad_amount, 0), value=self.pad_value)
            elif self.pad_direction == "both":
                left_pad = pad_amount // 2
                right_pad = pad_amount - left_pad
                x = F.pad(x, (left_pad, right_pad), value=self.pad_value)
            else:
                raise ValueError(f"Invalid pad_direction: {self.pad_direction}")
            seq_len = torch.tensor([x.shape[1]], dtype=torch.float, device=x.device)

        seq_len_time = seq_len
        seq_len_unfixed = self.get_seq_len(seq_len)
        seq_len = torch.where(seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
            ).squeeze(1)

        x = self._apply_dither(x, seq_len_time)

        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)

        x = self.stft(x)
        guard = 0 if not self.use_grads else DITHER_CONSTANT
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)
        if linear_spec:
            return x, seq_len

        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.matmul(self.fb.to(x.dtype), x)

        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        if self.frame_splicing > 1:
            x = self.splice_frames(x, self.frame_splicing)
        if self.normalize:
            x, _, _ = self.normalize_batch(x, seq_len, normalize_type=self.normalize)

        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), self.pad_value)
        del mask

        if self.pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, self.pad_to - pad_amt), value=self.pad_value)
        return x, seq_len


class CohereAsrFeatureExtractor(SequenceFeatureExtractor):
    """HF-compatible feature extractor wrapping FilterbankFeatures."""

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=64,
        sampling_rate=16000,
        padding_value=0.0,
        max_duration=30,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=DITHER_CONSTANT,
        pad_to=16,
        frame_splicing=1,
        exact_pad=False,
        mag_power=2.0,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,
        stft_conv=False,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.max_duration = max_duration
        self.hop_length = n_window_stride
        self._device = str(device)
        self._fb_config = dict(
            sample_rate=sampling_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=feature_size,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            max_duration=max_duration,
            frame_splicing=frame_splicing,
            exact_pad=exact_pad,
            pad_value=padding_value,
            mag_power=mag_power,
            nb_augmentation_prob=nb_augmentation_prob,
            nb_max_freq=nb_max_freq,
            mel_norm=mel_norm,
            stft_exact_pad=stft_exact_pad,
            stft_conv=stft_conv,
            device=device,
        )
        self._filterbank = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        fe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model_dir = Path(pretrained_model_name_or_path)
        if model_dir.is_dir():
            _maybe_load_preprocessor_buffers_from_checkpoint(feature_extractor=fe, model_dir=model_dir)
        return fe

    @property
    def filterbank(self):
        if self._filterbank is None:
            fb = FilterbankFeatures(**self._fb_config)
            fb.eval()
            self._filterbank = fb.to(self._device)
        return self._filterbank

    def get_seq_len(self, seq_len):
        return self.filterbank.get_seq_len(seq_len)

    def __call__(
        self,
        raw_speech,
        sampling_rate=None,
        return_tensors=None,
        **kwargs,
    ):
        """Extract mel features from raw waveform input."""
        if sampling_rate is not None and int(sampling_rate) != int(self.sampling_rate):
            raise ValueError(f"Expected sampling_rate={self.sampling_rate}, got {sampling_rate}")

        if isinstance(raw_speech, np.ndarray):
            if raw_speech.ndim == 1:
                raw_speech = [raw_speech]
            else:
                raw_speech = [s for s in raw_speech]
        elif isinstance(raw_speech, torch.Tensor):
            if raw_speech.ndim == 1:
                raw_speech = [raw_speech.detach().cpu().numpy()]
            else:
                raw_speech = [s.detach().cpu().numpy() for s in raw_speech]
        elif not isinstance(raw_speech, (list, tuple)):
            raise TypeError("raw_speech must be an array/tensor or list of arrays.")

        normalized = []
        for sample in raw_speech:
            arr = np.asarray(sample, dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError("Each audio sample must be 1D waveform.")
            normalized.append(arr)

        seq_len = torch.tensor([s.shape[0] for s in normalized], dtype=torch.long)
        max_len = max(s.shape[0] for s in normalized)
        padded = np.zeros((len(normalized), max_len), dtype=np.float32)
        for i, s in enumerate(normalized):
            padded[i, : s.shape[0]] = s

        audio_tensor = torch.from_numpy(padded).to(self._device)
        seq_len = seq_len.to(self._device)
        with torch.no_grad():
            input_features, length = self.filterbank(audio_tensor, seq_len)

        result = BatchFeature({"input_features": input_features.cpu(), "length": length.cpu()})
        if return_tensors is not None:
            result = result.convert_to_tensors(return_tensors)
        return result


class CohereAsrProcessor(ProcessorMixin):
    """HF-compatible processor for Cohere ASR.

    ``ProcessorMixin._get_arguments_from_pretrained`` resolves sub-component
    class names by looking them up inside the ``transformers`` package, which
    fails for custom remote-code classes.  We override ``from_pretrained`` to
    use ``AutoFeatureExtractor`` / ``AutoTokenizer`` instead -- those honour
    ``auto_map`` and ``trust_remote_code``.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "CohereAsrFeatureExtractor"
    tokenizer_class = "CohereAsrTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, **kwargs):
        if feature_extractor is None:
            raise ValueError(
                "CohereAsrProcessor requires a CohereAsrFeatureExtractor instance. " "Got feature_extractor=None."
            )
        if tokenizer is None:
            raise ValueError("CohereAsrProcessor requires a CohereAsrTokenizer instance. " "Got tokenizer=None.")
        # Bypass super().__init__ which calls get_possibly_dynamic_module to
        # validate sub-component types.  That lookup searches the transformers
        # package namespace and fails for remote-code classes.  We set the
        # attributes directly instead -- the type checks above are sufficient.
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.chat_template = kwargs.get("chat_template", None)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def __call__(
        self,
        audio=None,
        text=None,
        sampling_rate=None,
        return_tensors=None,
        **kwargs,
    ):
        """Run audio feature extraction and optional text tokenization."""
        if audio is None:
            raise ValueError("audio is required for CohereAsrProcessor.")

        result = self.feature_extractor(audio, sampling_rate=sampling_rate, return_tensors=return_tensors)

        if text is not None:
            add_special_tokens = kwargs.pop("add_special_tokens", False)
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens,
                **kwargs,
            )
            result["input_ids"] = text_inputs["input_ids"]
            if "attention_mask" in text_inputs:
                result["attention_mask"] = text_inputs["attention_mask"]
        return result

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


def _maybe_load_preprocessor_buffers_from_checkpoint(
    feature_extractor: CohereAsrFeatureExtractor, model_dir: Path
) -> None:
    """
    Load exported frontend buffers if they exist in checkpoint weights.
    """
    safetensor_path = model_dir / "model.safetensors"
    if not safetensor_path.exists():
        return
    try:
        state = safetensors_load_file(safetensor_path.as_posix())
    except Exception:
        return

    fb = state.get("preprocessor.featurizer.fb")
    window = state.get("preprocessor.featurizer.window")
    if fb is None or window is None:
        return

    fb_module = feature_extractor.filterbank
    target_device = fb_module.fb.device
    target_dtype = fb_module.fb.dtype
    fb_module.fb = fb.to(device=target_device, dtype=target_dtype)
    fb_module.window = window.to(device=target_device, dtype=target_dtype)
