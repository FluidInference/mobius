import torch
from transformers import PretrainedConfig

DEFAULT_SUPPORTED_LANGUAGES = ["ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"]
NO_SPACE_LANGS = {"ja", "zh"}


class CohereAsrConfig(PretrainedConfig):
    """Configuration for the Cohere ASR remote-code model."""

    model_type = "cohere_asr"

    def __init__(
        self,
        vocab_size=16384,
        encoder=None,
        transf_decoder=None,
        head=None,
        preprocessor=None,
        max_audio_clip_s=35,
        overlap_chunk_second=5,
        min_energy_window_samples=1600,
        batch_size=64,
        sample_rate=16000,
        supported_languages=None,
        **kwargs,
    ):
        kwargs.setdefault("is_encoder_decoder", True)
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.transf_decoder = transf_decoder
        self.head = head
        self.preprocessor = preprocessor
        self.max_audio_clip_s = max_audio_clip_s
        self.overlap_chunk_second = overlap_chunk_second
        self.min_energy_window_samples = min_energy_window_samples
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.supported_languages = (
            list(supported_languages) if supported_languages is not None else list(DEFAULT_SUPPORTED_LANGUAGES)
        )
        super().__init__(**kwargs)

    @property
    def num_hidden_layers(self):
        return self.transf_decoder["config_dict"]["num_layers"]


if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "disable"):
    _dynamo_disable = torch._dynamo.disable
else:

    def _dynamo_disable(fn):
        return fn
