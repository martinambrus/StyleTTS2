from typing import Optional, Union

import importlib
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from transformers.utils import TensorType

__all__ = ["DifferentiableWhisperFeatureExtractor"]


def _load_whisper():
    global whisper
    if "whisper" not in globals():
        whisper = importlib.import_module("whisper")


class DifferentiableWhisperFeatureExtractor(WhisperFeatureExtractor):
    """Wrapper around :class:`~WhisperFeatureExtractor` using differentiable STFT."""

    def __init__(self, wfe: WhisperFeatureExtractor):
        self.hop_length = wfe.hop_length
        self.n_fft = wfe.n_fft
        self.dither = getattr(wfe, "dither", 0.0)
        self.sampling_rate = wfe.sampling_rate
        self.feature_size = wfe.feature_size
        self.mel_filters = wfe.mel_filters
        self.n_samples = wfe.n_samples
        self.return_attention_mask = wfe.return_attention_mask
        self.padding_side = wfe.padding_side
        self.padding_value = wfe.padding_value

    def _hf_differentiable_extract_fbank_features(
        self, waveform: torch.Tensor, device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Return differentiable log-Mel spectrogram on the specified device and dtype."""
        _load_whisper()
        if device is None:
            device = waveform.device
        if dtype is None:
            dtype = waveform.dtype

        log_spec = whisper.audio.log_mel_spectrogram(
            waveform.to(device=device, dtype=dtype), n_mels=self.feature_size, device=device
        )
        return log_spec.to(device=device, dtype=dtype)

    def __call__(
        self,
        raw_speech: torch.Tensor,
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        return_token_timestamps: Optional[bool] = None,
        **kwargs,
    ) -> BatchFeature:
        max_length = max_length if max_length else self.n_samples
        current_max_len = raw_speech.shape[-1]

        if current_max_len > max_length:
            raw_speech = raw_speech[..., 0:max_length]
        elif current_max_len < max_length:
            batch_size = raw_speech.shape[0]
            new_len = max_length - current_max_len
            zeros = torch.zeros((batch_size, new_len), device=raw_speech.device, dtype=raw_speech.dtype)
            raw_speech = torch.concat([raw_speech, zeros], dim=1)

        input_features = self._hf_differentiable_extract_fbank_features(raw_speech, device, dtype)

        return BatchFeature({"input_features": input_features})
