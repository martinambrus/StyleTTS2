"""Utilities for computing Whisper features with autograd support."""

from __future__ import annotations

import importlib
from typing import Optional, Union

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)
from transformers.utils import TensorType


def _load_whisper() -> None:
    """Lazily import the OpenAI whisper package."""

    global whisper
    if "whisper" not in globals():
        whisper = importlib.import_module("whisper")


class DifferentiableWhisperFeatureExtractor(WhisperFeatureExtractor):
    """Whisper feature extractor that keeps the computation differentiable."""

    def __init__(self, wfe: WhisperFeatureExtractor):
        # copy only the attributes that are required for feature computation
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
        self, waveform: torch.Tensor, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
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
        """Featurise one or more waveforms into Whisper log-mel spectrograms."""

        if raw_speech.ndim == 3 and raw_speech.size(1) == 1:
            raw_speech = raw_speech.squeeze(1)
        elif raw_speech.ndim == 1:
            raw_speech = raw_speech.unsqueeze(0)
        elif raw_speech.ndim != 2:
            raise ValueError(f"Unexpected raw_speech shape: {raw_speech.shape}")

        max_length = max_length if max_length else self.n_samples
        current_max_len = raw_speech.shape[-1]

        if current_max_len > max_length:
            raw_speech = raw_speech[..., 0:max_length]
        elif current_max_len < max_length:
            batch_size = raw_speech.shape[0]
            new_len = max_length - current_max_len
            zeros = torch.zeros((batch_size, new_len), device=raw_speech.device, dtype=raw_speech.dtype)
            raw_speech = torch.cat([raw_speech, zeros], dim=1)

        input_features = self._hf_differentiable_extract_fbank_features(raw_speech, device, dtype)
        input_features = self._pad_or_trim_features(input_features)

        return BatchFeature({"input_features": input_features})

    def _pad_or_trim_features(self, input_features: torch.Tensor) -> torch.Tensor:
        """Match Whisper's expected 30s (3000 frame) feature length."""

        expected_len = int(self.n_samples // self.hop_length)
        if input_features.shape[-1] > expected_len:
            input_features = input_features[..., :expected_len]
        elif input_features.shape[-1] < expected_len:
            pad_width = expected_len - input_features.shape[-1]
            pad_shape = (*input_features.shape[:-1], pad_width)
            padding = torch.zeros(pad_shape, device=input_features.device, dtype=input_features.dtype)
            input_features = torch.cat([input_features, padding], dim=-1)
        return input_features
