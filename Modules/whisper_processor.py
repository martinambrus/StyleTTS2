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
        # Some checkpoints ship with mutated FFT parameters (e.g. a tiny
        # ``hop_length``) alongside the inflated ``chunk_length`` that caused
        # the original crash. Whisper's reference implementation, however,
        # always uses a 400-point FFT with a hop of 160 samples. Reinstating
        # those canonical numbers keeps the feature grid consistent with the
        # encoder's expectations.
        self.n_fft = 400
        self.hop_length = 160
        self.dither = getattr(wfe, "dither", 0.0)
        self.sampling_rate = wfe.sampling_rate
        self.feature_size = wfe.feature_size
        self.mel_filters = wfe.mel_filters
        # Some downstream configurations accidentally mutate ``chunk_length`` in the
        # serialized feature-extractor, which then propagates unrealistic budgets
        # (and ultimately leads to enormous mel tensors). Whisper's encoder,
        # however, is hard-coded for the 30s/3000-frame window, so we explicitly
        # pin those budgets here instead of trusting the possibly-modified
        # metadata from ``wfe``.
        self.chunk_length = 30.0
        self._expected_num_frames = 3000
        self.nb_max_frames = self._expected_num_frames
        default_samples = int(round(self.chunk_length * self.sampling_rate))
        self.max_input_samples = default_samples
        self.n_samples = default_samples
        self.return_attention_mask = wfe.return_attention_mask
        self.padding_side = wfe.padding_side
        self.padding_value = wfe.padding_value

    def _restore_audio_defaults(self) -> None:
        """Restore Whisper's canonical audio constants before feature extraction."""

        frames_per_second = int(round(self._expected_num_frames / self.chunk_length))
        tokens_per_second = max(frames_per_second // 2, 1)
        samples_per_token = max(int(round(self.sampling_rate / tokens_per_second)), 1)

        if (
            whisper.audio.N_FFT != self.n_fft
            or whisper.audio.HOP_LENGTH != self.hop_length
            or whisper.audio.N_FRAMES != self._expected_num_frames
            or whisper.audio.N_SAMPLES != self.n_samples
            or whisper.audio.SAMPLE_RATE != self.sampling_rate
        ):
            whisper.audio.N_FFT = self.n_fft
            whisper.audio.HOP_LENGTH = self.hop_length
            whisper.audio.N_FRAMES = self._expected_num_frames
            whisper.audio.N_SAMPLES = self.n_samples
            whisper.audio.SAMPLE_RATE = self.sampling_rate
            whisper.audio.FRAMES_PER_SECOND = frames_per_second
            whisper.audio.TOKENS_PER_SECOND = tokens_per_second
            whisper.audio.N_SAMPLES_PER_TOKEN = samples_per_token
            whisper.audio.CHUNK_LENGTH = self.chunk_length
            whisper.audio.mel_filters.cache_clear()

    def _hf_differentiable_extract_fbank_features(
        self, waveform: torch.Tensor, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Return differentiable log-Mel spectrogram on the specified device and dtype."""

        _load_whisper()
        self._restore_audio_defaults()

        if device is None:
            device = waveform.device
        if dtype is None:
            dtype = waveform.dtype

        waveform = waveform.to(device=device, dtype=dtype)
        waveform = whisper.audio.pad_or_trim(waveform, length=self.n_samples)

        log_spec = whisper.audio.log_mel_spectrogram(
            waveform, n_mels=self.feature_size, device=device
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

        target_num_samples = max_length if max_length else self.max_input_samples
        current_max_len = raw_speech.shape[-1]

        if current_max_len > target_num_samples:
            raw_speech = raw_speech[..., :target_num_samples]
        elif current_max_len < target_num_samples:
            batch_size = raw_speech.shape[0]
            new_len = target_num_samples - current_max_len
            zeros = torch.zeros((batch_size, new_len), device=raw_speech.device, dtype=raw_speech.dtype)
            raw_speech = torch.cat([raw_speech, zeros], dim=1)

        input_features = self._hf_differentiable_extract_fbank_features(raw_speech, device, dtype)
        input_features = self._pad_or_trim_features(input_features)

        return BatchFeature({"input_features": input_features})

    def _pad_or_trim_features(self, input_features: torch.Tensor) -> torch.Tensor:
        """Match Whisper's expected 30s (3000 frame) feature length."""

        expected_len = self._expected_num_frames
        if input_features.shape[-1] > expected_len:
            input_features = input_features[..., :expected_len]
        elif input_features.shape[-1] < expected_len:
            pad_width = expected_len - input_features.shape[-1]
            pad_shape = (*input_features.shape[:-1], pad_width)
            padding = torch.zeros(pad_shape, device=input_features.device, dtype=input_features.dtype)
            input_features = torch.cat([input_features, padding], dim=-1)
        return input_features
