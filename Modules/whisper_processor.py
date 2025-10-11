"""Differentiable Whisper feature extractor compatible with the HF API."""

from typing import Optional, Union

import importlib

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor
from transformers.utils import TensorType


def _load_whisper() -> None:
    """Lazily import the OpenAI Whisper package."""

    global whisper
    if "whisper" not in globals():
        whisper = importlib.import_module("whisper")


class DifferentiableWhisperFeatureExtractor:
    """Drop-in replacement for :class:`WhisperFeatureExtractor` using torch ops."""

    def __init__(self, feature_extractor: WhisperFeatureExtractor):
        self.hop_length = feature_extractor.hop_length
        self.n_fft = feature_extractor.n_fft
        self.dither = getattr(feature_extractor, "dither", 0.0)
        self.sampling_rate = feature_extractor.sampling_rate
        self.feature_size = feature_extractor.feature_size
        self.mel_filters = feature_extractor.mel_filters
        self.n_samples = feature_extractor.n_samples
        self.return_attention_mask = feature_extractor.return_attention_mask
        self.padding_side = feature_extractor.padding_side
        self.padding_value = feature_extractor.padding_value

    def _ensure_batch(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        if waveform.dim() != 2:
            raise ValueError(f"Unexpected raw_speech shape: {tuple(waveform.shape)}")
        return waveform

    def _hf_differentiable_extract_fbank_features(
        self, waveform: torch.Tensor, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Return differentiable log-Mel spectrogram on the specified device and dtype."""

        _load_whisper()
        device = device or waveform.device
        dtype = dtype or waveform.dtype

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
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling rate {self.sampling_rate}, but received {sampling_rate}."
            )

        if padding not in {"max_length", "longest", "do_not_pad"}:
            raise ValueError(f"Unsupported padding mode: {padding}")

        raw_speech = self._ensure_batch(raw_speech)

        if do_normalize:
            raw_speech = torch.nn.functional.layer_norm(raw_speech, raw_speech.shape[-1:])

        max_length = max_length if max_length is not None else self.n_samples
        if truncation and raw_speech.shape[-1] > max_length:
            raw_speech = raw_speech[..., :max_length]

        if padding == "max_length":
            current_len = raw_speech.shape[-1]
            if pad_to_multiple_of is not None:
                max_length = int(torch.ceil(torch.tensor(max_length / pad_to_multiple_of)) * pad_to_multiple_of)
            if current_len < max_length:
                pad_size = max_length - current_len
                pad_tensor = raw_speech.new_full((raw_speech.size(0), pad_size), float(self.padding_value))
                raw_speech = torch.cat([raw_speech, pad_tensor], dim=-1)

        input_features = self._hf_differentiable_extract_fbank_features(raw_speech, device=device, dtype=dtype)

        output = {"input_features": input_features}

        should_return_attention_mask = (
            return_attention_mask
            if return_attention_mask is not None
            else bool(self.return_attention_mask)
        )
        if should_return_attention_mask:
            attention_mask = torch.ones(
                input_features.size(0), input_features.size(-1), device=input_features.device, dtype=torch.long
            )
            output["attention_mask"] = attention_mask

        return BatchFeature(output, tensor_type=return_tensors)
