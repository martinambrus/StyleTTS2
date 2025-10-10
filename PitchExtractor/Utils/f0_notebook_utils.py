"""Shared helpers for Notebook-based evaluations.

The notebooks under ``Utils`` historically re-implemented small slices of the
dataset preprocessing pipeline which made it difficult to keep them in sync
whenever we added new functionality.  This module mirrors the key bits of the
runtime code (F0 extraction, resampling, cache configuration) so that the
notebooks stay aligned with the training stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio

from f0_backends import BackendComputationError, BackendResult, build_f0_extractor


@dataclass
class NotebookF0Result:
    f0: np.ndarray
    backend_name: str


def load_training_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    config_path = Path(config_path)
    if not config_path.is_file():
        return {}

    import yaml

    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def resolve_dataset_params(training_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset_params = training_config.get("dataset_params", {}) if training_config else {}
    mel_params = dataset_params.get("mel_params", {}) if dataset_params else {}
    f0_params = dataset_params.get("f0_params", {}) if dataset_params else {}
    return dict(mel_params), dict(f0_params)


def build_notebook_f0_extractor(
    mel_params: Dict[str, Any],
    f0_params: Dict[str, Any],
    *,
    fallback_sr: int,
    fallback_hop: int,
    verbose: bool = False,
):
    sample_rate = int(mel_params.get("sample_rate", fallback_sr))
    hop_length = int(mel_params.get("hop_length", mel_params.get("hop_len", fallback_hop)))
    return build_f0_extractor(sample_rate, hop_length, config=f0_params, verbose=verbose)


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 0:
        return audio.reshape(-1)
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio
    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, source_sr, target_sr)
    return resampled.squeeze(0).cpu().numpy()


def compute_f0_for_notebook(
    audio: np.ndarray,
    sr: int,
    extractor,
    *,
    target_frames: Optional[int] = None,
    zero_fill_value: float = 0.0,
) -> NotebookF0Result:
    waveform = ensure_mono(audio.astype(np.float32, copy=False))
    target_sr = extractor.sample_rate
    if sr != target_sr:
        waveform = resample_audio(waveform, sr, target_sr)
        sr = target_sr

    try:
        result: BackendResult = extractor.compute(waveform, sr=sr)
        f0 = result.f0
        backend_name = result.backend_name
    except BackendComputationError as exc:
        f0 = np.zeros((0,), dtype=np.float32)
        backend_name = ""
        print(f"Warning: all F0 backends failed ({exc}). Returning zeros.")

    if target_frames is not None:
        f0 = extractor.align_length(f0, target_frames)

    if np.any(np.isnan(f0)):
        f0 = np.nan_to_num(f0, nan=zero_fill_value)

    return NotebookF0Result(f0=f0.astype(np.float32), backend_name=backend_name)

