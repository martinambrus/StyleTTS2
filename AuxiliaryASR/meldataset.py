#coding: utf-8

import os
import os.path as osp
import time
import math
import random
import json
import hashlib
from pathlib import Path
import numpy as np
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.distributions import Beta
from torch.utils.data import DataLoader, Sampler
from torchaudio import transforms as T

from nltk.tokenize import word_tokenize
#import phonemizer
import torchaudio.functional as AF
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from text_utils import TextCleaner
np.random.seed(1)
random.seed(1)
DEFAULT_DICT_PATH = osp.join(osp.dirname(__file__), 'word_index_dict.txt')
# SPECT_PARAMS = {
#     "n_fft": 2048,
#     "win_length": 1200,
#     "hop_length": 300
# }
# MEL_PARAMS = {
#     "n_mels": 80,
#     "n_fft": 2048,
#     "win_length": 1200,
#     "hop_length": 300
# }

#global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)


def _ensure_list(value: Optional[Sequence]) -> List:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _collect_audio_files(sources: Optional[Sequence[str]]) -> List[str]:
    paths: List[str] = []
    for src in _ensure_list(sources):
        if not src:
            continue
        if osp.isdir(src):
            for root, _, files in os.walk(src):
                for name in files:
                    lowered = name.lower()
                    if lowered.endswith(('.wav', '.flac', '.ogg', '.mp3')):
                        paths.append(os.path.join(root, name))
        elif osp.isfile(src):
            paths.append(src)
    return paths


def _load_audio_sample(path: str, sample_rate: int) -> Optional[torch.Tensor]:
    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        return None

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    if sr != sample_rate:
        waveform = AF.resample(waveform, sr, sample_rate)
    return waveform


def _match_length(signal: torch.Tensor, target_length: int) -> torch.Tensor:
    if signal.size(0) == target_length:
        return signal
    if signal.size(0) > target_length:
        start = random.randint(0, max(0, signal.size(0) - target_length))
        return signal[start:start + target_length]
    repeat_times = math.ceil(target_length / signal.size(0))
    expanded = signal.repeat(repeat_times)[:target_length]
    return expanded


class MelFeatureCache:
    """Stores log-mel tensors inside memory-mapped ``.npy`` files.

    The cache mirrors the behaviour of :func:`prepare_data_list` metadata caching by
    persisting pre-computed features to disk and validating them against the
    current dataset settings.  When the mel computation options change, or the
    underlying audio files are modified, cached entries are invalidated
    automatically.
    """

    _META_FILENAME = "meta.json"
    _CACHE_VERSION = 1

    def __init__(
        self,
        config: Optional[Dict],
        mel_options: Dict,
        dataset_name: Optional[str] = None,
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.dataset_name = dataset_name or "default"
        self.directory = None
        self.dtype = np.float32
        self._options_digest = None

        if not self.enabled:
            return

        dataset_toggles = cfg.get("datasets") or {}
        if dataset_toggles and not dataset_toggles.get(self.dataset_name, True):
            self.enabled = False
            return

        directory = cfg.get("directory", "Data/mel_cache") or "Data/mel_cache"
        try:
            dtype = cfg.get("dtype", "float32")
            self.dtype = np.dtype(dtype)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid mel cache dtype %r. Falling back to float32.", cfg.get("dtype")
            )
            self.dtype = np.float32

        self.directory = Path(directory).expanduser() / self.dataset_name
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Failed to create mel cache directory %s: %s", self.directory, exc)
            self.enabled = False
            return

        self._meta_path = self.directory / self._META_FILENAME
        self._options_digest = self._ensure_metadata(mel_options)

    # ------------------------------------------------------------------
    # Metadata helpers
    def _normalise_options(self, mel_options: Dict) -> Dict:
        normalised = {}
        for key, value in sorted(mel_options.items()):
            if isinstance(value, (str, int, float, bool)) or value is None:
                normalised[key] = value
            elif isinstance(value, (list, tuple)):
                normalised[key] = list(value)
            else:
                normalised[key] = repr(value)
        return normalised

    def _ensure_metadata(self, mel_options: Dict) -> Optional[str]:
        if not self.enabled or self.directory is None:
            return None

        current = self._normalise_options(mel_options)
        current_payload = {
            "version": self._CACHE_VERSION,
            "options": current,
        }
        payload_bytes = json.dumps(current_payload, sort_keys=True).encode("utf-8")
        digest = hashlib.sha1(payload_bytes).hexdigest()

        existing: Optional[Dict] = None
        if self._meta_path.is_file():
            try:
                with self._meta_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
            except Exception:
                existing = None

        if not existing or existing.get("options_digest") != digest:
            self._clear_cache()
            meta_payload = {
                "version": self._CACHE_VERSION,
                "options": current,
                "options_digest": digest,
                "created": time.time(),
            }
            try:
                with self._meta_path.open("w", encoding="utf-8") as handle:
                    json.dump(meta_payload, handle, indent=2, sort_keys=True)
            except Exception as exc:
                logger.warning("Failed to write mel cache metadata: %s", exc)
                self.enabled = False
                return None

        return digest

    def _clear_cache(self) -> None:
        if not self.directory or not self.directory.exists():
            return
        for entry in self.directory.iterdir():
            if entry.is_file() and entry.name != self._META_FILENAME:
                try:
                    entry.unlink()
                except FileNotFoundError:
                    continue
                except Exception as exc:
                    logger.warning("Failed to remove mel cache file %s: %s", entry, exc)

    # ------------------------------------------------------------------
    # Per-entry helpers
    def _entry_key(self, wave_path: str) -> str:
        normalised = os.path.abspath(wave_path)
        return hashlib.sha1(normalised.encode("utf-8")).hexdigest()

    def _entry_paths(self, key: str) -> Tuple[Path, Path]:
        data_path = self.directory / f"{key}.npy"
        meta_path = self.directory / f"{key}.json"
        return data_path, meta_path

    def _stat_signature(self, wave_path: str) -> Optional[Dict]:
        try:
            stat = os.stat(wave_path)
        except OSError:
            return None
        return {
            "mtime": float(stat.st_mtime),
            "size": int(stat.st_size),
        }

    def _purge_entry(self, data_path: Path, meta_path: Path) -> None:
        for path in (data_path, meta_path):
            try:
                if path.exists():
                    path.unlink()
            except Exception as exc:
                logger.warning("Failed to remove stale mel cache file %s: %s", path, exc)

    # ------------------------------------------------------------------
    def try_load(self, wave_path: str) -> Optional[torch.Tensor]:
        if not self.enabled or self.directory is None:
            return None

        key = self._entry_key(wave_path)
        data_path, meta_path = self._entry_paths(key)

        if not data_path.is_file() or not meta_path.is_file():
            return None

        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception:
            self._purge_entry(data_path, meta_path)
            return None

        digest = meta.get("options_digest")
        if self._options_digest and digest and digest != self._options_digest:
            self._purge_entry(data_path, meta_path)
            return None

        stat_signature = self._stat_signature(wave_path)
        if stat_signature is None:
            self._purge_entry(data_path, meta_path)
            return None

        cached_sig_raw = meta.get("audio_signature")
        cached_sig = cached_sig_raw if isinstance(cached_sig_raw, dict) else {}
        if (
            abs(float(cached_sig.get("mtime", 0.0)) - stat_signature["mtime"]) >= 1e-3
            or int(cached_sig.get("size", -1)) != stat_signature["size"]
        ):
            self._purge_entry(data_path, meta_path)
            return None

        try:
            memmap = np.load(data_path, mmap_mode="r")
        except Exception:
            self._purge_entry(data_path, meta_path)
            return None

        cached_shape = tuple(meta.get("shape", ()))
        if cached_shape and tuple(memmap.shape) != cached_shape:
            self._purge_entry(data_path, meta_path)
            return None

        tensor = torch.tensor(np.array(memmap), dtype=torch.float32)
        del memmap
        return tensor

    def store(self, wave_path: str, log_mel: torch.Tensor) -> None:
        if not self.enabled or self.directory is None:
            return

        key = self._entry_key(wave_path)
        data_path, meta_path = self._entry_paths(key)

        signature = self._stat_signature(wave_path)
        if signature is None:
            logger.warning("Skipping mel cache storage for missing audio file %s", wave_path)
            return

        try:
            array = log_mel.detach().cpu().numpy().astype(self.dtype, copy=False)
            memmap = np.lib.format.open_memmap(
                data_path,
                mode="w+",
                dtype=self.dtype,
                shape=array.shape,
            )
            memmap[...] = array
            del memmap
        except Exception as exc:
            logger.warning("Failed to persist mel cache for %s: %s", wave_path, exc)
            try:
                if data_path.exists():
                    data_path.unlink()
            except Exception:
                pass
            return

        meta_payload = {
            "shape": list(array.shape),
            "dtype": str(self.dtype),
            "audio_signature": signature,
            "options_digest": self._options_digest,
            "version": self._CACHE_VERSION,
            "stored": time.time(),
        }

        try:
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(meta_payload, handle, indent=2, sort_keys=True)
        except Exception as exc:
            logger.warning("Failed to write mel cache metadata for %s: %s", wave_path, exc)
            self._purge_entry(data_path, meta_path)


class WaveformAugmenter:
    """Applies waveform-level augmentation such as noise, reverberation, and impulse responses."""

    def __init__(self, sample_rate: int, config: Optional[Dict] = None):
        self.sample_rate = int(sample_rate)
        self.config = config or {}
        self.enabled = bool(self.config.get('enabled', False))

        noise_cfg = self.config.get('noise', {}) or {}
        self.noise_enabled = bool(noise_cfg.get('enabled', False))
        self.noise_snr = noise_cfg.get('snr_db', [10, 40])
        self.noise_paths = _collect_audio_files(noise_cfg.get('paths'))
        self.noise_gaussian = bool(noise_cfg.get('gaussian', True))
        self.noise_prob = float(noise_cfg.get('probability', 0.5))

        musan_cfg = self.config.get('musan', {}) or {}
        self.musan_enabled = bool(musan_cfg.get('enabled', False))
        self.musan_paths = _collect_audio_files(musan_cfg.get('paths'))
        self.musan_snr = musan_cfg.get('snr_db', [0, 15])
        self.musan_prob = float(musan_cfg.get('probability', 0.3))

        reverb_cfg = self.config.get('reverberation', {}) or {}
        self.reverb_enabled = bool(reverb_cfg.get('enabled', False))
        self.reverb_paths = _collect_audio_files(reverb_cfg.get('rir_paths'))
        self.reverb_prob = float(reverb_cfg.get('probability', 0.3))

        impulse_cfg = self.config.get('impulse_response', {}) or {}
        self.impulse_enabled = bool(impulse_cfg.get('enabled', False))
        self.impulse_paths = _collect_audio_files(impulse_cfg.get('paths'))
        self.impulse_prob = float(impulse_cfg.get('probability', 0.3))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return waveform

        augmented = waveform.clone()

        augmented = self._apply_noise(augmented)
        augmented = self._apply_musan(augmented)
        augmented = self._apply_reverb_like(augmented, self.reverb_enabled, self.reverb_paths, self.reverb_prob)
        augmented = self._apply_reverb_like(augmented, self.impulse_enabled, self.impulse_paths, self.impulse_prob)

        return augmented

    def _apply_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.noise_enabled:
            return waveform
        if random.random() > self.noise_prob:
            return waveform

        if self.noise_gaussian or not self.noise_paths:
            snr = self._sample_snr(self.noise_snr)
            return self._additive_noise(waveform, snr)

        noise_path = random.choice(self.noise_paths)
        noise = _load_audio_sample(noise_path, self.sample_rate)
        if noise is None or noise.abs().sum() == 0:
            return waveform
        snr = self._sample_snr(self.noise_snr)
        noise = _match_length(noise, waveform.size(0))
        return self._mix_with_noise(waveform, noise, snr)

    def _apply_musan(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.musan_enabled or not self.musan_paths:
            return waveform
        if random.random() > self.musan_prob:
            return waveform

        noise_path = random.choice(self.musan_paths)
        noise = _load_audio_sample(noise_path, self.sample_rate)
        if noise is None or noise.abs().sum() == 0:
            return waveform
        noise = _match_length(noise, waveform.size(0))
        snr = self._sample_snr(self.musan_snr)
        return self._mix_with_noise(waveform, noise, snr)

    def _apply_reverb_like(self, waveform: torch.Tensor, enabled: bool, paths: Sequence[str], probability: float) -> torch.Tensor:
        if not enabled or not paths:
            return waveform
        if random.random() > probability:
            return waveform

        rir_path = random.choice(list(paths))
        rir = _load_audio_sample(rir_path, self.sample_rate)
        if rir is None or rir.abs().sum() == 0:
            return waveform

        rir = rir / (rir.abs().max() + 1e-8)
        return self._convolve(waveform, rir)

    def _sample_snr(self, snr_cfg: Sequence[float]) -> float:
        snr_values = list(_ensure_list(snr_cfg))
        if len(snr_values) == 0:
            return 20.0
        if len(snr_values) == 1:
            return float(snr_values[0])
        low, high = float(snr_values[0]), float(snr_values[1])
        if low > high:
            low, high = high, low
        return random.uniform(low, high)

    def _additive_noise(self, waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
        noise = torch.randn_like(waveform)
        return self._mix_with_noise(waveform, noise, snr_db)

    def _mix_with_noise(self, waveform: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        if noise_power <= 0 or signal_power <= 0:
            return waveform

        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        mixed = waveform + scale * noise
        return mixed.clamp_(-1.0, 1.0)

    def _convolve(self, waveform: torch.Tensor, impulse: torch.Tensor) -> torch.Tensor:
        kernel = impulse.view(1, 1, -1)
        signal = waveform.view(1, 1, -1)
        padding = kernel.size(-1) - 1
        convolved = F.conv1d(signal, kernel, padding=padding)
        return convolved.view(-1)[:waveform.size(0)]


class SpecAugment:
    """Extended SpecAugment implementation supporting multiple policies and augmentations."""

    _SPEC_POLICIES = {
        "LD": {"freq_mask_param": 27, "time_mask_param": 100, "num_freq_masks": 2, "num_time_masks": 4, "apply_prob": 1.0},
        "LF": {"freq_mask_param": 10, "time_mask_param": 100, "num_freq_masks": 4, "num_time_masks": 2, "apply_prob": 1.0},
    }

    def __init__(self,
                 freq_mask_param: int = 27,
                 time_mask_param: int = 40,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 2,
                 apply_prob: float = 1.0,
                 policy: Optional[str] = None,
                 mask_value: Optional[float] = None,
                 time_warp: Optional[Dict] = None,
                 adaptive_masking: Optional[Dict] = None,
                 random_frame_dropout: Optional[Dict] = None,
                 vtlp: Optional[Dict] = None):
        policy_cfg = {}
        if policy:
            policy_key = str(policy).upper()
            policy_cfg = self._SPEC_POLICIES.get(policy_key, {})

        self.freq_mask_param = int(policy_cfg.get("freq_mask_param", freq_mask_param))
        self.time_mask_param = int(policy_cfg.get("time_mask_param", time_mask_param))
        self.num_freq_masks = int(policy_cfg.get("num_freq_masks", num_freq_masks))
        self.num_time_masks = int(policy_cfg.get("num_time_masks", num_time_masks))
        self.apply_prob = float(policy_cfg.get("apply_prob", apply_prob))

        self.mask_value = mask_value

        self.time_warp_config = time_warp or {}
        self.time_warp_enabled = bool(self.time_warp_config.get("enabled", False))
        self.time_warp_window = int(self.time_warp_config.get("window", 5))

        self.adaptive_masking_config = adaptive_masking or {}
        self.adaptive_masking_enabled = bool(self.adaptive_masking_config.get("enabled", False))
        self.adaptive_time_ratio = float(self.adaptive_masking_config.get("max_time_ratio", 0.0))
        self.adaptive_freq_ratio = float(self.adaptive_masking_config.get("max_freq_ratio", 0.0))

        self.random_frame_dropout_config = random_frame_dropout or {}
        self.random_frame_dropout_enabled = bool(self.random_frame_dropout_config.get("enabled", False))
        self.random_frame_dropout_prob = float(self.random_frame_dropout_config.get("drop_prob", 0.0))
        self.random_frame_dropout_mode = str(self.random_frame_dropout_config.get("mode", "zero")).lower()

        self.vtlp_config = vtlp or {}
        self.vtlp_enabled = bool(self.vtlp_config.get("enabled", False))
        self.vtlp_alpha_low = float(self.vtlp_config.get("alpha_low", 0.9))
        self.vtlp_alpha_high = float(self.vtlp_config.get("alpha_high", 1.1))

    def __call__(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if self.apply_prob < 1.0 and random.random() > self.apply_prob:
            return mel_tensor

        squeeze = False
        if mel_tensor.dim() == 2:
            mel_tensor = mel_tensor.unsqueeze(0)
            squeeze = True

        augmented_batches = []
        for sample in mel_tensor:
            augmented_batches.append(self._augment_single(sample))
        augmented = torch.stack(augmented_batches, dim=0)

        if squeeze:
            augmented = augmented.squeeze(0)

        return augmented

    def _augment_single(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.dim() != 2:
            raise ValueError("SpecAugment expects 2D tensors of shape (freq, time) per sample")

        augmented = sample.clone()

        if self.time_warp_enabled:
            augmented = self._apply_time_warp(augmented)

        if self.vtlp_enabled:
            augmented = self._apply_vtlp(augmented)

        augmented = self._apply_masks(augmented)

        if self.random_frame_dropout_enabled:
            augmented = self._apply_random_frame_dropout(augmented)

        return augmented

    def _apply_masks(self, spec: torch.Tensor) -> torch.Tensor:
        freq, time = spec.shape
        mask_value = self._get_mask_value(spec)

        for _ in range(max(0, self.num_freq_masks)):
            mask_width = self._sample_mask_width(
                base_param=self.freq_mask_param,
                max_ratio=self.adaptive_freq_ratio,
                size=freq,
            )
            if mask_width <= 0:
                continue
            start = random.randint(0, max(0, freq - mask_width))
            spec[start:start + mask_width, :] = mask_value

        for _ in range(max(0, self.num_time_masks)):
            mask_width = self._sample_mask_width(
                base_param=self.time_mask_param,
                max_ratio=self.adaptive_time_ratio,
                size=time,
            )
            if mask_width <= 0:
                continue
            start = random.randint(0, max(0, time - mask_width))
            spec[:, start:start + mask_width] = mask_value

        return spec

    def _sample_mask_width(self, base_param: int, max_ratio: float, size: int) -> int:
        width_limit = int(base_param)
        if self.adaptive_masking_enabled and max_ratio > 0 and size > 0:
            adaptive_width = int(round(max_ratio * size))
            if adaptive_width > 0:
                width_limit = min(width_limit, adaptive_width) if width_limit > 0 else adaptive_width
        width_limit = max(0, min(width_limit, size))
        if width_limit <= 0:
            return 0
        return random.randint(0, width_limit)

    def _get_mask_value(self, spec: torch.Tensor) -> torch.Tensor:
        if self.mask_value is None or str(self.mask_value).lower() == "mean":
            return spec.mean()
        return torch.tensor(float(self.mask_value), device=spec.device, dtype=spec.dtype)

    def _apply_random_frame_dropout(self, spec: torch.Tensor) -> torch.Tensor:
        if self.random_frame_dropout_prob <= 0.0:
            return spec

        time = spec.size(1)
        num_drop = int(round(time * self.random_frame_dropout_prob))
        if num_drop <= 0:
            return spec

        drop_indices = torch.randperm(time, device=spec.device)[:num_drop]
        if self.random_frame_dropout_mode == "noise":
            noise = torch.randn(spec.size(0), drop_indices.numel(), device=spec.device, dtype=spec.dtype)
            spec[:, drop_indices] = noise
        elif self.random_frame_dropout_mode == "mean":
            spec[:, drop_indices] = spec.mean()
        else:
            spec[:, drop_indices] = 0.0
        return spec

    def _apply_time_warp(self, spec: torch.Tensor) -> torch.Tensor:
        _, time = spec.shape
        if time < 2 or self.time_warp_window <= 0 or time <= 2 * self.time_warp_window:
            return spec

        center = random.randint(self.time_warp_window, time - self.time_warp_window - 1)
        warped_center = center + random.randint(-self.time_warp_window, self.time_warp_window)
        if warped_center == center:
            return spec

        device = spec.device
        dtype = spec.dtype

        time_range = torch.arange(time, device=device, dtype=dtype)
        left = time_range[time_range < center]
        right = time_range[time_range >= center]

        warped = torch.empty_like(time_range)
        if center > 0:
            scale_left = warped_center / float(center)
            warped[left.long()] = left * scale_left
        if time - 1 - center > 0:
            scale_right = (time - 1 - warped_center) / float(time - 1 - center)
            warped[right.long()] = warped_center + (right - center) * scale_right
        warped = warped.clamp_(0, time - 1)

        grid_x = (warped / (time - 1) * 2) - 1
        freq = spec.size(0)
        grid_y = torch.linspace(-1.0, 1.0, freq, device=device, dtype=dtype)
        grid_x = grid_x.unsqueeze(0).expand(freq, -1)
        grid_y = grid_y.unsqueeze(1).expand(-1, time)
        grid = torch.stack((grid_x, grid_y), dim=-1)

        warped_spec = F.grid_sample(
            spec.unsqueeze(0).unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped_spec.squeeze(0).squeeze(0)

    def _apply_vtlp(self, spec: torch.Tensor) -> torch.Tensor:
        alpha = random.uniform(self.vtlp_alpha_low, self.vtlp_alpha_high)
        if abs(alpha - 1.0) < 1e-3:
            return spec

        freq, time = spec.shape
        if freq < 2:
            return spec

        device = spec.device
        dtype = spec.dtype

        freq_positions = torch.linspace(0.0, 1.0, freq, device=device, dtype=dtype)
        warped_positions = freq_positions * alpha
        warped_positions = warped_positions.clamp_(0.0, 1.0)

        grid_x = torch.linspace(-1.0, 1.0, time, device=device, dtype=dtype)
        grid_y = warped_positions * 2 - 1
        grid_x = grid_x.unsqueeze(0).expand(freq, -1)
        grid_y = grid_y.unsqueeze(1).expand(-1, time)
        grid = torch.stack((grid_x, grid_y), dim=-1)

        warped_spec = F.grid_sample(
            spec.unsqueeze(0).unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return warped_spec.squeeze(0).squeeze(0)


class LengthAwareBatchSampler(Sampler):
    """Batch sampler that groups items with similar lengths together."""

    def __init__(self,
                 lengths,
                 batch_size,
                 bucket_size=None,
                 shuffle_batches=True,
                 shuffle_within_bucket=True,
                 drop_last=False,
                 seed=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.lengths = list(lengths)
        if len(self.lengths) == 0:
            raise ValueError("lengths must not be empty")

        self.batch_size = int(batch_size)
        if bucket_size is None:
            bucket_size = self.batch_size * 50
        self.bucket_size = max(self.batch_size, int(bucket_size))
        self.shuffle_batches = bool(shuffle_batches)
        self.shuffle_within_bucket = bool(shuffle_within_bucket)
        self.drop_last = bool(drop_last)
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        if self.seed is not None:
            rng = random.Random(self.seed + self._epoch)
        else:
            rng = random.Random()
        self._epoch += 1

        indices = list(range(len(self.lengths)))
        indices.sort(key=lambda idx: self.lengths[idx])

        buckets = [indices[i:i + self.bucket_size] for i in range(0, len(indices), self.bucket_size)]
        if self.shuffle_batches:
            rng.shuffle(buckets)

        batches = []
        for bucket in buckets:
            if self.shuffle_within_bucket:
                rng.shuffle(bucket)

            for start in range(0, len(bucket), self.batch_size):
                batch = bucket[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.shuffle_batches:
            rng.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)


class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 dict_path=DEFAULT_DICT_PATH,
                 dictionary_config=None,
                 sr=24000,
                 spect_params={
                     "n_fft": 2048,
                     "win_length": 1200,
                     "hop_length": 300
                 },
                 mel_params={
                     "n_mels": 80
                 },
                 spec_augment_params=None,
                 waveform_augmentations=None,
                 mixup=None,
                 phoneme_dropout=None,
                 mel_cache=None,
                 dataset_name=None,
                 validation=False
                ):

        self.data_list = data_list
        self.text_cleaner = TextCleaner(dict_path, dictionary_config=dictionary_config)
        self.sr = sr
        self.validation = bool(validation)
        if dataset_name is None:
            dataset_name = "val" if self.validation else "train"
        self.cache_dataset_name = dataset_name

        self.blank_index = self.text_cleaner.word_index_dictionary.get(" ", 0)

        mel_opts = {**{'sample_rate': sr}, **mel_params, **spect_params}
        print("Options for MEL spectrogram calculations:", mel_opts)
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**mel_opts)
        self._mel_cache = MelFeatureCache(mel_cache, mel_opts, dataset_name=self.cache_dataset_name)

        self.spec_augment = None
        if spec_augment_params and not self.validation:
            try:
                spec_params = dict(spec_augment_params)
                enabled = spec_params.pop('enabled', True)
                if enabled:
                    self.spec_augment = SpecAugment(**spec_params)
            except TypeError:
                logger.warning(f"Invalid SpecAugment configuration: {spec_augment_params}. Skipping augmentation.")
                self.spec_augment = None

        self.waveform_augmenter = None
        if waveform_augmentations and not self.validation:
            try:
                self.waveform_augmenter = WaveformAugmenter(self.sr, waveform_augmentations)
            except Exception as exc:
                logger.warning(f"Failed to initialise waveform augmentations: {exc}")
                self.waveform_augmenter = None

        self.mixup_config = mixup or {}
        self.mixup_enabled = bool(self.mixup_config.get('enabled', False)) and not self.validation
        self.mixup_alpha = float(self.mixup_config.get('alpha', 0.4))
        self.mixup_prob = float(self.mixup_config.get('apply_prob', 0.0))
        self.mixup_dominant = bool(self.mixup_config.get('dominant_label', True))
        self.mixup_secondary_augment = bool(self.mixup_config.get('augment_secondary', True))
        self._mixup_beta = None
        self._mixup_alpha_cached = None
        self._last_mix = None

        self.phoneme_dropout_config = phoneme_dropout or {}
        self.phoneme_dropout_enabled = bool(self.phoneme_dropout_config.get('enabled', False)) and not self.validation
        keep_tokens = set([self.blank_index])
        for token in _ensure_list(self.phoneme_dropout_config.get('preserve_tokens')):
            if isinstance(token, str):
                idx = self.text_cleaner.word_index_dictionary.get(token)
                if idx is not None:
                    keep_tokens.add(int(idx))
            else:
                try:
                    keep_tokens.add(int(token))
                except (TypeError, ValueError):
                    continue
        self.phoneme_dropout_keep = keep_tokens

        # https://github.com/yl4579/StyleTTS/issues/57
        self.mean, self.std = -4, 4

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]
            wave_path = data[0]

            cache_allowed = (
                hasattr(self, "_mel_cache")
                and self._mel_cache is not None
                and self._mel_cache.enabled
                and not self.mixup_enabled
                and (self.waveform_augmenter is None or not getattr(self.waveform_augmenter, "enabled", False))
            )

            cached_log_mel = None
            if cache_allowed:
                cached_log_mel = self._mel_cache.try_load(wave_path)

            waveform, text_tensor, speaker_id = self._load_tensor(data)

            waveform = waveform.float()

            if self.waveform_augmenter is not None:
                waveform = self.waveform_augmenter(waveform)

            if self.mixup_enabled:
                waveform = self._maybe_mix_waveform(waveform, idx)

            if cached_log_mel is not None:
                log_mel_tensor = cached_log_mel
            else:
                mel_tensor = self.to_melspec(waveform)

                if (text_tensor.size(0)+1) >= (mel_tensor.size(1) // 3):
                    mel_tensor = F.interpolate(
                        mel_tensor.unsqueeze(0), size=(text_tensor.size(0)+1)*3, align_corners=False,
                        mode='linear').squeeze(0)

                log_mel_tensor = torch.log(1e-5 + mel_tensor)

                if cache_allowed:
                    self._mel_cache.store(wave_path, log_mel_tensor)

            if self.spec_augment is not None:
                log_mel_tensor = self.spec_augment(log_mel_tensor)

            acoustic_feature = (log_mel_tensor - self.mean)/self.std

            length_feature = acoustic_feature.size(1)
            acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

            return waveform, acoustic_feature, text_tensor, speaker_id
        except Exception as e:
            try:
                wave_path, text, speaker_id = data
                print(f"Error for wave path: {wave_path}, skipping - {e}")
            except Exception as e2:
                print(f"Error for wave data: {data}, skipping - {e2}")

            # Fallback to another index to keep training going
            new_idx = (idx + 1) % len(self.data_list)
            return self.__getitem__(new_idx)

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id) if speaker_id else 0

        waveform = self._load_waveform(wave_path)
        text_tensor = self._process_text(text)

        return waveform, text_tensor, speaker_id

    def _load_waveform(self, wave_path: str) -> torch.Tensor:
        wave_tensor, sr = torchaudio.load(wave_path)

        if wave_tensor.size(0) > 1:
            wave_tensor = wave_tensor.mean(dim=0)
            print("using mono track from stereo WAV file: ", wave_path)
        else:
            wave_tensor = wave_tensor.squeeze(0)

        if sr != self.sr:
            wave_tensor = AF.resample(wave_tensor, sr, self.sr)
            print("resampling: ", wave_path, ", from: ", sr, ", to: ", self.sr)

        return wave_tensor

    def _process_text(self, text: str) -> torch.LongTensor:
        tokens = word_tokenize(text)
        normalized = ' '.join(tokens)
        normalized = normalized.replace("(", "-").replace(")", "-")

        encoded = self.text_cleaner(normalized)

        if self.phoneme_dropout_enabled:
            encoded = self._apply_phoneme_dropout(encoded)

        encoded.insert(0, self.blank_index)
        encoded.append(self.blank_index)

        return torch.LongTensor(encoded)

    def _apply_phoneme_dropout(self, tokens: Sequence[int]) -> List[int]:
        drop_prob = float(self.phoneme_dropout_config.get('drop_prob', 0.0))
        min_tokens = int(self.phoneme_dropout_config.get('min_tokens', 1))
        if drop_prob <= 0:
            return list(tokens)

        kept: List[int] = []
        for token in tokens:
            if token in self.phoneme_dropout_keep or random.random() > drop_prob:
                kept.append(int(token))

        if len(kept) < max(1, min_tokens):
            return list(tokens)
        return kept

    def _maybe_mix_waveform(self, waveform: torch.Tensor, idx: int) -> torch.Tensor:
        if random.random() > self.mixup_prob:
            return waveform

        mix_idx = random.randrange(len(self.data_list))
        if mix_idx == idx:
            mix_idx = (mix_idx + 1) % len(self.data_list)

        mix_path = self.data_list[mix_idx][0]
        mix_waveform = self._load_waveform(mix_path)
        if self.waveform_augmenter is not None and self.mixup_secondary_augment:
            mix_waveform = self.waveform_augmenter(mix_waveform)

        primary, secondary = self._align_waveforms(waveform, mix_waveform)
        lam = self._sample_mix_lambda()

        mixed = lam * primary + (1.0 - lam) * secondary
        if not self.mixup_dominant:
            # Store metadata if downstream components need it in the future.
            self._last_mix = {'lambda': lam, 'secondary_index': mix_idx}
        return mixed

    def _align_waveforms(self, primary: torch.Tensor, secondary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_length = max(primary.size(0), secondary.size(0))
        primary_aligned = self._pad_or_trim(primary, target_length)
        secondary_aligned = self._pad_or_trim(secondary, target_length)
        return primary_aligned, secondary_aligned

    def _pad_or_trim(self, waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        if waveform.size(0) == target_length:
            return waveform
        if waveform.size(0) > target_length:
            start = random.randint(0, waveform.size(0) - target_length)
            return waveform[start:start + target_length]
        pad_amount = target_length - waveform.size(0)
        return F.pad(waveform, (0, pad_amount))

    def _sample_mix_lambda(self) -> float:
        alpha = max(1e-3, float(self.mixup_alpha))
        if self._mixup_beta is None or self._mixup_alpha_cached != alpha:
            concentration = torch.tensor([alpha], dtype=torch.float32)
            self._mixup_beta = Beta(concentration, concentration)
            self._mixup_alpha_cached = alpha
        lam = float(self._mixup_beta.sample().item())
        return min(max(lam, 0.0), 1.0)




class Collater(object):
    """
    Args:
      return_wave (bool): if true, will return the wave data along with spectrogram. 
    """

    def __init__(self, return_wave=False, return_speaker_ids=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.return_speaker_ids = return_speaker_ids

    def __call__(self, batch):
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])

        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        input_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        speaker_ids = torch.zeros(batch_size).long()
        for bid, (_, mel, text, speaker_id) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            input_lengths[bid] = text_size
            output_lengths[bid] = mel_size
            speaker_ids[bid] = int(speaker_id)
            assert(text_size < (mel_size//2))

        outputs = [texts, input_lengths, mels, output_lengths]

        if self.return_speaker_ids:
            outputs.append(speaker_ids)

        if self.return_wave:
            waves = [b[0] for b in batch]
            outputs.append(waves)

        return tuple(outputs)



def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     lengths=None,
                     bucket_sampler_config=None,
                     dataset_name=None):

    dataset_cfg = dict(dataset_config or {})
    mel_cache_cfg = dataset_cfg.pop('mel_cache', None)
    phoneme_dict_cfg = dataset_cfg.pop('phoneme_dictionary_config', None)

    dataset = MelDataset(
        path_list,
        validation=validation,
        mel_cache=mel_cache_cfg,
        dataset_name=dataset_name,
        dictionary_config=phoneme_dict_cfg,
        **dataset_cfg,
    )
    collate_fn = Collater(**collate_config)

    use_bucket_sampler = False
    batch_sampler = None

    if (not validation) and bucket_sampler_config and lengths is not None:
        lengths = list(lengths)
        if len(lengths) != len(dataset):
            raise ValueError("lengths must have the same length as path_list when using a bucket sampler")

        enabled = bucket_sampler_config.get('enabled', True)
        if enabled:
            bucket_size = bucket_sampler_config.get('bucket_size')
            if bucket_size is not None:
                bucket_size = int(bucket_size)
            else:
                multiplier = int(bucket_sampler_config.get('bucket_size_multiplier', 50))
                bucket_size = batch_size * max(1, multiplier)

            shuffle_batches = bucket_sampler_config.get('shuffle_batches', True)
            shuffle_within_bucket = bucket_sampler_config.get('shuffle_within_bucket', True)
            drop_last = bucket_sampler_config.get('drop_last', not validation)
            seed = bucket_sampler_config.get('seed')

            batch_sampler = LengthAwareBatchSampler(
                lengths=lengths,
                batch_size=batch_size,
                bucket_size=bucket_size,
                shuffle_batches=shuffle_batches,
                shuffle_within_bucket=shuffle_within_bucket,
                drop_last=drop_last,
                seed=seed,
            )
            use_bucket_sampler = True

    if use_bucket_sampler and batch_sampler is not None:
        data_loader = DataLoader(dataset,
                                 batch_sampler=batch_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'))
    else:
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=(not validation),
                                 num_workers=num_workers,
                                 drop_last=(not validation),
                                 collate_fn=collate_fn,
                                 pin_memory=(device != 'cpu'))

    return data_loader
