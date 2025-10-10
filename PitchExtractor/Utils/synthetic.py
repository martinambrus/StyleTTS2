# coding: utf-8
"""Utilities for generating synthetic speech with known F0.

This module currently focuses on synthesising simple vowel-like utterances
using the WORLD vocoder so that we can augment the training set with perfectly
labelled pitch contours.  The generator intentionally keeps its configuration
lightweight and dependency free beyond the optional ``pyworld`` package.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pyworld as pw
except ImportError:  # pragma: no cover - optional dependency
    pw = None


DEFAULT_VOWELS: Tuple[Dict[str, Iterable[Tuple[float, float, float]]], ...] = (
    {
        "label": "ah",
        "formants": (
            (730.0, 90.0, 1.0),
            (1090.0, 110.0, 0.6),
            (2440.0, 150.0, 0.4),
        ),
    },
    {
        "label": "ih",
        "formants": (
            (390.0, 80.0, 1.0),
            (1990.0, 120.0, 0.6),
            (2550.0, 160.0, 0.4),
        ),
    },
    {
        "label": "uh",
        "formants": (
            (440.0, 70.0, 1.0),
            (1020.0, 90.0, 0.6),
            (2240.0, 150.0, 0.4),
        ),
    },
)


def _db_to_amplitude(db_value: float) -> float:
    return float(10.0 ** (db_value / 20.0))


def _ensure_pyworld(verbose: bool = False):
    if pw is None:
        message = "pyworld is required for WORLD-based synthetic speech generation"
        if verbose:
            print(f"[Synthetic] {message}")
        raise ImportError(message)


@dataclass
class ModulationConfig:
    vibrato_probability: float = 0.6
    vibrato_semitones: float = 0.35
    vibrato_rate_range: Tuple[float, float] = (4.0, 7.0)
    max_segments: int = 4


class WorldSynthesizer:
    """Generate synthetic speech-like waveforms with known F0 using WORLD."""

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        fft_size: Optional[int] = None,
        config: Optional[Dict] = None,
        verbose: bool = False,
    ) -> None:
        _ensure_pyworld(verbose=verbose)

        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.fft_size = int(fft_size or 1024)
        self.verbose = verbose

        cfg = dict(config or {})

        duration_cfg = cfg.get("duration", {}) or {}
        self.min_duration = float(duration_cfg.get("min", 0.5))
        self.max_duration = float(duration_cfg.get("max", 1.8))
        if self.max_duration <= 0:
            raise ValueError("Synthetic duration must be positive")

        pitch_range = cfg.get("pitch_range", [110.0, 320.0])
        if len(pitch_range) != 2:
            raise ValueError("pitch_range must contain two values")
        self.pitch_min = float(min(pitch_range))
        self.pitch_max = float(max(pitch_range))

        noise_db_cfg = cfg.get("noise_db", -60.0)
        self.noise_db = None if noise_db_cfg is None else float(noise_db_cfg)
        gain_cfg = cfg.get("gain_db_range", [-18.0, -6.0])
        if isinstance(gain_cfg, (int, float)):
            gain_cfg = [gain_cfg, gain_cfg]
        if len(gain_cfg) != 2:
            raise ValueError("gain_db_range must provide two values")
        gain_min, gain_max = float(gain_cfg[0]), float(gain_cfg[1])
        if gain_min > gain_max:
            gain_min, gain_max = gain_max, gain_min
        self.gain_db_range = (gain_min, gain_max)
        self.modulation = ModulationConfig(**(cfg.get("modulation", {}) or {}))

        vowel_profiles = cfg.get("vowel_profiles") or DEFAULT_VOWELS
        self._templates = self._build_formant_templates(vowel_profiles)

        self.frame_period = 1000.0 * self.hop_length / self.sample_rate

    # ------------------------------------------------------------------
    def _build_formant_templates(self, profiles: Sequence[Dict]) -> List[np.ndarray]:
        freq_axis = np.linspace(0, self.sample_rate / 2, self.fft_size // 2 + 1)
        templates: List[np.ndarray] = []
        for profile in profiles:
            formants = profile.get("formants", [])
            if not formants:
                continue
            envelope = np.zeros_like(freq_axis)
            for formant in formants:
                if len(formant) < 2:
                    continue
                freq = float(formant[0])
                bandwidth = float(formant[1])
                amplitude = float(formant[2]) if len(formant) > 2 else 1.0
                if bandwidth <= 0:
                    bandwidth = 60.0
                envelope += amplitude * np.exp(
                    -0.5 * ((freq_axis - freq) / (bandwidth / 2.0)) ** 2
                )
            envelope = np.maximum(envelope, 1e-3)
            templates.append(envelope.astype(np.float64))

        if not templates:
            raise ValueError("No valid vowel templates provided for WORLD synthesis")

        return templates

    # ------------------------------------------------------------------
    def _sample_duration(self) -> float:
        if self.max_duration <= self.min_duration:
            return max(self.max_duration, 0.1)
        return random.uniform(self.min_duration, self.max_duration)

    def _sample_f0_curve(self, num_frames: int) -> np.ndarray:
        base = random.uniform(self.pitch_min, self.pitch_max)
        curve = np.full(num_frames, base, dtype=np.float64)

        max_segments = max(1, int(self.modulation.max_segments))
        num_segments = random.randint(1, max_segments)
        if num_segments > 1 and num_frames > 2:
            available = max(1, num_frames - 1)
            positions = sorted(
                random.sample(range(1, available), min(num_segments - 1, available - 1))
            )
            positions = [0] + positions + [num_frames - 1]
            segment_values = [
                random.uniform(self.pitch_min, self.pitch_max)
                for _ in range(len(positions))
            ]
            for idx in range(len(positions) - 1):
                start, end = positions[idx], positions[idx + 1]
                value_start, value_end = segment_values[idx], segment_values[idx + 1]
                if end <= start:
                    continue
                interp = np.linspace(value_start, value_end, end - start + 1)
                curve[start : end + 1] = interp

        if random.random() < self.modulation.vibrato_probability:
            depth = float(self.modulation.vibrato_semitones)
            depth = max(depth, 0.0)
            if depth > 0:
                rate = random.uniform(*self.modulation.vibrato_rate_range)
                t = np.arange(num_frames, dtype=np.float64) * (
                    self.frame_period / 1000.0
                )
                vibrato = np.sin(2.0 * math.pi * rate * t)
                ratio = 2.0 ** (vibrato * (depth / 12.0))
                curve *= ratio

        return curve

    # ------------------------------------------------------------------
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        duration = self._sample_duration()
        num_frames = max(2, int(np.ceil((duration * 1000.0) / self.frame_period)))

        template = random.choice(self._templates)
        spectral_envelope = np.tile(template, (num_frames, 1))
        aperiodicity = np.zeros_like(spectral_envelope)
        f0_curve = self._sample_f0_curve(num_frames)

        gain = _db_to_amplitude(random.uniform(*self.gain_db_range))
        waveform = pw.synthesize(
            f0_curve.astype(np.float64),
            spectral_envelope,
            aperiodicity,
            self.sample_rate,
            self.frame_period,
        ).astype(np.float64)

        waveform *= gain

        if self.noise_db is not None:
            noise_gain = _db_to_amplitude(self.noise_db)
            if noise_gain > 0:
                noise = np.random.normal(scale=noise_gain, size=waveform.shape)
                waveform += noise

        return waveform.astype(np.float32), f0_curve.astype(np.float32)


__all__ = ["WorldSynthesizer"]

