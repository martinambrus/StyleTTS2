from __future__ import annotations

from typing import Tuple

import numpy as np


def _apply_fade(audio: np.ndarray, sr: int, fade_time: float = 0.02) -> np.ndarray:
    """Apply a raised-cosine fade in/out to avoid clicks."""
    fade_samples = int(max(fade_time * sr, 0))
    if fade_samples <= 0:
        return audio.astype(np.float32, copy=False)

    window = np.ones_like(audio, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, fade_samples, dtype=np.float64))
    window[:fade_samples] = ramp
    window[-fade_samples:] = ramp[::-1]
    return (audio * window).astype(np.float32)


def synthesize_from_f0_curve(
    f0_curve: np.ndarray,
    sr: int,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Synthesize a sinusoid that follows the provided F0 contour."""
    omega = 2.0 * np.pi * f0_curve.astype(np.float64) / float(sr)
    phase = np.cumsum(omega)
    audio = amplitude * np.sin(phase)
    audio = _apply_fade(audio.astype(np.float32), sr)
    max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_val > 0.99:
        audio = audio / (max_val + 1e-6)
    return audio.astype(np.float32)


def generate_vibrato_waveform(
    rate_hz: float,
    depth_cents: float,
    base_freq: float,
    duration: float,
    sr: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a vibrato waveform and its analytical F0 trajectory."""
    t = np.linspace(0.0, duration, int(duration * sr), endpoint=False, dtype=np.float64)
    modulation = np.sin(2.0 * np.pi * rate_hz * t)
    f0_curve = base_freq * (2.0 ** ((depth_cents / 1200.0) * modulation))
    audio = synthesize_from_f0_curve(f0_curve, sr)
    return audio, t.astype(np.float32), f0_curve.astype(np.float32)


def generate_glide_waveform(
    duration: float,
    start_hz: float,
    end_hz: float,
    sr: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a linear glide between start and end frequencies."""
    t = np.linspace(0.0, duration, int(duration * sr), endpoint=False, dtype=np.float64)
    f0_curve = np.linspace(start_hz, end_hz, t.shape[0], dtype=np.float64)
    audio = synthesize_from_f0_curve(f0_curve, sr)
    return audio, t.astype(np.float32), f0_curve.astype(np.float32)


def sample_reference_f0(time_axis: np.ndarray, f0_curve: np.ndarray, num_frames: int) -> np.ndarray:
    """Resample the analytical F0 curve at the frame rate used by the model."""
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    if time_axis.size == 0:
        return np.zeros((num_frames,), dtype=np.float32)
    duration = time_axis[-1]
    if time_axis.size > 1:
        duration += time_axis[1] - time_axis[0]
    frame_times = np.linspace(0.0, duration, num=num_frames, endpoint=False, dtype=np.float64)
    reference = np.interp(frame_times, time_axis, f0_curve)
    return reference.astype(np.float32)


def hz_to_cents(f0: np.ndarray) -> np.ndarray:
    cents = np.zeros_like(f0, dtype=np.float32)
    positive = f0 > 0
    cents[positive] = 1200.0 * np.log2(f0[positive] / 55.0)
    return cents


def circular_cents_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    diff = np.mod(diff + 600.0, 1200.0) - 600.0
    return diff


def rms_cents_error(reference: np.ndarray, prediction: np.ndarray) -> float:
    length = min(reference.shape[0], prediction.shape[0])
    if length == 0:
        return float("nan")
    ref = reference[:length]
    pred = prediction[:length]
    mask = ref > 0
    if not np.any(mask):
        return float("nan")
    ref_cents = hz_to_cents(ref[mask])
    pred_cents = hz_to_cents(np.clip(pred[mask], a_min=1e-5, a_max=None))
    diff = pred_cents - ref_cents
    return float(np.sqrt(np.mean(diff**2)))


def estimate_tracking_delay_ms(
    reference: np.ndarray,
    prediction: np.ndarray,
    frame_period_ms: float,
) -> float:
    length = min(reference.shape[0], prediction.shape[0])
    if length == 0:
        return float("nan")
    ref = reference[:length]
    pred = prediction[:length]
    ref_centered = ref - np.mean(ref)
    pred_centered = pred - np.mean(pred)
    if np.allclose(ref_centered, 0) or np.allclose(pred_centered, 0):
        return float("nan")
    corr = np.correlate(pred_centered, ref_centered, mode="full")
    lag = np.argmax(corr) - (length - 1)
    return float(lag * frame_period_ms)


def compute_overshoot_cents(reference: np.ndarray, prediction: np.ndarray) -> float:
    length = min(reference.shape[0], prediction.shape[0])
    if length == 0:
        return float("nan")
    ref = reference[:length]
    pred = prediction[:length]
    target = ref[-1]
    peak = np.max(pred) if pred.size else 0.0
    if target <= 0 or peak <= 0:
        return float("nan")
    return float(1200.0 * np.log2(peak / target))
