"""Runtime-selectable F0 extraction backends.

This module centralises all fundamental F0 (pitch) extraction utilities and
provides a single ``F0Extractor`` facade that can iterate over multiple
backends until one succeeds.  The intent is to make it trivial to experiment
with different pitch trackers (speed vs. accuracy trade-offs) without having to
modify the training or evaluation pipelines.

Each backend is wrapped in a light-weight class that hides the dependency
surface and exposes a consistent ``compute`` method returning a NumPy array
containing Hertz values.  Optional dependencies are imported lazily and
reported via ``BackendUnavailableError`` so callers can gracefully fall back to
another backend defined in the configuration.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot be constructed due to missing deps."""


class BackendComputationError(RuntimeError):
    """Raised when a backend fails to compute an F0 trajectory."""


@dataclasses.dataclass
class BackendResult:
    f0: np.ndarray
    backend_name: str
    details: Optional[str] = None


class BaseF0Backend:
    """Base class for all backends."""

    backend_type: str = "base"

    def __init__(
        self,
        name: str,
        sr: int,
        hop_length: int,
        config: Optional[Dict] = None,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.sample_rate = sr
        self.hop_length = hop_length
        self.config = config or {}
        self.verbose = verbose
        # Indicates whether the backend expects CUDA availability in the
        # current process.  Individual implementations update this flag when
        # device preferences change (e.g. after a CPU fallback).
        self.requires_cuda = False

    @property
    def frame_period_ms(self) -> float:
        value = self.config.get("frame_period_ms")
        if value is None:
            value = self.hop_length * 1000.0 / self.sample_rate
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid float value for 'frame_period_ms' in backend '{self.name}': {value!r}"
            ) from exc

    @property
    def cache_key(self) -> str:
        suffix = self.config.get("cache_key_suffix")
        if suffix:
            return f"{self.name}-{suffix}"
        return self.name

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[{self.name}] {message}")
        LOGGER.debug("[%s] %s", self.name, message)

    def _coerce_float(self, key: str, default: float) -> float:
        value = self.config.get(key, default)
        if value is None:
            value = default
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid float value for '{key}' in backend '{self.name}': {value!r}"
            ) from exc

    # ------------------------------------------------------------------
    # API surface expected from subclasses
    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class PyWorldBackend(BaseF0Backend):
    backend_type = "pyworld"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            import pyworld as pw  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailableError("pyworld is not installed") from exc

        self._pw = pw
        self.algorithm = self.config.get("algorithm", "harvest")
        self.fallback_algorithm = self.config.get("fallback", "dio")
        self.use_stonemask = bool(self.config.get("stonemask", True))

    def _run_algorithm(self, algorithm: str, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        frame_period = self.frame_period_ms
        if algorithm == "harvest":
            return self._pw.harvest(audio, sr, frame_period=frame_period)
        if algorithm == "dio":
            return self._pw.dio(audio, sr, frame_period=frame_period)
        if algorithm == "stonemask":
            # stonemask is a refinement step and expects the initial F0 curve
            f0, t = self._pw.harvest(audio, sr, frame_period=frame_period)
            return self._pw.stonemask(audio, f0, t, sr), t
        raise ValueError(f"Unsupported PyWorld algorithm: {algorithm}")

    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = int(sr or self.sample_rate)
        signal = audio.astype("double", copy=False)
        algorithm = self.algorithm
        f0, t = self._run_algorithm(algorithm, signal, sr)
        if np.count_nonzero(f0) < self.config.get("min_voiced_frames", 5) and self.fallback_algorithm:
            self.log(
                f"Primary algorithm '{algorithm}' returned too few voiced frames; switching to '{self.fallback_algorithm}'."
            )
            f0, t = self._run_algorithm(self.fallback_algorithm, signal, sr)
        if self.use_stonemask and algorithm != "stonemask":
            f0 = self._pw.stonemask(signal, f0, t, sr)
        return f0.astype(np.float64)


class CrepeBackend(BaseF0Backend):
    backend_type = "crepe"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            import torch
            import torchcrepe
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailableError("torchcrepe is not installed") from exc

        self._torch = torch
        self._torchcrepe = torchcrepe

        device_preference = str(self.config.get("device", "auto") or "auto").strip().lower()
        if device_preference == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._force_gpu = False
        else:
            try:
                self._device = torch.device(device_preference)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid torch device specification '{device_preference}' for CREPE backend"
                ) from exc
            if self._device.type == "cuda" and not torch.cuda.is_available():
                raise BackendUnavailableError(
                    "CUDA device requested for CREPE backend but no CUDA devices are available"
                )
            self._force_gpu = self._device.type == "cuda"

        self.requires_cuda = self._device.type == "cuda"

        self.model = self.config.get("model", "full")
        self.step_size_ms = self._coerce_float("step_size_ms", self.frame_period_ms)
        self.fmin = self._coerce_float("fmin", 50.0)
        self.fmax = self._coerce_float("fmax", 1100.0)
        self.batch_size = int(self.config.get("batch_size", 1024) or 1024)
        self.pad = bool(self.config.get("pad", True))
        raw_pad_mode = self.config.get("pad_mode", "reflect")
        self.pad_mode = None if raw_pad_mode is None else str(raw_pad_mode)
        self.return_periodicity = bool(self.config.get("return_periodicity", True))
        self.periodicity_threshold = self._coerce_float("periodicity_threshold", 0.1)
        self.use_median_filter = int(self.config.get("median_filter_size", 0) or 0)
        if self.use_median_filter < 0:
            raise ValueError("median_filter_size must be >= 0")

        # ``torchcrepe.predict`` gains new keyword arguments over time.  Older
        # releases will raise ``TypeError`` when unexpected keywords are passed,
        # so we introspect the installed version once and only forward supported
        # options during inference.
        predict_signature = inspect.signature(self._torchcrepe.predict)
        self._predict_params = set(predict_signature.parameters)

        self._supports_batch_size = "batch_size" in self._predict_params
        self._supports_device = "device" in self._predict_params
        self._supports_pad = "pad" in self._predict_params
        self._supports_pad_mode = "pad_mode" in self._predict_params
        self._supports_return_periodicity = "return_periodicity" in self._predict_params

        if not self._supports_pad and self.pad:
            message = "Installed torchcrepe version does not support 'pad'; disabling padding."
            self.log(message)
            LOGGER.warning("[%s] %s", self.name, message)
            self.pad = False

        if not self._supports_pad_mode and self.pad_mode is not None:
            message = "Installed torchcrepe version does not support 'pad_mode'; ignoring configuration."
            self.log(message)
            LOGGER.warning("[%s] %s", self.name, message)
            self.pad_mode = None

        if not self._supports_return_periodicity and self.return_periodicity:
            message = (
                "Installed torchcrepe version does not support periodicity outputs; disabling confidence filtering."
            )
            self.log(message)
            LOGGER.warning("[%s] %s", self.name, message)
            self.return_periodicity = False

        # Track whether we have already warned about falling back to CPU when
        # CUDA initialisation fails inside a forked worker (e.g. PyTorch
        # ``DataLoader`` workers on Linux use ``fork`` by default).  Torch's
        # CUDA runtime cannot be re-initialised safely after ``fork`` so we
        # automatically fall back to CPU execution unless the user explicitly
        # forced GPU usage via the ``device`` configuration option.
        self._device_fallback_reported = False

    def _maybe_switch_to_cpu(self, exc: RuntimeError) -> bool:
        """Switch the backend to CPU when CUDA cannot be initialised."""

        if self._force_gpu or self._device.type != "cuda":
            return False

        message = str(exc)
        lower_message = message.lower()
        # We specifically guard against CUDA initialisation problems that are
        # typically seen in forked workers.  Broadly matching on "initialisation"
        # keeps the fallback permissive enough to recover from
        # ``CUDA_ERROR_NOT_INITIALIZED`` while avoiding unrelated runtime
        # errors (which should be surfaced to the caller).
        triggers = (
            "cannot re-initialize cuda",
            "cuda driver",
            "cuda initialization",
            "initialization error",
        )
        if not any(trigger in lower_message for trigger in triggers):
            return False

        warning = (
            "CUDA initialisation failed for CREPE in this worker; falling back to CPU execution."
        )
        if not self._device_fallback_reported:
            self.log(warning)
            LOGGER.warning("[%s] %s", self.name, warning)
            self._device_fallback_reported = True
        self._device = self._torch.device("cpu")
        self.requires_cuda = False
        return True

    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = int(sr or self.sample_rate)
        waveform = audio.astype(np.float32, copy=False)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)
        base_tensor = self._torch.from_numpy(waveform).unsqueeze(0)
        hop_length = max(1, int(round(self.step_size_ms * sr / 1000.0)))

        outputs = None
        periodicity = None
        while True:
            try:
                tensor = base_tensor.to(self._device)
            except RuntimeError as exc:
                if self._maybe_switch_to_cpu(exc):
                    continue
                raise

            try:
                with self._torch.no_grad():  # pragma: no cover - heavy dependency
                    predict_kwargs = {
                        "fmin": self.fmin,
                        "fmax": self.fmax,
                        "model": self.model,
                    }
                    if self._supports_batch_size:
                        predict_kwargs["batch_size"] = self.batch_size
                    if self._supports_device:
                        predict_kwargs["device"] = self._device
                    if self._supports_pad:
                        predict_kwargs["pad"] = self.pad
                    if self._supports_pad_mode and self.pad_mode is not None:
                        predict_kwargs["pad_mode"] = self.pad_mode
                    if self._supports_return_periodicity:
                        predict_kwargs["return_periodicity"] = self.return_periodicity

                    outputs = self._torchcrepe.predict(
                        tensor,
                        sr,
                        hop_length,
                        **predict_kwargs,
                    )
                break
            except RuntimeError as exc:
                if self._maybe_switch_to_cpu(exc):
                    continue
                raise

        if outputs is None:
            raise BackendComputationError("torchcrepe did not return any outputs")

        if self._supports_return_periodicity and self.return_periodicity:
            f0_tensor, periodicity = outputs
        else:
            f0_tensor = outputs
            periodicity = None

        if self.use_median_filter > 1:
            f0_tensor = self._torchcrepe.filter.median(f0_tensor, self.use_median_filter)
            if periodicity is not None:
                periodicity = self._torchcrepe.filter.median(periodicity, self.use_median_filter)

        f0 = f0_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)
        if periodicity is not None:
            confidence = periodicity.squeeze(0).detach().cpu().numpy()
        else:
            confidence = None

        if confidence is not None and self.periodicity_threshold > 0:
            f0[confidence < self.periodicity_threshold] = 0.0

        mean_conf = float(confidence.mean()) if confidence is not None else 1.0
        self.log(
            "CREPE analysed %d frames on %s with mean periodicity %.3f."
            % (f0.shape[0], self._device.type, mean_conf)
        )

        return f0


class SwiftF0Backend(BaseF0Backend):
    """SwiftF0 backend driven by the official ONNX implementation."""

    backend_type = "swiftf0"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            from swift_f0 import SwiftF0  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailableError(
                "swift-f0 is not installed; install it with 'pip install swift-f0'"
            ) from exc

        # ``SwiftF0`` operates on 16 kHz audio with a hop of 256 samples,
        # corresponding to a 16 ms frame period. Expose that as the default
        # frame period so downstream caches have consistent expectations even
        # when the dataset uses a different hop length.
        model_frame_period_ms = 1000.0 * SwiftF0.HOP_LENGTH / SwiftF0.TARGET_SAMPLE_RATE
        self.config.setdefault("frame_period_ms", model_frame_period_ms)

        def _maybe_float(key: str) -> Optional[float]:
            value = self.config.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid float value for '{key}' in SwiftF0 backend: {value!r}"
                ) from exc

        confidence_threshold = _maybe_float("confidence_threshold")
        fmin = _maybe_float("fmin")
        fmax = _maybe_float("fmax")

        try:
            self._model = SwiftF0(
                confidence_threshold=confidence_threshold,
                fmin=fmin,
                fmax=fmax,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise BackendUnavailableError(f"Failed to initialise SwiftF0: {exc}") from exc

        self.zero_unvoiced = bool(self.config.get("zero_unvoiced", True))
        unvoiced_value = self.config.get("unvoiced_value", 0.0)
        try:
            self.unvoiced_value = float(0.0 if unvoiced_value is None else unvoiced_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid float value for 'unvoiced_value' in SwiftF0 backend: {unvoiced_value!r}"
            ) from exc
        self.requires_cuda = False

    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = int(sr or self.sample_rate)
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.reshape(-1)

        try:
            result = self._model.detect_from_array(waveform, sr)
        except ImportError as exc:
            raise BackendUnavailableError(
                "SwiftF0 requires 'librosa' for resampling when the input sample rate is not 16 kHz."
            ) from exc
        except FileNotFoundError as exc:
            raise BackendUnavailableError(f"SwiftF0 model file is missing: {exc}") from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise BackendComputationError(f"SwiftF0 failed to compute F0: {exc}") from exc

        f0 = np.asarray(result.pitch_hz, dtype=np.float64)
        mean_conf = float(np.mean(result.confidence)) if result.confidence.size else 0.0
        self.log(f"SwiftF0 analysed {f0.size} frames with mean confidence {mean_conf:.3f}.")

        if self.zero_unvoiced:
            voicing = np.asarray(result.voicing, dtype=bool)
            if voicing.size:
                f0 = f0.copy()
                f0[~voicing] = self.unvoiced_value

        return f0


class PraatBackend(BaseF0Backend):
    backend_type = "praat"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        try:
            import parselmouth  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise BackendUnavailableError("parselmouth (Praat bindings) is not installed") from exc

        self._parselmouth = parselmouth
        self.min_pitch = self._coerce_float("min_pitch", 40.0)
        self.max_pitch = self._coerce_float("max_pitch", 1100.0)
        self.silence_threshold = self._coerce_float("silence_threshold", 0.03)
        self.voicing_threshold = self._coerce_float("voicing_threshold", 0.45)
        self.octave_cost = self._coerce_float("octave_cost", 0.01)
        self.octave_jump_cost = self._coerce_float("octave_jump_cost", 1.0)
        self.voiced_unvoiced_cost = self._coerce_float("voiced_unvoiced_cost", 0.3)
        self.pitch_unit = self.config.get("unit", "Hertz")
        self.very_accurate = _coerce_enabled_flag(self.config.get("very_accurate", False))
        self._raw_method = self.config.get("method")
        self._method_key = self._normalise_method(self._raw_method)

    @staticmethod
    def _normalise_method(method_value: Optional[object]) -> Optional[str]:
        if method_value is None:
            return None
        text = str(method_value).strip().lower()
        if not text:
            return None
        return re.sub(r"[^a-z0-9]+", "", text)

    def _resolve_method_enum(self, method_value):
        if method_value is None:
            return None
        enum_cls = getattr(self._parselmouth.Sound, "ToPitchMethod", None)
        if enum_cls is None:
            return None
        if isinstance(method_value, enum_cls):  # pragma: no cover - defensive
            return method_value
        method_key = self._normalise_method(method_value)
        if method_key is None:
            return None
        for attr in dir(enum_cls):
            if attr.startswith("_"):
                continue
            try:
                candidate = getattr(enum_cls, attr)
            except AttributeError:  # pragma: no cover - defensive
                continue
            if not isinstance(candidate, enum_cls):
                continue
            attr_key = re.sub(r"[^a-z0-9]+", "", attr.lower())
            if method_key == attr_key:
                return candidate
        return None

    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        sr = int(sr or self.sample_rate)
        sound = self._parselmouth.Sound(audio, sampling_frequency=sr)
        time_step = self.frame_period_ms / 1000.0
        method_key = self._method_key
        if method_key in {"ac", "autocorrelation"}:
            pitch = sound.to_pitch_ac(
                time_step=time_step,
                pitch_floor=self.min_pitch,
                pitch_ceiling=self.max_pitch,
                very_accurate=self.very_accurate,
                silence_threshold=self.silence_threshold,
                voicing_threshold=self.voicing_threshold,
                octave_cost=self.octave_cost,
                octave_jump_cost=self.octave_jump_cost,
                voiced_unvoiced_cost=self.voiced_unvoiced_cost,
            )
        elif method_key in {"cc", "crosscorrelation"}:
            pitch = sound.to_pitch_cc(
                time_step=time_step,
                pitch_floor=self.min_pitch,
                pitch_ceiling=self.max_pitch,
                very_accurate=self.very_accurate,
                silence_threshold=self.silence_threshold,
                voicing_threshold=self.voicing_threshold,
                octave_cost=self.octave_cost,
                octave_jump_cost=self.octave_jump_cost,
                voiced_unvoiced_cost=self.voiced_unvoiced_cost,
            )
        else:
            method_enum = self._resolve_method_enum(self._raw_method)
            kwargs = {
                "time_step": time_step,
                "pitch_floor": self.min_pitch,
                "pitch_ceiling": self.max_pitch,
            }
            if method_enum is not None:
                kwargs["method"] = method_enum
            pitch = sound.to_pitch(**kwargs)
        selected = pitch.selected_array
        unit_key = self.pitch_unit or "Hertz"
        candidate_keys: List[str] = []
        if isinstance(unit_key, str):
            candidate_keys.extend(
                [
                    unit_key,
                    unit_key.lower(),
                    unit_key.upper(),
                    unit_key.capitalize(),
                ]
            )
            if unit_key.lower() == "hertz":
                candidate_keys.append("frequency")
        else:
            candidate_keys.extend(["Hertz", "frequency"])

        # Preserve order while removing duplicates
        seen = set()
        candidate_keys = [key for key in candidate_keys if not (key in seen or seen.add(key))]

        last_error: Optional[Exception] = None
        for key in candidate_keys:
            try:
                values = selected[key]
                return np.asarray(values, dtype=np.float64)
            except Exception as exc:  # pragma: no cover - passthrough for unexpected APIs
                last_error = exc
                continue

        available: List[str] = []
        if hasattr(selected, "keys"):
            try:
                available = list(selected.keys())  # type: ignore[assignment]
            except Exception:  # pragma: no cover - defensive
                available = []
        dtype = getattr(selected, "dtype", None)
        if not available and getattr(dtype, "names", None):
            available = list(dtype.names)

        detail = (
            f"Available fields: {available!r}. Last error: {last_error}" if available or last_error else ""
        )
        raise ValueError(
            f"Unsupported Praat pitch unit '{self.pitch_unit}'. {detail}"
        )


class ParselmouthBackend(PraatBackend):
    """Alias backend for clarity when users explicitly select 'parselmouth'."""

    backend_type = "parselmouth"


BACKEND_REGISTRY = {
    "pyworld": PyWorldBackend,
    "crepe": CrepeBackend,
    "swiftf0": SwiftF0Backend,
    "praat": PraatBackend,
    "parselmouth": ParselmouthBackend,
}


def _normalise_backend_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _resolve_backend_config(
    name: str, backends_config: Dict[str, Dict]
) -> Tuple[Optional[Dict], str]:
    """Return a backend configuration entry using case-insensitive lookup."""

    if not backends_config:
        return None, name

    if name in backends_config:
        return backends_config[name], name

    normalised = _normalise_backend_name(name)
    for key, cfg in backends_config.items():
        if _normalise_backend_name(key) == normalised:
            return cfg, key

    return None, name


def _coerce_enabled_flag(value) -> bool:
    """Interpret configuration truthy/falsey values consistently."""

    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return False
        if normalized in {"0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
        # fall through for unexpected string values
    return bool(value)


class F0Extractor:
    """Facade for computing F0 with configurable backend fallbacks."""

    DEFAULT_SEQUENCE = [
        {"name": "pyworld_harvest", "type": "pyworld", "config": {"algorithm": "harvest", "fallback": "dio"}},
        {"name": "pyworld_dio", "type": "pyworld", "config": {"algorithm": "dio", "fallback": None}},
    ]

    def __init__(
        self,
        sr: int,
        hop_length: int,
        config: Optional[Dict] = None,
        verbose: bool = False,
    ) -> None:
        self.sample_rate = sr
        self.hop_length = hop_length
        self.verbose = verbose
        config = config or {}
        self.bad_f0_threshold = int(config.get("bad_f0_threshold", 5))
        zero_fill = config.get("zero_fill_value", 0.0)
        if zero_fill is None:
            zero_fill = 0.0
        try:
            self.zero_fill_value = float(zero_fill)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid zero_fill_value: {zero_fill!r}") from exc

        backends_config = config.get("backends") or {}
        sequence_config = config.get("backend_order")
        if sequence_config:
            sequence = list(sequence_config)
        elif backends_config:
            # Preserve the declaration order from the config mapping when no
            # explicit sequence is provided.
            sequence = list(backends_config.keys())
        else:
            sequence = [entry["name"] for entry in self.DEFAULT_SEQUENCE]

        # Merge defaults with user configuration.  Built-in defaults are only
        # applied when the user does not provide any backend configuration at
        # all; otherwise we respect the explicitly declared backends and skip
        # any names without a matching entry.  This prevents disabled backends
        # from silently reappearing when users trim the config down to a subset
        # of backends.
        defaults: Dict[str, Dict] = {entry["name"]: entry for entry in self.DEFAULT_SEQUENCE}
        merged_sequence: List[Dict] = []
        use_defaults_for_missing = not bool(backends_config)
        self._skipped_backends: List[str] = []

        for raw_name in sequence:
            if isinstance(raw_name, dict):
                entry = dict(raw_name)
                entry["enabled"] = _coerce_enabled_flag(entry.get("enabled", True))
                merged_sequence.append(entry)
                continue
            name = str(raw_name)
            backend_name = _normalise_backend_name(name)
            backend_cfg, backend_cfg_key = _resolve_backend_config(name, backends_config)
            if backend_cfg is None and not use_defaults_for_missing:
                # When the user provided at least one backend configuration we
                # assume any names missing from ``backends`` are intentionally
                # disabled.  Skip them instead of resurrecting the default
                # definition so the runtime chain mirrors the configuration.
                LOGGER.debug("Skipping backend '%s' because it is not defined in config", name)
                self._skipped_backends.append(f"{backend_name} (not configured)")
                continue

            default_entry = defaults.get(backend_cfg_key, defaults.get(name, {"name": name, "type": name}))
            merged_entry = {**default_entry, **(backend_cfg or {})}
            merged_entry.setdefault("name", backend_cfg_key or name)
            merged_entry.setdefault("type", merged_entry.get("backend", merged_entry.get("type", name)))
            merged_entry["enabled"] = _coerce_enabled_flag(merged_entry.get("enabled", True))
            merged_sequence.append(merged_entry)

        self.backends: List[BaseF0Backend] = []
        self._backend_chain: List[str] = []
        errors: List[str] = []
        for entry in merged_sequence:
            name = entry.get("name") or entry.get("type") or "backend"
            backend_name = _normalise_backend_name(str(name))
            if not entry.get("enabled", True):
                self._skipped_backends.append(f"{backend_name} (disabled)")
                continue
            backend_type = (entry.get("type") or entry.get("backend") or "pyworld").lower()
            backend_cls = BACKEND_REGISTRY.get(backend_type)
            if backend_cls is None:
                self._skipped_backends.append(
                    f"{backend_name} (unknown backend type '{backend_type}')"
                )
                errors.append(f"Unknown backend type '{backend_type}' (entry: {name})")
                continue
            backend_name = _normalise_backend_name(str(name))
            backend_config = entry.get("config") or {k: v for k, v in entry.items() if k not in {"name", "type", "backend", "enabled"}}
            try:
                instance = backend_cls(
                    name=backend_name,
                    sr=self.sample_rate,
                    hop_length=self.hop_length,
                    config=backend_config,
                    verbose=verbose,
                )
            except BackendUnavailableError as exc:
                message = f"Skipping backend '{backend_name}': {exc}"
                errors.append(message)
                LOGGER.warning(message)
                self._skipped_backends.append(f"{backend_name} (unavailable: {exc})")
                continue
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"Failed to initialise backend '{backend_name}': {exc}")
                LOGGER.exception("Failed to initialise backend '%s'", backend_name)
                self._skipped_backends.append(f"{backend_name} (initialisation error: {exc})")
                continue
            self.backends.append(instance)
            self._backend_chain.append(instance.name)

        if not self.backends:
            error_message = "No usable F0 backends are configured."
            if errors:
                error_message += " Details: " + "; ".join(errors)
            raise RuntimeError(error_message)

        cache_tag_components = [_normalise_backend_name(backend.cache_key) for backend in self.backends]
        self.cache_identifier = "-" + "_".join(cache_tag_components) if cache_tag_components else ""
        self.requires_cuda = any(getattr(backend, "requires_cuda", False) for backend in self.backends)

    # ------------------------------------------------------------------
    def compute(self, audio: np.ndarray, sr: Optional[int] = None) -> BackendResult:
        sr = int(sr or self.sample_rate)
        for backend in self.backends:
            try:
                f0 = backend.compute(audio, sr)
            except BackendUnavailableError as exc:
                LOGGER.warning("Backend '%s' became unavailable: %s", backend.name, exc)
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Backend '%s' failed with error", backend.name)
                continue

            if f0 is None:
                continue
            f0 = np.asarray(f0, dtype=np.float64)
            if np.count_nonzero(f0) < self.bad_f0_threshold:
                LOGGER.warning(
                    "Backend '%s' returned only %d voiced frames; attempting next backend.",
                    backend.name,
                    int(np.count_nonzero(f0)),
                )
                continue
            return BackendResult(f0=f0, backend_name=backend.name)

        raise BackendComputationError("All configured F0 backends failed to produce a valid contour.")

    # ------------------------------------------------------------------
    def align_length(self, values: np.ndarray, target_frames: int) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if target_frames <= 0:
            return np.zeros((0,), dtype=np.float32)
        if values.size == target_frames:
            return values.astype(np.float32)
        if values.size == 0:
            return np.zeros((target_frames,), dtype=np.float32)

        original_indices = np.linspace(0.0, values.size - 1, num=values.size)
        target_indices = np.linspace(0.0, values.size - 1, num=target_frames)
        resampled = np.interp(target_indices, original_indices, values)

        zero_mask = values == 0.0
        if np.any(zero_mask):
            nearest_indices = np.clip(np.round(target_indices).astype(int), 0, values.size - 1)
            resampled[zero_mask[nearest_indices]] = 0.0

        return resampled.astype(np.float32)

    # ------------------------------------------------------------------
    def describe_backends(self) -> List[str]:
        return list(self._backend_chain)

    # ------------------------------------------------------------------
    def describe_skipped_backends(self) -> List[str]:
        return list(self._skipped_backends)


def build_f0_extractor(
    sr: int,
    hop_length: int,
    config: Optional[Dict] = None,
    verbose: bool = False,
) -> F0Extractor:
    return F0Extractor(sr=sr, hop_length=hop_length, config=config, verbose=verbose)

