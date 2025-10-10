#coding: utf-8
import glob
import math
import os
import os.path as osp
import time
import random
import json
import numpy as np
import soundfile as sf
from soundfile import LibsndfileError
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

try:
    import librosa
except ImportError:  # pragma: no cover - optional dependency
    librosa = None

from Utils.synthetic import WorldSynthesizer

from f0_backends import BackendComputationError, build_f0_extractor

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

DEFAULT_MEL_PARAMS = {
    "sample_rate": 24000,
    "n_mels": 80,
    "n_fft": 1024,
    "win_length": 1024,
    "hop_length": 300,
}

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=DEFAULT_MEL_PARAMS["sample_rate"],
                 mel_params=None,
                 f0_params=None,
                 data_augmentation=False,
                 validation=False,
                 verbose=True,
                 synthetic_data=None,
                 ):

        self.verbose = verbose
        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [d[0] for d in _data_list]

        mel_params = mel_params or {}
        if 'win_len' in mel_params and 'win_length' not in mel_params:
            mel_params['win_length'] = mel_params.pop('win_len')

        self.mel_params = DEFAULT_MEL_PARAMS.copy()
        self.mel_params.update(mel_params)

        if sr is not None:
            self.sr = sr
        else:
            self.sr = self.mel_params.get('sample_rate', DEFAULT_MEL_PARAMS['sample_rate'])

        # ensure mel spectrogram uses the dataset sample rate
        self.mel_params['sample_rate'] = self.sr

        if self.verbose:
            print(f"[MelDataset] Using mel-spectrogram parameters: {self.mel_params}")
        logger.info("Using mel-spectrogram parameters: %s", self.mel_params)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**self.mel_params)

        self.f0_params = f0_params or {}
        try:
            self.f0_extractor = build_f0_extractor(
                sr=self.sr,
                hop_length=self.mel_params['hop_length'],
                config=self.f0_params,
                verbose=self.verbose,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise F0 extractor: {exc}") from exc

        self.requires_cuda_backend = getattr(self.f0_extractor, "requires_cuda", False)
        self.f0_cache_suffix = f"_f0{self.f0_extractor.cache_identifier}.npy"
        self.f0_meta_suffix = self.f0_cache_suffix.replace('.npy', '.json')
        if self.verbose:
            active_backends = self.f0_extractor.describe_backends()
            backend_summary = ', '.join(active_backends) if active_backends else 'none'
            print(f"[MelDataset] F0 backends in use: {backend_summary}")
            skipped_backends = self.f0_extractor.describe_skipped_backends()
            if skipped_backends:
                print(f"[MelDataset] Skipped F0 backends: {', '.join(skipped_backends)}")

        # cache management helpers
        self._mel_cache_suffix = "_mel.npy"
        self._mel_meta_suffix = "_mel_meta.json"
        self._mel_cache_invalidated = False
        self._cache_enabled = True

        # cache audio metadata to support lazy waveform loading
        self._audio_metadata_cache = {}
        self._invalid_paths = set()

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        self.mean, self.std = -4, 4

        # for silence detection
        self.zero_value = float(self.f0_params.get('zero_fill_value', 0.0))
        self.bad_F0 = int(self.f0_params.get('bad_f0_threshold', self.f0_extractor.bad_f0_threshold))

        # synthetic augmentation configuration
        self._base_length = len(self.data_list)
        self.synthetic_config = synthetic_data or {}
        self.synthetic_enabled = bool(self.synthetic_config.get('enabled', False))
        self.synthetic_apply_to_validation = bool(self.synthetic_config.get('apply_to_validation', False))
        if validation and not self.synthetic_apply_to_validation:
            self.synthetic_enabled = False
        self._synthetic_generators = []
        self._synthetic_count = 0
        self._world_synthesizer = None
        if self.synthetic_enabled:
            self._initialise_synthetic_generators()
        if self.verbose and self.synthetic_enabled:
            summary = {
                'count': self._synthetic_count,
                'strategies': self._synthetic_generators,
            }
            print(f"[MelDataset] Synthetic data enabled: {summary}")

    def __len__(self):
        if not self.synthetic_enabled:
            return self._base_length
        return self._base_length + self._synthetic_count

    # ------------------------------------------------------------------
    # Multiprocessing support helpers
    def __getstate__(self):
        """Make the dataset picklable for multiprocessing workers."""

        state = self.__dict__.copy()
        state['_f0_extractor_init'] = {
            'sr': self.sr,
            'hop_length': self.mel_params['hop_length'],
            'config': self.f0_params,
            'verbose': self.verbose,
        }
        # ``torchaudio`` transforms and backend instances hold module
        # references that cannot be pickled.  Drop them here and rebuild in
        # ``__setstate__`` when the worker process deserialises the dataset.
        state.pop('f0_extractor', None)
        state.pop('to_melspec', None)
        return state

    def __setstate__(self, state):
        extractor_init = state.pop('_f0_extractor_init')
        self.__dict__.update(state)
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**self.mel_params)
        self.f0_extractor = build_f0_extractor(
            sr=extractor_init['sr'],
            hop_length=extractor_init['hop_length'],
            config=extractor_init['config'],
            verbose=extractor_init['verbose'],
        )
        self.requires_cuda_backend = getattr(self.f0_extractor, 'requires_cuda', False)
        self.f0_cache_suffix = f"_f0{self.f0_extractor.cache_identifier}.npy"
        self.f0_meta_suffix = self.f0_cache_suffix.replace('.npy', '.json')
        self.bad_F0 = int(self.f0_params.get('bad_f0_threshold', self.f0_extractor.bad_f0_threshold))

    def path_to_mel_and_label(self, path):
        metadata = self._get_audio_metadata(path)
        source_sr = metadata.get('sample_rate') or metadata.get('samplerate') or metadata.get('sr')
        total_frames = int(metadata.get('frames', 0) or 0)

        hop_length = int(self.mel_params['hop_length'])
        target_frames = int(self.max_mel_length)

        segment_frames = None
        start_frame = 0
        use_full_file = True

        if target_frames > 0 and source_sr and total_frames > 0:
            base_duration = (target_frames * hop_length) / float(self.sr)
            window_size = int(self.mel_params.get('win_length') or self.mel_params.get('n_fft', hop_length))
            pad_duration = max(window_size, hop_length) / float(self.sr)
            requested_duration = base_duration + pad_duration
            segment_frames = int(np.ceil(requested_duration * float(source_sr)))
            if segment_frames <= 0:
                segment_frames = None
            elif segment_frames < total_frames:
                max_start = max(0, total_frames - segment_frames)
                start_frame = random.randint(0, max_start) if max_start > 0 else 0
                use_full_file = False

        wave_tensor, wave_sr = self._load_tensor(path, start_frame=start_frame, num_frames=segment_frames)
        waveform = wave_tensor.numpy()
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=-1)
        waveform = waveform.astype(np.float32)

        if wave_sr != self.sr:
            waveform = self._resample_waveform(waveform, wave_sr, self.sr)
            wave_sr = self.sr

        resampled_start_sample = 0
        if not use_full_file and source_sr:
            start_time = start_frame / float(source_sr)
            resampled_start_sample = int(round(start_time * self.sr))

        expected_frames = None
        if not use_full_file:
            expected_frames = int(np.ceil(len(waveform) / max(hop_length, 1))) + 2

        f0 = self._load_or_compute_f0(
            path,
            waveform,
            wave_sr,
            start_sample=resampled_start_sample,
            expected_frames=expected_frames,
            use_cache=True,
            write_cache=(use_full_file and not self.data_augmentation),
        )

        if self.data_augmentation:
            random_scale = 0.5 + 0.5 * np.random.random()
            waveform = random_scale * waveform

        cache_key = path if use_full_file else None
        allow_cache = (not self.data_augmentation) and use_full_file

        return self._build_training_example(
            waveform,
            sr=wave_sr,
            f0=f0,
            cache_key=cache_key,
            allow_cache=allow_cache,
        )


    def __getitem__(self, idx):
        if self.synthetic_enabled and idx >= self._base_length:
            return self._generate_synthetic_sample()

        total_items = len(self.data_list)
        if total_items == 0:
            raise IndexError("MelDataset is empty")

        attempts = 0
        while attempts < total_items:
            data_index = (idx + attempts) % total_items
            data = self.data_list[data_index]

            if data in self._invalid_paths:
                attempts += 1
                continue

            try:
                mel_tensor, f0, is_silence = self.path_to_mel_and_label(data)
            except (FileNotFoundError, LibsndfileError, RuntimeError, OSError, ValueError) as exc:
                self._mark_path_invalid(data, exc)
                attempts += 1
                continue

            return mel_tensor, f0, is_silence

        raise RuntimeError("No valid audio files could be loaded from the dataset")

    def _mark_path_invalid(self, path, exc):
        if path in self._invalid_paths:
            return
        self._invalid_paths.add(path)
        message = f"[MelDataset] Skipping unreadable audio file: {path} ({exc})"
        logger.warning(message)
        if self.verbose:
            print(message)

    def _load_tensor(self, data, start_frame=None, num_frames=None):
        wave_path = data
        try:
            if start_frame is None and num_frames is None:
                wave, sr = sf.read(wave_path, dtype='float32')
            else:
                start = int(start_frame or 0)
                frames = -1 if num_frames is None else int(num_frames)
                with sf.SoundFile(wave_path, mode='r') as sound_file:
                    sr = sound_file.samplerate
                    if start:
                        sound_file.seek(start)
                    wave = sound_file.read(frames=frames, dtype='float32', always_2d=False)
        except (FileNotFoundError, LibsndfileError, RuntimeError, OSError, ValueError) as exc:
            raise RuntimeError(f"Failed to load audio file '{wave_path}': {exc}") from exc
        wave_tensor = torch.from_numpy(np.asarray(wave, dtype=np.float32)).float()
        return wave_tensor, sr

    def _get_audio_metadata(self, path):
        metadata = self._audio_metadata_cache.get(path)
        if metadata is not None:
            return metadata
        try:
            info = sf.info(path)
        except RuntimeError:
            info = None
        if info is None:
            metadata = {'frames': 0, 'sample_rate': None}
        else:
            metadata = {
                'frames': getattr(info, 'frames', 0),
                'sample_rate': getattr(info, 'samplerate', None),
                'channels': getattr(info, 'channels', None),
            }
        self._audio_metadata_cache[path] = metadata
        return metadata

    # ------------------------------------------------------------------
    # Synthetic data helpers
    def _initialise_synthetic_generators(self):
        config = self.synthetic_config
        ratio = float(config.get('ratio', 0.0))
        absolute_count = config.get('absolute_count')
        max_items = config.get('max_items')
        min_items = config.get('min_items', 0)

        if absolute_count is not None:
            self._synthetic_count = max(0, int(absolute_count))
        else:
            target = int(round(self._base_length * ratio))
            if ratio > 0 and target == 0 and self._base_length > 0:
                target = 1
            self._synthetic_count = max(0, target)

        if max_items is not None:
            self._synthetic_count = min(self._synthetic_count, int(max_items))
        if min_items:
            self._synthetic_count = max(self._synthetic_count, int(min_items))

        pitch_shift_cfg = config.get('pitch_shift', {}) or {}
        pitch_shift_enabled = pitch_shift_cfg.get('enabled', True)
        if pitch_shift_enabled:
            if librosa is None:
                if self.verbose:
                    print("[MelDataset] Pitch-shift augmentation disabled: librosa not available.")
            elif not self.data_list:
                if self.verbose:
                    print("[MelDataset] Pitch-shift augmentation disabled: no base samples available.")
            else:
                self._synthetic_generators.append('pitch_shift')
        self.synthetic_pitch_shift_config = pitch_shift_cfg

        world_cfg = config.get('world_vocoder', {}) or {}
        world_enabled = world_cfg.get('enabled', False)
        if world_enabled:
            try:
                self._world_synthesizer = WorldSynthesizer(
                    sample_rate=self.sr,
                    hop_length=self.mel_params['hop_length'],
                    fft_size=self.mel_params.get('n_fft', 1024),
                    config=world_cfg,
                    verbose=self.verbose,
                )
            except (ImportError, RuntimeError, ValueError) as exc:
                self._world_synthesizer = None
                if self.verbose:
                    print(f"[MelDataset] WORLD vocoder synthetic generation disabled: {exc}")
            else:
                self._synthetic_generators.append('world_vocoder')

        if not self._synthetic_generators or self._synthetic_count <= 0:
            self.synthetic_enabled = False
            self._synthetic_generators = []
            self._synthetic_count = 0
            if self.verbose:
                print("[MelDataset] Synthetic data disabled: no valid generators or count is zero.")

    def _generate_synthetic_sample(self):
        if not self._synthetic_generators:
            raise RuntimeError("Synthetic generation requested but no generators are available")

        generator_name = random.choice(self._synthetic_generators)
        if generator_name == 'pitch_shift':
            result = self._generate_pitch_shift_sample()
            if result is not None:
                return result
            # fall back to another generator if pitch shift failed
            remaining = [g for g in self._synthetic_generators if g != 'pitch_shift']
            if remaining:
                generator_name = random.choice(remaining)
            else:
                # as a last resort, try again with pitch shift to avoid crashing
                result = self._generate_pitch_shift_sample(force=True)
                if result is not None:
                    return result
                raise RuntimeError("Unable to produce synthetic pitch-shift sample")

        if generator_name == 'world_vocoder' and self._world_synthesizer is not None:
            waveform, f0 = self._world_synthesizer.generate()
            return self._build_training_example(
                waveform.astype(np.float32),
                sr=self.sr,
                f0=f0.astype(np.float32),
                cache_key=None,
                allow_cache=False,
            )

        if generator_name != 'pitch_shift':
            raise RuntimeError(f"Unknown synthetic generator '{generator_name}'")
        # final attempt for pitch shift
        result = self._generate_pitch_shift_sample(force=True)
        if result is None:
            raise RuntimeError("Failed to generate synthetic sample")
        return result

    def _generate_pitch_shift_sample(self, force=False):
        if librosa is None:
            return None

        cfg = self.synthetic_pitch_shift_config or {}
        semitone_choices = cfg.get('semitones') or [-4, -2, -1, 1, 2, 4]
        if not semitone_choices:
            return None

        max_attempts = max(1, int(cfg.get('max_attempts', 5)))
        min_voiced_fraction = float(cfg.get('min_voiced_fraction', 0.05))
        gain_db_range = cfg.get('gain_db_range', [-6.0, 3.0])
        if gain_db_range is None:
            gain_db_range = None
        elif isinstance(gain_db_range, (int, float)):
            gain_db_range = (float(gain_db_range), float(gain_db_range))
        else:
            gain_db_range = tuple(float(v) for v in gain_db_range)
        noise_db = cfg.get('noise_db', None)
        if noise_db is not None:
            noise_db = float(noise_db)
        keep_original_when_zero = bool(cfg.get('keep_zero_pitch', True))
        res_type = cfg.get('resample_type', 'kaiser_best')

        for attempt in range(max_attempts):
            available_paths = [p for p in self.data_list if p not in self._invalid_paths]
            if not available_paths:
                if force and attempt == max_attempts - 1:
                    raise RuntimeError("No valid audio files available for pitch shifting")
                return None

            base_path = random.choice(available_paths)
            try:
                wave_tensor, wave_sr = self._load_tensor(base_path)
            except RuntimeError as exc:
                self._mark_path_invalid(base_path, exc)
                continue
            waveform = wave_tensor.numpy()
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=-1)
            waveform = waveform.astype(np.float32)
            if wave_sr != self.sr:
                waveform = self._resample_waveform(waveform, wave_sr, self.sr)
            base_f0 = self._load_or_compute_f0(base_path, waveform, self.sr)
            if base_f0.size == 0:
                if force and attempt == max_attempts - 1:
                    break
                continue
            voiced_fraction = float(np.count_nonzero(base_f0 > 0)) / max(1, base_f0.size)
            if voiced_fraction < min_voiced_fraction:
                if force and attempt == max_attempts - 1:
                    break
                continue

            semitone = random.choice(semitone_choices)
            if semitone == 0 and not force:
                if force and attempt == max_attempts - 1:
                    break
                continue

            try:
                shifted_waveform = librosa.effects.pitch_shift(
                    waveform,
                    sr=self.sr,
                    n_steps=float(semitone),
                    res_type=res_type,
                )
            except Exception:
                if force and attempt == max_attempts - 1:
                    raise
                continue

            ratio = float(2 ** (semitone / 12.0))
            shifted_f0 = base_f0.astype(np.float32) * ratio
            if keep_original_when_zero:
                shifted_f0[base_f0 == 0] = 0.0

            if gain_db_range is not None:
                low, high = gain_db_range
                if low > high:
                    low, high = high, low
                gain = 10.0 ** (random.uniform(low, high) / 20.0)
                shifted_waveform = shifted_waveform * gain

            if noise_db is not None:
                noise_gain = 10.0 ** (noise_db / 20.0)
                noise = np.random.normal(scale=noise_gain, size=shifted_waveform.shape)
                shifted_waveform = shifted_waveform + noise.astype(np.float32)

            return self._build_training_example(
                shifted_waveform.astype(np.float32),
                sr=self.sr,
                f0=shifted_f0,
                cache_key=None,
                allow_cache=False,
            )

        return None

    def _f0_cache_paths(self, path):
        data_path = path + self.f0_cache_suffix
        meta_path = path + self.f0_meta_suffix
        legacy_path = path + "_f0.npy"
        return data_path, meta_path, legacy_path

    def _load_or_compute_f0(self, path, waveform, sr, *, start_sample=0, expected_frames=None, use_cache=True, write_cache=True):
        cached_f0 = None
        if use_cache:
            cached_f0 = self._load_cached_f0(path)
            if cached_f0 is not None:
                if expected_frames is None:
                    return cached_f0
                hop = max(int(self.mel_params['hop_length']), 1)
                start_index = max(0, int(math.floor(start_sample / float(hop))))
                if start_index >= cached_f0.shape[0]:
                    return np.zeros((0,), dtype=np.float32)
                end_index = min(cached_f0.shape[0], start_index + int(expected_frames) + 4)
                return cached_f0[start_index:end_index]

        if (
            cached_f0 is None
            and use_cache
            and expected_frames is not None
            and not self.data_augmentation
        ):
            try:
                full_wave_tensor, full_wave_sr = self._load_tensor(path)
            except (FileNotFoundError, LibsndfileError, RuntimeError, OSError, ValueError):
                full_wave_tensor = None
            if full_wave_tensor is not None:
                full_waveform = full_wave_tensor.numpy()
                if full_waveform.ndim > 1:
                    full_waveform = np.mean(full_waveform, axis=-1)
                full_waveform = full_waveform.astype(np.float32)
                if full_wave_sr != self.sr:
                    full_waveform = self._resample_waveform(full_waveform, full_wave_sr, self.sr)
                    full_wave_sr = self.sr

                full_f0 = self._load_or_compute_f0(
                    path,
                    full_waveform,
                    full_wave_sr,
                    start_sample=0,
                    expected_frames=None,
                    use_cache=use_cache,
                    write_cache=True,
                )

                hop = max(int(self.mel_params['hop_length']), 1)
                start_index = max(0, int(math.floor(start_sample / float(hop))))
                if start_index >= full_f0.shape[0]:
                    return np.zeros((0,), dtype=np.float32)
                end_index = min(full_f0.shape[0], start_index + int(expected_frames) + 4)
                return full_f0[start_index:end_index]

        if self.verbose:
            active_backends = self.f0_extractor.describe_backends()
            backend_names = ', '.join(active_backends) if active_backends else 'none'
            print(f"[MelDataset] Computing F0 for {path} using backends: {backend_names}")

        try:
            result = self.f0_extractor.compute(waveform, sr=sr)
            f0 = np.asarray(result.f0, dtype=np.float32)
            backend_name = result.backend_name
            if self.verbose and backend_name:
                print(f"[MelDataset] Selected F0 backend '{backend_name}' for {path}")
        except BackendComputationError as exc:
            logger.warning("All configured F0 backends failed for %s: %s", path, exc)
            f0 = np.zeros((0,), dtype=np.float32)
            backend_name = ""
            if self.verbose:
                print(f"[MelDataset] F0 computation failed for {path}; using zeros")

        cache_entire = (
            use_cache and write_cache and self._cache_enabled and not self.data_augmentation
            and expected_frames is None and start_sample == 0
        )
        if cache_entire:
            self._save_f0_cache(path, f0, backend_name)

        return f0

    def _load_cached_f0(self, path):
        if not self._cache_enabled:
            return None

        data_path, meta_path, legacy_path = self._f0_cache_paths(path)

        if os.path.isfile(data_path):
            metadata = None
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as meta_file:
                        metadata = json.load(meta_file)
                except (OSError, json.JSONDecodeError):
                    self._remove_file_safely(meta_path)
                    metadata = None
            if metadata:
                expected = {
                    'cache_identifier': self.f0_extractor.cache_identifier,
                    'sample_rate': int(self.sr),
                    'hop_length': int(self.mel_params['hop_length']),
                }
                if all(metadata.get(key) == value for key, value in expected.items()):
                    try:
                        return np.load(data_path).astype(np.float32)
                    except (OSError, ValueError):
                        self._remove_file_safely(data_path)
                else:
                    self._remove_file_safely(data_path)
                    self._remove_file_safely(meta_path)
            else:
                self._remove_file_safely(data_path)

        if os.path.isfile(legacy_path):
            try:
                return np.load(legacy_path).astype(np.float32)
            except (OSError, ValueError):
                self._remove_file_safely(legacy_path)

        return None

    def _save_f0_cache(self, path, f0, backend_name):
        data_path, meta_path, _ = self._f0_cache_paths(path)
        try:
            np.save(data_path, np.asarray(f0, dtype=np.float32))
            metadata = {
                'cache_identifier': self.f0_extractor.cache_identifier,
                'backend': backend_name,
                'sample_rate': int(self.sr),
                'hop_length': int(self.mel_params['hop_length']),
            }
            with open(meta_path, 'w', encoding='utf-8') as meta_file:
                json.dump(metadata, meta_file, sort_keys=True)
        except OSError as exc:
            logger.warning("Failed to cache F0 for %s: %s", path, exc)

    @staticmethod
    def _resample_waveform(waveform, source_sr, target_sr):
        if source_sr == target_sr:
            return waveform
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor, source_sr, target_sr)
        return resampled.squeeze(0).cpu().numpy()

    def _build_training_example(self, waveform, sr, f0, cache_key=None, allow_cache=True):
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=-1)
        waveform = waveform.astype(np.float32)
        if sr != self.sr:
            waveform = self._resample_waveform(waveform, sr, self.sr)
            sr = self.sr

        wave_tensor = torch.from_numpy(waveform).float()
        expected_metadata = None
        mel_tensor = None
        if cache_key is not None and allow_cache:
            expected_metadata = self._build_mel_metadata(wave_tensor, sr)
            mel_tensor = self._load_cached_mel(cache_key, expected_metadata)
        if mel_tensor is None:
            mel_tensor = self.to_melspec(wave_tensor)
            if cache_key is not None and allow_cache and self._cache_enabled:
                if expected_metadata is None:
                    expected_metadata = self._build_mel_metadata(wave_tensor, sr)
                self._save_mel_cache(cache_key, mel_tensor, expected_metadata)

        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)

        if f0 is None:
            f0 = np.zeros((mel_length,), dtype=np.float32)
        else:
            f0 = self.f0_extractor.align_length(f0, mel_length)
        f0_tensor = torch.from_numpy(f0).float()

        f0_zero = (f0_tensor == 0)

        #######################################
        # You may want your own silence labels here
        # The more accurate the label, the better the results
        is_silence = torch.zeros(f0_tensor.shape)
        is_silence[f0_zero] = 1
        #######################################

        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]
            f0_tensor = f0_tensor[random_start:random_start + self.max_mel_length]
            is_silence = is_silence[random_start:random_start + self.max_mel_length]

        if torch.any(torch.isnan(f0_tensor)):  # failed
            f0_tensor[torch.isnan(f0_tensor)] = self.zero_value  # replace nan value with 0

        return mel_tensor, f0_tensor, is_silence

    def _build_mel_metadata(self, wave_tensor, wave_sr):
        num_samples = int(wave_tensor.shape[0]) if wave_tensor.ndim > 0 else int(wave_tensor.numel())
        num_channels = int(wave_tensor.shape[1]) if wave_tensor.ndim > 1 else 1

        def _serialize(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (np.generic,)):
                return value.item()
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                return value.item() if value.numel() == 1 else value.tolist()
            return value

        serialized_params = {k: _serialize(v) for k, v in self.mel_params.items()}

        return {
            "audio_sample_rate": int(wave_sr),
            "audio_num_samples": num_samples,
            "audio_num_channels": num_channels,
            "dataset_sample_rate": int(self.sr),
            "mel_params": serialized_params,
        }

    def _mel_cache_paths(self, path):
        return path + self._mel_cache_suffix, path + self._mel_meta_suffix

    def _load_cached_mel(self, path, expected_metadata):
        if not self._cache_enabled or self.data_augmentation:
            return None

        mel_cache_path, meta_cache_path = self._mel_cache_paths(path)

        if not os.path.isfile(mel_cache_path):
            # no cached mel available
            # remove stray metadata file if present to avoid stale comparisons later
            if os.path.isfile(meta_cache_path) and not self._mel_cache_invalidated:
                self._invalidate_mel_cache(meta_cache_path, reason="metadata_without_mel")
            return None

        if not os.path.isfile(meta_cache_path):
            # stale cache without metadata; invalidate the entire cache once
            self._invalidate_mel_cache(meta_cache_path, reason="missing_metadata")
            return None

        try:
            with open(meta_cache_path, "r", encoding="utf-8") as meta_file:
                cached_metadata = json.load(meta_file)
        except (OSError, json.JSONDecodeError):
            self._invalidate_mel_cache(meta_cache_path, reason="unreadable_metadata")
            return None

        if cached_metadata != expected_metadata:
            self._invalidate_mel_cache(meta_cache_path, reason="metadata_mismatch")
            return None

        try:
            mel_numpy = np.load(mel_cache_path)
        except (OSError, ValueError):
            self._invalidate_mel_cache(mel_cache_path, reason="unreadable_cache")
            return None

        return torch.from_numpy(mel_numpy)

    def _invalidate_mel_cache(self, reference_path, reason="unknown"):
        if self._mel_cache_invalidated:
            # ensure the reference file is removed even on subsequent calls
            self._remove_file_safely(reference_path)
            return

        self._mel_cache_invalidated = True
        if self.verbose:
            print(f"[MelDataset] Mel cache invalidation triggered ({reason}). Clearing cached spectrograms...")
        logger.info("Mel cache invalidation triggered (%s). Clearing cached spectrograms.", reason)

        for audio_path in self.data_list:
            mel_cache_path, meta_cache_path = self._mel_cache_paths(audio_path)
            f0_cache_path, f0_meta_path, legacy_path = self._f0_cache_paths(audio_path)
            self._remove_file_safely(mel_cache_path)
            self._remove_file_safely(meta_cache_path)
            self._remove_file_safely(f0_cache_path)
            self._remove_file_safely(f0_meta_path)
            self._remove_file_safely(legacy_path)
            for extra_path in glob.glob(audio_path + "_f0*.npy"):
                if extra_path not in {f0_cache_path, legacy_path}:
                    self._remove_file_safely(extra_path)
            for extra_meta in glob.glob(audio_path + "_f0*.json"):
                if extra_meta != f0_meta_path:
                    self._remove_file_safely(extra_meta)

    @staticmethod
    def _remove_file_safely(path):
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("Failed to remove cache file %s: %s", path, exc)

    def _save_mel_cache(self, path, mel_tensor, metadata):
        mel_cache_path, meta_cache_path = self._mel_cache_paths(path)
        mel_numpy = mel_tensor.detach().cpu().numpy()
        try:
            np.save(mel_cache_path, mel_numpy)
            with open(meta_cache_path, "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, sort_keys=True)
        except OSError as exc:
            logger.warning("Failed to save mel cache for %s: %s", path, exc)

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        f0s = torch.zeros((batch_size, self.max_mel_length)).float()
        is_silences = torch.zeros((batch_size, self.max_mel_length)).float()

        for bid, (mel, f0, is_silence) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            f0s[bid, :mel_size] = f0
            is_silences[bid, :mel_size] = is_silence

        if self.max_mel_length > self.min_mel_length:
            random_slice = np.random.randint(
                self.min_mel_length//self.mel_length_step,
                1+self.max_mel_length//self.mel_length_step) * self.mel_length_step + self.min_mel_length
            mels = mels[:, :, :random_slice]
            f0 = f0[:, :random_slice]

        mels = mels.unsqueeze(1)
        return mels, f0s, is_silences


def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config=None,
                     dataset_config=None):

    dataset_config = dict(dataset_config or {})
    dataloader_options = dataset_config.pop('dataloader', {}) or {}

    dataset = MelDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**(collate_config or {}))

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=(not validation),
        num_workers=num_workers,
        drop_last=(not validation),
        collate_fn=collate_fn,
        pin_memory=(device != 'cpu'),
    )

    start_method = dataloader_options.get('start_method')
    if start_method is None and num_workers > 0 and dataset.requires_cuda_backend:
        start_method = 'spawn'
        if dataset.verbose:
            print("[MelDataset] Using 'spawn' multiprocessing context for CUDA-enabled F0 backends.")

    if start_method:
        try:
            multiprocessing_context = torch.multiprocessing.get_context(start_method)
        except RuntimeError as exc:
            raise RuntimeError(f"Invalid DataLoader start method '{start_method}': {exc}") from exc
        loader_kwargs['multiprocessing_context'] = multiprocessing_context

    persistent_workers = dataloader_options.get('persistent_workers')
    if persistent_workers is not None and num_workers > 0:
        loader_kwargs['persistent_workers'] = bool(persistent_workers)

    prefetch_factor = dataloader_options.get('prefetch_factor')
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs['prefetch_factor'] = int(prefetch_factor)

    data_loader = DataLoader(dataset, **loader_kwargs)

    return data_loader
