from meldataset import build_dataloader
from optimizers import build_optimizer
from utils import *
from models import build_model
from trainer import Trainer

import copy
import itertools
import os
import os.path as osp
import math
import wave
import re
import sys
import json
import shutil
import hashlib
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import click
import importlib
import importlib.util
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# enable better memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True


class _DummySummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_figure(self, *args, **kwargs):
        return None


def _load_optional_tqdm():
    """Return tqdm.tqdm if available, otherwise ``None``."""
    spec = importlib.util.find_spec("tqdm")
    if spec is None:
        return None
    module = importlib.import_module("tqdm")
    return getattr(module, "tqdm", None)


_OPTIONAL_TQDM = _load_optional_tqdm()


_METADATA_CACHE_VERSION = 2


def prepare_data_list(
    raw_data_list,
    root_path="",
    cache_config=None,
    metadata_path=None,
    dataset_name=None,
):
    """Parse metadata lines and compute WAV durations with optional caching.

    The caching layer stores per-file audio format information (sample rate,
    frame count, number of channels and sample width) to detect dataset
    re-encodes even when file timestamps remain unchanged.

    Args:
        raw_data_list: Iterable with entries in the format
            ``path|phoneme sequence|speaker_id``.
        root_path: Base directory of the audio files.
        cache_config: Optional dictionary describing metadata caching
            behaviour.  When enabled the parsed metadata and durations are
            persisted to disk and reused on subsequent runs.
        metadata_path: Optional filesystem path to the metadata file the
            ``raw_data_list`` originated from.  Providing this allows the cache
            to track file modification times for invalidation.
        dataset_name: Identifier of the dataset split (e.g. ``"train"`` or
            ``"val"``).  Used when applying per-dataset cache toggles.

    Returns:
        Tuple ``(prepared_list, durations)`` where ``prepared_list`` contains
        ``[path, text, speaker_id]`` entries and ``durations`` is a list with
        their corresponding durations (in seconds).
    """

    def _stat_file(path):
        try:
            stat_result = os.stat(path)
        except OSError:
            return None
        return {"mtime": stat_result.st_mtime, "size": stat_result.st_size}

    def _audio_stats_match(expected, current):
        if expected is None or current is None:
            return expected is None and current is None
        return (
            abs(expected.get("mtime", 0.0) - current.get("mtime", 0.0)) < 1e-3
            and int(expected.get("size", -1)) == int(current.get("size", -1))
        )

    def _normalise_dataset_key(name):
        if name is None:
            return "dataset"
        return str(name).strip().lower() or "dataset"

    raw_data_sequence = list(raw_data_list)
    total_items = len(raw_data_sequence)
    prepared_list = []
    durations = []

    if total_items == 0:
        return prepared_list, durations

    dataset_key = _normalise_dataset_key(dataset_name)
    root_path_str = str(root_path or "")
    cache_cfg = cache_config if isinstance(cache_config, dict) else {}
    cache_enabled = bool(cache_cfg.get("enabled", False))
    dataset_toggles = cache_cfg.get("datasets") if isinstance(cache_cfg, dict) else None
    if isinstance(dataset_toggles, dict) and dataset_toggles:
        toggles = {str(k).strip().lower(): bool(v) for k, v in dataset_toggles.items()}
        if dataset_key in toggles:
            cache_enabled = cache_enabled and toggles.get(dataset_key, True)

    metadata_path_obj = Path(metadata_path).expanduser() if metadata_path else None
    metadata_stat = None
    raw_hash = None
    cache_file = None

    if cache_enabled:
        cache_dir = cache_cfg.get("directory", "Data/cache") or "Data/cache"
        cache_dir = Path(cache_dir).expanduser()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(f"Failed to create metadata cache directory {cache_dir!s}: {exc}")
            cache_enabled = False
        else:
            identifier_parts = [dataset_key, root_path_str]
            if metadata_path_obj is not None:
                identifier_parts.append(str(metadata_path_obj.resolve()))
                try:
                    metadata_stat = metadata_path_obj.stat()
                except OSError:
                    metadata_stat = None
            if metadata_path_obj is None:
                hasher = hashlib.sha1()
                for line in raw_data_sequence:
                    hasher.update(line.encode("utf-8"))
                raw_hash = hasher.hexdigest()
                identifier_parts.append(raw_hash)

            digest = hashlib.sha1("|".join(identifier_parts).encode("utf-8")).hexdigest()
            safe_base = re.sub(r"[^a-zA-Z0-9_.-]+", "_", dataset_key or "metadata")
            cache_file = cache_dir / f"{safe_base}-{digest}.json"

    cache_payload = None
    if cache_enabled and cache_file and cache_file.is_file():
        try:
            with cache_file.open("r", encoding="utf-8") as f:
                cache_payload = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Failed to read metadata cache {cache_file}: {exc}")
            cache_payload = None

    validate_audio = bool(cache_cfg.get("validate_audio", True))
    if cache_payload and cache_payload.get("version") == _METADATA_CACHE_VERSION:
        if cache_payload.get("dataset") != dataset_key:
            cache_payload = None
        elif cache_payload.get("root_path", "") != root_path_str:
            cache_payload = None
        else:
            cached_metadata_path = cache_payload.get("metadata_path")
            metadata_info = cache_payload.get("metadata_info", {}) or {}
            cache_valid = True

            if metadata_path_obj is not None:
                if metadata_stat is None:
                    try:
                        metadata_stat = metadata_path_obj.stat()
                    except OSError:
                        metadata_stat = None
                if metadata_stat is None:
                    cache_valid = False
                else:
                    cached_mtime = metadata_info.get("mtime")
                    cached_size = metadata_info.get("size")
                    if (
                        cached_metadata_path is None
                        or str(cached_metadata_path) != str(metadata_path_obj)
                        or abs(float(cached_mtime or 0.0) - float(metadata_stat.st_mtime)) >= 1e-3
                        or int(cached_size or -1) != int(metadata_stat.st_size)
                    ):
                        cache_valid = False
            else:
                if raw_hash is None:
                    hasher = hashlib.sha1()
                    for line in raw_data_sequence:
                        hasher.update(line.encode("utf-8"))
                    raw_hash = hasher.hexdigest()
                if metadata_info.get("hash") != raw_hash:
                    cache_valid = False

            if cache_valid and validate_audio:
                entries = cache_payload.get("entries", [])

                first_signature = cache_payload.get("first_audio_signature")
                if entries and first_signature:
                    first_path = first_signature.get("path")
                    if first_path:
                        first_wav_path = os.path.join(root_path_str, first_path)
                        try:
                            with wave.open(first_wav_path, "rb") as wf:
                                current_signature = {
                                    "sample_rate": wf.getframerate(),
                                    "frames": wf.getnframes(),
                                    "channels": wf.getnchannels(),
                                    "sample_width": wf.getsampwidth(),
                                }
                        except Exception:
                            cache_valid = False
                        else:
                            for key in ("sample_rate", "frames", "channels", "sample_width"):
                                cached_value = int(first_signature.get(key, -1))
                                current_value = int(current_signature.get(key, -2))
                                if cached_value != current_value:
                                    cache_valid = False
                                    break
                    else:
                        cache_valid = False

                if cache_valid:
                    for item in entries:
                        wave_rel_path = item.get("path")
                        if not wave_rel_path:
                            cache_valid = False
                            break
                        wave_path = os.path.join(root_path_str, wave_rel_path)
                        audio_stat = item.get("audio")
                        if audio_stat is None:
                            cache_valid = False
                            break
                        current_stat = _stat_file(wave_path)
                        if not _audio_stats_match(audio_stat, current_stat):
                            cache_valid = False
                            break

            if cache_valid:
                cached_entries = cache_payload.get("entries", [])
                prepared_list = []
                durations = []
                for item in cached_entries:
                    path = item.get("path")
                    text = item.get("text", "")
                    speaker_id = item.get("speaker_id", "")
                    duration = float(item.get("duration", 0.0))
                    if path is None:
                        cache_valid = False
                        break
                    prepared_list.append([path, text, speaker_id])
                    durations.append(duration)

                if cache_valid:
                    logger.info(
                        "Loaded %d metadata entries for '%s' from cache %s",
                        len(prepared_list),
                        dataset_key,
                        cache_file,
                    )
                    return prepared_list, durations

    progress_desc = "Computing audio durations"
    iterator = raw_data_sequence
    use_tqdm = _OPTIONAL_TQDM is not None

    if use_tqdm:
        iterator = _OPTIONAL_TQDM(
            raw_data_sequence, desc=progress_desc, total=total_items, unit="files"
        )
    else:
        print(f"{progress_desc} for {total_items} files...")
        update_interval = max(1, total_items // 20)

    cached_audio_stats = []
    cached_audio_formats = []

    for index, line in enumerate(iterator):
        cleaned_line = line.rstrip("\n")
        if not cleaned_line.strip():
            continue

        parts = cleaned_line.split("|")
        if len(parts) < 2:
            print(f"Parse error for line: {cleaned_line}")
            continue

        path = parts[0].strip()
        if len(parts) == 2:
            text = parts[1]
            speaker_id = ""
        else:
            text = "|".join(parts[1:-1])
            speaker_id = parts[-1].strip()

        wav_path = os.path.join(root_path_str, path)

        try:
            audio_stat = _stat_file(wav_path)
            with wave.open(wav_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                duration = frames / float(rate) if rate else 0.0
        except Exception as e:
            print(f"Error for wave path: {wav_path}, {e}")
            continue

        prepared_list.append([path, text, speaker_id])
        durations.append(duration)
        cached_audio_stats.append(audio_stat)
        cached_audio_formats.append(
            {
                "sample_rate": int(rate),
                "frames": int(frames),
                "channels": int(channels),
                "sample_width": int(sample_width),
            }
        )

        if not use_tqdm:
            should_update = ((index + 1) % update_interval == 0) or ((index + 1) == total_items)
            if should_update:
                print(f"Computed durations for {index + 1}/{total_items} files", flush=True)

    if cache_enabled and cache_file:
        metadata_info = {}
        if metadata_stat is not None:
            metadata_info = {"mtime": metadata_stat.st_mtime, "size": metadata_stat.st_size}
        else:
            if raw_hash is None:
                hasher = hashlib.sha1()
                for line in raw_data_sequence:
                    hasher.update(line.encode("utf-8"))
                raw_hash = hasher.hexdigest()
            metadata_info = {"hash": raw_hash}

        cache_entries = []
        for entry, duration, audio_stat, audio_format in zip(
            prepared_list, durations, cached_audio_stats, cached_audio_formats
        ):
            cache_entries.append(
                {
                    "path": entry[0],
                    "text": entry[1],
                    "speaker_id": entry[2],
                    "duration": float(duration),
                    "audio": audio_stat,
                    "audio_format": audio_format,
                }
            )

        first_audio_signature = None
        if cache_entries:
            first_entry = cache_entries[0]
            audio_format = first_entry.get("audio_format", {}) or {}
            first_audio_signature = {
                "path": first_entry.get("path"),
                "sample_rate": audio_format.get("sample_rate"),
                "frames": audio_format.get("frames"),
                "channels": audio_format.get("channels"),
                "sample_width": audio_format.get("sample_width"),
            }

        payload = {
            "version": _METADATA_CACHE_VERSION,
            "dataset": dataset_key,
            "root_path": root_path_str,
            "metadata_path": str(metadata_path_obj) if metadata_path_obj else None,
            "metadata_info": metadata_info,
            "entries": cache_entries,
            "first_audio_signature": first_audio_signature,
        }

        try:
            with cache_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
        except OSError as exc:
            logger.warning(f"Failed to write metadata cache {cache_file}: {exc}")
        else:
            logger.info(
                "Stored %d metadata entries for '%s' in cache %s",
                len(prepared_list),
                dataset_key,
                cache_file,
            )

    return prepared_list, durations


def sort_prepared_data_list(data_list, durations):
    if len(data_list) != len(durations):
        raise ValueError("data_list and durations must have the same length")

    paired = sorted(zip(durations, data_list), key=lambda x: x[0])
    sorted_list = [item for _, item in paired]
    sorted_durations = [duration for duration, _ in paired]
    return sorted_list, sorted_durations


def sort_data_list_by_duration(raw_data_list=None, root_path="", precomputed=None):
    """
    Sort metadata entries by ascending audio duration.

    Args:
        raw_data_list: Iterable with raw metadata lines. Optional when
            ``precomputed`` is provided.
        root_path: Base directory of the audio files (used when parsing the
            raw metadata).
        precomputed: Optional tuple ``(prepared_list, durations)`` as returned
            by :func:`prepare_data_list`.

    Returns:
        A tuple ``(sorted_list, sorted_durations)``.
    """
    if precomputed is not None:
        data_list, durations = precomputed
    else:
        if raw_data_list is None:
            raise ValueError("Either raw_data_list or precomputed must be provided")
        data_list, durations = prepare_data_list(raw_data_list, root_path=root_path)

    return sort_prepared_data_list(data_list, durations)

def cfg_get_nested(cfg: dict, path, default=None, sep="."):
    """
    Get a nested value from a dict using a list of keys or a dot-separated string.

    Examples:
        cfg_get_nested(config, ["model_params", "input_dim"], 80)
        cfg_get_nested(config, "model_params.input_dim", 80)
        cfg_get_nested(config, "top_key", 80)
    """
    if isinstance(path, str):
        keys = path.split(sep)
    else:
        keys = path

    cur = cfg
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

class EarlyStoppingWithNoLearningRate:
    def __init__(self, patience=5):
        self.patience = patience  # Number of epochs to wait for improvement
        self.counter = 0
        self.stop_training = False

    def __call__(self, value):
        if round( value, 5 ) <= 0.0000:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True

        return self.stop_training


@click.command()
@click.option('-p', '--config_path', default='./Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    desired_device = str(cfg_get_nested(config, 'device', 'cpu')).lower()
    accelerator_kwargs = {}
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator_kwargs['kwargs_handlers'] = [ddp_kwargs]
    # ``split_batches`` defaults to ``True`` which would further subdivide each
    # dataloader batch across data-parallel workers.  The training scripts
    # already size batches per device, so keep them intact by disabling that
    # behaviour explicitly.
    accelerator_kwargs['split_batches'] = False
    if desired_device.startswith('cpu'):
        accelerator_kwargs['cpu'] = True

    accelerator = Accelerator(**accelerator_kwargs)
    print_fn = accelerator.print

    print_fn(f"Loading config data from: {config_path}")

    log_dir = cfg_get_nested(config, 'log_dir', 'Checkpoint')
    print_fn(f"Using logs and models folder: {log_dir}")

    if accelerator.is_main_process:
        if not osp.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")
    else:
        writer = _DummySummaryWriter()

    if accelerator.is_main_process:
        file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
        logger.addHandler(file_handler)

    configured_global_batch_size = int(cfg_get_nested(config, 'batch_size', 10))
    epochs = cfg_get_nested(config, 'epochs', 200)
    save_freq = cfg_get_nested(config, 'save_freq', 10)
    train_path = cfg_get_nested(config, 'train_data', None)
    val_path = cfg_get_nested(config, 'val_data', None)
    enable_early_stopping = cfg_get_nested(config, 'enable_early_stopping', True)

    device = accelerator.device
    device_str = str(device)

    data_parallel_world_size = max(1, int(accelerator.num_processes))
    grad_accumulation_steps = max(1, int(getattr(accelerator, 'gradient_accumulation_steps', 1)))
    effective_batch_factor = max(1, data_parallel_world_size * grad_accumulation_steps)
    batch_size = max(1, int(math.ceil(configured_global_batch_size / float(effective_batch_factor))))
    effective_global_batch_size = batch_size * effective_batch_factor

    if accelerator.is_main_process:
        if data_parallel_world_size > 1 or grad_accumulation_steps > 1:
            print_fn(
                "Using %d data parallel process(es) with %d gradient accumulation step(s)."
                % (data_parallel_world_size, grad_accumulation_steps)
            )
        if effective_global_batch_size != configured_global_batch_size:
            print_fn(
                "Configured global batch size %d mapped to per-device batch size %d for an effective global size of %d."
                % (
                    configured_global_batch_size,
                    batch_size,
                    effective_global_batch_size,
                )
            )

    dataset_params = {
        'dict_path': cfg_get_nested(config, 'phoneme_maps_path', 'Data/word_index_dict.txt'),
        'sr': cfg_get_nested(config, 'preprocess_params.sr', 24000),
        'spect_params': cfg_get_nested(config, 'preprocess_params.spect_params', {
            'n_fft': 1024,
            'win_length': 1024,
            'hop_length': 300
        }),
        'mel_params': cfg_get_nested(config, 'preprocess_params.mel_params', {'n_mels': 80})
    }

    phoneme_dict_config = cfg_get_nested(config, 'phoneme_dictionary', {}) or {}
    dataset_params['phoneme_dictionary_config'] = phoneme_dict_config

    mel_cache_config = cfg_get_nested(config, 'mel_cache', {}) or {}
    dataset_params['mel_cache'] = mel_cache_config

    dataset_additional_params = cfg_get_nested(config, 'dataset_params', {})
    if isinstance(dataset_additional_params, dict):
        for override_key in ('dict_path', 'sr', 'spect_params', 'mel_params', 'phoneme_dictionary_config'):
            if override_key in dataset_additional_params:
                dataset_params[override_key] = dataset_additional_params[override_key]

        if 'spec_augment' in dataset_additional_params:
            dataset_params['spec_augment_params'] = dataset_additional_params['spec_augment']

        for override_key, override_value in dataset_additional_params.items():
            if override_key in ('dict_path', 'sr', 'spect_params', 'mel_params', 'spec_augment'):
                continue
            dataset_params[override_key] = override_value

    metadata_cache_config = cfg_get_nested(config, 'metadata_cache', {}) or {}
    (
        raw_train_list,
        raw_val_list,
        train_metadata_path,
        val_metadata_path,
    ) = get_data_path_list(train_path, val_path, return_paths=True)

    train_entries, train_durations = prepare_data_list(
        raw_train_list,
        root_path="",
        cache_config=metadata_cache_config,
        metadata_path=train_metadata_path,
        dataset_name="train",
    )
    val_entries, val_durations = prepare_data_list(
        raw_val_list,
        root_path="",
        cache_config=metadata_cache_config,
        metadata_path=val_metadata_path,
        dataset_name="val",
    )
    num_train_items = len(train_entries)

    train_list_sorted, _ = sort_data_list_by_duration(precomputed=(train_entries, train_durations))
    val_list_sorted, _ = sort_data_list_by_duration(precomputed=(val_entries, val_durations))

    dataloader_params = cfg_get_nested(config, 'dataloader_params', {})
    train_num_workers = int(dataloader_params.get('train_num_workers', 8))
    val_num_workers = int(dataloader_params.get('val_num_workers', 2))
    train_bucket_sampler_config = dataloader_params.get('train_bucket_sampler', {})

    base_eval_global_batch_size = int(
        cfg_get_nested(config, 'eval_params.batch_size', configured_global_batch_size)
    )
    base_eval_batch_size = max(
        1, int(math.ceil(base_eval_global_batch_size / float(data_parallel_world_size)))
    )
    effective_eval_global_batch_size = base_eval_batch_size * data_parallel_world_size
    if accelerator.is_main_process and effective_eval_global_batch_size != base_eval_global_batch_size:
        print_fn(
            "Configured evaluation batch size %d mapped to per-device batch size %d for an effective global size of %d."
            % (
                base_eval_global_batch_size,
                base_eval_batch_size,
                effective_eval_global_batch_size,
            )
        )

    curriculum_batch_cfg = cfg_get_nested(config, 'training_curriculum.batch_size_schedule', {}) or {}

    def _scale_curriculum_batch_sizes(schedule_cfg, divisor):
        if not schedule_cfg or divisor <= 1:
            return schedule_cfg

        scaled_cfg = copy.deepcopy(schedule_cfg)

        def _scale_value(value):
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return value
            return max(1, int(math.ceil(numeric / float(divisor))))

        for key in ('base_batch_size', 'initial_batch_size'):
            if key in scaled_cfg:
                scaled_cfg[key] = _scale_value(scaled_cfg[key])

        strategies = scaled_cfg.get('strategies')
        if isinstance(strategies, dict):
            milestone_cfg = strategies.get('milestones')
            if isinstance(milestone_cfg, dict):
                schedule = milestone_cfg.get('schedule')
                if isinstance(schedule, dict):
                    for sched_key, sched_value in list(schedule.items()):
                        schedule[sched_key] = _scale_value(sched_value)
                elif isinstance(schedule, list):
                    new_schedule = []
                    for entry in schedule:
                        if isinstance(entry, dict):
                            if 'batch_size' in entry:
                                entry['batch_size'] = _scale_value(entry['batch_size'])
                            if 'value' in entry:
                                entry['value'] = _scale_value(entry['value'])
                            new_schedule.append(entry)
                        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                            entry_list = list(entry)
                            entry_list[1] = _scale_value(entry_list[1])
                            if isinstance(entry, tuple):
                                new_schedule.append(tuple(entry_list))
                            else:
                                new_schedule.append(entry_list)
                        else:
                            new_schedule.append(entry)
                    milestone_cfg['schedule'] = new_schedule

            linear_cfg = strategies.get('linear')
            if isinstance(linear_cfg, dict):
                for key in ('start_batch_size', 'end_batch_size'):
                    if key in linear_cfg:
                        linear_cfg[key] = _scale_value(linear_cfg[key])

        return scaled_cfg

    scaled_curriculum_cfg = _scale_curriculum_batch_sizes(curriculum_batch_cfg, effective_batch_factor)
    batch_scheduler = BatchSizeScheduler(scaled_curriculum_cfg, default_batch_size=batch_size, total_epochs=epochs)

    def _curriculum_summary(scale_factor):
        transitions = []
        last_value = None
        for epoch_idx in range(1, epochs + 1):
            value = batch_scheduler.batch_size_for_epoch(epoch_idx)
            if value != last_value:
                transitions.append((epoch_idx, value))
                last_value = value
        if not transitions:
            return ''

        parts = []
        for epoch_idx, value in transitions:
            if scale_factor == 1:
                parts.append(f"{epoch_idx}\u2192{value}")
            else:
                parts.append(f"{epoch_idx}\u2192{value * scale_factor} (per-device {value})")
        return ', '.join(parts)

    if accelerator.is_main_process:
        summary = _curriculum_summary(effective_batch_factor)
        if batch_scheduler.enabled:
            print_fn(f"[Curriculum] Batch size schedule: {summary}")
        elif summary:
            print_fn(f"[Curriculum] Static batch size configured: {summary}")

    collate_config = {'return_speaker_ids': True}

    def _prepare_dataloaders(sorted_train, shuffled_train, sorted_val, shuffled_val):
        dataloaders = [sorted_train, shuffled_train, sorted_val, shuffled_val]
        to_prepare = [dl for dl in dataloaders if dl is not None]
        if not to_prepare:
            return tuple(dataloaders)

        prepared = accelerator.prepare(*to_prepare)
        if len(to_prepare) == 1:
            prepared = (prepared,)
        iterator = iter(prepared)
        prepared_loaders = []
        for dl in dataloaders:
            if dl is None:
                prepared_loaders.append(None)
            else:
                prepared_loaders.append(next(iterator))
        return tuple(prepared_loaders)

    def _build_dataloaders_for_batch(current_batch_size):
        current_batch_size = int(current_batch_size)
        validation_batch_size = int(current_batch_size if batch_scheduler.apply_to_validation else base_eval_batch_size)

        sorted_train_loader = build_dataloader(
            train_list_sorted,
            batch_size=current_batch_size,
            num_workers=train_num_workers,
            dataset_config=dataset_params,
            device=device_str,
            collate_config=collate_config,
            dataset_name="train")

        shuffled_train_loader = build_dataloader(
            train_entries,
            batch_size=current_batch_size,
            num_workers=train_num_workers,
            dataset_config=dataset_params,
            device=device_str,
            lengths=train_durations,
            bucket_sampler_config=train_bucket_sampler_config,
            collate_config=collate_config,
            dataset_name="train")

        sorted_val_loader = build_dataloader(
            val_list_sorted,
            batch_size=validation_batch_size,
            validation=True,
            num_workers=val_num_workers,
            device=device_str,
            dataset_config=dataset_params,
            collate_config=collate_config,
            dataset_name="val")

        shuffled_val_loader = build_dataloader(
            val_entries,
            batch_size=validation_batch_size,
            validation=True,
            num_workers=val_num_workers,
            device=device_str,
            dataset_config=dataset_params,
            collate_config=collate_config,
            dataset_name="val")

        raw_steps = len(sorted_train_loader) if sorted_train_loader is not None else len(shuffled_train_loader)
        raw_steps = int(raw_steps)
        if raw_steps <= 0:
            raise ValueError(
                f"Curriculum batch size {current_batch_size} is too large for the training set and would produce no steps."
            )

        prepared_sorted_train, prepared_shuffled_train, prepared_sorted_val, prepared_shuffled_val = _prepare_dataloaders(
            sorted_train_loader,
            shuffled_train_loader,
            sorted_val_loader,
            shuffled_val_loader,
        )

        prepared_train_loader = (
            prepared_sorted_train if prepared_sorted_train is not None else prepared_shuffled_train
        )
        per_rank_steps = raw_steps
        inferred_world_size = 1

        if prepared_train_loader is not None:
            try:
                prepared_steps = int(len(prepared_train_loader))
            except TypeError:
                prepared_steps = None

            if prepared_steps is not None and prepared_steps > 0:
                max_prepared_steps = prepared_steps
                if data_parallel_world_size > 1:
                    steps_tensor = torch.tensor(
                        [prepared_steps],
                        device=accelerator.device,
                        dtype=torch.long,
                    )
                    gathered_steps = accelerator.gather(steps_tensor)
                    if gathered_steps is not None and gathered_steps.numel() > 0:
                        max_prepared_steps = int(gathered_steps.max().item())
                        min_prepared_steps = int(gathered_steps.min().item())
                        if (
                            accelerator.is_main_process
                            and max_prepared_steps != min_prepared_steps
                        ):
                            print_fn(
                                "[Scheduler] Detected uneven per-rank dataloader lengths (%d vs %d). "
                                "Planning the LR schedule with the maximum to avoid early decay."
                                % (min_prepared_steps, max_prepared_steps)
                            )

                planned_steps = max(raw_steps, max_prepared_steps)
                per_rank_steps = planned_steps
                if (
                    accelerator.is_main_process
                    and max_prepared_steps < raw_steps
                ):
                    print_fn(
                        "[Scheduler] Detected that accelerate split dataloader batches "
                        "(%d -> %d steps per rank). Using the unsplit length for LR "
                        "planning; consider launching with split_batches=False if this "
                        "was unintended."
                        % (raw_steps, max_prepared_steps)
                    )
                if raw_steps > 0 and max_prepared_steps > 0:
                    inferred = int(round(raw_steps / float(max_prepared_steps)))
                    inferred_world_size = max(
                        1, min(data_parallel_world_size, inferred)
                    )
            else:
                per_rank_steps = raw_steps
        else:
            per_rank_steps = raw_steps

        per_rank_steps = max(1, int(per_rank_steps))
        dataset_items = max(len(train_entries), len(train_list_sorted))
        per_rank_dataset_estimate = max(
            1,
            int(
                math.ceil(
                    dataset_items
                    / float(max(1, inferred_world_size))
                )
            ),
        )
        per_rank_samples = max(per_rank_steps * current_batch_size, per_rank_dataset_estimate)

        return (
            prepared_sorted_train,
            prepared_shuffled_train,
            prepared_sorted_val,
            prepared_shuffled_val,
            per_rank_steps,
            per_rank_samples,
            inferred_world_size,
        )

    initial_batch_size = batch_scheduler.batch_size_for_epoch(1)
    (
        sorted_train_dataloader,
        shuffled_train_dataloader,
        sorted_val_dataloader,
        shuffled_val_dataloader,
        steps_per_epoch,
        train_samples_per_rank,
        inferred_scheduler_world_size,
    ) = _build_dataloaders_for_batch(initial_batch_size)

    word_indexes = set(
        line.strip() for line in open(cfg_get_nested(config, 'phoneme_maps_path', 'Data/word_index_dict.txt'))
        if line.strip()
    )

    model_params = cfg_get_nested(config, 'model_params', {
        'input_dim': 80,
        'hidden_dim': 256,
        'n_token': len(word_indexes),
        'token_embedding_dim': 512,
        'n_layers': 5,
        'location_kernel_size': 31
    })

    if 'n_token' not in model_params:
        model_params['n_token'] = len(word_indexes)

    multi_task_config = cfg_get_nested(config, 'multi_task', {}) or {}

    speaker_cfg = multi_task_config.get('speaker', {}) or {}
    if speaker_cfg.get('enabled', False) and int(speaker_cfg.get('num_speakers', 0)) <= 0:
        speaker_ids = set()
        for entry in train_entries + val_entries:
            if len(entry) >= 3:
                speaker_id = str(entry[2]).strip()
                if speaker_id:
                    speaker_ids.add(speaker_id)
        inferred = max(1, len(speaker_ids))
        print_fn(f"Inferred {inferred} unique speaker id(s) from metadata")
        speaker_cfg = dict(speaker_cfg)
        speaker_cfg['num_speakers'] = inferred
        multi_task_config['speaker'] = speaker_cfg

    stabilization_config = cfg_get_nested(config, 'stabilization', {}) or {}
    memory_optimization_config = cfg_get_nested(config, 'memory_optimizations', {}) or {}

    model_params = dict(model_params)
    model_params['multi_task_config'] = multi_task_config
    model_params['stabilization_config'] = stabilization_config
    model_params['memory_optimization_config'] = memory_optimization_config

    print_fn("Using model parameters:", model_params)

    model = build_model(model_params=model_params)

    if accelerator.is_main_process and data_parallel_world_size > 1 and inferred_scheduler_world_size < data_parallel_world_size:
        print_fn(
            "[Scheduler] Detected that the prepared dataloader does not shard the dataset across all %d workers. "
            "Estimating scheduler steps with an effective world size of %d to avoid premature LR decay." % (
                data_parallel_world_size,
                inferred_scheduler_world_size,
            )
        )

    samples_per_rank = max(1, int(train_samples_per_rank))
    per_epoch_optimizer_steps = max(
        1, int(math.ceil(steps_per_epoch / float(grad_accumulation_steps)))
    )

    total_training_steps = 0
    for epoch_idx in range(1, epochs + 1):
        per_device_batch = max(1, int(batch_scheduler.batch_size_for_epoch(epoch_idx)))
        epoch_step_estimate = int(math.ceil(samples_per_rank / float(per_device_batch)))
        if grad_accumulation_steps > 1:
            epoch_step_estimate = int(
                math.ceil(epoch_step_estimate / float(grad_accumulation_steps))
            )
        total_training_steps += max(1, epoch_step_estimate)

    if accelerator.is_main_process:
        print_fn(
            "[Scheduler] Planning %d optimiser steps (~%d per epoch) for OneCycleLR"
            % (total_training_steps, per_epoch_optimizer_steps)
        )
    scheduler_params = {
        'max_lr': float(cfg_get_nested(config, 'optimizer_params.lr', 5e-4)),
        'pct_start': float(cfg_get_nested(config, 'optimizer_params.pct_start', 0.1)),
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'per_rank_samples': train_samples_per_rank,
        'effective_world_size': inferred_scheduler_world_size,
        'gradient_accumulation_steps': grad_accumulation_steps,
        'optimizer_steps_per_epoch': per_epoch_optimizer_steps,
        'total_steps': total_training_steps,
    }
    config['scheduler_params'] = scheduler_params

    entropy_params = cfg_get_nested(config, 'entropy_params', {"label_smoothing": 0.1})

    model.to(device)
    optimizer, scheduler = build_optimizer(
        {"params": model.parameters(), "optimizer_params": {}, "scheduler_params": scheduler_params})

    blank_index = sorted_train_dataloader.dataset.text_cleaner.word_index_dictionary[" "]
    criterion = build_criterion(critic_params={
                'ctc': {'blank': blank_index, 'reduction': 'none', 'zero_infinity': True},
        }, entropy_params=entropy_params, multi_task_config=multi_task_config)

    ctc_loss_config = cfg_get_nested(config, 'ctc_loss', {}) or {}
    if not isinstance(ctc_loss_config, dict):
        ctc_loss_config = {}
    ctc_blank_bias = float(ctc_loss_config.get('blank_logit_bias', 0.0))
    ctc_logit_temperature = float(ctc_loss_config.get('logit_temperature', 1.0))
    if ctc_logit_temperature <= 0:
        ctc_logit_temperature = 1.0
    ctc_regularization_config = ctc_loss_config.get('regularization', {}) or {}
    if not isinstance(ctc_regularization_config, dict):
        ctc_regularization_config = {}
    ctc_blank_scale_config = ctc_loss_config.get('blank_scale', None)
    if isinstance(ctc_blank_scale_config, dict) and not ctc_blank_scale_config:
        ctc_blank_scale_config = None

    if enable_early_stopping:
        patience = max([3, int(math.floor(int(cfg_get_nested(config, 'save_freq', 10)) / 2))])
        early_stopping = EarlyStoppingWithNoLearningRate(patience=patience)
    else:
        early_stopping = None

    loss_weight_config = cfg_get_nested(config, 'loss_weights', {}) or {}
    regularization_config = cfg_get_nested(config, 'regularization', {}) or {}
    entropy_regularization_config = cfg_get_nested(regularization_config, 'entropy', {}) or {}
    use_ctc = bool(multi_task_config.get('use_ctc', True))
    use_s2s = bool(multi_task_config.get('use_seq2seq', True))
    frame_cfg = multi_task_config.get('frame_phoneme', {}) or {}
    speaker_cfg = multi_task_config.get('speaker', {}) or {}
    pron_cfg = multi_task_config.get('pronunciation_error', {}) or {}

    ctc_weight = float(loss_weight_config.get('ctc', 1.0 if use_ctc else 0.0)) if use_ctc else 0.0
    s2s_weight = float(loss_weight_config.get('s2s', 1.0 if use_s2s else 0.0)) if use_s2s else 0.0
    frame_weight = float(loss_weight_config.get('frame_phoneme', 0.0)) if frame_cfg.get('enabled', False) else 0.0
    speaker_weight = float(loss_weight_config.get('speaker', 0.0)) if speaker_cfg.get('enabled', False) else 0.0
    pron_weight = float(loss_weight_config.get('pronunciation_error', 0.0)) if pron_cfg.get('enabled', False) else 0.0
    mixspeech_config = stabilization_config.get('mix_speech', {}) or {}
    intermediate_ctc_config = stabilization_config.get('intermediate_ctc', {}) or {}
    if isinstance(intermediate_ctc_config, dict) and 'loss_weight' not in intermediate_ctc_config:
        intermediate_ctc_config = dict(intermediate_ctc_config)
        intermediate_ctc_config['loss_weight'] = float(loss_weight_config.get('intermediate_ctc', 0.0))
    self_conditioned_ctc_config = stabilization_config.get('self_conditioned_ctc', {}) or {}
    if isinstance(self_conditioned_ctc_config, dict) and 'loss_weight' not in self_conditioned_ctc_config:
        self_conditioned_ctc_config = dict(self_conditioned_ctc_config)
        self_conditioned_ctc_config['loss_weight'] = float(loss_weight_config.get('self_conditioned_ctc', 0.0))

    if sorted_train_dataloader is not None:
        try:
            local_steps = int(len(sorted_train_dataloader))
        except TypeError:
            local_steps = None
    else:
        try:
            local_steps = int(len(shuffled_train_dataloader))
        except TypeError:
            local_steps = None

    if local_steps is not None and local_steps > 0:
        steps_per_epoch = max(int(steps_per_epoch), local_steps)

    to_prepare = [model, optimizer]
    scheduler_included = scheduler is not None
    if scheduler_included:
        to_prepare.append(scheduler)

    prepared_objects = accelerator.prepare(*to_prepare)
    if scheduler_included:
        model, optimizer, scheduler = prepared_objects
    else:
        model, optimizer = prepared_objects

    trainer = Trainer(model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    device=device,
                    accelerator=accelerator,
                    sorted_train_dataloader=sorted_train_dataloader,
                    shuffled_train_dataloader=shuffled_train_dataloader,
                    sorted_val_dataloader=sorted_val_dataloader,
                    shuffled_val_dataloader=shuffled_val_dataloader,
                    logger=logger,
                    switch_sortagrad_dataset_epoch=cfg_get_nested(config, 'sortagrad_switch_to_shuffled_dataset_epoch', 10),
                    use_diagonal_attention_prior=(cfg_get_nested(config, 'use_diagonal_attention_prior', True) and use_s2s),
                    diagonal_attention_prior_weight=cfg_get_nested(config, 'diagonal_attention_prior_weight', 0.1),
                    diagonal_attention_prior_sigma=cfg_get_nested(config, 'diagonal_attention_prior_sigma', 0.5),
                    ctc_weight=ctc_weight,
                    s2s_weight=s2s_weight,
                    frame_weight=frame_weight,
                    speaker_weight=speaker_weight,
                    pron_error_weight=pron_weight,
                    ctc_blank_id=blank_index,
                    ctc_logit_bias=ctc_blank_bias,
                    ctc_logit_temperature=ctc_logit_temperature,
                    ctc_regularization_config=ctc_regularization_config,
                    ctc_blank_scale_config=ctc_blank_scale_config,
                    enable_frame_classifier=frame_cfg.get('enabled', False),
                    enable_speaker=speaker_cfg.get('enabled', False),
                    enable_pronunciation_error=pron_cfg.get('enabled', False),
                    mixspeech_config=mixspeech_config,
                    intermediate_ctc_config=intermediate_ctc_config,
                    self_conditioned_ctc_config=self_conditioned_ctc_config,
                    entropy_regularization_config=entropy_regularization_config,
                    memory_optimization_config=memory_optimization_config,
                    steps_per_epoch=steps_per_epoch,
                    total_epochs=epochs
                    )

    current_train_batch_size = int(initial_batch_size)
    current_samples_per_rank = int(train_samples_per_rank)

    checkpoint_selection_cfg = cfg_get_nested(config, 'checkpoint_selection', {}) or {}
    joint_selection_enabled = bool(checkpoint_selection_cfg.get('enabled', False))
    joint_lambda_diag = float(checkpoint_selection_cfg.get('lambda_diag', 0.0))
    joint_lambda_length = float(checkpoint_selection_cfg.get('lambda_length', 0.0))
    joint_target_len_diff = float(checkpoint_selection_cfg.get('target_length_diff', 0.0))

    length_penalty_mode = str(
        checkpoint_selection_cfg.get('length_penalty_mode', 'signed_normalized')
    ).lower()
    if length_penalty_mode not in {
        'signed_normalized',
        'absolute',
        'absolute_normalized',
    }:
        length_penalty_mode = 'signed_normalized'

    lambda_length_decay_cfg = checkpoint_selection_cfg.get('lambda_length_decay', {}) or {}
    if not isinstance(lambda_length_decay_cfg, dict):
        lambda_length_decay_cfg = {}

    lambda_length_decay_enabled = bool(
        lambda_length_decay_cfg.get('enabled', True)
    ) and joint_lambda_length > 0.0
    lambda_length_decay_target = float(lambda_length_decay_cfg.get('target', 2.0))
    lambda_length_decay_tolerance = float(lambda_length_decay_cfg.get('tolerance', 0.05))
    lambda_length_decay_confirmations = max(
        1, int(lambda_length_decay_cfg.get('confirmations', 2))
    )
    lambda_length_decay_ratio = float(lambda_length_decay_cfg.get('target_ratio', 0.6))
    lambda_length_decay_ratio = min(max(lambda_length_decay_ratio, 0.0), 1.0)
    lambda_length_decay_span = float(lambda_length_decay_cfg.get('span_fraction', 0.12))
    lambda_length_decay_span = min(max(lambda_length_decay_span, 0.0), 1.0)
    lambda_length_min_value = float(lambda_length_decay_cfg.get('min_value', 0.0))
    lambda_length_allow_retrigger = bool(lambda_length_decay_cfg.get('allow_retrigger', False))

    lambda_length_state = {
        'enabled': lambda_length_decay_enabled,
        'confirmations': 0,
        'triggered': False,
        'completed': False,
        'start_step': 0,
        'end_step': 0,
        'start_value': joint_lambda_length,
        'target_value': joint_lambda_length * lambda_length_decay_ratio,
        'current_value': joint_lambda_length,
    }

    grid_lambda_values = checkpoint_selection_cfg.get('lambda_length_grid', []) or []
    grid_delta_values = checkpoint_selection_cfg.get('target_length_diff_grid', []) or []
    if not isinstance(grid_lambda_values, (list, tuple)):
        grid_lambda_values = []
    if not isinstance(grid_delta_values, (list, tuple)):
        grid_delta_values = []

    grid_lambda_values = sorted({float(value) for value in grid_lambda_values})
    grid_delta_values = sorted({float(value) for value in grid_delta_values})
    grid_enabled = joint_selection_enabled and grid_lambda_values and grid_delta_values
    grid_best_scores = {}
    grid_best_paths = {}

    def _compute_length_penalty_value(len_diff, len_diff_norm, delta, mode):
        if delta is None:
            return None
        try:
            delta_value = float(delta)
        except (TypeError, ValueError):
            return None

        if mode == 'absolute':
            if len_diff is None:
                return None
            return max(0.0, abs(float(len_diff)) - delta_value)
        if mode == 'absolute_normalized':
            if len_diff_norm is None:
                return None
            return max(0.0, abs(float(len_diff_norm)) - delta_value)

        if len_diff_norm is None:
            return None
        return max(0.0, delta_value - float(len_diff_norm))

    def _maybe_update_lambda_length(metric_value, current_step):
        nonlocal joint_lambda_length
        if not lambda_length_state['enabled']:
            return False

        if lambda_length_state['completed'] and not lambda_length_allow_retrigger:
            return False

        updated = False

        if lambda_length_state['triggered']:
            span = max(1, lambda_length_state['end_step'] - lambda_length_state['start_step'])
            progress = (current_step - lambda_length_state['start_step']) / float(span)
            progress = min(max(progress, 0.0), 1.0)
            start = lambda_length_state['start_value']
            target = lambda_length_state['target_value']
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            value = target + (start - target) * cosine
            value = max(lambda_length_min_value, float(value))
            lambda_length_state['current_value'] = value
            joint_lambda_length = value
            updated = True
            if progress >= 1.0 - 1e-6:
                lambda_length_state['triggered'] = False
                lambda_length_state['completed'] = True
            return updated

        if metric_value is None or not math.isfinite(metric_value):
            lambda_length_state['confirmations'] = 0
            return updated

        if abs(metric_value - lambda_length_decay_target) <= lambda_length_decay_tolerance:
            lambda_length_state['confirmations'] += 1
        else:
            lambda_length_state['confirmations'] = 0

        if lambda_length_state['confirmations'] < lambda_length_decay_confirmations:
            return updated

        lambda_length_state['confirmations'] = 0
        lambda_length_state['triggered'] = True
        lambda_length_state['start_step'] = current_step
        total_steps = max(1, int(scheduler_params.get('total_steps', 0)))
        if total_steps <= 0:
            total_steps = max(1, trainer._get_optimizer_step_count())
        span_steps = max(1, int(round(lambda_length_decay_span * total_steps)))
        lambda_length_state['end_step'] = current_step + span_steps
        lambda_length_state['start_value'] = lambda_length_state['current_value']
        lambda_length_state['target_value'] = max(
            lambda_length_min_value,
            lambda_length_state['start_value'] * lambda_length_decay_ratio,
        )
        updated = True
        return updated

    best_joint_score = None
    best_checkpoint_path = None

    def _ensure_curriculum_for_epoch(epoch):
        nonlocal sorted_train_dataloader
        nonlocal shuffled_train_dataloader
        nonlocal sorted_val_dataloader
        nonlocal shuffled_val_dataloader
        nonlocal steps_per_epoch
        nonlocal current_train_batch_size
        nonlocal current_samples_per_rank

        desired_batch_size = int(batch_scheduler.batch_size_for_epoch(epoch))
        if desired_batch_size != current_train_batch_size:
            (
                sorted_train_dataloader,
                shuffled_train_dataloader,
                sorted_val_dataloader,
                shuffled_val_dataloader,
                steps_per_epoch,
                current_samples_per_rank,
                _,
            ) = _build_dataloaders_for_batch(desired_batch_size)

            trainer.update_dataloaders(
                sorted_train=sorted_train_dataloader,
                shuffled_train=shuffled_train_dataloader,
                sorted_val=sorted_val_dataloader,
                shuffled_val=shuffled_val_dataloader,
                steps_per_epoch=steps_per_epoch,
            )
            current_train_batch_size = desired_batch_size
            effective_value = desired_batch_size * effective_batch_factor
            if effective_batch_factor == 1:
                print_fn(f"[Curriculum] Epoch {epoch}: updated batch size to {effective_value}")
            else:
                print_fn(
                    f"[Curriculum] Epoch {epoch}: updated batch size to {effective_value} (per-device {desired_batch_size})"
                )
        else:
            trainer.update_dataloaders(steps_per_epoch=steps_per_epoch)

    def _curriculum_completed_steps(num_epochs_completed):
        completed = 0
        total_epochs_completed = max(0, int(num_epochs_completed))
        samples_per_rank = max(1, int(current_samples_per_rank))
        for epoch_idx in range(1, total_epochs_completed + 1):
            batch_sz = max(1, batch_scheduler.batch_size_for_epoch(epoch_idx))
            completed += max(1, int(math.ceil(samples_per_rank / float(batch_sz))))
        return completed

    pretrained_model = cfg_get_nested(config, 'pretrained_model', '')
    load_only_params = cfg_get_nested(config, 'load_only_params', False)
    if isinstance(pretrained_model, bool) and pretrained_model is True:
        try:
            ckpts = [f for f in os.listdir(log_dir) if f.startswith("epoch_") and f.endswith(".pth")]
            if ckpts:
                iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
                iters = sorted(iters)[-1]
                checkpoint_file = osp.join(log_dir, f"epoch_{iters:05}.pth")
                print_fn(f"Starting to train from checkpoint {checkpoint_file}")
                start_epoch = trainer.load_checkpoint(checkpoint_file, load_only_params=load_only_params)
            else:
                print_fn("No previous checkpoints found, starting training from epoch 1.")
                start_epoch = 1
        except Exception as e:
            print_fn(f"Failed to load latest checkpoint, starting training from epoch 1 - {e}")
            start_epoch = 1
    elif isinstance(pretrained_model, str) and pretrained_model != "":
        start_epoch = trainer.load_checkpoint(pretrained_model, load_only_params=load_only_params)
        start_epoch += 1
        print_fn(f"Checkpoint {pretrained_model} loaded, starting training from epoch {start_epoch}.")
    elif (isinstance(pretrained_model, str) and pretrained_model == "") or (
        isinstance(pretrained_model, bool) and pretrained_model is False
    ):
        print_fn("Starting training from epoch 1.")
        start_epoch = 1
    else:
        print_fn(f"Unrecognized value for load_checkpoint config option, starting training from epoch 1 - {pretrained_model}")
        start_epoch = 1

    _ensure_curriculum_for_epoch(start_epoch)
    if getattr(trainer, "_resumed_from_checkpoint", False):
        trainer.handle_sortagrad_after_resume()
        completed_steps = _curriculum_completed_steps(trainer.epochs)
        trainer.sync_scheduler_to_progress(completed_steps=completed_steps)

    for epoch in range(start_epoch, epochs + 1):
        _ensure_curriculum_for_epoch(epoch)
        train_results = trainer._train_epoch()
        eval_results = trainer._eval_epoch()

        learning_rate = train_results.get('train/learning_rate', None)

        if learning_rate is None:
            raise Exception("learning_rate not found in training results! Please check the metric calculation.")

        results = train_results.copy()
        results.update(eval_results)

        p50_metric = results.get('diagnostics/attn_duration_p50')
        current_optimizer_step = trainer._get_optimizer_step_count()
        lambda_updated = _maybe_update_lambda_length(p50_metric, current_optimizer_step)
        if lambda_updated or lambda_length_state['triggered']:
            joint_lambda_length = lambda_length_state['current_value']
        results['eval/checkpoint_lambda_length'] = joint_lambda_length
        results['eval/checkpoint_lambda_length_decay_active'] = float(
            1.0 if lambda_length_state['triggered'] else 0.0
        )
        results['eval/checkpoint_lambda_length_completed'] = float(
            1.0 if lambda_length_state['completed'] else 0.0
        )
        if lambda_length_state['triggered']:
            span = max(1, lambda_length_state['end_step'] - lambda_length_state['start_step'])
            decay_progress = (
                current_optimizer_step - lambda_length_state['start_step']
            ) / float(span)
        elif lambda_length_state['completed']:
            decay_progress = 1.0
        else:
            decay_progress = 0.0
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        results['eval/checkpoint_lambda_length_decay_progress'] = decay_progress

        joint_score = None
        length_penalty = None
        is_best = False
        grid_improved_keys = []
        grid_scores = {}

        if joint_selection_enabled:
            per_metric = results.get('eval/wer')
            diag_metric = results.get('eval/diag_coherence')
            len_diff_norm_metric = results.get('eval/ctc_len_diff_norm')
            len_diff_metric = results.get('eval/ctc_len_diff')
            if per_metric is not None and diag_metric is not None:
                length_penalty = _compute_length_penalty_value(
                    len_diff=len_diff_metric,
                    len_diff_norm=len_diff_norm_metric,
                    delta=joint_target_len_diff,
                    mode=length_penalty_mode,
                )
                if length_penalty is None:
                    length_penalty = 0.0
                joint_score = (
                    per_metric
                    + joint_lambda_diag * (1.0 - diag_metric)
                    + joint_lambda_length * length_penalty
                )
                results['eval/joint_score'] = joint_score
                results['eval/length_penalty'] = length_penalty
                tolerance = 1.0e-6
                if best_joint_score is None or joint_score < best_joint_score - tolerance:
                    is_best = True

                if grid_enabled:
                    penalties_by_delta = {}
                    for delta in grid_delta_values:
                        penalties_by_delta[delta] = _compute_length_penalty_value(
                            len_diff=len_diff_metric,
                            len_diff_norm=len_diff_norm_metric,
                            delta=delta,
                            mode=length_penalty_mode,
                        )

                    for lam, delta in itertools.product(grid_lambda_values, grid_delta_values):
                        penalty_value = penalties_by_delta.get(delta)
                        if penalty_value is None:
                            continue
                        score = per_metric + joint_lambda_diag * (1.0 - diag_metric) + lam * penalty_value
                        key = (lam, delta)
                        grid_scores[key] = score
                        results[
                            f'eval/joint_score_grid/lambda_{lam:.3f}_delta_{delta:.3f}'
                        ] = score
                        best_score = grid_best_scores.get(key)
                        if best_score is None or score < best_score - tolerance:
                            grid_best_scores[key] = score
                            grid_improved_keys.append(key)

                    for lam, delta in itertools.product(grid_lambda_values, grid_delta_values):
                        key = (lam, delta)
                        if key in grid_best_scores:
                            results[
                                f'eval/joint_score_best/lambda_{lam:.3f}_delta_{delta:.3f}'
                            ] = grid_best_scores[key]

        if joint_selection_enabled:
            display_best = best_joint_score
            if is_best and joint_score is not None:
                display_best = joint_score
            if display_best is not None:
                results['eval/best_joint_score'] = display_best

        if accelerator.is_main_process:
            logger.info('--- epoch %d ---' % epoch)
            for key, value in results.items():
                if isinstance(value, float):
                    logger.info('%-15s: %.5f' % (key, value))
                    writer.add_scalar(key, value, epoch)
                else:
                    for v in value:
                        writer.add_figure('eval_attn', plot_image(v), epoch)

        should_trigger_early_stop = False
        if early_stopping is not None:
            should_trigger_early_stop = early_stopping(learning_rate)

        grid_force_save = False
        if grid_enabled:
            results['eval/joint_score_grid_improved'] = float(1.0 if grid_improved_keys else 0.0)
            if grid_improved_keys:
                grid_force_save = bool(
                    checkpoint_selection_cfg.get('save_grid_checkpoints', True)
                )
        else:
            results['eval/joint_score_grid_improved'] = 0.0

        save_checkpoint = (
            (epoch % save_freq) == 0
            or should_trigger_early_stop
            or is_best
            or grid_force_save
        )
        checkpoint_path = None
        if save_checkpoint and accelerator.is_main_process:
            checkpoint_path = osp.join(log_dir, 'epoch_%05d.pth' % epoch)
            trainer.save_checkpoint(checkpoint_path)
            if grid_enabled and grid_improved_keys:
                for key in grid_improved_keys:
                    grid_best_paths[key] = checkpoint_path
        accelerator.wait_for_everyone()

        if joint_selection_enabled and is_best and joint_score is not None:
            best_joint_score = joint_score
            if accelerator.is_main_process and checkpoint_path is not None:
                best_checkpoint_path = checkpoint_path
                best_symlink = osp.join(log_dir, 'best_joint.pth')
                try:
                    if os.path.islink(best_symlink) or os.path.exists(best_symlink):
                        os.remove(best_symlink)
                    os.symlink(os.path.basename(checkpoint_path), best_symlink)
                except OSError:
                    shutil.copyfile(checkpoint_path, best_symlink)

        if joint_selection_enabled and best_joint_score is not None:
            results['eval/best_joint_score'] = best_joint_score

        if should_trigger_early_stop:
            if accelerator.is_main_process:
                logger.info(f"Early stopping triggered at epoch {epoch}, learning_rate: {learning_rate:.5f}")
            break

    return 0

if __name__=="__main__":
    main()
