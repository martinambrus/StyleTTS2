import math
import os
import os.path as osp
import sys
import time
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import numpy as np
import soundfile as sf
import torch
from torch import nn
import jiwer

import matplotlib.pylab as plt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from decoding import build_decoder_from_config
from meldataset import build_dataloader
from models import build_model
from phoneme_dictionary import load_phoneme_dictionary
from token_map import build_token_map_from_data


def select_logits_from_output(model_output, preferred_order=(
    "primary_logits",
    "ctc_logits",
    "s2s_logits",
    "logits",
)):
    """Return the primary logit tensor from a model forward pass.

    Recent multi-task changes make :class:`~models.ASRCNN` return a dictionary
    containing logits for every enabled objective. The utility notebooks were
    written against the previous behaviour where ``model(mels)`` yielded a
    tensor, so helpers that consume model outputs now need a consistent way to
    retrieve the main ASR logits.  This function inspects the output structure
    and returns the first tensor that matches the preferred key order.  If a
    tensor is passed directly it is returned unchanged.

    Args:
        model_output: The object returned by ``model.forward``.
        preferred_order: Sequence of dictionary keys to probe.  The first
            existing tensor value is returned.

    Returns:
        torch.Tensor: The logits tensor suitable for decoding.

    Raises:
        TypeError: If ``model_output`` is neither a tensor nor a mapping.
        KeyError: If no tensor is found for any of the preferred keys.
    """

    if isinstance(model_output, torch.Tensor):
        return model_output

    if isinstance(model_output, dict):
        for key in preferred_order:
            tensor = model_output.get(key)
            if isinstance(tensor, torch.Tensor):
                return tensor
        raise KeyError(
            "Could not find logits in model output. Available keys: "
            + ", ".join(model_output.keys())
        )

    raise TypeError(
        "Expected model output to be a tensor or dict, got "
        f"{type(model_output)!r} instead"
    )


def _cfg_get_nested(cfg: Dict, path, default=None, sep: str = "."):
    """Retrieve a nested value from a dictionary.

    Args:
        cfg: Configuration dictionary to traverse.
        path: Either a dot-separated string or a sequence with the keys that
            should be followed.
        default: Value returned when any of the keys is missing.
        sep: Separator used when ``path`` is provided as a string.

    Returns:
        The resolved value or ``default`` when the path cannot be followed.
    """

    if isinstance(path, str):
        keys = path.split(sep)
    else:
        keys = list(path)

    current = cfg
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def load_asr_model_from_config(config: Dict, model_path: str, device: torch.device):
    """Instantiate and load an ASR model using the training configuration.

    The helper mirrors the logic used by the training entry-point so notebooks
    can easily bootstrap a model for analysis without re-implementing the setup
    steps.

    Args:
        config: Parsed configuration dictionary.
        model_path: Filesystem path to the checkpoint that should be restored.
        device: Target device where the model will be moved.

    Returns:
        Tuple ``(model, token_map)`` containing the loaded model in evaluation
        mode together with the phoneme-to-index mapping used during training.
    """

    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    if not model_path:
        raise ValueError("model_path must be provided")

    checkpoint_path = Path(model_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file '{model_path}' not found")

    dictionary_cfg = _cfg_get_nested(config, "phoneme_dictionary", {}) or {}
    dictionary_path = _cfg_get_nested(config, "phoneme_maps_path", None)

    if dictionary_path and Path(dictionary_path).is_file():
        token_map = load_phoneme_dictionary(dictionary_path, config=dictionary_cfg)
    else:
        train_path = _cfg_get_nested(config, "train_data", None)
        val_path = _cfg_get_nested(config, "val_data", None)
        token_map = build_token_map_from_data(train_path, val_path)

    if not token_map:
        raise RuntimeError("Failed to build phoneme dictionary from configuration")

    inferred_n_token = max(int(idx) for idx in token_map.values()) + 1

    default_model_params = {
        "input_dim": 80,
        "hidden_dim": 256,
        "token_embedding_dim": 512,
        "n_layers": 5,
        "location_kernel_size": 31,
    }
    model_params = dict(_cfg_get_nested(config, "model_params", default_model_params) or default_model_params)
    model_params.setdefault("input_dim", default_model_params["input_dim"])
    model_params.setdefault("hidden_dim", default_model_params["hidden_dim"])
    model_params.setdefault("token_embedding_dim", default_model_params["token_embedding_dim"])
    model_params.setdefault("n_layers", default_model_params["n_layers"])
    model_params.setdefault("location_kernel_size", default_model_params["location_kernel_size"])

    configured_tokens = int(model_params.get("n_token", inferred_n_token))
    model_params["n_token"] = max(configured_tokens, inferred_n_token)

    multi_task_config = dict(_cfg_get_nested(config, "multi_task", {}) or {})
    frame_cfg = multi_task_config.get("frame_phoneme")
    if isinstance(frame_cfg, dict) and frame_cfg.get("enabled", False):
        classes = int(frame_cfg.get("num_classes", 0))
        if classes <= 0:
            frame_cfg = dict(frame_cfg)
            frame_cfg["num_classes"] = model_params["n_token"]
            multi_task_config["frame_phoneme"] = frame_cfg

    model_params["multi_task_config"] = multi_task_config
    model_params["stabilization_config"] = dict(_cfg_get_nested(config, "stabilization", {}) or {})
    model_params["memory_optimization_config"] = dict(_cfg_get_nested(config, "memory_optimizations", {}) or {})

    model = build_model(model_params=model_params)

    checkpoint = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint at '{model_path}' does not contain a valid state dict")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        warnings.warn(f"Missing parameters when loading the checkpoint: {missing_keys}")
    if unexpected_keys:
        warnings.warn(f"Unexpected parameters in checkpoint: {unexpected_keys}")

    model.to(device)
    model.eval()

    return model, token_map


def _parse_metadata_file(path: str) -> List[List[str]]:
    entries: List[List[str]] = []
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Validation metadata file '{path}' not found")

    with metadata_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            cleaned = raw.rstrip("\n")
            if not cleaned.strip():
                continue

            parts = cleaned.split("|")
            if len(parts) < 2:
                raise ValueError(f"Malformed metadata line {line_number} in '{path}': {cleaned!r}")

            wave_path = parts[0].strip()
            if len(parts) == 2:
                text = parts[1]
                speaker_id = ""
            else:
                text = "|".join(parts[1:-1])
                speaker_id = parts[-1].strip()

            entries.append([wave_path, text, speaker_id])

    if not entries:
        raise RuntimeError(f"No entries found in metadata file '{path}'")

    return entries


def build_dev_dataloader_from_config(
    config: Dict,
    device: torch.device,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
):
    """Construct the validation dataloader using configuration defaults."""

    if not isinstance(config, dict):
        raise TypeError("config must be a dictionary")

    val_metadata_path = _cfg_get_nested(config, "val_data", "Data/val_list.txt")
    val_entries = _parse_metadata_file(val_metadata_path)

    dataset_config = {
        "dict_path": _cfg_get_nested(config, "phoneme_maps_path", "Data/word_index_dict.txt"),
        "sr": int(_cfg_get_nested(config, "preprocess_params.sr", 24000)),
        "spect_params": dict(_cfg_get_nested(config, "preprocess_params.spect_params", {
            "n_fft": 1024,
            "win_length": 1024,
            "hop_length": 300,
        }) or {}),
        "mel_params": dict(_cfg_get_nested(config, "preprocess_params.mel_params", {"n_mels": 80}) or {}),
        "phoneme_dictionary_config": _cfg_get_nested(config, "phoneme_dictionary", {}) or {},
        "mel_cache": _cfg_get_nested(config, "mel_cache", {}) or {},
    }

    dataset_overrides = _cfg_get_nested(config, "dataset_params", {}) or {}
    if isinstance(dataset_overrides, dict):
        if "spec_augment" in dataset_overrides:
            dataset_config["spec_augment_params"] = dataset_overrides["spec_augment"]
        for key, value in dataset_overrides.items():
            if key in {"dict_path", "sr", "spect_params", "mel_params"}:
                dataset_config[key] = value
            elif key == "spec_augment":
                # SpecAugment configuration is passed via the dedicated
                # ``spec_augment_params`` argument expected by ``MelDataset``.
                continue
            elif key not in dataset_config:
                dataset_config[key] = value

    eval_batch_size = batch_size if batch_size is not None else int(
        _cfg_get_nested(
            config,
            "eval_params.batch_size",
            _cfg_get_nested(config, "batch_size", 1),
        )
    )

    dataloader_params = _cfg_get_nested(config, "dataloader_params", {}) or {}
    val_workers = num_workers if num_workers is not None else int(dataloader_params.get("val_num_workers", 2))

    collate_config = {"return_speaker_ids": True}
    device_str = str(device)
    if isinstance(device, torch.device):
        device_str = device.type

    dataloader = build_dataloader(
        val_entries,
        batch_size=max(1, int(eval_batch_size)),
        validation=True,
        num_workers=max(0, int(val_workers)),
        device=device_str,
        dataset_config=dataset_config,
        collate_config=collate_config,
        dataset_name="val",
    )

    return dataloader, val_entries


def calc_wer(target, pred, ignore_indexes=[0]):
    target_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(target)))))
    pred_chars = drop_duplicated(list(filter(lambda x: x not in ignore_indexes, map(str, list(pred)))))
    target_str = ' '.join(target_chars)
    pred_str = ' '.join(pred_chars)
    error = jiwer.wer(target_str, pred_str)
    return error

def drop_duplicated(chars):
    ret_chars = [chars[0]]
    for prev, curr in zip(chars[:-1], chars[1:]):
        if prev != curr:
            ret_chars.append(curr)
    return ret_chars

def build_criterion(critic_params={}, entropy_params={}, multi_task_config=None):
    multi_task_config = multi_task_config or {}

    ctc_params = dict(critic_params.get('ctc', {}))
    ctc_params.setdefault('reduction', 'none')

    criterion = {
        "ce": nn.CrossEntropyLoss(ignore_index=-1, **entropy_params),
        "ctc": torch.nn.CTCLoss(**ctc_params),
    }

    frame_cfg = multi_task_config.get('frame_phoneme', {}) or {}
    if frame_cfg.get('enabled', False):
        criterion["frame_ce"] = nn.CrossEntropyLoss(ignore_index=-1, **entropy_params)

    speaker_cfg = multi_task_config.get('speaker', {}) or {}
    if speaker_cfg.get('enabled', False):
        criterion["speaker_ce"] = nn.CrossEntropyLoss()

    pron_cfg = multi_task_config.get('pronunciation_error', {}) or {}
    if pron_cfg.get('enabled', False):
        criterion["pron_error_ce"] = nn.CrossEntropyLoss(ignore_index=-1, **entropy_params)

    return criterion

def get_data_path_list(train_path=None, val_path=None, return_paths=False):
    """Return the metadata entries for the train/validation splits.

    Args:
        train_path: Optional path to the training metadata file.  When ``None``
            the default ``Data/train_list.txt`` is used.
        val_path: Optional path to the validation metadata file.  When ``None``
            the default ``Data/val_list.txt`` is used.
        return_paths: If ``True`` the resolved metadata file paths are returned
            alongside the contents.  This is useful for caching layers that
            need to reason about file modification times.

    Returns:
        Tuple containing the training and validation metadata lines.  When
        ``return_paths`` is enabled, the resolved file system paths are appended
        to the tuple.
    """

    train_path = Path(train_path) if train_path is not None else Path("Data") / "train_list.txt"
    val_path = Path(val_path) if val_path is not None else Path("Data") / "val_list.txt"

    with train_path.open('r', encoding='utf-8') as f:
        train_list = f.readlines()
    with val_path.open('r', encoding='utf-8') as f:
        val_list = f.readlines()

    if return_paths:
        return train_list, val_list, str(train_path), str(val_path)
    return train_list, val_list


def build_beam_search_decoder(config=None, vocab_size=None):
    """Construct a :class:`~decoding.CTCBeamSearchDecoder` from a config dict."""

    if config is None:
        return None
    try:
        if (
            vocab_size is not None
            and isinstance(config, dict)
            and "decoding" in config
            and "shallow_fusion" in config["decoding"]
        ):
            config = dict(config)
            decoding_cfg = dict(config.get("decoding", {}))
            shallow_cfg = dict(decoding_cfg.get("shallow_fusion", {}))
            shallow_cfg["vocab_size"] = int(vocab_size)
            decoding_cfg["shallow_fusion"] = shallow_cfg
            config["decoding"] = decoding_cfg
        decoder = build_decoder_from_config(config)
    except Exception:
        return None
    return decoder


def plot_image(image):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(image, aspect="auto", origin="lower",
                   interpolation='none')

    fig.canvas.draw()
    plt.close(fig)

    return fig

def diagonal_attention_prior(attn, text_lengths, mel_lengths, sigma=0.5, eps=1.0e-6):
    """Calculate diagonal attention loss with length-aware masking.

    Args:
        attn: Attention weights of shape ``[B, T_text, T_mel]``.
        text_lengths: Tensor containing the valid number of text tokens per batch item.
        mel_lengths: Tensor containing the valid number of mel frames per batch item.
        sigma: Controls the width of the diagonal Gaussian prior.
        eps: Numerical stability constant.
    """

    if attn is None:
        raise ValueError("'attn' must be a tensor")

    if sigma <= 0:
        raise ValueError("'sigma' must be positive")

    B, T_text, T_mel = attn.size()
    device = attn.device

    text_lengths = text_lengths.to(device=device, dtype=torch.float32)
    mel_lengths = mel_lengths.to(device=device, dtype=torch.float32)

    text_positions = torch.arange(T_text, device=device, dtype=torch.float32).view(1, T_text, 1)
    mel_positions = torch.arange(T_mel, device=device, dtype=torch.float32).view(1, 1, T_mel)

    text_scale = torch.clamp(text_lengths.view(B, 1, 1) - 1.0, min=1.0)
    mel_scale = torch.clamp(mel_lengths.view(B, 1, 1) - 1.0, min=1.0)

    text_norm = torch.clamp(text_positions / text_scale, 0.0, 1.0)
    mel_norm = torch.clamp(mel_positions / mel_scale, 0.0, 1.0)

    # Broadcast to [B, T_text, T_mel]
    text_norm = text_norm.expand(B, -1, -1)
    mel_norm = mel_norm.expand(B, -1, -1)

    expected = torch.exp(-((text_norm - mel_norm) ** 2) / (2 * sigma ** 2))
    expected = expected / expected.amax(dim=(1, 2), keepdim=True).clamp_min(eps)

    text_mask = (text_positions.long() < text_lengths.view(B, 1, 1).long()).expand(B, -1, T_mel)
    mel_mask = (mel_positions.long() < mel_lengths.view(B, 1, 1).long()).expand(B, T_text, -1)
    valid_mask = text_mask & mel_mask

    mask = valid_mask.to(attn.dtype)
    denom = mask.sum()
    if denom.item() <= 0:
        return torch.tensor(0.0, device=device, dtype=attn.dtype)

    loss = (attn * (1.0 - expected) * mask).sum() / denom
    return loss


def diagonal_attention_coherence(attn, text_lengths, mel_lengths, sigma=0.5, eps=1.0e-6):
    """Return the per-sample diagonal alignment score in ``[0, 1]``."""

    if attn is None:
        raise ValueError("'attn' must be a tensor")

    if sigma <= 0:
        raise ValueError("'sigma' must be positive")

    if not torch.is_tensor(text_lengths):
        text_lengths = torch.as_tensor(text_lengths, device=attn.device)
    if not torch.is_tensor(mel_lengths):
        mel_lengths = torch.as_tensor(mel_lengths, device=attn.device)

    B, T_text, T_mel = attn.size()
    device = attn.device

    text_lengths = text_lengths.to(device=device, dtype=torch.float32)
    mel_lengths = mel_lengths.to(device=device, dtype=torch.float32)

    text_positions = torch.arange(T_text, device=device, dtype=torch.float32).view(1, T_text, 1)
    mel_positions = torch.arange(T_mel, device=device, dtype=torch.float32).view(1, 1, T_mel)

    text_scale = torch.clamp(text_lengths.view(B, 1, 1) - 1.0, min=1.0)
    mel_scale = torch.clamp(mel_lengths.view(B, 1, 1) - 1.0, min=1.0)

    text_norm = torch.clamp(text_positions / text_scale, 0.0, 1.0).expand(B, -1, -1)
    mel_norm = torch.clamp(mel_positions / mel_scale, 0.0, 1.0).expand(B, -1, -1)

    expected = torch.exp(-((text_norm - mel_norm) ** 2) / (2 * sigma ** 2))
    expected = expected / expected.amax(dim=(1, 2), keepdim=True).clamp_min(eps)

    text_mask = (text_positions.long() < text_lengths.view(B, 1, 1).long()).expand(B, -1, T_mel)
    mel_mask = (mel_positions.long() < mel_lengths.view(B, 1, 1).long()).expand(B, T_text, -1)
    mask = (text_mask & mel_mask).to(attn.dtype)

    expected_mass = (expected * mask).sum(dim=(1, 2)).clamp_min(eps)
    alignment_mass = (attn * expected * mask).sum(dim=(1, 2))

    return alignment_mass / expected_mass


class BatchSizeScheduler:
    """Utility to manage curriculum batch-size schedules.

    The scheduler consumes the ``training_curriculum.batch_size_schedule``
    section from the configuration dictionary and produces the batch size that
    should be used for each epoch.  Two complementary strategies are supported:

    * ``milestones`` – Specify explicit ``epoch`` → ``batch_size`` pairs.  The
      batch size is held constant between milestones.
    * ``linear`` – Optionally enable linear interpolation between the milestone
      anchors.  The interpolation frequency can be controlled with
      ``update_interval`` so that the batch size only changes every ``n``
      epochs.

    Both strategies can be enabled at the same time.  In that case milestones
    provide the anchor points and the linear strategy interpolates between
    them.  When no milestones are supplied the linear configuration is used to
    derive the start and end anchors, falling back to ``initial_batch_size`` and
    ``base_batch_size`` respectively.

    The scheduler is deterministic and pre-computes the curriculum for all
    epochs, making it safe to query in notebooks or in the training loop.
    """

    def __init__(
        self,
        schedule_config: Optional[Dict] = None,
        default_batch_size: int = 1,
        total_epochs: int = 1,
    ) -> None:
        schedule_config = schedule_config or {}
        if not isinstance(schedule_config, dict):
            schedule_config = {}

        self.config = schedule_config
        self.enabled = bool(schedule_config.get("enabled", False))
        self.total_epochs = max(1, int(total_epochs))
        self.default_batch_size = max(1, int(default_batch_size))
        self.base_batch_size = max(
            1,
            int(schedule_config.get("base_batch_size", self.default_batch_size)),
        )
        self.initial_batch_size = max(
            1,
            int(schedule_config.get("initial_batch_size", self.default_batch_size)),
        )
        self.apply_to_validation = bool(schedule_config.get("apply_to_validation", False))

        eval_epoch = schedule_config.get("evaluation_epoch")
        try:
            self.evaluation_epoch = int(eval_epoch) if eval_epoch is not None else None
        except (TypeError, ValueError):
            self.evaluation_epoch = None

        strategies = schedule_config.get("strategies", {})
        if not isinstance(strategies, dict):
            strategies = {}
        self._milestone_cfg = strategies.get("milestones", {}) or {}
        if not isinstance(self._milestone_cfg, dict):
            self._milestone_cfg = {}
        self._linear_cfg = strategies.get("linear", {}) or {}
        if not isinstance(self._linear_cfg, dict):
            self._linear_cfg = {}

        self.linear_enabled = bool(self._linear_cfg.get("enabled", False))
        self.linear_update_interval = max(
            1,
            int(self._linear_cfg.get("update_interval", 1)),
        )

        self._anchors = self._build_anchors()
        self._epoch_schedule = self._build_epoch_schedule()

    def _build_anchors(self) -> List[Tuple[int, int]]:
        if not self.enabled:
            return [(1, self.default_batch_size), (self.total_epochs, self.default_batch_size)]

        anchors: Dict[int, int] = {}
        schedule_entries: Iterable = []
        if isinstance(self._milestone_cfg, dict):
            entries = self._milestone_cfg.get("schedule", [])
            if isinstance(entries, dict):
                schedule_entries = entries.items()
            else:
                schedule_entries = entries

        for entry in schedule_entries:
            if isinstance(entry, dict):
                epoch = entry.get("epoch", entry.get("start_epoch"))
                batch_size = entry.get("batch_size", entry.get("value"))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                epoch, batch_size = entry[0], entry[1]
            else:
                continue

            try:
                epoch = int(epoch)
                batch_size = int(batch_size)
            except (TypeError, ValueError):
                continue

            if epoch < 1:
                epoch = 1

            anchors[epoch] = max(1, batch_size)

        if not anchors and self.linear_enabled:
            start_epoch = max(1, int(self._linear_cfg.get("start_epoch", 1)))
            end_epoch = max(start_epoch, int(self._linear_cfg.get("end_epoch", self.total_epochs)))
            start_bs = max(1, int(self._linear_cfg.get("start_batch_size", self.initial_batch_size)))
            end_bs = max(1, int(self._linear_cfg.get("end_batch_size", self.base_batch_size)))
            anchors[start_epoch] = start_bs
            anchors[end_epoch] = end_bs

        if 1 not in anchors:
            anchors[1] = self.initial_batch_size

        final_epoch = self.total_epochs
        final_value = anchors.get(final_epoch, self.base_batch_size)
        if self.linear_enabled:
            final_value = int(self._linear_cfg.get("end_batch_size", final_value))
        anchors[final_epoch] = max(1, final_value)

        filtered: Dict[int, int] = {}
        for epoch, batch_size in anchors.items():
            epoch = int(epoch)
            if epoch < 1:
                continue
            if epoch > self.total_epochs:
                epoch = self.total_epochs
            filtered[epoch] = max(1, int(batch_size))

        return sorted(filtered.items(), key=lambda item: item[0])

    def _build_epoch_schedule(self) -> List[int]:
        schedule = [self.default_batch_size] * (self.total_epochs + 1)
        if not self.enabled:
            for epoch in range(1, self.total_epochs + 1):
                schedule[epoch] = self.default_batch_size
            return schedule

        anchors = self._anchors
        if len(anchors) == 1:
            value = max(1, anchors[0][1])
            for epoch in range(1, self.total_epochs + 1):
                schedule[epoch] = value
            return schedule

        last_value = self.initial_batch_size
        for epoch in range(1, self.total_epochs + 1):
            prev_anchor = anchors[0]
            next_anchor = anchors[-1]
            for anchor in anchors:
                if anchor[0] <= epoch:
                    prev_anchor = anchor
                if anchor[0] >= epoch:
                    next_anchor = anchor
                    break

            if self.linear_enabled and next_anchor[0] > prev_anchor[0]:
                span = max(1, next_anchor[0] - prev_anchor[0])
                progress = (epoch - prev_anchor[0]) / span
                progress = min(max(progress, 0.0), 1.0)
                interpolated = prev_anchor[1] + progress * (next_anchor[1] - prev_anchor[1])
                value = int(round(interpolated))
                if (
                    self.linear_update_interval > 1
                    and epoch != prev_anchor[0]
                    and (epoch - prev_anchor[0]) % self.linear_update_interval != 0
                ):
                    value = last_value
            else:
                value = prev_anchor[1]

            value = max(1, int(value))
            schedule[epoch] = value
            last_value = value

        return schedule

    def batch_size_for_epoch(self, epoch: int) -> int:
        epoch = int(epoch)
        if epoch < 1:
            epoch = 1
        if epoch > self.total_epochs:
            epoch = self.total_epochs
        return self._epoch_schedule[epoch]

    def epoch_schedule(self) -> Dict[int, int]:
        return {epoch: self._epoch_schedule[epoch] for epoch in range(1, self.total_epochs + 1)}

    def summary(self, max_entries: int = 10) -> str:
        """Return a readable summary of schedule transitions."""

        transitions: List[Tuple[int, int]] = []
        last_value = None
        for epoch in range(1, self.total_epochs + 1):
            value = self._epoch_schedule[epoch]
            if value != last_value:
                transitions.append((epoch, value))
                last_value = value

        if len(transitions) > max_entries:
            head = transitions[: max_entries - 1]
            tail = transitions[-1:]
            summary_parts = ["%i→%i" % (e, v) for e, v in head]
            summary_parts.append("…")
            summary_parts.extend("%i→%i" % (e, v) for e, v in tail)
        else:
            summary_parts = ["%i→%i" % (e, v) for e, v in transitions]

        return ", ".join(summary_parts)

    def final_batch_size(self) -> int:
        return self.batch_size_for_epoch(self.total_epochs)

    def expected_total_steps(
        self,
        dataset_size: int,
        *,
        world_size: int = 1,
        per_rank_samples: Optional[int] = None,
    ) -> int:
        """Return the number of optimiser steps implied by the schedule.

        Args:
            dataset_size: Total number of training items across all processes.
            world_size: Number of data-parallel workers processing the dataset.
            per_rank_samples: Optional override for the number of samples seen by
                each rank in an epoch.  When provided, ``world_size`` is ignored
                and the step estimation is derived directly from this value.
        """

        samples_per_rank: int
        if per_rank_samples is not None:
            samples_per_rank = max(1, int(per_rank_samples))
        else:
            dataset_size = max(1, int(dataset_size))
            world_size = max(1, int(world_size))
            samples_per_rank = max(1, math.ceil(dataset_size / float(world_size)))

        total_steps = 0
        for epoch in range(1, self.total_epochs + 1):
            batch_size = max(1, self.batch_size_for_epoch(epoch))
            # ``len(dataloader)`` effectively performs a ``math.ceil`` over the
            # batch size, so mirror that behaviour here using the per-rank
            # dataset size.  This keeps OneCycle/linear warm-up schedulers in
            # sync with the actual number of optimisation steps executed on each
            # worker.
            steps = max(1, math.ceil(samples_per_rank / float(batch_size)))
            total_steps += steps
        return total_steps
