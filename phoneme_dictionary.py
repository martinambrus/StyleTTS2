"""Utilities for loading phoneme-to-index dictionaries."""

from __future__ import annotations

import csv
import os
import threading
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import yaml

DictionaryLike = Mapping[str, int]


DEFAULT_DICTIONARY_PATH = os.path.join(
    os.path.dirname(__file__), "Data", "word_index_dict.txt"
)


__all__ = [
    "DEFAULT_DICTIONARY_PATH",
    "load_phoneme_dictionary",
    "resolve_phoneme_dictionary_settings",
    "infer_phoneme_dictionary_token_count",
]


# Process-local cache keyed by absolute dictionary paths.
_LOCAL_CACHE: Dict[str, Dict[str, int]] = {}
_LOCAL_LOCK = threading.RLock()


def _resolve_flags(config: Optional[Mapping]) -> Tuple[bool, bool]:
    """Return the resolved (lazy_enabled, share_enabled) flags.

    The loader accepts the same configuration schema as the Auxiliary ASR
    training scripts.  ``share_enabled`` is currently unused but recognised so
    existing configuration files remain compatible.
    """

    lazy_enabled = True
    share_enabled = False

    if not isinstance(config, Mapping):
        return lazy_enabled, share_enabled

    lazy_cfg = config.get("lazy_loading")
    share_cfg = config.get("shared_cache")

    if isinstance(lazy_cfg, Mapping):
        lazy_enabled = bool(lazy_cfg.get("enabled", True))
    elif isinstance(lazy_cfg, bool):
        lazy_enabled = lazy_cfg

    if isinstance(share_cfg, Mapping):
        share_enabled = bool(share_cfg.get("enabled", False))
    elif isinstance(share_cfg, bool):
        share_enabled = share_cfg

    return lazy_enabled, share_enabled


def _load_dictionary(path: str) -> Dict[str, int]:
    """Parse ``phoneme,index`` pairs from ``path`` into a dictionary."""

    dictionary: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            try:
                phoneme, index = row[0], row[1]
            except IndexError:
                continue
            phoneme = phoneme.strip()
            if phoneme.startswith("\"") and phoneme.endswith("\""):
                phoneme = phoneme[1:-1]
            try:
                dictionary[phoneme] = int(index)
            except ValueError:
                continue
    return dictionary


def load_phoneme_dictionary(
    path_or_mapping: Union[str, DictionaryLike],
    config: Optional[Mapping] = None,
) -> Dict[str, int]:
    """Return a phoneme dictionary from ``path_or_mapping``.

    The loader honours ``config['lazy_loading']['enabled']`` to control whether
    parsed dictionaries are cached within the current process.  The
    ``shared_cache`` option is accepted for compatibility but does not trigger
    any additional behaviour in this simplified port.
    """

    if isinstance(path_or_mapping, Mapping):
        return dict(path_or_mapping)

    path = os.path.abspath(str(path_or_mapping))
    lazy_enabled, _ = _resolve_flags(config)

    if not lazy_enabled:
        return _load_dictionary(path)

    with _LOCAL_LOCK:
        cached = _LOCAL_CACHE.get(path)
        if cached is not None:
            return dict(cached)
        dictionary = _load_dictionary(path)
        _LOCAL_CACHE[path] = dictionary
        return dict(dictionary)


def infer_phoneme_dictionary_token_count(
    path_or_mapping: Union[str, DictionaryLike, None],
    config: Optional[Mapping] = None,
) -> Optional[int]:
    """Return the number of tokens addressed by ``path_or_mapping``.

    Args:
        path_or_mapping: Either a dictionary mapping phonemes to indexes or a
            filesystem path to a CSV file containing the mapping.
        config: Optional configuration passed to :func:`load_phoneme_dictionary`.

    Returns:
        The inferred vocabulary size (maximum index + 1) or ``None`` when the
        dictionary could not be loaded or does not contain numeric indexes.
    """

    if path_or_mapping is None:
        return None

    try:
        mapping = load_phoneme_dictionary(path_or_mapping, config=config)
    except FileNotFoundError:
        return None

    if not mapping:
        return None

    max_index: Optional[int] = None
    for value in mapping.values():
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue

        if max_index is None or index > max_index:
            max_index = index

    if max_index is None or max_index < 0:
        return None

    return max_index + 1


def _cfg_get_nested(cfg: Mapping[str, Any], path, default=None, sep: str = "."):
    if isinstance(path, str):
        keys = path.split(sep)
    else:
        keys = list(path or [])

    current: Any = cfg
    for key in keys:
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return default
    return current


def _deep_merge_dict(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_relative_path(path: Optional[Union[str, DictionaryLike]], base_dir: Optional[str]):
    if not isinstance(path, str):
        return path

    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded

    if base_dir:
        candidate = os.path.abspath(os.path.join(base_dir, expanded))
        if os.path.exists(candidate):
            return candidate

    return expanded


def resolve_phoneme_dictionary_settings(
    data_params: Optional[Mapping[str, Any]] = None,
    asr_config_path: Optional[str] = None,
    default_path: Optional[Union[str, DictionaryLike]] = DEFAULT_DICTIONARY_PATH,
) -> Tuple[Union[str, DictionaryLike, None], Dict[str, Any]]:
    """Resolve the phoneme dictionary source and configuration.

    Args:
        data_params: Optional mapping containing dataset configuration values.
        asr_config_path: Optional path to an Auxiliary ASR configuration file.
        default_path: Fallback dictionary source when neither overrides nor
            configuration provide one.

    Returns:
        Tuple of ``(dictionary_source, dictionary_config)`` where
        ``dictionary_source`` is either a filesystem path, a mapping, or
        ``None`` when no source could be determined, and ``dictionary_config``
        contains any dictionary loading options from the configuration.
    """

    dictionary_source: Union[str, DictionaryLike, None] = None
    dictionary_config: Dict[str, Any] = {}

    config_data: Mapping[str, Any] = {}
    config_dir: Optional[str] = None

    if asr_config_path:
        try:
            with open(asr_config_path, "r", encoding="utf-8") as handle:
                config_data = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            config_data = {}
        config_dir = os.path.dirname(os.path.abspath(asr_config_path))

    if config_data:
        config_dictionary = _cfg_get_nested(config_data, "phoneme_dictionary", {}) or {}
        if isinstance(config_dictionary, Mapping):
            dictionary_config = dict(config_dictionary)
        dictionary_path = _cfg_get_nested(config_data, "phoneme_maps_path", None)
        dictionary_source = _resolve_relative_path(dictionary_path, config_dir)

    if isinstance(data_params, Mapping):
        override_path = data_params.get("phoneme_dict_path")
        if override_path is None:
            override_path = data_params.get("dict_path")
        if override_path is not None:
            dictionary_source = _resolve_relative_path(override_path, config_dir)

        override_cfg = data_params.get("phoneme_dictionary_config")
        if isinstance(override_cfg, Mapping):
            dictionary_config = _deep_merge_dict(dictionary_config, override_cfg)

    if dictionary_source is None:
        dictionary_source = default_path

    return dictionary_source, dictionary_config
