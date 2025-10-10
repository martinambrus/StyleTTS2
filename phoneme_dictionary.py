"""Utilities for loading phoneme-to-index dictionaries."""

from __future__ import annotations

import csv
import os
import threading
from typing import Dict, Mapping, Optional, Tuple, Union

DictionaryLike = Mapping[str, int]


__all__ = ["load_phoneme_dictionary"]


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
