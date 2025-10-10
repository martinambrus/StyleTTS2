"""Utilities for loading and sharing phoneme dictionaries across workers.

This module centralises the logic for parsing phoneme-to-index dictionaries and
optionally sharing the parsed mapping between dataloader workers.  The primary
entry point is :func:`load_phoneme_dictionary`, which accepts a path or mapping
and returns a standard Python ``dict`` instance.

The loader honours configuration flags that control lazy loading and
cross-process sharing.  When sharing is enabled the first process that touches a
dictionary parses it and stores the result inside a ``multiprocessing.Manager``
dictionary.  Subsequent workers simply copy the cached mapping instead of
re-parsing the CSV file.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
from typing import Dict, Mapping, MutableMapping, Optional, Tuple, Union

import pandas as pd

LOGGER = logging.getLogger(__name__)


# Local in-process cache keyed by the absolute path of the dictionary file.
_LOCAL_CACHE: Dict[str, Dict[str, int]] = {}
_LOCAL_LOCK = threading.RLock()


class _SharedState:
    """Encapsulates the multiprocessing manager used for cross-worker sharing."""

    __slots__ = ("manager", "cache", "lock")

    def __init__(self, manager: mp.Manager, cache: MutableMapping[str, Dict[str, int]], lock: mp.RLock):
        self.manager = manager
        self.cache = cache
        self.lock = lock


_SHARED_STATE: Optional[_SharedState] = None
_SHARED_STATE_LOCK = threading.Lock()


def _get_shared_state() -> Optional[_SharedState]:
    """Initialise or return the shared multiprocessing cache.

    On some environments (for example, restricted sandboxes) spawning a
    ``multiprocessing.Manager`` can fail.  In that case we silently disable
    cross-worker sharing and fall back to the process-local cache.
    """

    global _SHARED_STATE
    if _SHARED_STATE is not None:
        return _SHARED_STATE

    with _SHARED_STATE_LOCK:
        if _SHARED_STATE is not None:
            return _SHARED_STATE
        try:
            manager = mp.Manager()
            cache = manager.dict()  # type: ignore[assignment]
            lock = manager.RLock()  # type: ignore[assignment]
        except (OSError, ValueError) as exc:
            LOGGER.warning(
                "Failed to initialise multiprocessing manager for phoneme dictionary sharing: %s. "
                "Falling back to process-local caching only.",
                exc,
            )
            return None

        _SHARED_STATE = _SharedState(manager=manager, cache=cache, lock=lock)
        return _SHARED_STATE


def _load_dictionary_from_disk(path: str) -> Dict[str, int]:
    """Parse a phoneme dictionary CSV into a mapping.

    Args:
        path: Path to the CSV file containing ``phoneme,index`` pairs.

    Returns:
        A dictionary mapping phoneme symbols to integer ids.
    """

    csv = pd.read_csv(path, header=None).values
    dictionary: Dict[str, int] = {}
    for word, index in csv:
        dictionary[str(word)] = int(index)
    return dictionary


def _resolve_config(config: Optional[Mapping]) -> Tuple[bool, bool]:
    """Return the resolved (lazy_enabled, share_enabled) flags."""

    if not isinstance(config, Mapping):
        return True, True

    lazy_cfg = config.get("lazy_loading", {}) if isinstance(config.get("lazy_loading"), Mapping) else config.get("lazy_loading")
    share_cfg = config.get("shared_cache", {}) if isinstance(config.get("shared_cache"), Mapping) else config.get("shared_cache")

    lazy_enabled = True
    share_enabled = True

    if isinstance(lazy_cfg, Mapping):
        lazy_enabled = bool(lazy_cfg.get("enabled", True))
    elif isinstance(lazy_cfg, bool):
        lazy_enabled = lazy_cfg

    if isinstance(share_cfg, Mapping):
        share_enabled = bool(share_cfg.get("enabled", True))
    elif isinstance(share_cfg, bool):
        share_enabled = share_cfg

    return lazy_enabled, share_enabled


def load_phoneme_dictionary(
    path_or_mapping: Union[str, Mapping[str, int]],
    config: Optional[Mapping] = None,
) -> Dict[str, int]:
    """Return a phoneme dictionary.

    The function accepts either a file path or an in-memory mapping.  When a
    path is provided, the loading behaviour is controlled by ``config``:

    * ``lazy_loading.enabled`` (default ``True``) keeps the parsed dictionary in
      memory so it is reused within the process.
    * ``shared_cache.enabled`` (default ``True``) stores the parsed dictionary in
      a ``multiprocessing.Manager`` dictionary to avoid re-parsing in dataloader
      workers spawned from other processes.
    """

    if isinstance(path_or_mapping, Mapping):
        return dict(path_or_mapping)

    path = os.path.abspath(str(path_or_mapping))
    lazy_enabled, share_enabled = _resolve_config(config)

    if not lazy_enabled:
        dictionary = _load_dictionary_from_disk(path)
        if share_enabled:
            shared_state = _get_shared_state()
            if shared_state is not None:
                with shared_state.lock:
                    shared_state.cache[path] = dictionary
        return dictionary

    with _LOCAL_LOCK:
        cached = _LOCAL_CACHE.get(path)
        if cached is not None:
            return cached

    if share_enabled:
        shared_state = _get_shared_state()
        if shared_state is not None:
            cache = shared_state.cache
            # Fast path: if another worker already cached the dictionary we can reuse it.
            if path in cache:
                dictionary = dict(cache[path])
                with _LOCAL_LOCK:
                    _LOCAL_CACHE[path] = dictionary
                return dictionary
            # Slow path: take the shared lock, double-check, and populate the cache.
            with shared_state.lock:
                if path not in cache:
                    cache[path] = _load_dictionary_from_disk(path)
                dictionary = dict(cache[path])
            with _LOCAL_LOCK:
                _LOCAL_CACHE[path] = dictionary
            return dictionary

    dictionary = _load_dictionary_from_disk(path)
    with _LOCAL_LOCK:
        _LOCAL_CACHE[path] = dictionary
    return dictionary


__all__ = ["load_phoneme_dictionary"]

