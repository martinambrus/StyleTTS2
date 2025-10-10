"""Utilities for converting phoneme strings into token indices."""

import logging
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

from phoneme_dictionary import load_phoneme_dictionary

LOGGER = logging.getLogger(__name__)

# Default location for the phoneme dictionary.  Matches the AuxiliaryASR layout
# where the mapping lives inside the Data directory.
DEFAULT_DICT_PATH = os.path.join("Data", "word_index_dict.txt")

# Fallback symbols used when a custom phoneme dictionary is unavailable.  This
# preserves backwards compatibility with earlier releases that generated the
# mapping at runtime.
_PAD = "$"
_PUNCTUATION = ';:,.!?¡¿—…"«»“” '
_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

_FALLBACK_SYMBOLS: List[str] = [_PAD] + list(_PUNCTUATION) + list(_LETTERS) + list(_LETTERS_IPA)
_FALLBACK_DICTIONARY: Dict[str, int] = {symbol: index for index, symbol in enumerate(_FALLBACK_SYMBOLS)}


def _normalise_mapping(mapping: Mapping[str, int]) -> Dict[str, int]:
    """Return a copy of *mapping* with string keys and integer values."""

    normalised: Dict[str, int] = {}
    for key, value in mapping.items():
        string_key = str(key)
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        normalised[string_key] = index
    if not normalised:
        return dict(_FALLBACK_DICTIONARY)
    return normalised


class TextCleaner:
    """Convert phoneme strings into index sequences using a dictionary."""

    def __init__(
        self,
        word_index_dict_path: Union[str, os.PathLike, Mapping[str, int]] = DEFAULT_DICT_PATH,
        dictionary_config: Optional[Mapping] = None,
    ) -> None:
        self._dictionary_config = dictionary_config
        self._dictionary_source: Union[str, os.PathLike, Mapping[str, int]] = word_index_dict_path
        self._word_index_dictionary: Optional[Dict[str, int]] = None
        self._inverse_mapping: Optional[Dict[int, str]] = None

        if isinstance(word_index_dict_path, Mapping):
            self._word_index_dictionary = _normalise_mapping(word_index_dict_path)
        else:
            lazy_enabled = True
            if isinstance(dictionary_config, Mapping):
                lazy_section = dictionary_config.get("lazy_loading")
                if isinstance(lazy_section, Mapping):
                    lazy_enabled = bool(lazy_section.get("enabled", True))
                elif isinstance(lazy_section, bool):
                    lazy_enabled = lazy_section

            if not lazy_enabled:
                self._word_index_dictionary = self.load_dictionary(word_index_dict_path)

        if self._word_index_dictionary is not None:
            self._inverse_mapping = {index: token for token, index in self._word_index_dictionary.items()}

    def __call__(self, text: Union[str, Sequence[str]]) -> List[int]:
        """Return the token indices corresponding to *text*.

        The cleaner accepts either a single string or a sequence of pre-tokenised
        phoneme strings.  When a string is supplied the cleaner first attempts to
        split it on whitespace and match whole tokens against the dictionary.
        Falling back to character-level processing preserves compatibility with
        legacy pipelines that emitted character sequences.
        """

        dictionary = self.word_index_dictionary

        if isinstance(text, str):
            stripped = text.strip()
            candidates: Iterable[str]
            if not stripped:
                candidates = []
            else:
                whitespace_tokens = stripped.split()
                if whitespace_tokens and all(token in dictionary for token in whitespace_tokens):
                    candidates = whitespace_tokens
                elif stripped in dictionary:
                    candidates = [stripped]
                else:
                    candidates = list(text)
        else:
            candidates = list(text)

        indexes: List[int] = []
        for token in candidates:
            try:
                indexes.append(dictionary[token])
            except KeyError:
                LOGGER.warning(
                    "(TextCleaner) Phoneme %r not found in dictionary. Text: %s",
                    token,
                    text,
                )
        return indexes

    @property
    def word_index_dictionary(self) -> Dict[str, int]:
        self._ensure_dictionary_loaded()
        # ``_word_index_dictionary`` is always initialised by ``_ensure_dictionary_loaded``
        # so the ``or`` fallback is purely for type checkers.
        return self._word_index_dictionary or dict(_FALLBACK_DICTIONARY)

    @property
    def inverse_mapping(self) -> Dict[int, str]:
        self._ensure_dictionary_loaded()
        if self._inverse_mapping is None and self._word_index_dictionary is not None:
            self._inverse_mapping = {index: token for token, index in self._word_index_dictionary.items()}
        return self._inverse_mapping or {index: token for token, index in _FALLBACK_DICTIONARY.items()}

    def _ensure_dictionary_loaded(self) -> None:
        if self._word_index_dictionary is None:
            self._word_index_dictionary = self.load_dictionary(self._dictionary_source)
            self._inverse_mapping = {index: token for token, index in self._word_index_dictionary.items()}

    def load_dictionary(self, path_or_dict: Union[str, os.PathLike, Mapping[str, int]]) -> Dict[str, int]:
        """Load a phoneme dictionary from *path_or_dict*.

        When loading from disk the method defers to
        :func:`phoneme_dictionary.load_phoneme_dictionary`.  Failures gracefully
        fall back to the historical static symbol table.
        """

        if isinstance(path_or_dict, Mapping):
            return _normalise_mapping(path_or_dict)

        path = os.fspath(path_or_dict) if path_or_dict is not None else ""
        if not path:
            LOGGER.warning("No phoneme dictionary path provided. Falling back to default mapping.")
            return dict(_FALLBACK_DICTIONARY)

        try:
            mapping = load_phoneme_dictionary(path, config=self._dictionary_config)
        except FileNotFoundError:
            LOGGER.warning(
                "Phoneme dictionary %s not found. Falling back to default mapping.",
                path,
            )
            return dict(_FALLBACK_DICTIONARY)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.warning(
                "Failed to load phoneme dictionary %s: %s. Falling back to default mapping.",
                path,
                exc,
            )
            return dict(_FALLBACK_DICTIONARY)

        normalised = _normalise_mapping(mapping)
        if not normalised:
            LOGGER.warning(
                "Phoneme dictionary %s was empty. Falling back to default mapping.",
                path,
            )
            return dict(_FALLBACK_DICTIONARY)
        return normalised


__all__ = ["TextCleaner", "DEFAULT_DICT_PATH"]
