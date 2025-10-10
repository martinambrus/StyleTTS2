import os
from typing import Mapping, Optional

from phoneme_dictionary import load_phoneme_dictionary

DEFAULT_DICT_PATH = os.path.join('word_index_dict.txt')


class TextCleaner:
    def __init__(self, word_index_dict_path=DEFAULT_DICT_PATH, dictionary_config: Optional[Mapping] = None):
        """Create a TextCleaner.

        Parameters
        ----------
        word_index_dict_path : str or dict
            Either a path to a CSV file mapping phonemes to indices or a
            dictionary object already containing the mapping.
        dictionary_config : mapping, optional
            Behaviour overrides for lazy loading and cross-worker sharing.  See
            :func:`phoneme_dictionary.load_phoneme_dictionary` for the supported
            keys.
        """
        self._dictionary_config = dictionary_config
        self._dictionary_source = word_index_dict_path
        self._word_index_dictionary = None
        self._inverse_mapping = None

        if isinstance(word_index_dict_path, dict):
            self._word_index_dictionary = dict(word_index_dict_path)
        else:
            lazy_enabled = True
            if isinstance(dictionary_config, Mapping):
                lazy_section = dictionary_config.get('lazy_loading')
                if isinstance(lazy_section, Mapping):
                    lazy_enabled = bool(lazy_section.get('enabled', True))
                elif isinstance(lazy_section, bool):
                    lazy_enabled = lazy_section

            if not lazy_enabled:
                self._word_index_dictionary = self.load_dictionary(word_index_dict_path)

        if self._word_index_dictionary is not None:
            self._inverse_mapping = {index: word for word, index in self._word_index_dictionary.items()}

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(f"(TextCleaner) Warning: Phoneme '{char}' not found in dictionary. Text: " + "".join(text))
        return indexes

    @property
    def word_index_dictionary(self):
        self._ensure_dictionary_loaded()
        return self._word_index_dictionary

    @property
    def inverse_mapping(self):
        self._ensure_dictionary_loaded()
        if self._inverse_mapping is None and self._word_index_dictionary is not None:
            self._inverse_mapping = {index: word for word, index in self._word_index_dictionary.items()}
        return self._inverse_mapping

    def _ensure_dictionary_loaded(self):
        if self._word_index_dictionary is None:
            self._word_index_dictionary = self.load_dictionary(self._dictionary_source)
            self._inverse_mapping = {index: word for word, index in self._word_index_dictionary.items()}

    def load_dictionary(self, path_or_dict):
        """Load phoneme to index mapping from a path or return the given dict."""
        if isinstance(path_or_dict, dict):
            return dict(path_or_dict)

        return load_phoneme_dictionary(path_or_dict, config=self._dictionary_config)
