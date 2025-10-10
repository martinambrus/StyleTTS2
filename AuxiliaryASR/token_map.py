import os
import os.path as osp


def build_token_map_from_data(*paths, apply_asr_tokenizer=False):
    """Return a phoneme-to-index mapping inferred from dataset listing files.

    Parameters
    ----------
    *paths : list of str
        One or more listing files where the second column (pipe separated)
        contains text as phoneme sequences.
    """
    word_index_dict = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3, " ": 4}
    idx = 4

    def _collect(path):
        if not path or not osp.isfile(path):
            return []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [
                line.strip().split("|")[1] if "|" in line else line.strip()
                for line in f
            ]

    sentences = []
    for p in paths:
        sentences.extend(_collect(p))

    for sentence in sentences:
        if apply_asr_tokenizer:
            sentence = sentence.replace("(", "-").replace(")", "-")
        for phoneme in "".join(sentence.split(" ")):
            ph = '""' if phoneme == '"' else phoneme
            if ph not in word_index_dict:
                idx += 1
                word_index_dict[ph] = idx

    return word_index_dict