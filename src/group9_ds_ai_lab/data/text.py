"""Single transcript normaliser used for training labels AND evaluation references.

Fixes the M3 inconsistency (apostrophes dropped at data stage, kept by the vocab).
Output alphabet matches the wav2vec2 32-token character vocab: A-Z, apostrophe, space.
"""
import re

_NON_VOCAB = re.compile(r"[^A-Z' ]")
_SPACES = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = str(text).upper().replace("’", "'").replace("`", "'")
    text = _NON_VOCAB.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


# 32-token vocab of facebook/wav2vec2-base-960h (reused so all models share it)
VOCAB = {
    "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7, "O": 8,
    "N": 9, "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16, "M": 17,
    "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26,
    "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31,
}
ID2CH = {i: c for c, i in VOCAB.items()}
BLANK_ID = 0
DELIM_ID = 4


def encode(text: str) -> list[int]:
    """Normalised text -> label ids (space -> word delimiter '|')."""
    return [DELIM_ID if c == " " else VOCAB.get(c, 3) for c in text]
