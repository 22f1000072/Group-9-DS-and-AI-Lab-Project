"""Interview glossary: optional names / places / terms the user expects to hear (novelty, M6).

How it works (no retraining; the acoustic model is run once):
  1. parse_glossary()     user text -> at most 20 normalised terms (A-Z, apostrophe, space; <= 4 words each)
  2. decode twice from the SAME log-probabilities:
       standard  : the normal beam search + n-gram LM (exactly the evaluated system)
       boosted   : same beam search, plus
                   - the glossary's rare words get a small score bonus (pyctcdecode "hotwords")
                   - glossary words the LM has never seen (e.g. NEELADRI) are scored like a rare word
                     instead of receiving the large unknown-word penalty (GlossaryLM)
     Only the rare words of a term are boosted: for "NEELADRI ROAD" only NEELADRI is boosted, so the common
     word ROAD is not pushed into the rest of the transcript.
  3. find_suggestions()   align the two transcripts; every place where they differ AND the boosted side
                          contains a complete glossary term becomes a suggestion with timestamps
  4. audio check          the suggestion is kept only if the audio supports it almost as well as the standard
                          words (CTC log-likelihood over the same frames, difference >= -max_acoustic_cost).
                          This is what keeps unspoken terms (e.g. "Aniruddha") out.
Suggestions never change the transcript by themselves: the reviewer replays the region and accepts or rejects.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass

import numpy as np

from group9_ds_ai_lab.data.text import BLANK_ID, DELIM_ID, VOCAB, normalize_text

MAX_TERMS = 20
MAX_WORDS = 4
MAX_CHARS = 40
FRAME_SEC = 0.02

# defaults chosen on the dev set with scripts/glossary_eval.py (see experiments/results/glossary/)
DEFAULT_WEIGHT = 8.0          # hotword bonus (natural-log units) per boosted word
DEFAULT_OOV_LOGP = -5.0       # log10 LM probability given to glossary words the LM does not know
DEFAULT_MAX_COST = 8.0        # max acoustic cost (nats) a suggestion may have against the standard words
RARE_LOGP = -4.5              # words with unigram log10 p below this (or unknown) count as rare -> boosted


@dataclass
class Term:
    text: str                 # normalised, e.g. "NEELADRI ROAD"
    words: tuple              # ("NEELADRI", "ROAD")
    boost: tuple              # words that get the bonus, e.g. ("NEELADRI",)
    original: str             # as typed by the user


def parse_glossary(raw, unigram_logp: dict | None = None) -> tuple[list[Term], list[str]]:
    """Text (one term per line, or separated by commas/semicolons) or a list -> (terms, warnings)."""
    unigram_logp = unigram_logp or {}
    items = raw if isinstance(raw, (list, tuple)) else re.split(r"[\n,;]+", str(raw or ""))
    terms, seen, warn = [], set(), []
    for item in items:
        orig = str(item).strip()
        if not orig:
            continue
        t = normalize_text(orig.replace("-", " "))
        if not t:
            warn.append(f"'{orig}' has no letters the model can write (A-Z, apostrophe): skipped")
            continue
        if t in seen:
            continue
        words = tuple(t.split())
        if len(words) > MAX_WORDS or len(t) > MAX_CHARS:
            warn.append(f"'{orig}' is longer than {MAX_WORDS} words / {MAX_CHARS} characters: skipped")
            continue
        if len(terms) >= MAX_TERMS:
            warn.append(f"only the first {MAX_TERMS} terms are used: '{orig}' skipped")
            continue
        seen.add(t)
        terms.append(Term(t, words, _boost_words(words, unigram_logp), orig))
    return terms, warn


def _boost_words(words: tuple, unigram_logp: dict) -> tuple:
    """Rare / unknown words of a term (common words like ROAD, MR, THE are not boosted on their own).
    A term made only of common words boosts its rarest word."""
    lp = {w: unigram_logp.get(w, -99.0) for w in words}
    rare = tuple(w for w in words if lp[w] < RARE_LOGP and len(w) >= 3)
    return rare or (min(words, key=lambda w: lp[w]),)


# ------------------------------------------------------------------------------------------ boosted decoding
def _glossary_lm_class():
    from pyctcdecode.constants import LOG_BASE_CHANGE_FACTOR
    from pyctcdecode.language_model import AbstractLanguageModel, _get_empty_lm_state

    class GlossaryLM(AbstractLanguageModel):
        """Wraps the normal KenLM model; only glossary words the LM does not know are scored differently."""

        def __init__(self, base, oov_words: set, oov_logp: float):
            self.base, self.oov, self.oov_logp = base, set(oov_words), float(oov_logp)
            self.prefixes = {w[:i] for w in self.oov for i in range(1, len(w) + 1)}

        @property
        def order(self):
            return self.base.order

        def get_start_state(self):
            return self.base.get_start_state()

        def score_partial_token(self, partial_token: str) -> float:
            return 0.0 if partial_token in self.prefixes else self.base.score_partial_token(partial_token)

        def score(self, prev_state, word: str, is_last_word: bool = False):
            if word not in self.oov:
                return self.base.score(prev_state, word, is_last_word)
            end_state = _get_empty_lm_state()
            self.base._kenlm_model.BaseScore(prev_state, word, end_state)  # advances the LM state (as <unk>)
            s = self.oov_logp + (self.base._get_raw_end_score(end_state) if is_last_word else 0.0)
            return self.base.alpha * s * LOG_BASE_CHANGE_FACTOR + self.base.beta, end_state

        def reset_params(self, **params):
            self.base.reset_params(**params)

    return GlossaryLM


def boosted_beam(decoder, lp: np.ndarray, terms: list[Term], beam: int = 50, weight: float = DEFAULT_WEIGHT,
                 oov_logp: float = DEFAULT_OOV_LOGP):
    """Beam search with the glossary bonus. Returns [(word, start_frame, end_frame), ...].
    `decoder` is the normal pyctcdecode decoder; its KenLM model is shared (nothing is reloaded)."""
    from pyctcdecode.decoder import BeamSearchDecoderCTC
    hot = sorted({w for t in terms for w in t.boost})
    base = decoder._language_model
    if base is None:  # no LM: hotwords only
        out = decoder.decode_beams(lp, beam_width=beam, hotwords=hot, hotword_weight=weight)[0]
        return _frames(out)
    known = base._unigram_set
    oov = {w for t in terms for w in t.words if w not in known}
    dec = BeamSearchDecoderCTC(decoder._alphabet, _glossary_lm_class()(base, oov, oov_logp))
    try:
        out = dec.decode_beams(lp, beam_width=beam, hotwords=hot, hotword_weight=weight)[0]
    finally:
        BeamSearchDecoderCTC.model_container.pop(dec._model_key, None)  # do not keep one entry per request
    return _frames(out)


def _frames(beam_out):
    tf = beam_out.text_frames if hasattr(beam_out, "text_frames") else beam_out[2]
    return [(w, s, e) for w, (s, e) in tf]


# ------------------------------------------------------------------------------------------ audio check
def ctc_loglik(lp: np.ndarray, text: str) -> float:
    """log P(text | frames) under the CTC model (forward algorithm). lp: [T, 32] log-probabilities."""
    import torch
    import torch.nn.functional as F
    ids = [DELIM_ID if c == " " else VOCAB.get(c, 3) for c in text.strip()]
    T = lp.shape[0]
    if T == 0:
        return -np.inf if ids else 0.0
    x = torch.from_numpy(np.ascontiguousarray(lp, dtype=np.float32))[:, None, :]
    tgt = torch.tensor(ids or [0], dtype=torch.long)[None]
    loss = F.ctc_loss(x, tgt[:, :len(ids)] if ids else tgt[:, :0], torch.tensor([T]), torch.tensor([len(ids)]),
                      blank=BLANK_ID, reduction="sum", zero_infinity=False)
    v = -float(loss)
    return v if np.isfinite(v) else -np.inf


def acoustic_support(lp: np.ndarray, f0: int, f1: int, standard: str, suggested: str, pad: int = 3) -> float:
    """CTC log-likelihood of the suggested words minus that of the standard words, on the same frames."""
    a, b = max(f0 - pad, 0), min(f1 + pad, lp.shape[0])
    seg = lp[a:b]
    ls, lb = ctc_loglik(seg, suggested), ctc_loglik(seg, standard)
    if not np.isfinite(ls):
        return -np.inf
    if not np.isfinite(lb):
        return np.inf
    return ls - lb


# ------------------------------------------------------------------------------------------ suggestions
@dataclass
class Suggestion:
    std_start: int            # index range [std_start, std_end) into the standard word list
    std_end: int
    standard: str             # standard words in that range ('' if the suggestion inserts words)
    suggested: str            # glossary-assisted words
    words: list               # suggested words with times: [{"word", "start", "end"}]
    terms: list               # glossary terms this suggestion contains
    start: float              # seconds
    end: float
    support: float            # acoustic support (nats; >= -max_cost to be kept)

    def to_dict(self):
        return asdict(self)


def _find(seq: list, sub: tuple, lo: int, hi: int) -> list[int]:
    n = len(sub)
    return [k for k in range(max(lo, 0), min(hi, len(seq)) - n + 1) if tuple(seq[k:k + n]) == sub]


def find_suggestions(std: list[dict], bst: list[dict], terms: list[Term], lp: np.ndarray | None = None,
                     max_cost: float = DEFAULT_MAX_COST) -> tuple[list[Suggestion], list[Suggestion]]:
    """std/bst: word dicts with 'word', 'start', 'end' (seconds). Returns (kept, rejected_by_audio_check)."""
    a, b = [w["word"] for w in std], [w["word"] for w in bst]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    kept, dropped = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        hits = []
        for t in terms:
            n = len(t.words)
            for k in _find(b, t.words, j1 - n + 1, j2 + n - 1):
                if k + n > j1 and k < j2:  # the occurrence overlaps the changed words
                    hits.append((t, k, n))
        if not hits:
            continue
        jl = min([j1] + [k for _, k, _ in hits])
        jh = max([j2] + [k + n for _, k, n in hits])
        il, ih = max(i1 - (j1 - jl), 0), min(i2 + (jh - j2), len(a))
        # skip if the standard transcript already has the term at this place
        names = [t.text for t, _, _ in hits if not _find(a, t.words, il - len(t.words) + 1, ih + len(t.words) - 1)]
        if not names:
            continue
        sw = bst[jl:jh]
        ow = std[il:ih]
        t0 = min([w["start"] for w in sw + ow])
        t1 = max([w["end"] for w in sw + ow])
        standard, suggested = " ".join(a[il:ih]), " ".join(b[jl:jh])
        sup = float("nan")
        if lp is not None:
            sup = acoustic_support(lp, int(round(t0 / FRAME_SEC)), int(round(t1 / FRAME_SEC)), standard, suggested)
        s = Suggestion(il, ih, standard, suggested,
                       [{"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)} for w in sw],
                       sorted(set(names)), round(t0, 3), round(t1, 3), round(sup, 3) if np.isfinite(sup) else sup)
        (kept if (lp is None or sup >= -max_cost) else dropped).append(s)
    return kept, dropped


def term_in(words: list[str], term: Term) -> list[int]:
    """Positions where a glossary term occurs in a word list."""
    return _find(words, term.words, 0, len(words))
