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
  3. candidates()        (a) "diff": align the two transcripts; where they differ AND the boosted side contains
                              a complete glossary term -> one candidate per term occurrence, covering only the
                              standard words at the same time as the term (not the whole differing stretch)
                          (b) "near": spans of 1..n+2 words, in either transcript, whose letters are close to a
                              glossary term (edit distance on the spelling and on a rough sound key, e.g.
                              RAMILAST PAST ONE ~ RAM VILAS PASWAN, KUCHENKUMAR ~ GULSHAN KUMAR); these catch
                              terms the boosted search spelled only partly or not at all
  4. audio check          CTC log-likelihood of the term minus that of the standard words, over the same frames.
     select()             diff candidates are kept if >= -max_cost (else shown as "weak"); near candidates need
                          distance <= near_dist and >= -near_cost (else dropped); overlaps are resolved by score.
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

# defaults chosen on the dev set with glossary_eval.py (see experiments/results/glossary/)
DEFAULT_WEIGHT = 8.0          # hotword bonus (natural-log units) per boosted word
DEFAULT_OOV_LOGP = -5.0       # log10 LM probability given to glossary words the LM does not know
DEFAULT_MAX_COST = 4.0        # max acoustic cost (nats) of a "diff" suggestion against the standard words (v1: 8)
DEFAULT_NEAR_DIST = 0.1       # max normalised spelling distance of a "near" candidate
DEFAULT_NEAR_COST = 16.0      # max acoustic cost of a "near" candidate
NEAR_MIN_LETTERS = 5          # shorter terms are too ambiguous for near matching
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
    bst_start: int = -1       # index range [bst_start, bst_end) into the glossary-assisted word list
    bst_end: int = -1
    kind: str = "diff"        # "diff" (the two transcripts disagree on the term) or "near" (close spelling)
    dist: float = 0.0         # spelling distance (near candidates)

    def to_dict(self):
        return asdict(self)


def _find(seq: list, sub: tuple, lo: int, hi: int) -> list[int]:
    n = len(sub)
    return [k for k in range(max(lo, 0), min(hi, len(seq)) - n + 1) if tuple(seq[k:k + n]) == sub]


def _sugg(std, bst, il, ih, jl, jh, names, lp, suggested=None, kind="diff", dist=0.0):
    """Build a Suggestion replacing std[il:ih] by bst[jl:jh] (or by `suggested`, spread over the same time)."""
    sw, ow = bst[jl:jh], std[il:ih]
    t0 = min([w["start"] for w in sw + ow])
    t1 = max([w["end"] for w in sw + ow])
    standard = " ".join(w["word"] for w in ow)
    if suggested is None:
        suggested = " ".join(w["word"] for w in sw)
        words = [{"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)} for w in sw]
    else:
        tw = suggested.split()
        step = (t1 - t0) / len(tw)
        words = [{"word": w, "start": round(t0 + k * step, 3), "end": round(t0 + (k + 1) * step, 3)}
                 for k, w in enumerate(tw)]
    sup = float("nan")
    if lp is not None:
        sup = acoustic_support(lp, int(round(t0 / FRAME_SEC)), int(round(t1 / FRAME_SEC)), standard, suggested)
    return Suggestion(il, ih, standard, suggested, words, sorted(set(names)), round(t0, 3), round(t1, 3),
                      round(sup, 3) if np.isfinite(sup) else sup, jl, jh, kind, round(dist, 4))


def _span_at(words, t0, t1, lo=0, hi=None):
    """Index range of the words whose centre lies in [t0, t1] (an empty range at the right place if none)."""
    hi = len(words) if hi is None else hi
    sel = [i for i in range(max(lo, 0), min(hi, len(words))) if t0 - 0.05 <= (words[i]["start"] + words[i]["end"]) / 2 <= t1 + 0.05]
    if sel:
        return min(sel), max(sel) + 1
    k = sum((w["start"] + w["end"]) / 2 < t0 for w in words)
    return k, k


_KEY_RULES = [("PH", "F"), ("BH", "B"), ("DH", "D"), ("TH", "T"), ("KH", "K"), ("GH", "G"), ("JH", "J"),
              ("SH", "S"), ("CH", "S"), ("CK", "K"), ("Q", "K"), ("C", "K"), ("Z", "S"), ("W", "V"), ("X", "KS"),
              ("Y", "I"), ("E", "I"), ("O", "U"), ("H", "")]


def sound_key(letters: str) -> str:
    """Rough sound key for Indian / English spellings: GULSHAN ~ GULCHEN, VILAS ~ WILAS, PRADESH ~ PRADES."""
    k = letters
    for a, b in _KEY_RULES:
        k = k.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", k)


def spelling_distance(a: str, b: str) -> float:
    """Normalised edit distance between two letter strings (min over spelling and sound key)."""
    from rapidfuzz.distance import Levenshtein
    d1 = Levenshtein.distance(a, b) / max(len(a), len(b), 1)
    ka, kb = sound_key(a), sound_key(b)
    d2 = Levenshtein.distance(ka, kb) / max(len(ka), len(kb), 1)
    return min(d1, d2)


def _letters(ws) -> str:
    return "".join(ws).replace("'", "")


def candidates(std: list[dict], bst: list[dict], terms: list[Term], lp: np.ndarray | None = None,
               split: bool = True, near: bool = True, near_max: float = 0.45) -> list[Suggestion]:
    """All glossary candidates (with audio-check scores), before selection.
    split=False, near=False reproduces the first version (one candidate per differing stretch)."""
    a, b = [w["word"] for w in std], [w["word"] for w in bst]
    out, seen = [], set()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
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
        if not split:
            jl = min([j1] + [k for _, k, _ in hits])
            jh = max([j2] + [k + n for _, k, n in hits])
            il, ih = max(i1 - (j1 - jl), 0), min(i2 + (jh - j2), len(a))
            groups = [(il, ih, jl, jh, hits)]
        else:  # one candidate per (overlapping group of) term occurrence(s)
            groups = []
            for t, k, n in sorted(hits, key=lambda h: h[1]):
                if groups and k < groups[-1][3]:
                    g = groups[-1]
                    groups[-1] = (0, 0, g[2], max(g[3], k + n), g[4] + [(t, k, n)])
                else:
                    groups.append((0, 0, k, k + n, [(t, k, n)]))
            m = max(n for _, _, n in hits)
            groups = [(*_span_at(std, bst[jl]["start"], bst[jh - 1]["end"], i1 - m - 1, i2 + m + 1), jl, jh, hs)
                      for _, _, jl, jh, hs in groups]
        for il, ih, jl, jh, hs in groups:
            # skip terms the standard transcript already has at this place
            names = [t.text for t, _, _ in hs if not _find(a, t.words, il - len(t.words) + 1, ih + len(t.words) - 1)]
            if names and (jl, jh) not in seen:
                seen.add((jl, jh))
                out.append(_sugg(std, bst, il, ih, jl, jh, names, lp))
    if not near:
        return out
    # near matches: spans whose spelling is close to a term, in either transcript
    for t in terms:
        tl = _letters(t.words)
        if len(tl) < NEAR_MIN_LETTERS:
            continue
        n = len(t.words)
        found = []
        for src, ws in (("A", std), ("B", bst)):
            words = [w["word"] for w in ws]
            for i in range(len(words)):
                for L in range(1, n + 3):
                    if i + L > len(words):
                        break
                    span = words[i:i + L]
                    sl = _letters(span)
                    if not (0.6 <= len(sl) / len(tl) <= 1.6):
                        continue
                    d = spelling_distance(sl, tl)
                    if d <= near_max:
                        found.append((d, ws[i]["start"], ws[i + L - 1]["end"]))
        # group overlapping spans (one place in the audio); at each place try the closest few spellings and keep
        # the one the audio supports best (a span with an extra neighbouring word scores much worse)
        places = []
        for d, t0, t1 in sorted(found, key=lambda f: f[1]):
            if places and t0 < places[-1][1]:
                places[-1][1] = max(places[-1][1], t1)
                places[-1][2].append((d, t0, t1))
            else:
                places.append([t0, t1, [(d, t0, t1)]])
        for _, _, opts in places:
            best, tried = None, set()
            for d, t0, t1 in sorted(opts)[:6]:
                il, ih = _span_at(std, t0, t1)
                if il == ih or (il, ih) in tried or _find(a, t.words, il - n + 1, ih + n - 1):
                    continue
                tried.add((il, ih))
                jl, jh = _span_at(bst, std[il]["start"], std[ih - 1]["end"])
                c = _sugg(std, bst, il, ih, jl, jh, [t.text], lp, suggested=t.text, kind="near", dist=d)
                if best is None or np.nan_to_num(c.support, nan=-d) > np.nan_to_num(best.support, nan=-best.dist):
                    best = c
                if len(tried) >= 3:
                    break
            if best is not None:
                out.append(best)
    return out


def select(cands: list[Suggestion], max_cost: float = DEFAULT_MAX_COST, near_dist: float = DEFAULT_NEAR_DIST,
           near_cost: float = DEFAULT_NEAR_COST) -> tuple[list[Suggestion], list[Suggestion]]:
    """(kept, weak). Diff candidates failing the audio check are 'weak' (shown, not queued); near candidates must
    pass both thresholds. Overlapping candidates: the best audio score wins (diff before near)."""
    def ok(c):
        if np.isnan(c.support):
            return c.kind == "diff" or c.dist <= near_dist
        if c.kind == "diff":
            return c.support >= -max_cost
        return c.dist <= near_dist and c.support >= -near_cost
    good = sorted((c for c in cands if ok(c)), key=lambda c: (c.kind != "diff", -np.nan_to_num(c.support, nan=0.0)))
    kept, used = [], []

    def overlaps(c, spans):
        a0, a1 = c.std_start, max(c.std_end, c.std_start + 1)
        return any(a0 < y and x < a1 for x, y in spans)
    for c in good:
        if not overlaps(c, used):
            kept.append(c)
            used.append((c.std_start, max(c.std_end, c.std_start + 1)))
    weak = [c for c in cands if c.kind == "diff" and not ok(c) and not overlaps(c, used)]
    return sorted(kept, key=lambda c: c.start), sorted(weak, key=lambda c: c.start)


def find_suggestions(std: list[dict], bst: list[dict], terms: list[Term], lp: np.ndarray | None = None,
                     max_cost: float = DEFAULT_MAX_COST, near_dist: float = DEFAULT_NEAR_DIST,
                     near_cost: float = DEFAULT_NEAR_COST, split: bool = True, near: bool = True
                     ) -> tuple[list[Suggestion], list[Suggestion]]:
    """std/bst: word dicts with 'word', 'start', 'end' (seconds). Returns (kept, weak = failed the audio check)."""
    return select(candidates(std, bst, terms, lp, split=split, near=near and lp is not None), max_cost, near_dist,
                  near_cost)


def term_in(words: list[str], term: Term) -> list[int]:
    """Positions where a glossary term occurs in a word list."""
    return _find(words, term.words, 0, len(words))
