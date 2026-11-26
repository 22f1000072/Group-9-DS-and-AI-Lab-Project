"""Interview glossary on top of the project's transcription assistant (novelty, no retraining).

`GlossaryAssistant` extends `group9_ds_ai_lab.assistant.engine.Assistant` without changing it:
  1. the trained model (Hugging Face: kumargulshan/wav2vec2-base-ls960-noise-robust) runs once -> CTC log-probabilities
  2. transcript A: the project's standard beam search + LM, unchanged (the evaluated system)
  3. transcript B: a second beam search on the same log-probabilities with a small boost for the glossary terms
  4. where A and B disagree on a glossary term (or A has a close spelling of one), a suggestion with its times is
     made, checked against the audio (glossary.py); the reviewer replays it and decides.
Without a glossary the output is exactly the project's standard transcript.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import glossary as G
from group9_ds_ai_lab.assistant.engine import HUB_MODEL, Assistant, Transcript, Word, load_audio
from group9_ds_ai_lab.data.text import BLANK_ID, DELIM_ID
from group9_ds_ai_lab.inference.decode import FRAME_SEC
from group9_ds_ai_lab.utils.paths import SAMPLE_RATE


@dataclass
class GWord(Word):
    glossary: bool = False  # part of a glossary term already in the standard transcript


@dataclass
class GlossaryTranscript(Transcript):
    glossary: list = field(default_factory=list)          # normalised glossary terms used
    suggestions: list = field(default_factory=list)       # suggestions that passed the audio check (dicts)
    boosted: list = field(default_factory=list)           # transcript B: [{"word", "start", "end"}]
    weak_suggestions: list = field(default_factory=list)  # differences on a term that failed the audio check


class GlossaryAssistant(Assistant):
    def __init__(self, model=None, **kw):
        super().__init__(model=model or HUB_MODEL, **kw)
        self.glossary_params = {"weight": G.DEFAULT_WEIGHT, "oov_logp": G.DEFAULT_OOV_LOGP}
        self.glossary_max_cost = G.DEFAULT_MAX_COST
        self.glossary_near = {"near_dist": G.DEFAULT_NEAR_DIST, "near_cost": G.DEFAULT_NEAR_COST}

    def parse_glossary(self, raw):
        """User text / list -> (terms, warnings); at most 20 terms."""
        return G.parse_glossary(raw, self.unigram_logp)

    def _decode_boosted(self, lpn: np.ndarray, terms) -> list[dict]:
        """Transcript B: glossary-boosted beam search on the same (normalised) log-probabilities."""
        words = []
        for w, s, e in G.boosted_beam(self.decoder, lpn, terms, beam=self.beam, **self.glossary_params):
            seg = lpn[s:max(e, s + 1)]
            pmax, am = np.exp(seg.max(-1)), seg.argmax(-1)
            keep = (am != BLANK_ID) & (am != DELIM_ID)
            v = pmax[keep] if keep.any() else pmax
            words.append({"word": w, "start": s * FRAME_SEC, "end": e * FRAME_SEC,
                          "conf_min": float(v.min()), "conf_mean": float(v.mean())})
        return words

    def transcribe(self, audio, sr: int | None = None, threshold: float | None = None, glossary=None):
        """glossary: None, user text (one term per line) or a list of terms / parsed glossary.Term objects."""
        thr = self.threshold if threshold is None else threshold
        t = {}
        t0 = time.perf_counter()
        x = load_audio(audio, sr)
        t["load_audio"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        lp = self.log_probs(x)
        t["acoustic_model"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        lpn = self._norm(lp)
        raw = self._decode(lpn, normalized=True)  # transcript A, exactly the project's decoding
        t["decoding"] = time.perf_counter() - t0
        terms, suggestions, weak, bst = [], [], [], []
        if glossary and self.decoder is not None:
            t0 = time.perf_counter()
            terms = glossary if all(isinstance(g, G.Term) for g in glossary) else self.parse_glossary(glossary)[0]
            if terms:
                bst = self._decode_boosted(lpn, terms)
                suggestions, weak = G.find_suggestions(raw, bst, terms, lpn, self.glossary_max_cost, **self.glossary_near)
                ws = [w["word"] for w in raw]
                for term in terms:
                    for k in G.term_in(ws, term):
                        for j in range(k, k + len(term.words)):
                            raw[j]["glossary"] = True
            t["glossary"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = self.risk(raw)
        t["error_detector"] = time.perf_counter() - t0
        dur = len(x) / SAMPLE_RATE
        words = [GWord(w["word"], round(w["start"], 3), round(w["end"], 3), round(w["conf_mean"], 4), round(float(pe), 4),
                       bool(pe >= thr), round(max(w["start"] - self.replay_s, 0.0), 3),
                       round(min(w["end"] + self.replay_s, dur), 3), bool(w.get("glossary", False)))
                 for w, pe in zip(raw, p)]

        def sdict(s):
            d = s.to_dict()
            d["replay_start"] = round(max(s.start - self.replay_s, 0.0), 3)
            d["replay_end"] = round(min(s.end + self.replay_s, dur), 3)
            return d
        settings = {**self.settings, "threshold": thr}
        if terms:
            settings["glossary"] = {**self.glossary_params, "max_acoustic_cost": self.glossary_max_cost, **self.glossary_near}
        return GlossaryTranscript(" ".join(w.word for w in words), words, round(dur, 3),
                                  {k: round(v, 4) for k, v in t.items()}, settings,
                                  [x.text for x in terms], [sdict(s) for s in suggestions],
                                  [{"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)} for w in bst],
                                  [sdict(s) for s in weak])
