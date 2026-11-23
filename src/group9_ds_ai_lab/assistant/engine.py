"""Transcription assistant engine: audio file -> words with timestamps, confidence, error risk and replay windows.

Pipeline (same components as the evaluated system):
  1. load audio, convert to 16 kHz mono
  2. acoustic model (fine-tuned wav2vec2-base + CTC) over overlapping windows, so audio of any length fits in memory;
     the log-probabilities of the window centres are stitched into one matrix (word times stay global)
  3. beam search with a KenLM n-gram LM (pyctcdecode), word times from the CTC frames (20 ms)
  4. word error detector (gradient-boosted trees trained on the dev set, scripts/error_detector.py)
  5. flags = words with P(error) >= threshold; replay window = word start - 1 s ... word end + 1 s (objective O4)
  6. optional interview glossary (novelty): a second, glossary-boosted beam search on the same log-probabilities;
     where it differs on a glossary term, a suggestion (with times) goes to the review queue (assistant/glossary.py)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from group9_ds_ai_lab.data.text import BLANK_ID, DELIM_ID, ID2CH
from group9_ds_ai_lab.inference.decode import FRAME_SEC
from group9_ds_ai_lab.utils.paths import DATA_RAW, EVAL, RUNS, SAMPLE_RATE

HUB_MODEL = "kumargulshan/wav2vec2-base-ls960-noise-robust"
LOCAL_MODEL = RUNS / "B960_rev" / "best"
LMS = {"3gram": DATA_RAW / "librispeech_lm" / "3-gram.pruned.1e-7.arpa",
       "4gram": DATA_RAW / "librispeech_lm" / "4-gram.arpa"}
DETECTORS = {"3gram": EVAL / "B960_rev_lm3" / "error_detector" / "model.joblib",
             "4gram": EVAL / "B960_rev_lm4" / "error_detector" / "model.joblib"}
SPECIAL = (1, 2, 3)  # <s> </s> <unk>: never emitted


@dataclass
class Word:
    word: str
    start: float
    end: float
    conf: float          # mean max-posterior over the word's frames
    p_error: float       # detector probability that the word is wrong
    flagged: bool
    replay_start: float
    replay_end: float
    glossary: bool = False  # part of a glossary term in the standard transcript


@dataclass
class Transcript:
    text: str
    words: list[Word]
    duration_s: float
    timings_s: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    glossary: list = field(default_factory=list)       # normalised glossary terms used
    suggestions: list = field(default_factory=list)    # glossary suggestions (dicts, see glossary.Suggestion)

    @property
    def rtf(self) -> float:
        return sum(self.timings_s.values()) / max(self.duration_s, 1e-9)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rtf"] = self.rtf
        d["n_flagged"] = sum(w.flagged for w in self.words)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_txt(self, mark_flags: bool = True) -> str:
        return " ".join(f"[{w.word}?]" if (mark_flags and w.flagged) else w.word for w in self.words)

    def to_srt(self, max_words: int = 12, max_gap: float = 0.8) -> str:
        """Subtitles: groups of words split at pauses or every `max_words`; flagged words marked with '?'."""
        lines, cur = [], []
        for w in self.words:
            if cur and (len(cur) >= max_words or w.start - cur[-1].end > max_gap):
                lines.append(cur)
                cur = []
            cur.append(w)
        if cur:
            lines.append(cur)

        def ts(t):
            ms = int(round(t * 1000))
            return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
        out = []
        for i, ws in enumerate(lines, 1):
            txt = " ".join(w.word + ("?" if w.flagged else "") for w in ws)
            out.append(f"{i}\n{ts(ws[0].start)} --> {ts(ws[-1].end)}\n{txt}\n")
        return "\n".join(out)


def load_audio(path_or_array, sr: int | None = None) -> np.ndarray:
    """File path or (array, sample rate) -> float32 mono 16 kHz."""
    from group9_ds_ai_lab.data.audio import to_mono_16k
    if isinstance(path_or_array, (str, Path)):
        try:
            import soundfile as sf
            x, sr = sf.read(str(path_or_array), dtype="float32", always_2d=True)
        except Exception:  # formats soundfile cannot read (e.g. mp3, m4a): fall back to librosa/audioread
            import librosa
            x, sr = librosa.load(str(path_or_array), sr=None, mono=False)
            x = np.atleast_2d(x).T
        return to_mono_16k(x, sr)
    x = np.asarray(path_or_array, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] < x.shape[1]:  # (channels, samples) -> (samples, channels)
        x = x.T
    if np.issubdtype(np.asarray(path_or_array).dtype, np.integer):
        x = x / np.iinfo(np.asarray(path_or_array).dtype).max
    return to_mono_16k(x, sr or SAMPLE_RATE)


def _ctc_labels():
    lab = [ID2CH[i] for i in range(len(ID2CH))]
    lab[BLANK_ID], lab[DELIM_ID] = "", " "
    for i, c in zip(SPECIAL, "#$%"):  # dummy symbols, masked out of the log-probabilities
        lab[i] = c
    return lab


def _arpa_unigrams(path: Path) -> tuple[list[str], dict]:
    words, logp, on = [], {}, False
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("\\1-grams"):
                on = True
                continue
            if on:
                if line.startswith("\\2-grams"):
                    break
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2:
                    logp[p[1].strip()] = float(p[0])
                    if p[1].strip() not in ("<s>", "</s>", "<unk>"):
                        words.append(p[1].strip())
    return words, logp


class Assistant:
    def __init__(self, model: str | Path | None = None, lm: str | None = "3gram", threshold: float = 0.5,
                 threads: int = 4, quantize: bool = False, device: str = "cpu", window_s: float = 30.0,
                 context_s: float = 2.0, alpha: float = 0.8, beta: float = 2.0, beam: int = 50, replay_s: float = 1.0):
        import torch
        from transformers import Wav2Vec2ForCTC
        t0 = time.time()
        torch.set_num_threads(threads)
        self.torch, self.device = torch, device
        src = model or (LOCAL_MODEL if (LOCAL_MODEL / "model.safetensors").exists() else HUB_MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(str(src)).eval().to(device)
        if quantize:  # int8 dynamic quantisation of all Linear layers (CPU only)
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.decoder, self.unigram_logp = None, {}
        if lm:
            from pyctcdecode import build_ctcdecoder
            words, self.unigram_logp = _arpa_unigrams(LMS[lm])
            self.decoder = build_ctcdecoder(_ctc_labels(), str(LMS[lm]), unigrams=words, alpha=alpha, beta=beta)
        if not self.unigram_logp:
            _, self.unigram_logp = _arpa_unigrams(LMS["3gram"])
        import joblib
        det = joblib.load(DETECTORS[lm or "3gram"])
        self.detector, self.features = det["model"], det["features"]
        self.threshold, self.beam, self.replay_s = threshold, beam, replay_s
        self.window, self.context = int(window_s * SAMPLE_RATE), int(context_s * SAMPLE_RATE)
        self.settings = {"model": str(src), "lm": lm, "threshold": threshold, "threads": threads, "quantize": quantize,
                         "device": device, "window_s": window_s, "context_s": context_s, "beam": beam,
                         "alpha": alpha, "beta": beta, "replay_s": replay_s}
        from group9_ds_ai_lab.assistant import glossary as G
        self.glossary_params = {"weight": G.DEFAULT_WEIGHT, "oov_logp": G.DEFAULT_OOV_LOGP}
        self.glossary_max_cost = G.DEFAULT_MAX_COST
        self.load_time_s = time.time() - t0

    def parse_glossary(self, raw):
        """User text / list -> (terms, warnings); at most 20 terms."""
        from group9_ds_ai_lab.assistant.glossary import parse_glossary
        return parse_glossary(raw, self.unigram_logp)

    # ------------------------------------------------------------------------------------------ acoustic model
    def log_probs(self, x: np.ndarray) -> np.ndarray:
        """CTC log-probabilities (frames x 32) for audio of any length, via overlapping windows."""
        torch = self.torch
        hop = 320  # samples per output frame (approximately; the exact count comes from the conv stack below)
        # exact number of frames the model produces for the whole file (conv stack: kernel 400, stride 320 overall);
        # using len(x) // 320 would leave an empty last frame that the beam search turns into a stray letter
        n_frames = int(self.model._get_feat_extract_output_lengths(torch.tensor(len(x))))
        if len(x) <= self.window + 2 * self.context:
            spans = [(0, len(x), 0, n_frames)]
        else:
            spans, s = [], 0
            while s < len(x):
                e = min(s + self.window, len(x))
                spans.append((max(s - self.context, 0), min(e + self.context, len(x)), s // hop, e // hop))
                s = e
        out = np.zeros((n_frames, 32), np.float32)
        filled = 0
        for a, b, f0, f1 in spans:
            seg = x[a:b]
            seg = (seg - seg.mean()) / np.sqrt(seg.var() + 1e-7)  # same normalisation as in training
            with torch.inference_mode():
                lg = self.model(torch.from_numpy(seg.astype(np.float32))[None].to(self.device)).logits[0]
            lp = torch.log_softmax(lg.float(), -1).cpu().numpy()
            off = a // hop
            k0, k1 = f0 - off, min(f1 - off, lp.shape[0], n_frames - f0 + (f0 - off))
            out[f0:f0 + (k1 - k0)] = lp[k0:k1]
            filled = max(filled, f0 + (k1 - k0))
        return out[:filled]  # never return frames the model did not produce

    # ------------------------------------------------------------------------------------------ decoding
    @staticmethod
    def _norm(lp: np.ndarray) -> np.ndarray:
        lp = lp.copy()
        lp[:, SPECIAL] = -1e4
        return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)

    def _decode(self, lp: np.ndarray, terms: list | None = None, normalized: bool = False) -> list[dict]:
        """Standard decoding, or glossary-boosted decoding when `terms` is given."""
        lp = lp if normalized else self._norm(lp)
        words = []
        if terms:
            from group9_ds_ai_lab.assistant.glossary import boosted_beam
            spans = boosted_beam(self.decoder, lp, terms, beam=self.beam, **self.glossary_params)
        elif self.decoder is None:
            from group9_ds_ai_lab.inference.decode import greedy_words
            _, gw = greedy_words(lp)
            spans = [(w["word"], int(round(w["start"] / FRAME_SEC)), int(round(w["end"] / FRAME_SEC))) for w in gw]
        else:
            beam = self.decoder.decode_beams(lp, beam_width=self.beam)[0]
            spans = [(w, s, e) for w, (s, e) in (beam.text_frames if hasattr(beam, "text_frames") else beam[2])]
        for w, s, e in spans:
            seg = lp[s:max(e, s + 1)]
            pmax, am = np.exp(seg.max(-1)), seg.argmax(-1)
            keep = (am != BLANK_ID) & (am != DELIM_ID)
            v = pmax[keep] if keep.any() else pmax
            words.append({"word": w, "start": s * FRAME_SEC, "end": e * FRAME_SEC,
                          "conf_min": float(v.min()), "conf_mean": float(v.mean())})
        return words

    def _segments(self, words: list[dict], max_s: float = 20.0, pause_s: float = 0.6) -> list[int]:
        """Utterance-like segments (split at pauses, at most ~max_s long) for the detector's utterance features,
        because the detector was trained on single LibriSpeech utterances (1-35 s)."""
        seg, sid, t0 = [], 0, None
        for i, w in enumerate(words):
            if i and (w["start"] - words[i - 1]["end"] > pause_s or w["end"] - t0 > max_s):
                sid += 1
                t0 = None
            t0 = w["start"] if t0 is None else t0
            seg.append(sid)
        return seg

    def risk(self, words: list[dict]) -> np.ndarray:
        """P(word is wrong) with exactly the feature definitions of scripts/error_detector.py."""
        if not words:
            return np.zeros(0)
        import pandas as pd
        w = pd.DataFrame(words)
        w["seg"] = self._segments(words)
        eps = 1e-4
        for k in ("mean", "min"):
            c = w[f"conf_{k}"].clip(eps, 1 - eps)
            w[f"logit_{k}"] = np.log(c / (1 - c))
        w["n_chars"] = w.word.str.len()
        w["dur"] = (w.end - w.start).clip(lower=0.02)
        w["chars_per_sec"] = w.n_chars / w.dur
        w["unigram_logp"] = w.word.map(self.unigram_logp).fillna(-8.0)
        g = w.groupby("seg")
        w["pos"] = g.cumcount()
        w["utt_n_words"] = g.word.transform("size")
        w["is_first"] = (w.pos == 0).astype(int)
        w["is_last"] = (w.pos == w.utt_n_words - 1).astype(int)
        w["utt_conf_mean"] = g.conf_mean.transform("mean")
        w["utt_conf_min"] = g.conf_mean.transform("min")
        w["rel_conf"] = w.conf_mean - w.utt_conf_mean
        return self.detector.predict_proba(w[self.features].to_numpy(np.float64))[:, 1]

    # ------------------------------------------------------------------------------------------ public API
    def transcribe(self, audio, sr: int | None = None, threshold: float | None = None, glossary=None) -> Transcript:
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
        raw = self._decode(lpn, normalized=True)
        t["decoding"] = time.perf_counter() - t0
        terms, suggestions = [], []
        if glossary:
            from group9_ds_ai_lab.assistant.glossary import Term, find_suggestions, term_in
            t0 = time.perf_counter()
            terms = glossary if all(isinstance(g, Term) for g in glossary) else self.parse_glossary(glossary)[0]
            if terms:
                bst = self._decode(lpn, terms=terms, normalized=True)
                suggestions, _ = find_suggestions(raw, bst, terms, lpn, self.glossary_max_cost)
                ws = [w["word"] for w in raw]
                for term in terms:
                    for k in term_in(ws, term):
                        for j in range(k, k + len(term.words)):
                            raw[j]["glossary"] = True
            t["glossary"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = self.risk(raw)
        t["error_detector"] = time.perf_counter() - t0
        dur = len(x) / SAMPLE_RATE
        words = [Word(w["word"], round(w["start"], 3), round(w["end"], 3), round(w["conf_mean"], 4), round(float(pe), 4),
                      bool(pe >= thr), round(max(w["start"] - self.replay_s, 0.0), 3), round(min(w["end"] + self.replay_s, dur), 3),
                      bool(w.get("glossary", False)))
                 for w, pe in zip(raw, p)]
        sug = []
        for s in suggestions:
            d = s.to_dict()
            d["replay_start"] = round(max(s.start - self.replay_s, 0.0), 3)
            d["replay_end"] = round(min(s.end + self.replay_s, dur), 3)
            sug.append(d)
        return Transcript(" ".join(w.word for w in words), words, round(dur, 3), {k: round(v, 4) for k, v in t.items()},
                          {**self.settings, "threshold": thr, **({"glossary": {**self.glossary_params,
                                                                     "max_acoustic_cost": self.glossary_max_cost}} if terms else {})},
                          [x.text for x in terms], sug)
