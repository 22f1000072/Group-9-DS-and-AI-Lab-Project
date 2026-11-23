"""Session logic of the live transcription app (no UI code here, so it can be tested on its own).

A session holds the audio recorded so far and the transcript built from it:
  * audio arrives in chunks (microphone, every ~1 s) or all at once (uploaded file)
  * it is cut into segments at pauses between words (6-15 s long); each finished segment is transcribed right away by the
    noise-robust model (+ LM, word-error detector and glossary), so work starts while the speaker is still talking
  * the not-yet-finished tail is shown as a fast greedy preview ("listening ...")
  * the reviewer can replay any word, edit / delete / revert it, and accept / reject glossary suggestions
The session is a plain dict (numpy arrays + lists), so it can live in a gr.State.
"""
from __future__ import annotations

import json
import threading

import numpy as np

SR = 16000
FRAME = 320                  # 20 ms
MIN_SEG_S, MAX_SEG_S = 6.0, 15.0
PAUSE_S = 0.3
CTX_L, CTX_R = 4.0, 1.0       # context (s) decoded around each segment; only the segment's own words are kept
# (chosen on 20 one-minute noisy recordings from test-other: live WER 13.97 % vs 14.26 % decoding the whole file)
REPLAY_S = 1.0
LOCK = threading.Lock()      # one model call at a time (the model is shared by all browser sessions)


# ------------------------------------------------------------------------------------------ audio buffer
def new_session() -> dict:
    return {"buf": np.zeros(SR * 60, np.float32), "n": 0, "committed": 0, "words": [], "suggestions": [],
            "partial": "", "partial_at": 0, "next_id": 0, "segments": 0, "proc_s": 0.0, "stopped": False,
            "glossary": [], "source": None}


def append_audio(S: dict, x: np.ndarray) -> None:
    """Append 16 kHz mono float32 audio (amortised O(1): the buffer doubles when full)."""
    n = S["n"] + len(x)
    if n > len(S["buf"]):
        nb = np.zeros(max(n, 2 * len(S["buf"])), np.float32)
        nb[: S["n"]] = S["buf"][: S["n"]]
        S["buf"] = nb
    S["buf"][S["n"]:n] = x
    S["n"] = n


def audio(S: dict, a: float = 0.0, b: float | None = None) -> np.ndarray:
    i, j = int(a * SR), S["n"] if b is None else min(int(b * SR), S["n"])
    return S["buf"][max(i, 0):j]


def duration(S: dict) -> float:
    return S["n"] / SR


def find_cut(S: dict, final: bool = False) -> int | None:
    """Sample index where the next segment ends, or None if more audio is needed.
    Cuts in the middle of the first pause (>= 0.3 s of low energy) after 4 s; forces a cut at the quietest
    point of the last 5 s once 15 s are pending (noisy audio may have no clear pause)."""
    c, n = S["committed"], S["n"]
    pend = n - c
    if final:
        return n if pend > 0 else None
    if pend < MIN_SEG_S * SR:
        return None
    x = S["buf"][c:n]
    nf = len(x) // FRAME
    e = 10 * np.log10(np.mean(x[: nf * FRAME].reshape(nf, FRAME).astype(np.float64) ** 2, axis=1) + 1e-10)
    lo, hi = np.percentile(e, 10), np.percentile(e, 90)
    quiet = e < lo + 0.3 * (hi - lo)
    need = int(PAUSE_S / 0.02)
    f = 0
    while f < nf:
        if not quiet[f]:
            f += 1
            continue
        g = f
        while g + 1 < nf and quiet[g + 1]:
            g += 1
        mid = (f + g) // 2
        if g - f + 1 >= need and mid * 0.02 >= MIN_SEG_S:
            return c + mid * FRAME
        f = g + 1
    if pend >= MAX_SEG_S * SR:
        a, b = int((MAX_SEG_S - 5) / 0.02), int(MAX_SEG_S / 0.02)
        k = a + int(np.argmin(np.convolve(e[a:b], np.ones(5) / 5, mode="same")))
        return c + k * FRAME
    return None


# ------------------------------------------------------------------------------------------ transcription
def commit(S: dict, A, cut: int, terms: list, thr: float) -> bool:
    """Transcribe audio [committed, cut) and append its words / glossary suggestions (global times).
    The model sees CTX_L s before and up to CTX_R s after the segment, so words at the cut keep their context;
    a word belongs to the segment whose span contains the word's centre."""
    c = S["committed"]
    S["committed"] = cut
    if cut - c < 0.25 * SR:
        return False
    a, b = max(c - int(CTX_L * SR), 0), min(cut + int(CTX_R * SR), S["n"])
    with LOCK:
        tr = A.transcribe(S["buf"][a:b].copy(), glossary=terms or None, threshold=thr)
    off = a / SR
    lo, hi = (c - a) / SR, (cut - a) / SR
    own = [lo <= (w.start + w.end) / 2 < hi for w in tr.words]
    S["proc_s"] += sum(tr.timings_s.values())
    S["segments"] += 1
    prev_last = S["words"][-1]["id"] if S["words"] else None
    ids = []
    for w, keep in zip(tr.words, own):
        if not keep:
            ids.append(None)
            continue
        S["words"].append({"id": S["next_id"], "word": w.word, "start": round(w.start + off, 3),
                           "end": round(w.end + off, 3), "conf": w.conf, "p_error": w.p_error,
                           "glossary": w.glossary, "original": None, "deleted": False, "accepted": None})
        ids.append(S["next_id"])
        S["next_id"] += 1
    for s in tr.suggestions:
        span = own[s["std_start"]:s["std_end"]]
        mid = (s["start"] + s["end"]) / 2
        if (span and not all(span)) or not (lo <= mid < hi):
            continue  # belongs to the neighbouring segment
        S["suggestions"].append({
            "id": len(S["suggestions"]), "word_ids": ids[s["std_start"]:s["std_end"]],
            "after_id": next((i for i in reversed(ids[:s["std_start"]]) if i is not None), prev_last),
            "standard": s["standard"], "suggested": s["suggested"], "terms": s["terms"],
            "words": [{**w, "start": round(w["start"] + off, 3), "end": round(w["end"] + off, 3)} for w in s["words"]],
            "start": round(s["start"] + off, 3), "end": round(s["end"] + off, 3), "support": s["support"],
            "status": "pending", "replaced": None})
    return True


def preview(S: dict, A) -> list[dict]:
    """Fast greedy pass over the audio not yet in a finished segment: sets the live preview text and returns
    its words (times relative to the start of the pending audio)."""
    from group9_ds_ai_lab.inference.decode import greedy_words
    c, n = S["committed"], S["n"]
    if n - c < 0.4 * SR:
        S["partial"] = ""
        return []
    e = min(n, c + int((MAX_SEG_S + 3) * SR))  # an uploaded file: look at most 18 s ahead
    with LOCK:
        lp = A.log_probs(S["buf"][c:e].copy())
    S["partial"], words = greedy_words(A._norm(lp))
    S["partial_at"] = n
    S["view_s"] = (e - c) / SR
    return words


def find_cut_ctc(S: dict, words: list[dict]) -> int | None:
    """Cut between two words, using the model's own word timings (robust in noise, where loudness-based pause
    detection fails): the first gap of >= 0.25 s after 6 s that is not at the still-growing end; after 15 s of
    pending audio, the widest gap before 15 s (or the energy-based fallback if there is none)."""
    c, n = S["committed"], S["n"]
    pend = min((n - c) / SR, S.get("view_s", np.inf))
    if pend < MIN_SEG_S:
        return None
    gaps = [(words[i + 1]["start"] - words[i]["end"], (words[i]["end"] + words[i + 1]["start"]) / 2)
            for i in range(len(words) - 1)]
    for g, mid in gaps:
        if g >= 0.25 and MIN_SEG_S <= mid <= pend - max(0.8, CTX_R):
            return c + int(mid * SR)
    if words and pend - words[-1]["end"] >= 0.8 and words[-1]["end"] >= MIN_SEG_S:  # speaker stopped talking
        return c + int((words[-1]["end"] + 0.3) * SR)
    if pend >= MAX_SEG_S:
        inside = [(g, mid) for g, mid in gaps if 2.0 <= mid <= MAX_SEG_S]
        if inside:
            return c + int(max(inside)[1] * SR)
        if not words:
            return c + int(MAX_SEG_S * SR)
        return find_cut(S)
    return None


def process(S: dict, A, terms: list, thr: float, final: bool = False, max_segments: int = 1) -> int:
    """Commit up to `max_segments` finished segments; returns how many were committed."""
    k = 0
    while k < max_segments:
        if final:
            cut = S["n"] if S["n"] > S["committed"] else None
        else:
            cut = find_cut_ctc(S, preview(S, A))
        if cut is None:
            break
        commit(S, A, cut, terms, thr)
        k += 1
    if final:
        S["partial"] = ""
    elif k:
        preview(S, A)
    return k


# ------------------------------------------------------------------------------------------ review / editing
def _pos(S, wid):
    for i, w in enumerate(S["words"]):
        if w["id"] == wid:
            return i
    return None


def word(S, wid):
    i = _pos(S, wid)
    return None if i is None else S["words"][i]


def pending_of(S) -> dict:
    """word id -> pending suggestion id."""
    return {wid: s["id"] for s in S["suggestions"] if s["status"] == "pending" for wid in s["word_ids"]}


def _added_to(S, wid) -> list:
    """Words that a manual edit of `wid` added after it (when one word was corrected into several)."""
    return [w for w in S["words"] if w.get("added_to") == wid]


def _drop_added(S, wid) -> None:
    for w in _added_to(S, wid):
        _drop_added(S, w["id"])
        S["words"].remove(w)


def edit_word(S, wid, text: str) -> None:
    """Correct a word. Several words of text become separate words (each clickable), sharing the
    original word's time span; an empty text deletes the word."""
    w = word(S, wid)
    if w is None:
        return
    parts = str(text or "").split()
    if w.get("added_to") is not None and not parts:  # a word added by an earlier edit: just remove it
        _drop_added(S, wid)
        S["words"].remove(w)
        return
    _drop_added(S, wid)
    if w["original"] is None:
        w["original"] = w["word"]
        w["span"] = (w["start"], w["end"])
    a, b = w.get("span") or (w["start"], w["end"])
    if not parts:
        w["deleted"] = True
    else:
        n, step = len(parts), (b - a) / len(parts)
        w["word"], w["deleted"], w["start"], w["end"] = parts[0], False, a, a + step
        at = _pos(S, wid) + 1
        extra = []
        for k, t in enumerate(parts[1:], 1):
            extra.append({"id": S["next_id"], "word": t, "start": a + k * step, "end": a + (k + 1) * step,
                          "conf": None, "p_error": 0.0, "glossary": False, "original": None, "deleted": False,
                          "accepted": None, "added_to": wid})
            S["next_id"] += 1
        S["words"][at:at] = extra
    if w["word"] == w["original"] and not w["deleted"] and not _added_to(S, wid):
        w["original"] = None
        w["start"], w["end"] = w.pop("span", (w["start"], w["end"]))


def revert_word(S, wid) -> None:
    """Undo a manual edit. On a word that an edit added, removes just that word."""
    w = word(S, wid)
    if w is None:
        return
    if w.get("added_to") is not None:
        _drop_added(S, wid)
        S["words"].remove(w)
        return
    _drop_added(S, wid)
    if w["original"] is not None:
        w["word"], w["original"] = w["original"], None
        w["start"], w["end"] = w.pop("span", (w["start"], w["end"]))
    w["deleted"] = False


def edited_text(S, wid) -> str:
    """The full correction of a word: the word plus any words its edit added."""
    w = word(S, wid)
    if w is None or w["deleted"]:
        return ""
    return " ".join([w["word"]] + [edited_text(S, x["id"]) for x in _added_to(S, wid)]).strip()


def accept(S, sid) -> None:
    s = S["suggestions"][sid]
    if s["status"] != "pending":
        return
    ids = set(s["word_ids"])
    for w in S["words"]:  # words added by manual edits of the replaced words go too
        if w.get("added_to") in ids:
            ids.add(w["id"])
    pos = [p for p in (_pos(S, i) for i in ids) if p is not None]
    if pos:
        at = min(pos)
    else:
        a = _pos(S, s["after_id"]) if s["after_id"] is not None else -1
        at = (a if a is not None else -1) + 1
    s["replaced"] = [S["words"][p] for p in sorted(pos)]
    for p in sorted(pos, reverse=True):
        del S["words"][p]
    new = []
    for w in s["words"]:
        new.append({"id": S["next_id"], "word": w["word"], "start": w["start"], "end": w["end"], "conf": None,
                    "p_error": 0.0, "glossary": True, "original": None, "deleted": False, "accepted": sid})
        S["next_id"] += 1
    S["words"][at:at] = new
    s["status"], s["new_ids"] = "accepted", [w["id"] for w in new]


def reject(S, sid) -> None:
    if S["suggestions"][sid]["status"] == "pending":
        S["suggestions"][sid]["status"] = "rejected"


def undo(S, sid) -> None:
    """Put an accepted / rejected suggestion back to pending (restores the standard words)."""
    s = S["suggestions"][sid]
    if s["status"] == "accepted":
        pos = [p for p in (_pos(S, i) for i in s.get("new_ids", [])) if p is not None]
        at = min(pos) if pos else len(S["words"])
        for p in sorted(pos, reverse=True):
            del S["words"][p]
        S["words"][at:at] = s["replaced"] or []
    s["status"], s["replaced"] = "pending", None


def final_text(S) -> str:
    return " ".join(w["word"] for w in S["words"] if not w["deleted"])


def label(w, thr, pend) -> str | None:
    if w["original"] is not None or w.get("added_to") is not None:
        return "edited"
    if w["accepted"] is not None:
        return "glossary accepted"
    if w["id"] in pend:
        return "glossary suggestion"
    if w["p_error"] >= max(thr, 0.7):
        return "likely wrong"
    if w["p_error"] >= thr:
        return "check"
    if w["glossary"]:
        return "glossary term"
    return None


def highlighted(S, thr) -> tuple[list, list]:
    """(HighlightedText value, word id for each item; None for the live preview)."""
    pend = pending_of(S)
    out, ids = [], []
    for w in S["words"]:
        if w["deleted"]:
            continue
        out.append((w["word"] + " ", label(w, thr, pend)))
        ids.append(w["id"])
    if S["partial"]:
        out.append((S["partial"].lower() + " …", "listening…"))
        ids.append(None)
    return out, ids


QUEUE_COLS = ["ref", "time", "type", "heard", "suggestion", "score", "status"]


def queue_rows(S, thr) -> list[list]:
    """Review queue: glossary suggestions + flagged / edited words, in time order."""
    rows = []
    for s in S["suggestions"]:
        sup = s["support"]
        rows.append((s["start"], [f"s{s['id']}", _fmt(s["start"]), "glossary", s["standard"] or "(nothing)",
                                  s["suggested"], f"audio {sup:+.1f}" if isinstance(sup, (int, float)) and np.isfinite(sup) else "",
                                  s["status"]]))
    pend = pending_of(S)
    for w in S["words"]:
        if w["id"] in pend or w["accepted"] is not None or w.get("added_to") is not None:
            continue
        if w["p_error"] >= thr or w["original"] is not None:
            st = "deleted" if w["deleted"] else ("edited" if w["original"] is not None else "to check")
            rows.append((w["start"], [f"w{w['id']}", _fmt(w["start"]), "flag" if w["p_error"] >= thr else "edit",
                                      w["original"] or w["word"], "" if w["original"] is None else ("-" if w["deleted"] else edited_text(S, w["id"])),
                                      f"risk {100 * w['p_error']:.0f} %", st]))
    return [r for _, r in sorted(rows, key=lambda t: t[0])]


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def stats(S, thr) -> str:
    vis = [w for w in S["words"] if not w["deleted"]]
    nf = sum(w["p_error"] >= thr for w in vis)
    np_ = sum(s["status"] == "pending" for s in S["suggestions"])
    d = duration(S)
    rtf = S["proc_s"] / max(S["committed"] / SR, 1e-9)
    return (f"**Audio** {_fmt(d)} · **words** {len(vis)} · **flagged** {nf} · **glossary suggestions** "
            f"{np_} pending / {len(S['suggestions'])} · **edits** {sum(w['original'] is not None for w in S['words'])} · "
            f"**segments** {S['segments']} · real-time factor {rtf:.2f}")


def to_srt(S, max_words=12, max_gap=0.8) -> str:
    lines, cur = [], []
    for w in (w for w in S["words"] if not w["deleted"]):
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > max_gap):
            lines.append(cur)
            cur = []
        cur.append(w)
    if cur:
        lines.append(cur)

    def ts(t):
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    return "\n".join(f"{i}\n{ts(ws[0]['start'])} --> {ts(ws[-1]['end'])}\n{' '.join(w['word'] for w in ws)}\n"
                     for i, ws in enumerate(lines, 1))


def to_json(S, thr, settings: dict) -> str:
    return json.dumps({"text": final_text(S), "duration_s": round(duration(S), 3), "threshold": thr,
                       "glossary": S["glossary"], "settings": settings,
                       "words": [{**{k: w[k] for k in ("word", "start", "end", "conf", "p_error", "original", "deleted",
                                                       "glossary", "accepted")}, "added_by_edit_of": w.get("added_to")}
                                 for w in S["words"]],
                       "glossary_suggestions": [{k: v for k, v in s.items() if k not in ("replaced",)}
                                                for s in S["suggestions"]]}, indent=2, default=float)
