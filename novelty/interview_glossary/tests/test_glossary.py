"""Unit tests for the interview glossary and the glossary app's session logic (app2/; no model needed)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]  # novelty/interview_glossary
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "app2"))  # the glossary app: app2/core + app2/glossary_ext

import glossary as G  # noqa: E402
from group9_ds_ai_lab.data.text import VOCAB  # noqa: E402

def _session():
    """The app's session API: core/session.py plus the glossary additions in glossary_ext/session.py."""
    from types import SimpleNamespace

    from core import session as core_ss
    from glossary_ext import session as gloss_ss
    names = {**{k: getattr(core_ss, k) for k in dir(core_ss) if not k.startswith("__")},
             **{k: getattr(gloss_ss, k) for k in dir(gloss_ss) if not k.startswith("__")}}
    return SimpleNamespace(**names)


ULP = {"ROAD": -3.2, "WE": -2.0, "MET": -3.9, "NEAR": -3.5, "NEELA": -6.5, "PARK": -4.0, "STREET": -3.8}


def test_parse_limits_and_normalisation():
    raw = "Neeladri Road\nAniruddha, neeladri-road;\n\n" + "\n".join(f"Term{chr(65 + i % 26)}{chr(65 + i // 26)}" for i in range(30))
    terms, warn = G.parse_glossary(raw, ULP)
    assert len(terms) == G.MAX_TERMS
    assert terms[0].text == "NEELADRI ROAD" and terms[0].boost == ("NEELADRI",)  # common word ROAD not boosted
    assert terms[1].text == "ANIRUDDHA"
    assert sum("only the first" in w for w in warn) > 0
    t, _ = G.parse_glossary(["Park Street"], ULP)
    assert t[0].boost == ("PARK",)  # all-common term: rarest word only
    t, w = G.parse_glossary(["one two three four five"], ULP)
    assert not t and w


def _words(text, t0=0.0):
    return [{"word": w, "start": t0 + 0.4 * i, "end": t0 + 0.4 * i + 0.3} for i, w in enumerate(text.split())]


def test_suggestion_only_for_glossary_terms():
    terms, _ = G.parse_glossary(["Neeladri Road", "Aniruddha"], ULP)
    std = _words("WE MET NEAR NEELA ROAD AND THE CAT SAT")
    bst = _words("WE MET NEAR NEELADRI ROAD AND THE HAT SAT")  # CAT->HAT is not a glossary change
    kept, dropped = G.find_suggestions(std, bst, terms, lp=None)
    assert len(kept) == 1 and not dropped
    s = kept[0]
    assert s.standard == "NEELA ROAD" and s.suggested == "NEELADRI ROAD" and s.terms == ["NEELADRI ROAD"]
    assert s.std_start == 3 and s.std_end == 5 and abs(s.start - std[3]["start"]) < 1e-3


def test_no_suggestion_when_term_already_present_or_identical():
    terms, _ = G.parse_glossary(["Neeladri Road"], ULP)
    std = _words("WE MET NEAR NEELADRI ROAD")
    assert G.find_suggestions(std, std, terms)[0] == []


def _lp_for(text, frames_per_char=3, T_extra=4):
    """Synthetic log-probs that spell `text` (with blanks) very clearly."""
    ids = [VOCAB["|"] if c == " " else VOCAB[c] for c in text]
    rows = []
    for i in ids:
        rows += [i] * frames_per_char + [0]
    rows += [0] * T_extra
    lp = np.full((len(rows), 32), np.log(1e-4), np.float32)
    for t, i in enumerate(rows):
        lp[t, i] = np.log(1 - 31e-4)
    return lp


def test_audio_check_rejects_unspoken_term():
    lp = _lp_for("WE MET NEAR NEELA ROAD")
    T = lp.shape[0]
    std = [{"word": "NEELA", "start": 0.0, "end": T * 0.02}]
    fake = [{"word": "ANIRUDDHA", "start": 0.0, "end": T * 0.02}]
    terms, _ = G.parse_glossary(["Aniruddha"], ULP)
    kept, dropped = G.find_suggestions(std, fake, terms, lp, max_cost=8.0)
    assert not kept and len(dropped) == 1
    good = G.acoustic_support(lp, 0, T, "WE MET NEAR NEELA ROAD", "WE MET NEAR NEELA ROAD")
    assert abs(good) < 1e-3


def test_session_buffer_cut_and_review():
    ss = _session()
    S = ss.new_session()
    rng = np.random.default_rng(0)
    speech = lambda s: (0.3 * rng.standard_normal(int(s * ss.SR))).astype(np.float32)  # noqa: E731
    quiet = lambda s: (0.001 * rng.standard_normal(int(s * ss.SR))).astype(np.float32)  # noqa: E731
    for part in (speech(6.5), quiet(0.6), speech(3.0)):
        for k in range(0, len(part), 16000):
            ss.append_audio(S, part[k:k + 16000])
    assert S["n"] == int(10.1 * ss.SR)
    cut = ss.find_cut(S)
    assert cut is not None and 6.5 * ss.SR <= cut <= 7.1 * ss.SR
    # review operations on a hand-made transcript
    S["words"] = [{"id": i, "word": w, "start": i * 0.5, "end": i * 0.5 + 0.4, "conf": 0.9, "p_error": p,
                   "glossary": False, "original": None, "deleted": False, "accepted": None}
                  for i, (w, p) in enumerate([("WE", .1), ("MET", .2), ("NEELA", .8), ("ROAD", .3)])]
    S["next_id"] = 4
    S["suggestions"] = [{"id": 0, "word_ids": [2, 3], "after_id": 1, "standard": "NEELA ROAD",
                         "suggested": "NEELADRI ROAD", "terms": ["NEELADRI ROAD"],
                         "words": [{"word": "NEELADRI", "start": 1.0, "end": 1.5}, {"word": "ROAD", "start": 1.5, "end": 1.9}],
                         "start": 1.0, "end": 1.9, "support": -0.5, "status": "pending", "replaced": None}]
    hl, ids = ss.highlighted(S, 0.5)
    assert [l for _, l in hl] == [None, None, "glossary suggestion", "glossary suggestion"]
    ss.accept(S, 0)
    assert ss.final_text(S) == "WE MET NEELADRI ROAD"
    ss.undo(S, 0)
    assert ss.final_text(S) == "WE MET NEELA ROAD"
    ss.reject(S, 0)
    assert ss.final_text(S) == "WE MET NEELA ROAD"
    ss.edit_word(S, 1, "MEET")
    ss.edit_word(S, 0, "")
    assert ss.final_text(S) == "MEET NEELA ROAD"
    ss.revert_word(S, 0)
    ss.revert_word(S, 1)
    assert ss.final_text(S) == "WE MET NEELA ROAD"
    rows = ss.queue_rows(S, 0.5)
    assert rows[0][2] == "glossary" and rows[0][-1] == "rejected"


def test_manual_edit_into_several_words():
    ss = _session()
    S = ss.new_session()
    S["words"] = [{"id": i, "word": w, "start": i * 0.5, "end": i * 0.5 + 0.4, "conf": 0.9, "p_error": 0.1,
                   "glossary": False, "original": None, "deleted": False, "accepted": None}
                  for i, w in enumerate(["WE", "MET", "NEELADROAD", "TODAY"])]
    S["next_id"] = 4
    ss.edit_word(S, 2, "Neeladri Road")
    hl, ids = ss.highlighted(S, 0.5)
    assert [t.strip() for t, _ in hl] == ["WE", "MET", "Neeladri", "Road", "TODAY"]
    assert [l for _, l in hl][2:4] == ["edited", "edited"] and len(set(ids)) == 5  # two separate clickable words
    new = ss.word(S, ids[3])
    assert abs(new["start"] - 1.2) < 1e-6 and abs(new["end"] - 1.4) < 1e-6  # time span shared out
    assert ss.final_text(S) == "WE MET Neeladri Road TODAY"
    assert ss.queue_rows(S, 0.5)[0][3:5] == ["NEELADROAD", "Neeladri Road"]
    ss.edit_word(S, 2, "Neeladri Rd Junction")  # re-editing replaces the earlier split
    assert ss.final_text(S) == "WE MET Neeladri Rd Junction TODAY"
    _, ids = ss.highlighted(S, 0.5)
    ss.edit_word(S, ids[3], "")  # deleting an added word removes it
    assert ss.final_text(S) == "WE MET Neeladri Junction TODAY"
    ss.revert_word(S, 2)  # revert restores the single original word
    assert ss.final_text(S) == "WE MET NEELADROAD TODAY" and len(S["words"]) == 4
    assert ss.word(S, 2)["end"] == 1.4


def test_boosted_view_and_glossary_status():
    ss = _session()
    S = ss.new_session()
    S["terms"], _ = G.parse_glossary(["Neeladri Road", "Ram Vilas Paswan", "Aniruddha"], ULP)
    S["standard"] = _words("WE MET NEAR NEELA ROAD WITH RAM")
    S["words"] = [{"id": i, **w, "conf": 0.9, "p_error": 0.1, "glossary": False, "original": None, "deleted": False,
                   "accepted": None} for i, w in enumerate(S["standard"])]
    S["next_id"] = len(S["words"])
    bw = _words("WE MET NEAR NEELADRI ROAD WITH RAM VILAS")
    S["boosted"] = [{"id": j, **w, "diff": j in (3, 7)} for j, w in enumerate(bw)]
    base = {"after_id": 2, "terms": ["NEELADRI ROAD"], "words": bw[3:5], "start": 1.2, "end": 1.9,
            "status": "pending", "replaced": None}
    S["suggestions"] = [{**base, "id": 0, "strength": "weak", "b_ids": [3, 4], "word_ids": [3, 4],
                         "standard": "NEELA ROAD", "suggested": "NEELADRI ROAD", "support": -10.0}]
    hb, bids = ss.boosted_highlighted(S)
    assert [l for _, l in hb][3:5] == ["weak suggestion"] * 2 and hb[7][1] == "differs"
    assert ss.suggestion_of_boosted(S, 4)["id"] == 0
    assert [r for r in ss.queue_rows(S, 0.5) if r[2] == "glossary"] == []  # weak: not in the queue
    hl, _ = ss.highlighted(S, 0.5)
    assert hl[3][1] is None  # and not highlighted as a suggestion in transcript A
    st = {g["term"]: g for g in ss.glossary_status(S)}
    assert len(st["NEELADRI ROAD"]["suggestions"]) == 1
    assert [w for _, w in st["RAM VILAS PASWAN"]["partial"]] == ["RAM", "VILAS"]
    assert not st["ANIRUDDHA"]["partial"] and "not heard" in ss.glossary_status_md(S).split("ANIRUDDHA")[1]
    ss.accept(S, 0)  # the reviewer can still accept a weak suggestion
    assert ss.final_text(S) == "WE MET NEAR NEELADRI ROAD WITH RAM"


def _timed(spec):
    """'WORD@start-end ...' -> word dicts (both transcripts come from the same audio, so times line up)."""
    out = []
    for item in spec.split():
        w, t = item.split("@")
        a, b = map(float, t.split("-"))
        out.append({"word": w, "start": a, "end": b})
    return out


def test_split_one_suggestion_per_term():
    terms, _ = G.parse_glossary(["Gulshan Kumar", "Uttar Pradesh"], ULP)
    std = _timed("MY@0-.2 NAME@.3-.5 IS@.6-.7 GOSHEN@.8-1.2 COME@1.3-1.6 ALIVE@1.7-2.0 IN@2.1-2.2 UTAPREDACE@2.3-3.0 AND@3.1-3.2")
    bst = _timed("MY@0-.2 NAME@.3-.5 IS@.6-.7 GULSHAN@.8-1.2 KUMAR@1.3-1.6 A@1.7-1.75 LIVIN@1.8-2.2 UTTAR@2.3-2.6 "
                 "PRADESH@2.7-3.0 AND@3.1-3.2")
    old = G.candidates(std, bst, terms, None, split=False, near=False)
    assert [c.standard for c in old] == ["GOSHEN COME ALIVE IN UTAPREDACE"]  # first version: one big suggestion
    new = G.candidates(std, bst, terms, None, split=True, near=False)
    assert [(c.standard, c.suggested) for c in new] == [("GOSHEN COME", "GULSHAN KUMAR"), ("UTAPREDACE", "UTTAR PRADESH")]


def test_near_match_and_selection():
    terms, _ = G.parse_glossary(["Ram Vilas Paswan", "Aniruddha"], ULP)
    assert G.sound_key("GULCHEN") == G.sound_key("GULSHAN")[:-2] + "IN"  # CH ~ SH, A/E/I vowels
    assert G.spelling_distance("RAMMILASPASRAN", "RAMVILASPASWAN") < 0.2
    std = _timed("THE@0-.1 CHIEF@.2-.5 MINISTER@.6-1.0 RAMMILASPASRAN@1.1-2.0 AND@2.1-2.2 HIS@2.3-2.4 SON@2.5-2.7")
    bst = _timed("THE@0-.1 CHIEF@.2-.5 MINISTER@.6-1.0 RAM@1.1-1.3 MELAS@1.4-1.6 PASWAN@1.7-2.0 AND@2.1-2.2 HIS@2.3-2.4 "
                 "SON@2.5-2.7")  # B got the term only partly
    assert G.candidates(std, bst, terms, None, near=False) == []
    c = G.candidates(std, bst, terms, None, near=True)
    assert len(c) == 1 and c[0].kind == "near" and c[0].standard == "RAMMILASPASRAN" and c[0].suggested == "RAM VILAS PASWAN"
    assert [w["word"] for w in c[0].words] == ["RAM", "VILAS", "PASWAN"]
    kept, _ = G.select(c, near_dist=0.2)
    assert len(kept) == 1
    kept, _ = G.select(c, near_dist=0.05)  # too far for a strict setting
    assert kept == []
    # overlapping candidates: the diff one wins, a weak diff one is not kept
    d = G.Suggestion(3, 4, "X", "Y", [], ["T"], 1.0, 1.4, -1.0, kind="diff")
    n = G.Suggestion(3, 5, "X Z", "Y", [], ["T"], 1.0, 1.8, -0.5, kind="near", dist=0.05)
    w = G.Suggestion(6, 7, "Q", "R", [], ["T"], 2.0, 2.4, -20.0, kind="diff")
    kept, weak = G.select([n, d, w], max_cost=4, near_dist=0.1, near_cost=16)
    assert kept == [d] and weak == [w]
