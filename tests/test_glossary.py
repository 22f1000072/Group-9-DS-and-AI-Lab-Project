"""Unit tests for the interview glossary and the live-app session logic (no model needed)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "deployment" / "live_app"))

from group9_ds_ai_lab.assistant import glossary as G  # noqa: E402
from group9_ds_ai_lab.data.text import VOCAB  # noqa: E402

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
    import session as ss
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
    import session as ss
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
