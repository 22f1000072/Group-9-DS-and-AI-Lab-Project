"""Evaluate the interview glossary (novelty): the same held-out noisy clips decoded with the glossary OFF and ON.

Uses the stored CTC log-probabilities (scripts/dump_logits.py), so the acoustic model is not re-run.
Every clip gets a 20-term glossary (the app's maximum):
  spoken terms   : up to 2 rare words of the clip's reference (unigram log10 p < -5 in the LM, or unknown to the LM),
                   i.e. names and unusual words, the kind of term an interviewer would type
  unspoken terms : the rest, drawn from rare words of OTHER chapters (not in this clip) + Indian names / places
                   that never occur in LibriSpeech (like "Aniruddha" in the brief)
The same utterance gets the same glossary in every noise condition (paired comparison).

Transcripts compared (word lists):
  off        : standard beam search + 3-gram LM (the evaluated system)
  boosted    : glossary-boosted beam search, used as is (no review)            -> for reference only
  accept_all : off + every glossary suggestion accepted                         -> pessimistic reviewer
  oracle     : off + only the suggestions that are right accepted               -> careful reviewer
Metrics: spoken-term accuracy (% of spoken glossary-term occurrences transcribed correctly), unspoken-term rate
(% of (clip, unspoken term) pairs where the term appears in the output), suggestions per minute, suggestion
precision, WER, decoding time.

Two versions of the suggestion step are compared on the same decodes:
  v1 : one suggestion per differing stretch that contains a complete glossary term (first version)
  v2 : one suggestion per term occurrence + "near" matches (spans spelled close to a term, checked against the audio)

Usage:
  python novelty/interview_glossary/glossary_eval.py --set dev  --grid            # v1: choose weight and max acoustic cost on dev
  python novelty/interview_glossary/glossary_eval.py --set test --weight W --max-cost C   # v1: held-out test, unseen noise + clean
  python novelty/interview_glossary/glossary_eval.py --set dev  --grid-v2         # v2: choose max cost / near distance / near cost
  python novelty/interview_glossary/glossary_eval.py --set test --v2              # v1 and v2 on the held-out test (chosen settings)
Outputs -> experiments/results/glossary/
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glossary as G  # noqa: E402
from group9_ds_ai_lab.assistant.engine import LMS, _arpa_unigrams, _ctc_labels  # noqa: E402
from group9_ds_ai_lab.inference.decode import FRAME_SEC  # noqa: E402
from group9_ds_ai_lab.utils.paths import EVAL, PROCESSED  # noqa: E402

SPECIAL = (1, 2, 3)
OUT = EVAL / "glossary"
INDIAN = ["Aniruddha", "Neeladri Road", "Kothapalli", "Subramaniam", "Vishwanathan", "Chattopadhyay", "Malleswaram",
          "Thiruvanmiyur", "Bhubaneswar", "Priyadarshini", "Kondapur", "Rajarajeswari", "Gulmohar Colony",
          "Hiranandani", "Venkataraman", "Purushottam", "Srinivasan", "Lakshmipuram", "Basavanagudi", "Dwarakanath",
          "Jayalakshmi", "Pattabhiram", "Ramakrishnapuram", "Indiranagar"]
_G = {}


def utt_logp(i):
    lp = _G["logp"][_G["off"][i]:_G["off"][i + 1]].astype(np.float32)
    lp[:, SPECIAL] = -1e4
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)


def rare_words(text, ulp, thr=-5.0):
    ws = [w for w in text.split() if len(w) >= 4 and "'" not in w and ulp.get(w, -99.0) < thr]
    return sorted(set(ws), key=lambda w: ulp.get(w, -99.0))


def build_glossaries(meta, ulp, size=20, seed=0):
    """audio_id -> (spoken terms, unspoken terms)."""
    utt = meta.drop_duplicates("audio_id")[["audio_id", "text", "chapter_id"]]
    pool = {}  # chapter -> rare words
    for r in utt.itertuples():
        pool.setdefault(r.chapter_id, set()).update(rare_words(r.text, ulp))
    chapters = sorted(pool)
    out = {}
    for r in utt.itertuples():
        rng = np.random.default_rng(G_seed(r.audio_id, seed))
        spoken = rare_words(r.text, ulp)[:2]
        ref = set(r.text.split())
        # sample other-chapter rare words cheaply: pick random chapters, then a word from each
        cand = []
        while len(cand) < 12:
            c = chapters[int(rng.integers(len(chapters)))]
            if c == r.chapter_id or not pool[c]:
                continue
            w = sorted(pool[c])[int(rng.integers(len(pool[c])))]
            if w not in ref and w not in cand:
                cand.append(w)
        ind = [INDIAN[k] for k in rng.permutation(len(INDIAN))[: size - len(spoken) - len(cand)]]
        out[r.audio_id] = (spoken, cand + [G.normalize_text(x) for x in ind])
    return out


def G_seed(*parts):
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def spans_to_words(spans):
    return [{"word": w, "start": s * FRAME_SEC, "end": e * FRAME_SEC} for w, s, e in spans]


def work(i):
    lp = utt_logp(i)
    dec, beam = _G["decoder"], _G["beam"]
    t0 = time.perf_counter()
    std = spans_to_words(G._frames(dec.decode_beams(lp, beam_width=beam)[0]))
    t_std = time.perf_counter() - t0
    spoken, unspoken = _G["gloss"][_G["ids"][i]]
    terms, _ = G.parse_glossary(spoken + unspoken, _G["ulp"])
    res = {"i": i, "std": [w["word"] for w in std], "t_std": t_std, "runs": {}}
    for wgt in _G["weights"]:
        t0 = time.perf_counter()
        bst = spans_to_words(G.boosted_beam(dec, lp, terms, beam=beam, weight=wgt, oov_logp=_G["oov_logp"]))
        if _G.get("v2"):
            c1 = G.candidates(std, bst, terms, lp, split=False, near=False)
            t1 = time.perf_counter()
            c2 = G.candidates(std, bst, terms, lp, split=True, near=True)
            res["runs"][wgt] = {"bst": [w["word"] for w in bst], "t": t1 - t0, "t_v2": time.perf_counter() - t0,
                                "cands": {"v1": c1, "v2": c2}}
        else:
            kept, _ = G.find_suggestions(std, bst, terms, lp, max_cost=np.inf, split=False, near=False)
            res["runs"][wgt] = {"bst": [w["word"] for w in bst], "t": time.perf_counter() - t0,
                                "sug": [(s.std_start, s.std_end, s.suggested, s.terms, float(s.support)) for s in kept]}
    return res


def count_term(words, term):
    tw = term.split()
    n = len(tw)
    return sum(words[k:k + n] == tw for k in range(len(words) - n + 1))


def apply(std, sugs):
    out = list(std)
    for s0, s1, sug, *_ in sorted(sugs, key=lambda s: -s[0]):
        out[s0:s1] = sug.split()
    return out


def edits(ref, hyp):
    import jiwer
    if not hyp:
        return len(ref.split())
    o = jiwer.process_words(ref, " ".join(hyp))
    return o.substitutions + o.deletions + o.insertions


def score(meta, results, gloss, wgt, max_cost, v2=None):
    """v2: None (v1 results), or (version, max_cost, near_dist, near_cost) to select from stored candidates."""
    rows = []
    for r in results:
        m = meta.iloc[r["i"]]
        ref = m.text.split()
        spoken, unspoken = gloss[m.audio_id]
        run = r["runs"][wgt]
        if v2 is None:
            sugs = [s for s in run["sug"] if s[4] >= -max_cost]
        else:
            ver, mc, nd, nc = v2
            kept, _ = G.select(run["cands"][ver], mc, nd, nc)
            sugs = [(s.std_start, s.std_end, s.suggested, s.terms, float(s.support), s.kind) for s in kept]
        e_std = edits(m.text, r["std"])
        good = [s for s in sugs if edits(m.text, apply(r["std"], [s])) < e_std]
        hyps = {"off": r["std"], "boosted": run["bst"], "accept_all": apply(r["std"], sugs), "oracle": apply(r["std"], good)}
        row = {"audio_id": m.audio_id, "noise_type": m.noise_type, "snr_db": m.snr_db, "dur": m.duration,
               "n_ref": len(ref), "n_sug": len(sugs), "n_sug_good": len(good), "t_std": r["t_std"],
               "t_gloss": run["t_v2"] if (v2 is not None and v2[0] == "v2") else run["t"],
               "n_sug_near": sum(len(s) > 5 and s[5] == "near" for s in sugs)}
        for k, h in hyps.items():
            row[f"spk_ref_{k}"] = sum(count_term(ref, t) for t in spoken)
            row[f"spk_ok_{k}"] = sum(min(count_term(ref, t), count_term(h, t)) for t in spoken)
            row[f"unspk_in_{k}"] = sum(count_term(h, t) > 0 for t in unspoken)
            row[f"edits_{k}"] = e_std if k == "off" else edits(m.text, h)
        row["n_unspoken"] = len(unspoken)
        row["spk_oov"] = sum(t not in _G["ulp"] for t in spoken)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df, by=None):
    def agg(d):
        o = {"clips": len(d), "spoken_terms": int(d.spk_ref_off.sum()),
             "suggestions_per_min": 60 * d.n_sug.sum() / d.dur.sum(),
             "suggestion_precision_%": 100 * d.n_sug_good.sum() / max(d.n_sug.sum(), 1)}
        for k in ("off", "boosted", "accept_all", "oracle"):
            o[f"spoken_acc_{k}_%"] = 100 * d[f"spk_ok_{k}"].sum() / max(d[f"spk_ref_{k}"].sum(), 1)
            o[f"unspoken_rate_{k}_%"] = 100 * d[f"unspk_in_{k}"].sum() / max(d.n_unspoken.sum(), 1)
            o[f"wer_{k}_%"] = 100 * d[f"edits_{k}"].sum() / d.n_ref.sum()
        o["extra_decode_ms_per_clip"] = 1000 * d.t_gloss.mean()
        return pd.Series(o)
    if by is None:
        return agg(df).to_frame().T
    return df.groupby(by, dropna=False).apply(agg).reset_index()


def bootstrap_ci(df, col_a, col_b, ref_col, n=1000, seed=0):
    """95 % CI of the difference in spoken-term accuracy (b - a), resampling utterances (paired)."""
    g = df.groupby("audio_id")[[col_a, col_b, ref_col]].sum()
    g = g[g[ref_col] > 0]
    a, b, r = g[col_a].to_numpy(), g[col_b].to_numpy(), g[ref_col].to_numpy()
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        k = rng.integers(len(g), size=len(g))
        d.append(100 * (b[k].sum() - a[k].sum()) / r[k].sum())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="B960_rev")
    ap.add_argument("--set", choices=["dev", "test"], default="dev")
    ap.add_argument("--lm", default="3gram")
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--beam", type=int, default=50)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--grid-v2", action="store_true")
    ap.add_argument("--v2", action="store_true", help="test: compare v1 and v2 (v2 settings from the dev choice)")
    ap.add_argument("--weight", type=float, default=G.DEFAULT_WEIGHT)
    ap.add_argument("--max-cost", type=float, default=G.DEFAULT_MAX_COST)
    ap.add_argument("--oov-logp", type=float, default=G.DEFAULT_OOV_LOGP)
    ap.add_argument("--conds", default="clean,esc50:0,esc50:5,esc50:10,brown:0,brown:5,brown:10",
                    help="test only: noise_type[:snr] list")
    ap.add_argument("--limit", type=int, default=0, help="max clips (debug)")
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    d = PROCESSED / "logits" / a.name
    meta = pd.read_parquet(d / f"{a.set}_meta.parquet")
    z = np.load(d / f"{a.set}.npz")
    _G["logp"], _G["off"] = z["logp"], z["offsets"]
    if a.set == "test":
        keep = np.zeros(len(meta), bool)
        for c in a.conds.split(","):
            nt, _, snr = c.partition(":")
            m = meta.noise_type.eq(nt) & (meta.snr_db.eq(float(snr)) if snr else True)
            keep |= m.to_numpy()
        idx = np.flatnonzero(keep)
    else:
        idx = np.arange(len(meta))
    if a.limit:
        idx = idx[:: max(len(idx) // a.limit, 1)][: a.limit]

    words, ulp = _arpa_unigrams(LMS[a.lm])
    from pyctcdecode import build_ctcdecoder
    _G["decoder"] = build_ctcdecoder(_ctc_labels(), str(LMS[a.lm]), unigrams=words, alpha=a.alpha, beta=a.beta)
    _G["ulp"], _G["beam"], _G["oov_logp"] = ulp, a.beam, a.oov_logp
    _G["ids"] = meta.audio_id.to_numpy()
    gloss = build_glossaries(meta, ulp)
    _G["gloss"] = gloss
    weights = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0] if a.grid else [a.weight]
    _G["weights"] = weights
    _G["v2"] = a.grid_v2 or a.v2
    print(f"{a.set}: {len(idx)} clips, weights {weights}, {a.workers} workers", flush=True)

    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        results = []
        for k, r in enumerate(pool.imap_unordered(work, idx.tolist(), chunksize=8)):
            results.append(r)
            if (k + 1) % 2000 == 0:
                print(f"  {k + 1}/{len(idx)}  {time.time() - t0:.0f} s", flush=True)
    results.sort(key=lambda r: r["i"])
    print(f"decoded in {time.time() - t0:.0f} s", flush=True)

    tag = f"{a.name}_lm{a.lm[0]}_{a.set}"
    if _G["v2"]:
        return run_v2(a, meta, results, gloss, weights[0], tag)
    with open(OUT / f"{tag}_glossaries.json", "w") as f:
        json.dump({k: {"spoken": v[0], "unspoken": v[1]} for k, v in gloss.items()
                   if k in set(meta.audio_id.iloc[idx])}, f)
    if a.grid:
        rows = []
        for w in weights:
            for c in (1.0, 2.0, 4.0, 8.0, 16.0, np.inf):
                s = summarize(score(meta, results, gloss, w, c)).iloc[0].to_dict()
                rows.append({"weight": w, "max_cost": c, **s})
                print(f"w={w:<4} cost={c:<4} spoken acc off {s['spoken_acc_off_%']:.1f} -> accept_all "
                      f"{s['spoken_acc_accept_all_%']:.1f} / oracle {s['spoken_acc_oracle_%']:.1f}; unspoken "
                      f"{s['unspoken_rate_off_%']:.3f} -> {s['unspoken_rate_accept_all_%']:.3f}; "
                      f"prec {s['suggestion_precision_%']:.1f}; WER {s['wer_off_%']:.2f} -> {s['wer_accept_all_%']:.2f}",
                      flush=True)
        grid = pd.DataFrame(rows)
        grid.to_csv(OUT / f"{tag}_grid.csv", index=False)
        # choice: among settings with unspoken-term rate <= 0.1 %, suggestion precision >= 95 % (a reviewer should be
        # able to trust the queue: at most 1 wrong suggestion in 20) and WER not worse,
        # take the smallest boost (then the strictest audio check) within 1 point of the best accept-all accuracy
        ok = grid[(grid["unspoken_rate_accept_all_%"] <= 0.1) & (grid["wer_accept_all_%"] <= grid["wer_off_%"])
                  & (grid["suggestion_precision_%"] >= 95)]
        ok = ok if len(ok) else grid
        top = ok["spoken_acc_accept_all_%"].max()
        best = ok[ok["spoken_acc_accept_all_%"] >= top - 1.0].sort_values(["weight", "max_cost"]).iloc[0]
        print("chosen:", best[["weight", "max_cost"]].to_dict())
        json.dump({"weight": float(best.weight), "max_cost": float(best.max_cost), "oov_logp": a.oov_logp,
                   "rule": "smallest weight, then smallest max_cost, within 1 pt of the best accept-all spoken-term accuracy, "
                           "s.t. unspoken-term rate <= 0.1 %, suggestion precision >= 95 %, WER <= off"},
                  open(OUT / f"{tag}_choice.json", "w"), indent=2)
        return
    df = score(meta, results, gloss, weights[0], a.max_cost)
    df["condition"] = df.noise_type + df.snr_db.map(lambda s: "" if pd.isna(s) else f" {int(s)} dB")
    df.to_parquet(OUT / f"{tag}_clips.parquet", index=False)
    overall_noisy = summarize(df[df.noise_type != "clean"]).assign(condition="unseen noise (ESC-50 + brown, 0-10 dB)")
    overall_clean = summarize(df[df.noise_type == "clean"]).assign(condition="clean")
    by_cond = summarize(df, "condition")
    noisy = df[df.noise_type != "clean"]
    ci = bootstrap_ci(noisy, "spk_ok_off", "spk_ok_accept_all", "spk_ref_off")
    ci_or = bootstrap_ci(noisy, "spk_ok_off", "spk_ok_oracle", "spk_ref_off")
    oov = noisy.assign(kind=np.where(noisy.spk_oov > 0, "contains LM-unknown term", "LM-known terms only"))
    by_kind = summarize(oov[oov.spk_ref_off > 0], "kind")
    pd.concat([overall_noisy, overall_clean]).to_csv(OUT / f"{tag}_summary.csv", index=False)
    by_cond.to_csv(OUT / f"{tag}_by_condition.csv", index=False)
    by_kind.to_csv(OUT / f"{tag}_by_term_kind.csv", index=False)
    json.dump({"weight": weights[0], "max_cost": a.max_cost, "oov_logp": a.oov_logp, "clips": int(len(df)),
               "ci95_spoken_acc_gain_accept_all_pts_noisy": ci, "ci95_spoken_acc_gain_oracle_pts_noisy": ci_or,
               "noisy": overall_noisy.iloc[0].drop("condition").to_dict(),
               "clean": overall_clean.iloc[0].drop("condition").to_dict()},
              open(OUT / f"{tag}_metrics.json", "w"), indent=2, default=float)
    pd.set_option("display.width", 250)
    print(pd.concat([overall_noisy, overall_clean]).T.to_string())
    print("95 % CI gain (accept all):", ci, " (oracle):", ci_or)
    print(by_kind.T.to_string())


# ------------------------------------------------------------------------------------------ v2 (step 3)
V2_RULE = ("among settings with unspoken-term rate <= 0.1 %, suggestion precision >= 95 % and WER <= off: the "
           "highest accept-all spoken-term accuracy; within 1 pt of it, the highest suggestion precision "
           "(rule fixed before running the grid)")


def _score_job(args):
    w, cfg = args
    return cfg, summarize(score(_G["meta"], _G["results"], _G["gloss_all"], w, None, v2=cfg)).iloc[0].to_dict()


def run_v2(a, meta, results, gloss, w, tag):
    _G["meta"], _G["results"], _G["gloss_all"] = meta, results, gloss
    if a.grid_v2:
        cfgs = [("v1", 8.0, -1.0, 0.0)] + [("v2", mc, nd, nc) for mc in (4.0, 8.0, 12.0, 16.0)
                                           for nd in (-1.0, 0.1, 0.2, 0.3, 0.4) for nc in (2.0, 4.0, 8.0, 12.0, 16.0)]
        with mp.get_context("fork").Pool(min(a.workers, len(cfgs))) as pool:
            out = pool.map(_score_job, [(w, c) for c in cfgs])
        rows = [{"version": c[0], "max_cost": c[1], "near_dist": c[2], "near_cost": c[3], **s} for c, s in out]
        grid = pd.DataFrame(rows)
        grid.to_csv(OUT / f"{tag}_grid_v2.csv", index=False)
        v2 = grid[grid.version == "v2"]
        ok = v2[(v2["unspoken_rate_accept_all_%"] <= 0.1) & (v2["suggestion_precision_%"] >= 95)
                & (v2["wer_accept_all_%"] <= v2["wer_off_%"])]
        ok = ok if len(ok) else v2
        top = ok["spoken_acc_accept_all_%"].max()
        best = ok[ok["spoken_acc_accept_all_%"] >= top - 1.0].sort_values(
            ["suggestion_precision_%", "spoken_acc_accept_all_%"], ascending=False).iloc[0]
        cols = ["version", "max_cost", "near_dist", "near_cost", "spoken_acc_off_%", "spoken_acc_accept_all_%",
                "spoken_acc_oracle_%", "unspoken_rate_accept_all_%", "suggestion_precision_%", "suggestions_per_min",
                "wer_off_%", "wer_accept_all_%"]
        pd.set_option("display.width", 250)
        print(grid[cols].sort_values("spoken_acc_accept_all_%", ascending=False).head(25).to_string(index=False))
        print("v1:", grid[grid.version == "v1"][cols].to_string(index=False))
        print("chosen:", best[cols].to_dict())
        json.dump({"weight": w, "max_cost": float(best.max_cost), "near_dist": float(best.near_dist),
                   "near_cost": float(best.near_cost), "rule": V2_RULE},
                  open(OUT / f"{tag}_choice_v2.json", "w"), indent=2)
        return
    ch = json.load(open(OUT / f"{a.name}_lm{a.lm[0]}_dev_choice_v2.json"))
    parts = []
    for ver, cfg in (("v1", ("v1", a.max_cost, -1.0, 0.0)),
                     ("v2", ("v2", ch["max_cost"], ch["near_dist"], ch["near_cost"]))):
        df = score(meta, results, gloss, w, None, v2=cfg)
        df["condition"] = df.noise_type + df.snr_db.map(lambda s: "" if pd.isna(s) else f" {int(s)} dB")
        noisy = df[df.noise_type != "clean"]
        oov = noisy.assign(kind=np.where(noisy.spk_oov > 0, "contains LM-unknown term", "LM-known terms only"))
        res = pd.concat([summarize(noisy).assign(condition="unseen noise (ESC-50 + brown, 0-10 dB)"),
                         summarize(df[df.noise_type == "clean"]).assign(condition="clean"),
                         summarize(df, "condition"),
                         summarize(oov[oov.spk_ref_off > 0], "kind").rename(columns={"kind": "condition"})
                         .assign(condition=lambda d: "unseen noise, " + d.condition)])
        res.insert(0, "version", ver)
        ci = bootstrap_ci(noisy, "spk_ok_off", "spk_ok_accept_all", "spk_ref_off")
        res["ci95_gain_accept_all_noisy"] = str(tuple(round(x, 2) for x in ci))
        res["near_share_%"] = 100 * df.n_sug_near.sum() / max(df.n_sug.sum(), 1)
        parts.append(res)
        df.to_parquet(OUT / f"{tag}_clips_{ver}.parquet", index=False)
    out = pd.concat(parts)
    out.to_csv(OUT / f"{tag}_v1_vs_v2.csv", index=False)
    json.dump({"v1": {"weight": w, "max_cost": a.max_cost}, "v2": ch}, open(OUT / f"{tag}_v1_vs_v2_settings.json", "w"),
              indent=2)
    cols = ["version", "condition", "spoken_acc_off_%", "spoken_acc_accept_all_%", "spoken_acc_oracle_%",
            "unspoken_rate_off_%", "unspoken_rate_accept_all_%", "suggestion_precision_%", "suggestions_per_min",
            "wer_off_%", "wer_accept_all_%", "extra_decode_ms_per_clip"]
    pd.set_option("display.width", 250)
    print(out[cols].to_string(index=False))
    print(out.groupby("version")["ci95_gain_accept_all_noisy"].first())


if __name__ == "__main__":
    main()
