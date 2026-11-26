"""Objective O4: do flagged words point to the right place in the audio?

Target (M1 proposal): at least 90 % of 50 sampled flags have the target word inside the replay window
(flagged word's start - 1 s ... end + 1 s).

1. Reference word times: CTC forced alignment (Viterbi) of the REFERENCE transcript on the CLEAN version of each
   test utterance (B960_rev log-probabilities). Noise is added without any time shift, so the same times hold for
   every noisy version. Sanity check: on correctly recognised clean words, compare with the decoder's own times.
2. Flags: words of B960_rev + 4-gram LM with detector probability >= threshold (GBM detector).
   Target word of a flag = the reference word it is aligned to (equal / substitution); for an inserted word,
   the nearest reference word before or after it.
3. Automatic check on ALL flags (hit = target word lies completely inside the window), per condition and threshold.
4. 50 randomly sampled flags (seeded, stratified over conditions) exported as audio clips of the replay window
   + a CSV for the team to check by listening (column `heard_target_word` to fill in: y / n).

Usage : python scripts/replay_window.py [--threshold 0.5] [--n-sample 50]
Output: milestones/M5_evaluation/replay_check/ (summary.json, per_condition.csv, sample_flags.csv, clips/*.flac)
        data/processed/ref_word_times_test.parquet (git-ignored)
"""
import argparse
import json
import sys
from pathlib import Path

import jiwer
import joblib
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_detector import features, unigram_table  # noqa: E402
from group9_ds_ai_lab.data.audio import ArrowAudioReader  # noqa: E402
from group9_ds_ai_lab.data.noise import apply_condition  # noqa: E402
from group9_ds_ai_lab.data.text import BLANK_ID, DELIM_ID, encode  # noqa: E402
from group9_ds_ai_lab.inference.decode import FRAME_SEC  # noqa: E402
from group9_ds_ai_lab.utils.paths import EVAL, PROCESSED, REPO  # noqa: E402

OUT = REPO / "milestones" / "M5_evaluation" / "replay_check"
RUN = "B960_rev"
WINDOW = 1.0


def ctc_force_align(logp: np.ndarray, labels: list[int]):
    """Viterbi CTC alignment. Returns, for every label position, (first frame, last frame)."""
    T = logp.shape[0]
    ext = [BLANK_ID]
    for c in labels:
        ext += [c, BLANK_ID]
    S = len(ext)
    ext = np.array(ext)
    lp = logp[:, ext].astype(np.float64)                       # T x S
    neg = -1e30
    alpha = np.full(S, neg)
    alpha[0], alpha[1] = lp[0, 0], lp[0, 1]
    back = np.zeros((T, S), np.int8)                           # 0 = stay, 1 = from s-1, 2 = from s-2
    skip_ok = np.zeros(S, bool)
    skip_ok[2:] = (ext[2:] != BLANK_ID) & (ext[2:] != ext[:-2])
    for t in range(1, T):
        a0 = alpha
        a1 = np.concatenate(([neg], alpha[:-1]))
        a2 = np.where(skip_ok, np.concatenate(([neg, neg], alpha[:-2])), neg)
        stack = np.stack([a0, a1, a2])
        back[t] = stack.argmax(0)
        alpha = stack.max(0) + lp[t]
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    path = np.empty(T, int)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    first = np.full(len(labels), -1)
    last = np.full(len(labels), -1)
    for t, s in enumerate(path):
        if s % 2 == 1:
            k = s // 2
            if first[k] < 0:
                first[k] = t
            last[k] = t
    return first, last


def ref_word_times(meta: pd.DataFrame, logp_all: np.ndarray, offsets: np.ndarray) -> pd.DataFrame:
    rows = []
    clean = meta[meta.noise_type == "clean"]
    for i, r in clean.iterrows():
        lp = logp_all[offsets[i]:offsets[i + 1]]
        words = r.text.split()
        labels, owner = [], []
        for wi, w in enumerate(words):
            if wi:
                labels.append(DELIM_ID)  # word separator '|', as emitted by the model; owned by no word
                owner.append(-1)
            ids = encode(w)
            labels += ids
            owner += [wi] * len(ids)
        first, last = ctc_force_align(lp, labels)
        owner = np.array(owner)
        for wi, w in enumerate(words):
            sel = owner == wi
            rows.append((r.audio_id, wi, w, first[sel].min() * FRAME_SEC, (last[sel].max() + 1) * FRAME_SEC))
    return pd.DataFrame(rows, columns=["audio_id", "ref_idx", "ref_word", "ref_start", "ref_end"])


def flag_targets(words: pd.DataFrame, preds: pd.DataFrame, times: pd.DataFrame) -> pd.DataFrame:
    """For every hypothesis word: index of the reference word it belongs to (aligned or nearest)."""
    tgt = np.full(len(words), -1)
    kind = np.empty(len(words), object)
    words = words.reset_index(drop=True)
    for (aid, nt, snr), g in words.groupby(["audio_id", "noise_type", "snr_k"], sort=False):
        ref = preds[(aid, nt, snr)]
        hyp = " ".join(g.word)
        al = jiwer.process_words(ref, hyp).alignments[0]
        idx = g.index.to_numpy()
        n_ref = len(ref.split())
        for ch in al:
            for k in range(ch.hyp_end_idx - ch.hyp_start_idx):
                j = ch.hyp_start_idx + k
                if ch.type in ("equal", "substitute"):
                    tgt[idx[j]], kind[idx[j]] = ch.ref_start_idx + k, ch.type
                elif ch.type == "insert":  # nearest reference word (the one before, else the one after)
                    tgt[idx[j]] = min(max(ch.ref_start_idx - 1, 0), n_ref - 1)
                    kind[idx[j]] = "insert"
    words["ref_idx"], words["align_type"] = tgt, kind
    return words.merge(times, on=["audio_id", "ref_idx"], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5, help="detector probability above which a word is flagged")
    ap.add_argument("--n-sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clips").mkdir(exist_ok=True)

    times_path = PROCESSED / "ref_word_times_test.parquet"
    meta = pd.read_parquet(PROCESSED / "logits" / RUN / "test_meta.parquet")
    if times_path.exists():
        times = pd.read_parquet(times_path)
    else:
        z = np.load(PROCESSED / "logits" / RUN / "test.npz")
        print("forced alignment of", int((meta.noise_type == "clean").sum()), "clean utterances ...", flush=True)
        times = ref_word_times(meta, z["logp"], z["offsets"])
        times.to_parquet(times_path, index=False)
    print("reference words with times:", len(times), flush=True)

    words = pd.read_parquet(EVAL / f"{RUN}_lm4" / "test_words.parquet")
    det = joblib.load(EVAL / f"{RUN}_lm4" / "error_detector" / "model.joblib")
    fw = features(words, unigram_table())
    words["p_error"] = det["model"].predict_proba(fw[det["features"]].to_numpy(np.float64))[:, 1]
    p = pd.read_parquet(EVAL / f"{RUN}_lm4" / "test_predictions.parquet")
    preds = dict(zip(zip(p.audio_id, p.noise_type, p.snr_db.fillna(-99)), p.text.str.strip()))
    words["snr_k"] = words.snr_db.fillna(-99)
    words = flag_targets(words, preds, times)

    # sanity check of the forced alignment: correctly recognised words on clean audio
    ok = words[(words.noise_type == "clean") & (words.align_type == "equal")]
    d_start, d_end = (ok.start - ok.ref_start).abs(), (ok.end - ok.ref_end).abs()
    sanity = {"n_words": int(len(ok)), "median_abs_start_diff_s": float(d_start.median()),
              "p95_abs_start_diff_s": float(d_start.quantile(0.95)), "median_abs_end_diff_s": float(d_end.median()),
              "p95_abs_end_diff_s": float(d_end.quantile(0.95)), "share_start_within_0.1s": float((d_start <= 0.1).mean())}
    print("alignment sanity:", json.dumps(sanity), flush=True)

    words["hit"] = (words.ref_start >= words.start - WINDOW) & (words.ref_end <= words.end + WINDOW)
    words["cond"] = np.where(words.noise_type.eq("clean"), "clean",
                             words.noise_type + " " + words.snr_db.fillna(-99).astype(int).astype(str) + " dB")
    res = {"threshold": args.threshold, "window_s": WINDOW, "alignment_sanity": sanity, "by_threshold": {}}
    for thr in (0.3, 0.5, 0.7):
        f = words[words.p_error >= thr]
        res["by_threshold"][str(thr)] = {"n_flags": int(len(f)), "hit_rate": float(f.hit.mean()),
                                         "hit_rate_true_errors": float(f[f.is_error == 1].hit.mean()),
                                         "precision": float(f.is_error.mean())}
    flags = words[words.p_error >= args.threshold]
    pc = flags.groupby("cond").agg(n_flags=("hit", "size"), hit_rate=("hit", "mean"), precision=("is_error", "mean")).round(4)
    pc.to_csv(OUT / "per_condition.csv")
    by_kind = flags.groupby("align_type").hit.agg(["size", "mean"]).round(4)
    res["by_alignment_type"] = by_kind.reset_index().to_dict("records")
    res["misses_window_extent"] = {
        "median_miss_s": float(np.maximum(flags.start - WINDOW - flags.ref_start, flags.ref_end - flags.end - WINDOW)[~flags.hit].median())
        if (~flags.hit).any() else 0.0}

    # 50 sampled flags, stratified over conditions (seeded), exported as audio clips
    rng = np.random.default_rng(args.seed)
    conds = sorted(flags.cond.unique())
    per = int(np.ceil(args.n_sample / len(conds)))
    samp = pd.concat([g.sample(min(per, len(g)), random_state=int(rng.integers(1 << 31))) for _, g in flags.groupby("cond")])
    samp = samp.sample(args.n_sample, random_state=args.seed).reset_index(drop=True)
    meta_k = meta.assign(snr_k=meta.snr_db.fillna(-99)).set_index(["audio_id", "noise_type", "snr_k"])
    reader = ArrowAudioReader()
    rows = []
    for i, r in samp.iterrows():
        m = meta_k.loc[(r.audio_id, r.noise_type, r.snr_k)]
        x = reader.load(m.shard, int(m.row))
        x, _ = apply_condition(x, r.noise_type, r.snr_db, int(m.cond_seed))
        a, b = max(r.start - WINDOW, 0.0), min(r.end + WINDOW, len(x) / 16000)
        clip = f"flag_{i + 1:02d}.flac"
        sf.write(OUT / "clips" / clip, x[int(a * 16000):int(b * 16000)], 16000)
        hyp_words = words[(words.audio_id == r.audio_id) & (words.noise_type == r.noise_type) & (words.snr_k == r.snr_k)]
        hyp_ctx = " ".join(hyp_words.word.where(hyp_words.start != r.start, "[" + r.word + "]"))
        rows.append({"clip": clip, "condition": r.cond, "audio_id": r.audio_id, "flagged_word": r.word,
                     "p_error": round(r.p_error, 3), "is_error": int(r.is_error), "target_word": r.ref_word,
                     "window_start_s": round(a, 2), "window_end_s": round(b, 2),
                     "target_start_s": round(r.ref_start, 2), "target_end_s": round(r.ref_end, 2),
                     "auto_hit": bool(r.hit), "hypothesis": hyp_ctx, "reference": preds[(r.audio_id, r.noise_type, r.snr_k)],
                     "heard_target_word": ""})
    pd.DataFrame(rows).to_csv(OUT / "sample_flags.csv", index=False)
    res["sample"] = {"n": len(rows), "auto_hit_rate": float(np.mean([r["auto_hit"] for r in rows]))}
    res["all_flags"] = {"n": int(len(flags)), "hit_rate": float(flags.hit.mean())}
    res["target_O4"] = ">= 90 % of 50 sampled flags have the target word inside the replay window"
    with open(OUT / "summary.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: res[k] for k in ("by_threshold", "by_alignment_type", "sample", "all_flags", "misses_window_extent")}, indent=1))


if __name__ == "__main__":
    main()
