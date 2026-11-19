"""Word-level error detector (objective O3): predicts P(word is wrong) from decoder outputs.

Trained on the DEV set only, evaluated on the TEST grid. Compares:
  raw        : P(error) = 1 - conf_mean (no training)
  isotonic   : monotone calibration of conf_mean (1 feature)
  logreg     : logistic regression on word + utterance features
  gbm        : gradient-boosted trees on the same features

Usage:
  python scripts/error_detector.py --run B_noisy_lm4
Inputs : experiments/results/<run>/{dev,test}_words.parquet   (scripts/decode_words.py)
Outputs: experiments/results/<run>/error_detector/  metrics.json, per_condition.csv, reliability.png, model.joblib
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.utils.paths import DATA_RAW, EVAL  # noqa: E402
from group9_ds_ai_lab.utils.plotstyle import INK2, SERIES, apply_style  # noqa: E402

FEATURES = ["conf_mean", "conf_min", "logit_mean", "logit_min", "n_chars", "dur", "chars_per_sec", "unigram_logp",
            "is_first", "is_last", "utt_conf_mean", "utt_conf_min", "utt_n_words", "rel_conf"]


def unigram_table():
    path = DATA_RAW / "librispeech_lm" / "3-gram.pruned.1e-7.arpa"
    out, on = {}, False
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
                    out[p[1]] = float(p[0])
    return out


def features(w: pd.DataFrame, uni: dict) -> pd.DataFrame:
    w = w.copy()
    eps = 1e-4
    for k in ("mean", "min"):
        c = w[f"conf_{k}"].clip(eps, 1 - eps)
        w[f"logit_{k}"] = np.log(c / (1 - c))
    w["n_chars"] = w.word.str.len()
    w["dur"] = (w.end - w.start).clip(lower=0.02)
    w["chars_per_sec"] = w.n_chars / w.dur
    w["unigram_logp"] = w.word.map(uni).fillna(-8.0)
    key = ["audio_id", "noise_type", "snr_db"]
    g = w.groupby(key, dropna=False)
    w["pos"] = g.cumcount()
    w["utt_n_words"] = g.word.transform("size")
    w["is_first"] = (w.pos == 0).astype(int)
    w["is_last"] = (w.pos == w.utt_n_words - 1).astype(int)
    w["utt_conf_mean"] = g.conf_mean.transform("mean")
    w["utt_conf_min"] = g.conf_mean.transform("min")
    w["rel_conf"] = w.conf_mean - w.utt_conf_mean
    return w


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def recall_at(y, p, frac=0.30):
    """Share of all word errors that fall inside the riskiest `frac` of words."""
    k = int(np.ceil(frac * len(p)))
    top = np.argsort(-p, kind="stable")[:k]
    return float(y[top].sum() / max(y.sum(), 1))


def evaluate(y, p, thr=0.5):
    pred = p >= thr
    tp = float((pred & (y == 1)).sum())
    return {"auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
            "ece": ece(y, p), "brier": float(brier_score_loss(y, p)), "recall_at_30pct": recall_at(y, p),
            "precision_at_0.5": tp / max(pred.sum(), 1), "recall_at_0.5": tp / max((y == 1).sum(), 1),
            "flagged_frac_at_0.5": float(pred.mean()), "error_rate": float(y.mean()), "n_words": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder under experiments/results with dev_/test_words.parquet")
    args = ap.parse_args()
    src = EVAL / args.run
    out = src / "error_detector"
    out.mkdir(exist_ok=True)
    uni = unigram_table()
    dev = features(pd.read_parquet(src / "dev_words.parquet"), uni)
    test = features(pd.read_parquet(src / "test_words.parquet"), uni)
    Xd, yd = dev[FEATURES].to_numpy(np.float64), dev.is_error.to_numpy()
    Xt, yt = test[FEATURES].to_numpy(np.float64), test.is_error.to_numpy()
    print(f"dev words {len(yd)} (err {yd.mean():.3f}), test words {len(yt)} (err {yt.mean():.3f})", flush=True)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(-dev.conf_mean, yd)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(Xd, yd)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=0).fit(Xd, yd)
    preds = {"raw": 1 - test.conf_mean.to_numpy(), "isotonic": iso.predict(-test.conf_mean),
             "logreg": lr.predict_proba(Xt)[:, 1], "gbm": gbm.predict_proba(Xt)[:, 1]}

    metrics = {k: evaluate(yt, p) for k, p in preds.items()}
    best = max(("isotonic", "logreg", "gbm"), key=lambda k: metrics[k]["auroc"])
    metrics["selected"] = best
    metrics["targets"] = {"auroc": ">= 0.75", "recall_at_30pct": ">= 0.70", "ece": "<= 0.10"}
    joblib.dump({"model": {"isotonic": iso, "logreg": lr, "gbm": gbm}[best], "features": FEATURES}, out / "model.joblib")

    rows = []
    test["p"] = preds[best]
    for (n, snr), g in test.groupby(["noise_type", "snr_db"], dropna=False):
        r = evaluate(g.is_error.to_numpy(), g.p.to_numpy())
        rows.append({"noise_type": n, "snr_db": "clean" if pd.isna(snr) else int(snr), **r})
    pd.DataFrame(rows).round(4).to_csv(out / "per_condition.csv", index=False)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    apply_style()
    fig, ax = plt.subplots(figsize=(5, 4.6))
    edges = np.linspace(0, 1, 16)
    for (k, p), c in zip(preds.items(), SERIES):
        idx = np.clip(np.digitize(p, edges) - 1, 0, 14)
        xs = [p[idx == b].mean() for b in range(15) if (idx == b).sum() > 50]
        ys = [yt[idx == b].mean() for b in range(15) if (idx == b).sum() > 50]
        ax.plot(xs, ys, "-o", ms=4, color=c, label=f"{k} (ECE {metrics[k]['ece']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=INK2, lw=1)
    ax.set(xlabel="predicted P(word is wrong)", ylabel="observed error rate", title=f"Reliability - {args.run}",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=150)
    for k in ("raw", "isotonic", "logreg", "gbm"):
        print(f"{k:9s}", {kk: round(vv, 4) for kk, vv in metrics[k].items()})
    print("selected:", best)


if __name__ == "__main__":
    main()
