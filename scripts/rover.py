"""ROVER-style word-level system combination (Fiscus, 1997), simplified to a pivot alignment.

Frame-level averaging of separately trained CTC models does not work (their letter spikes fall on different frames),
so systems are combined after decoding, at the word level:
  1. the first system is the pivot; every other hypothesis is aligned to it (word edit distance, jiwer);
  2. for each pivot word the candidates are the aligned words of all systems (or 'nothing' for a deletion);
  3. each candidate scores  sum over systems voting for it of  (1 - lam) + lam * word confidence
     ('nothing' votes use confidence 0.5); the best candidate wins, ties go to the pivot.
  Words that other systems insert between pivot words are ignored (the pivot decides the word count).

Usage:
  python scripts/rover.py --runs B2_lr1e-4_lm4 B460_lm4 B960_rev_lm4 --set dev --name rover3
Inputs : experiments/results/<run>/<set>_predictions.parquet + <set>_words.parquet  (scripts/decode_words.py)
Outputs: experiments/results/<name>/<set>_predictions.parquet, <set>_summary.csv, <set>_metrics.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.inference.decode import wer_cer  # noqa: E402
from group9_ds_ai_lab.utils.paths import EVAL  # noqa: E402

KEY = ["audio_id", "noise_type", "snr_db"]


def load(run, set_):
    p = pd.read_parquet(EVAL / run / f"{set_}_predictions.parquet").reset_index(drop=True)
    w = pd.read_parquet(EVAL / run / f"{set_}_words.parquet")
    conf = w.groupby("row").conf_mean.apply(list).to_dict()
    return p, [conf.get(i, []) for i in range(len(p))]


def combine(hyps, confs, lam):
    piv = hyps[0].split()
    if not piv:
        return hyps[0]
    votes = [defaultdict(float) for _ in piv]
    for w, c, v in zip(piv, confs[0], votes):
        v[w] += (1 - lam) + lam * c
    for hyp, conf in zip(hyps[1:], confs[1:]):
        hw = hyp.split()
        got = [None] * len(piv)  # word this system puts at each pivot slot (None = deletion)
        if hw:
            for ch in jiwer.process_words(" ".join(piv), hyp).alignments[0]:
                if ch.type in ("equal", "substitute"):
                    for k in range(ch.ref_end_idx - ch.ref_start_idx):
                        got[ch.ref_start_idx + k] = (hw[ch.hyp_start_idx + k], conf[ch.hyp_start_idx + k])
        for g, v in zip(got, votes):
            if g is None:
                v[""] += (1 - lam) + lam * 0.5
            else:
                v[g[0]] += (1 - lam) + lam * g[1]
    out = []
    for w, v in zip(piv, votes):
        best = max(v.items(), key=lambda kv: (kv[1], kv[0] == w))[0]
        if best:
            out.append(best)
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="result folders; the first one is the pivot")
    ap.add_argument("--set", choices=["dev", "test"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lam", type=float, default=0.5, help="weight of confidence vs plain vote count")
    args = ap.parse_args()
    systems = [load(r, args.set) for r in args.runs]
    base = systems[0][0]
    for r, (p, _) in zip(args.runs[1:], systems[1:]):
        assert p[KEY].astype(str).equals(base[KEY].astype(str)), f"{r}: rows differ from {args.runs[0]}"
    hyp = [combine([s[0].hyp.iloc[i] or "" for s in systems], [s[1][i] for s in systems], args.lam)
           for i in range(len(base))]
    m = base.drop(columns=["hyp"]).assign(hyp=hyp)
    out = EVAL / args.name
    out.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out / f"{args.set}_predictions.parquet", index=False)
    rows = []
    for (s, n, snr), g in m.groupby(["ls_split", "noise_type", "snr_db"], dropna=False):
        w, c = wer_cer(g.text.tolist(), g.hyp.tolist())
        rows.append({"model": args.name, "ls_split": s, "noise_type": n,
                     "snr_db": "clean" if pd.isna(snr) else str(int(snr)), "n_utts": len(g), "wer": w, "cer": c})
    pd.DataFrame(rows).to_csv(out / f"{args.set}_summary.csv", index=False)
    met = {"model": args.name, "set": args.set, "runs": args.runs, "lam": args.lam}
    for k, sel in (("clean", m.noise_type.eq("clean")), ("noisy", m.noise_type.ne("clean"))):
        met[f"wer_{k}_pooled"] = wer_cer(m.text[sel].tolist(), m.hyp[sel].tolist())[0]
        for r, (p, _) in zip(args.runs, systems):
            met[f"{r}_wer_{k}_pooled"] = wer_cer(p.text[sel].tolist(), p.hyp[sel].fillna("").tolist())[0]
    for n in m.noise_type.unique():
        g = m[(m.noise_type == n) & m.snr_db.isin([0.0, 5.0])]
        if len(g):
            met[f"wer_{n}_0_5dB"] = wer_cer(g.text.tolist(), g.hyp.tolist())[0]
    with open(out / f"{args.set}_metrics.json", "w") as f:
        json.dump(met, f, indent=2)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()}, indent=1))


if __name__ == "__main__":
    main()
