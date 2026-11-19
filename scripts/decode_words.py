"""Decode stored CTC log-probabilities (scripts/dump_logits.py) with greedy or beam search + KenLM n-gram LM,
and write per-word confidences/timestamps/error labels in the same format as scripts/evaluate.py.

Usage:
  # tune LM weight (alpha) and word bonus (beta) on the dev set
  python scripts/decode_words.py --name B_noisy --set dev --decoder lm --lm 3gram --grid
  # decode with chosen params
  python scripts/decode_words.py --name B_noisy --set test --decoder lm --lm 4gram --alpha 0.5 --beta 1.0

Outputs -> experiments/results/<name>_<tag>/  (tag = greedy | lm3 | lm4)
  <set>_predictions.parquet, <set>_words.parquet   (git-ignored)
  <set>_summary.csv, <set>_metrics.json, lm_grid_<set>.csv   (small, committed)
"""
import argparse
import gzip
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.data.text import BLANK_ID, DELIM_ID, ID2CH  # noqa: E402
from group9_ds_ai_lab.inference.decode import FRAME_SEC, greedy_words, wer_cer, word_error_labels  # noqa: E402
from group9_ds_ai_lab.utils.paths import DATA_RAW, EVAL, PROCESSED  # noqa: E402

LMS = {"3gram": DATA_RAW / "librispeech_lm" / "3-gram.pruned.1e-7.arpa",
       "4gram": DATA_RAW / "librispeech_lm" / "4-gram.arpa"}
SPECIAL = (1, 2, 3)
_G = {}  # globals shared with forked workers: logp, offsets, decoder, beam


def labels():
    lab = [ID2CH[i] for i in range(len(ID2CH))]
    lab[BLANK_ID], lab[DELIM_ID] = "", " "
    for i, c in zip(SPECIAL, "#$%"):  # never emitted (masked below); unique dummy symbols for pyctcdecode
        lab[i] = c
    return lab


def ensure_arpa(path: Path) -> Path:
    if not path.exists():
        gz = path.with_suffix(path.suffix + ".gz")
        print(f"decompressing {gz.name} ...", flush=True)
        with gzip.open(gz, "rb") as fi, open(path, "wb") as fo:
            while chunk := fi.read(1 << 24):
                fo.write(chunk)
    return path


def arpa_unigrams(path: Path):
    words, on = [], False
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("\\1-grams"):
                on = True
                continue
            if on:
                if line.startswith("\\2-grams"):
                    break
                p = line.split("\t")
                if len(p) >= 2 and p[1].strip() not in ("<s>", "</s>", "<unk>"):
                    words.append(p[1].strip())
    return words


def utt_logp(i):
    lp = _G["logp"][_G["off"][i]:_G["off"][i + 1]].astype(np.float32)
    lp[:, SPECIAL] = -1e4
    return lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)


def span_conf(lp, s, e):
    """Confidence of a word decoded on frames [s, e): max-posterior over its non-blank, non-delimiter frames."""
    seg = lp[s:max(e, s + 1)]
    pmax, am = np.exp(seg.max(-1)), seg.argmax(-1)
    keep = (am != BLANK_ID) & (am != DELIM_ID)
    v = pmax[keep] if keep.any() else pmax
    return float(v.min()), float(v.mean())


def decode_one(i):
    lp = utt_logp(i)
    if _G["decoder"] is None:
        return greedy_words(lp)
    beam = _G["decoder"].decode_beams(lp, beam_width=_G["beam"])[0]
    words = []
    for w, (s, e) in beam.text_frames if hasattr(beam, "text_frames") else beam[2]:
        cmin, cmean = span_conf(lp, s, e)
        words.append({"word": w, "start": round(s * FRAME_SEC, 3), "end": round(e * FRAME_SEC, 3),
                      "conf_min": cmin, "conf_mean": cmean})
    return " ".join(w["word"] for w in words), words


def run(idx, workers):
    if workers <= 1:
        return [decode_one(i) for i in idx]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(decode_one, idx, chunksize=8)


def subset(meta, limit):
    """Deterministic stratified random subset: up to `limit` rows per noise type (None -> all rows)."""
    if not limit:
        return np.arange(len(meta))
    return np.sort(meta.groupby("noise_type", group_keys=False)
                   .apply(lambda g: g.sample(n=min(limit, len(g)), random_state=0)).index.to_numpy())


def score(meta, out, name, set_):
    meta = meta.copy()
    meta["hyp"] = [h for h, _ in out]
    wrows = []
    for r, (ref, (hyp, words)) in enumerate(zip(meta.text, out)):
        for w, e in zip(words, word_error_labels(ref, hyp)):
            wrows.append((r, w["word"], w["start"], w["end"], w["conf_min"], w["conf_mean"], e))
    words = pd.DataFrame(wrows, columns=["row", "word", "start", "end", "conf_min", "conf_mean", "is_error"])
    cols = ["audio_id", "ls_split", "noise_type", "snr_db", "duration"]
    words = pd.concat([meta[cols].iloc[words.row].reset_index(drop=True), words], axis=1)
    rows = []
    for (s, n, snr), g in meta.groupby(["ls_split", "noise_type", "snr_db"], dropna=False):
        w, c = wer_cer(g.text.tolist(), g.hyp.tolist())
        rows.append({"model": name, "ls_split": s, "noise_type": n,
                     "snr_db": "clean" if pd.isna(snr) else str(int(snr)), "n_utts": len(g), "wer": w, "cer": c})
    return meta, words, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="logits folder under data/processed/logits/")
    ap.add_argument("--set", choices=["dev", "test"], required=True)
    ap.add_argument("--decoder", choices=["greedy", "lm"], default="lm")
    ap.add_argument("--lm", choices=list(LMS), default="4gram")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beam", type=int, default=50)
    ap.add_argument("--grid", action="store_true", help="dev tuning of alpha/beta on a stratified subset")
    ap.add_argument("--limit", type=int, default=None, help="rows per noise type")
    ap.add_argument("--workers", type=int, default=96)
    args = ap.parse_args()

    src = PROCESSED / "logits" / args.name
    z = np.load(src / f"{args.set}.npz")
    _G["logp"], _G["off"] = z["logp"], z["offsets"]
    meta = pd.read_parquet(src / f"{args.set}_meta.parquet")
    tag = "greedy" if args.decoder == "greedy" else f"lm{args.lm[0]}"
    out_dir = EVAL / f"{args.name}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _G["decoder"], _G["beam"] = None, args.beam
    if args.decoder == "lm":
        from pyctcdecode import build_ctcdecoder
        arpa = ensure_arpa(LMS[args.lm])
        t0 = time.time()
        _G["decoder"] = build_ctcdecoder(labels(), str(arpa), unigrams=arpa_unigrams(arpa),
                                         alpha=args.alpha, beta=args.beta)
        print(f"LM {arpa.name} loaded in {time.time() - t0:.0f}s", flush=True)

    if args.grid:
        idx = subset(meta, args.limit)
        res = []
        for a in (0.4, 0.6, 0.8, 1.0, 1.3):
            for b in (0.0, 1.0, 2.0, 3.0):
                _G["decoder"].reset_params(alpha=a, beta=b)
                t0 = time.time()
                out = run(idx, args.workers)
                m = meta.iloc[idx]
                hyp = np.array([h for h, _ in out], dtype=object)
                r = {"alpha": a, "beta": b}
                for k, sel in (("clean", m.noise_type.eq("clean").to_numpy()), ("noisy", m.noise_type.ne("clean").to_numpy())):
                    r[f"wer_{k}"] = wer_cer(m.text[sel].tolist(), hyp[sel].tolist())[0]
                r["wer_avg"] = (r["wer_clean"] + r["wer_noisy"]) / 2
                r["sec"] = time.time() - t0
                res.append(r)
                print(json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
        g = pd.DataFrame(res)
        g.to_csv(out_dir / f"lm_grid_{args.set}.csv", index=False)
        print("best:", g.loc[g.wer_avg.idxmin()].to_dict())
        return

    idx = subset(meta, args.limit)
    t0 = time.time()
    out = run(idx, args.workers)
    dec_s = time.time() - t0
    m, words, summ = score(meta.iloc[idx].reset_index(drop=True), out, f"{args.name}_{tag}", args.set)
    m.to_parquet(out_dir / f"{args.set}_predictions.parquet", index=False)
    words.to_parquet(out_dir / f"{args.set}_words.parquet", index=False)
    summ.to_csv(out_dir / f"{args.set}_summary.csv", index=False)
    metrics = {"model": f"{args.name}_{tag}", "set": args.set, "rows": len(m), "decode_sec": dec_s,
               "decoder": args.decoder, "lm": args.lm if args.decoder == "lm" else None,
               "alpha": args.alpha, "beta": args.beta, "beam": args.beam}
    for (s, n), g in m[m.noise_type == "clean"].groupby(["ls_split", "noise_type"]):
        metrics[f"wer_{s}_clean"] = wer_cer(g.text.tolist(), g.hyp.tolist())[0]
    g = m[m.noise_type == "clean"]
    metrics["wer_clean_pooled"] = wer_cer(g.text.tolist(), g.hyp.tolist())[0]
    g = m[m.noise_type != "clean"]
    metrics["wer_noisy_pooled"] = wer_cer(g.text.tolist(), g.hyp.tolist())[0]
    for n in m.noise_type.unique():
        g = m[(m.noise_type == n) & m.snr_db.isin([0.0, 5.0])]
        if len(g):
            metrics[f"wer_{n}_0_5dB"] = wer_cer(g.text.tolist(), g.hyp.tolist())[0]
    from sklearn.metrics import roc_auc_score
    if words.is_error.nunique() == 2:
        metrics["word_auroc_conf_mean"] = float(roc_auc_score(words.is_error, 1 - words.conf_mean))
        metrics["word_auroc_conf_min"] = float(roc_auc_score(words.is_error, 1 - words.conf_min))
    with open(out_dir / f"{args.set}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=1))


if __name__ == "__main__":
    main()
