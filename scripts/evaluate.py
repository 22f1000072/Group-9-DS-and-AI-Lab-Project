"""Evaluate a CTC model on the fixed test-condition grid (test-clean + test-other x noise x SNR).

Usage:
  python scripts/evaluate.py --model facebook/wav2vec2-base-960h --name ref_base960h
  python scripts/evaluate.py --model experiments/runs/A_clean/best --name A_clean
  # quick check: --limit 200 --noise clean esc50

Outputs -> experiments/results/<name>/
  summary.csv          WER/CER per (test split, noise type, SNR)   [small, committed]
  metrics.json         headline numbers                              [small, committed]
  predictions.parquet  per-utterance ref/hyp                         [large, git-ignored]
  words.parquet        per-hypothesis-word confidence, timestamps, error label (for O3/O4)
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.utils.paths import EVAL, MANIFESTS, absp, setup_env  # noqa: E402

setup_env()
from transformers import Wav2Vec2ForCTC  # noqa: E402

from group9_ds_ai_lab.data.dataset import ASRDataset  # noqa: E402
from group9_ds_ai_lab.inference.decode import run_inference, wer_cer, word_error_labels  # noqa: E402


def _labels(args):
    return [word_error_labels(r, h) for r, h in zip(*args)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or path to a saved model dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--splits", nargs="*", default=["test.clean", "test.other"])
    ap.add_argument("--noise", nargs="*", default=None, help="subset of noise types (clean white pink brown esc50)")
    ap.add_argument("--limit", type=int, default=None, help="utterances per condition (quick runs)")
    ap.add_argument("--max-seconds", type=float, default=800.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--manifest", default="test_conditions.csv", help="test_conditions_ext.csv for the M5 extended grid")
    args = ap.parse_args()

    out = EVAL / args.name
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFESTS / args.manifest)
    df = df[df.ls_split.isin(args.splits)]
    if args.noise:
        df = df[df.noise_type.isin(args.noise)]
    if args.limit:
        df = df.groupby(["ls_split", "noise_type", "snr_db"], dropna=False, group_keys=False).head(args.limit)
    df = df.reset_index(drop=True)
    print(f"{len(df)} rows, {df.duration.sum() / 3600:.1f} h audio", flush=True)

    src = absp(args.model) if Path(absp(args.model)).exists() else args.model
    model = Wav2Vec2ForCTC.from_pretrained(src).cuda().eval()
    t0 = time.time()
    res = run_inference(model, ASRDataset(df, fixed_conditions=True), max_seconds=args.max_seconds,
                        num_workers=args.workers)
    infer_s = time.time() - t0
    df["hyp"] = [r["hyp"] for r in res]
    print(f"inference {infer_s:.0f}s ({df.duration.sum() / infer_s:.0f}x real time)", flush=True)
    if "reverb" in df:  # after inference: encode reverb in the condition name so all tables below separate it
        rev = df.reverb.fillna(False).astype(bool)
        df["noise_type"] = np.where(rev, "reverb+" + df.noise_type, df.noise_type)

    # word-level error labels (parallel)
    chunks = np.array_split(np.arange(len(df)), 64)
    with ProcessPoolExecutor(32) as ex:
        labs = ex.map(_labels, [(df.text.iloc[c].tolist(), df.hyp.iloc[c].tolist()) for c in chunks])
        labels = [l for part in labs for l in part]
    wrows = []
    for i, (r, lab) in enumerate(zip(res, labels)):
        for w, e in zip(r["words"], lab):
            wrows.append((i, w["word"], w["start"], w["end"], w["conf_min"], w["conf_mean"], e))
    words = pd.DataFrame(wrows, columns=["row", "word", "start", "end", "conf_min", "conf_mean", "is_error"])
    meta = df[["audio_id", "ls_split", "noise_type", "snr_db"]]
    words = pd.concat([meta.iloc[words.row].reset_index(drop=True), words], axis=1)
    words.to_parquet(out / "words.parquet", index=False)
    df[["audio_id", "ls_split", "speaker_id", "duration", "noise_type", "snr_db", "cond_seed", "text", "hyp"]] \
        .to_parquet(out / "predictions.parquet", index=False)

    rows = []
    for (s, n, snr), g in df.groupby(["ls_split", "noise_type", "snr_db"], dropna=False):
        w, c = wer_cer(g.text.tolist(), g.hyp.tolist())
        rows.append({"model": args.name, "ls_split": s, "noise_type": n,
                     "snr_db": "clean" if pd.isna(snr) else str(int(snr)), "n_utts": len(g), "wer": w, "cer": c})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)

    metrics = {"model": args.name, "model_path": str(args.model), "rows": int(len(df)),
               "audio_hours": float(df.duration.sum() / 3600), "inference_sec": infer_s}
    for s in args.splits:
        g = df[(df.ls_split == s) & (df.noise_type == "clean")]
        if len(g):
            metrics[f"wer_{s}_clean"] = wer_cer(g.text.tolist(), g.hyp.tolist())[0]
    for snrs, tag in (([0.0, 5.0], "0_5dB"), ([-5.0], "m5dB")):
        for n in df.noise_type.unique():
            g = df[(df.noise_type == n) & df.snr_db.isin(snrs)]
            if len(g):
                metrics[f"wer_{n}_{tag}"] = wer_cer(g.text.tolist(), g.hyp.tolist())[0]
    try:
        from sklearn.metrics import roc_auc_score
        if words.is_error.nunique() == 2:
            metrics["word_auroc_conf_min"] = float(roc_auc_score(words.is_error, 1 - words.conf_min))
            metrics["word_auroc_conf_mean"] = float(roc_auc_score(words.is_error, 1 - words.conf_mean))
    except Exception as e:  # noqa: BLE001
        metrics["auroc_error"] = str(e)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.set_option("display.width", 200)
    print(summ.pivot_table(index=["ls_split", "noise_type"], columns="snr_db", values="wer").round(4).to_string())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
