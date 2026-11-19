"""Run a CTC model once and store its per-frame log-probabilities, so decoders (greedy, beam+LM) and the
word-error detector can be developed on CPU without re-running the GPU.

Usage:
  python scripts/dump_logits.py --model experiments/runs/B_noisy/best --name B_noisy --set dev
  python scripts/dump_logits.py --model experiments/runs/B_noisy/best --name B_noisy --set test

Outputs -> data/processed/logits/<name>/<set>.npz  (git-ignored)
  logp    float16 [sum_T, 32]  all utterances concatenated
  offsets int64   [N + 1]      utterance i = logp[offsets[i]:offsets[i+1]]
and <set>_meta.parquet with the manifest rows (audio_id, ls_split, noise_type, snr_db, text, ...).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.utils.paths import MANIFESTS, PROCESSED, absp, setup_env  # noqa: E402

setup_env()
from transformers import Wav2Vec2ForCTC  # noqa: E402

from group9_ds_ai_lab.data.dataset import ASRDataset  # noqa: E402
from group9_ds_ai_lab.inference.decode import run_inference  # noqa: E402

SETS = {"dev": "valid_eval.csv", "test": "test_conditions.csv", "test_ext": "test_conditions_ext.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", choices=SETS, required=True)
    ap.add_argument("--max-seconds", type=float, default=800.0)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    df = pd.read_csv(MANIFESTS / SETS[args.set]).reset_index(drop=True)
    src = absp(args.model) if Path(absp(args.model)).exists() else args.model
    import torch
    model = Wav2Vec2ForCTC.from_pretrained(src).cuda().eval()
    t0 = time.time()
    res = run_inference(model, ASRDataset(df, fixed_conditions=True), max_seconds=args.max_seconds,
                        num_workers=args.workers, keep_words=False, keep_logp=True)
    del model
    torch.cuda.empty_cache()
    lens = np.array([len(r["logp"]) for r in res], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    out = PROCESSED / "logits" / args.name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.set}.npz", logp=np.concatenate([r["logp"] for r in res]), offsets=offsets)
    df.to_parquet(out / f"{args.set}_meta.parquet", index=False)
    print(f"{len(df)} utts, {offsets[-1]} frames, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
