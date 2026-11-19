"""Milestone 5 data preparation (adds to, never changes, the M4 outputs of scripts/preprocess.py).

1. Larger training manifests from the already-QC'd full LibriSpeech table:
     train_460.csv = train-clean-100 + train-clean-360      train_960.csv = + train-other-500
   (same filters as M4: decodable, non-empty, CTC-feasible, 1-20 s)
2. Room-impulse-response banks from OpenSLR 28 (RIRS_NOISES):
     rir_sim_train / rir_sim_val : SIMULATED rooms (room-level 90/10 split, 3,000 RIRs per room size)
     rir_real_test               : REAL measured RIRs (RWCP / AIR / REVERB) - test only, never trained on
3. Extended test grid (test_conditions_ext.csv): real-room reverb, reverb + ESC-50, and -5 dB SNR.

Usage: python scripts/preprocess_m5.py [--skip-rirs]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.data.audio import to_mono_16k  # noqa: E402
from group9_ds_ai_lab.data.noise import stable_seed  # noqa: E402
from group9_ds_ai_lab.utils.paths import DATA_RAW, MANIFESTS, NOISE_DIR, SAMPLE_RATE, rel  # noqa: E402

RIR_ROOT = DATA_RAW / "rirs" / "RIRS_NOISES"


def train_manifests(min_dur=1.0, max_dur=20.0):
    full = pd.read_parquet(MANIFESTS / "librispeech_all.parquet")
    ok = full.decode_ok & (full.n_chars > 0) & full.ctc_feasible & full.duration.between(min_dur, max_dur)
    cols = pd.read_csv(MANIFESTS / "train.csv", nrows=1).columns
    stats = {}
    for name, splits in (("train_460", ["train.clean.100", "train.clean.360"]),
                         ("train_960", ["train.clean.100", "train.clean.360", "train.other.500"])):
        d = full[ok & full.ls_split.isin(splits)].copy()
        d["audio_path"] = d["shard"] + "#" + d["row"].astype(str)
        d["noise_type"], d["snr_db"], d["split"] = "clean", np.nan, "train"
        d["duration"] = d["duration"].round(4)
        d = d[list(cols)].reset_index(drop=True)
        d.to_csv(MANIFESTS / f"{name}.csv", index=False)
        # no leakage: speakers of the extra training data must not appear in dev / test
        held = set(pd.read_csv(MANIFESTS / "valid.csv").speaker_id) | set(pd.read_csv(MANIFESTS / "test.csv").speaker_id)
        shared = len(set(d.speaker_id) & held)
        assert shared == 0, f"{name}: {shared} speakers shared with dev/test"
        stats[name] = {"utts": int(len(d)), "hours": float(d.duration.sum() / 3600), "speakers": int(d.speaker_id.nunique())}
        print(name, stats[name], flush=True)
    return stats


def _read_list(path):
    out = []
    for line in open(path):
        p = line.split()
        if p:
            out.append((p[p.index("--room-id") + 1] if "--room-id" in p else "", p[-1]))
    return out


def _load(path):
    x, sr = sf.read(RIR_ROOT.parent / path, dtype="float32", always_2d=True)
    return to_mono_16k(x[:, 0], sr)  # first channel of multichannel measurements


def write_bank(name, files, arrs):
    lengths = [len(a) for a in arrs]
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    np.save(NOISE_DIR / f"{name}.npy", np.concatenate(arrs).astype(np.float16))
    with open(NOISE_DIR / f"{name}.index.json", "w") as f:
        json.dump({"files": files, "starts": [int(s) for s in starts], "lengths": lengths}, f)
    print(f"bank {name}: {len(files)} RIRs, median {np.median(lengths) / SAMPLE_RATE:.2f} s", flush=True)


def rir_banks(seed=42, per_size=3000, val_frac=0.1):
    rng = np.random.default_rng(seed)
    stats = {}
    train_f, val_f = [], []
    for size in ("smallroom", "mediumroom", "largeroom"):
        items = _read_list(RIR_ROOT / "simulated_rirs" / size / "rir_list")
        rooms = sorted({r for r, _ in items})
        val_rooms = set(rng.choice(rooms, size=max(1, int(round(len(rooms) * val_frac))), replace=False))
        pick = [items[i] for i in rng.permutation(len(items))[:per_size]]
        train_f += [p for r, p in pick if r not in val_rooms]
        val_f += [p for r, p in pick if r in val_rooms]
    real = [p for _, p in _read_list(RIR_ROOT / "real_rirs_isotropic_noises" / "rir_list")]
    for name, files in (("rir_sim_train", train_f), ("rir_sim_val", val_f), ("rir_real_test", real)):
        arrs = [_load(f) for f in files]
        keep = [i for i, a in enumerate(arrs) if len(a) > 0 and np.abs(a).max() > 0]
        write_bank(name, [rel(RIR_ROOT.parent / files[i]) for i in keep], [arrs[i] for i in keep])
        stats[name] = len(keep)
    return stats


def extended_test():
    test = pd.read_csv(MANIFESTS / "test.csv")
    conds = [("clean", np.nan, True), ("esc50", 10.0, True), ("esc50", 5.0, True), ("esc50", 0.0, True),
             ("esc50", -5.0, False), ("white", -5.0, False), ("pink", -5.0, False)]
    parts = []
    for n, s, rev in conds:
        d = test.copy()
        d["noise_type"], d["snr_db"], d["reverb"] = n, s, rev
        d["cond_seed"] = [stable_seed(a, n, s, "rev" if rev else "", "test_ext") for a in d.audio_id]
        parts.append(d)
    out = pd.concat(parts).reset_index(drop=True)
    out["noise_part"] = "test"
    out.to_csv(MANIFESTS / "test_conditions_ext.csv", index=False)
    print(f"test_conditions_ext.csv: {len(out)} rows", flush=True)
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rirs", action="store_true")
    args = ap.parse_args()
    qc = {"train_manifests": train_manifests()}
    if not args.skip_rirs:
        qc["rir_banks"] = rir_banks()
        qc["test_ext_rows"] = extended_test()
    with open(MANIFESTS / "qc_summary_m5.json", "w") as f:
        json.dump(qc, f, indent=2)


if __name__ == "__main__":
    main()
