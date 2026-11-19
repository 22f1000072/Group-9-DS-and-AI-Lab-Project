"""Milestone 3/4 preprocessing for the noise-robust ASR project.

What it does (one parallel pass, CPU only):
  1. LibriSpeech (all 7 splits, 960 h): decode every utterance from the local HF arrow
     shards, normalise transcript, compute duration + signal-quality stats -> full manifest.
  2. Model-ready split manifests in the M2 schema (metadata.csv + per-split CSVs):
       train = train-clean-100 (filtered), valid = dev-clean + dev-other, test = test-clean + test-other.
     Audio is NOT duplicated: `audio_path` points to "<arrow shard>#<row>" (16 kHz FLAC inside).
  3. Noise banks at 16 kHz: MUSAN noise/music split by file 90/10 (train/val), ESC-50 (test only).
  4. Fixed, seeded validation and test corruption manifests so every model hears identical audio.

Usage:  python scripts/preprocess.py [--workers 120]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.data.audio import decode_flac_bytes, frame_stats, open_arrow, to_mono_16k  # noqa: E402
from group9_ds_ai_lab.data.noise import TEST_NOISE_TYPES, TEST_SNRS, TRAIN_NOISE_TYPES, stable_seed  # noqa: E402
from group9_ds_ai_lab.utils.paths import (ESC50_ROOT, LS_SPLITS, MANIFESTS, MUSAN_ROOT, NOISE_DIR, PROCESSED,  # noqa: E402
                       SAMPLE_RATE, absp, librispeech_shards, rel)
from group9_ds_ai_lab.data.text import encode, normalize_text  # noqa: E402

SPLIT_ROLE = {
    "train.clean.100": "train", "train.clean.360": "unused", "train.other.500": "unused",
    "validation.clean": "valid", "validation.other": "valid",
    "test.clean": "test", "test.other": "test",
}


def ctc_frames(n_samples: int) -> int:
    """Output frames of the wav2vec2 conv stack (kernels 10,3,3,3,3,2,2 / strides 5,2,2,2,2,2,2)."""
    L = n_samples
    for k, s in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        L = (L - k) // s + 1
    return max(L, 0)


def ctc_min_frames(ids: list[int]) -> int:
    """Minimum CTC frames for a label sequence: length + number of adjacent repeats."""
    return len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)


def process_shard(args):
    split, shard = args
    t = open_arrow(shard)
    texts = t.column("text").to_pylist()
    ids = t.column("id").to_pylist()
    spk = t.column("speaker_id").to_pylist()
    chap = t.column("chapter_id").to_pylist()
    audio = t.column("audio").combine_chunks().field("bytes")
    rows = []
    for i in range(t.num_rows):
        rec = {"audio_id": ids[i], "ls_split": split, "shard": rel(shard), "row": i,
               "speaker_id": spk[i], "chapter_id": chap[i], "raw_text": texts[i]}
        try:
            x, sr, ch = decode_flac_bytes(audio[i].as_py())
            n16 = len(x) if sr == SAMPLE_RATE else int(round(len(x) * SAMPLE_RATE / sr))
            rec.update(original_sample_rate=sr, channels=ch, duration=n16 / SAMPLE_RATE,
                       n_samples_16k=n16, decode_ok=True, **frame_stats(x, sr))
        except Exception as e:  # corrupted audio -> recorded, filtered later
            rec.update(decode_ok=False, error=str(e)[:200])
            n16 = 0
        text = normalize_text(texts[i] or "")
        lab = encode(text)
        rec.update(text=text, n_chars=len(text), n_words=len(text.split()) if text else 0,
                   ctc_frames=ctc_frames(n16), ctc_min_frames=ctc_min_frames(lab),
                   text_changed=(text != (texts[i] or "").strip()))
        rows.append(rec)
    return rows


def build_librispeech(workers: int) -> pd.DataFrame:
    jobs = [(s, f) for s in LS_SPLITS for f in librispeech_shards(s)]
    print(f"[librispeech] {len(jobs)} shards, {workers} workers")
    t0, rows = time.time(), []
    with ProcessPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(process_shard, jobs, chunksize=1), 1):
            rows.extend(r)
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} shards, {len(rows)} utts, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["role"] = df["ls_split"].map(SPLIT_ROLE)
    df["chars_per_sec"] = df["n_chars"] / df["duration"].replace(0, np.nan)
    df["ctc_feasible"] = df["ctc_frames"] >= df["ctc_min_frames"]
    return df


def m2_schema(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Rows in the schema promised in Milestone 2 (+ extra columns needed by the loaders)."""
    out = pd.DataFrame({
        "audio_id": df["audio_id"],
        "audio_path": df["shard"] + "#" + df["row"].astype(str),
        "text": df["text"],
        "duration": df["duration"].round(4),
        "original_sample_rate": df["original_sample_rate"],
        "noise_type": "clean",
        "snr_db": np.nan,
        "split": split_name,
        "ls_split": df["ls_split"],
        "shard": df["shard"],
        "row": df["row"],
        "speaker_id": df["speaker_id"],
        "chapter_id": df["chapter_id"],
    })
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- noise
def _load_noise_file(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    ch = x.shape[1]
    y = to_mono_16k(x.mean(axis=1), sr)
    return path, sr, ch, y


def build_noise(workers: int, seed: int, val_frac: float) -> pd.DataFrame:
    NOISE_DIR.mkdir(parents=True, exist_ok=True)
    musan = sorted(str(p) for p in MUSAN_ROOT.glob("*/*/*.wav"))
    esc = sorted(str(p) for p in (ESC50_ROOT / "audio").glob("*.wav"))
    esc_meta = pd.read_csv(ESC50_ROOT / "meta" / "esc50.csv").set_index("filename")
    print(f"[noise] MUSAN {len(musan)} files, ESC-50 {len(esc)} files")
    loaded = {}
    with ProcessPoolExecutor(workers) as ex:
        for path, sr, ch, y in ex.map(_load_noise_file, musan + esc, chunksize=4):
            loaded[path] = (sr, ch, y)

    recs = []
    rng = np.random.default_rng(seed)
    for path in musan + esc:
        sr, ch, y = loaded[path]
        p = Path(path)
        if path in esc:
            corpus, category = "esc50", esc_meta.loc[p.name, "category"]
            sub = "esc50"
        else:
            corpus, category, sub = "musan", p.parts[-3], p.parts[-2]
        recs.append({"corpus": corpus, "category": category, "subset": sub, "file": rel(path),
                     "original_sample_rate": sr, "channels": ch, "duration": len(y) / SAMPLE_RATE,
                     **frame_stats(y)})
    meta = pd.DataFrame(recs)

    # MUSAN file-level train/val split (per category, seeded). MUSAN speech is NOT used
    # (babble is out of scope) but is profiled for EDA.
    meta["role"] = "unused"
    for cat in ("noise", "music"):
        idx = meta.index[(meta.corpus == "musan") & (meta.category == cat)].to_numpy()
        idx = idx[rng.permutation(len(idx))]
        n_val = int(round(len(idx) * val_frac))
        meta.loc[idx[:n_val], "role"] = "val"
        meta.loc[idx[n_val:], "role"] = "train"
    meta.loc[meta.corpus == "esc50", "role"] = "test"
    silent = meta["rms_dbfs"] < -70  # digital-silence clips cannot set an SNR
    meta.loc[silent & (meta.role != "unused"), "role"] = "excluded_silent"

    def write_bank(name, sel):
        files = meta.loc[sel, "file"].tolist()
        arrs = [loaded[absp(f)][2] for f in files]
        lengths = [len(a) for a in arrs]
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
        np.save(NOISE_DIR / f"{name}.npy", np.concatenate(arrs).astype(np.float16))
        with open(NOISE_DIR / f"{name}.index.json", "w") as f:
            json.dump({"files": files, "starts": [int(s) for s in starts], "lengths": lengths}, f)
        print(f"  bank {name}: {len(files)} clips, {sum(lengths) / SAMPLE_RATE / 3600:.2f} h")

    for cat in ("noise", "music"):
        for part in ("train", "val"):
            write_bank(f"musan_{cat}_{part}", (meta.corpus == "musan") & (meta.category == cat) & (meta.role == part))
    write_bank("esc50_test", (meta.corpus == "esc50") & (meta.role == "test"))
    return meta


# ----------------------------------------------------------------------------- fixed conditions
def build_valid_eval(valid: pd.DataFrame, per_split: int, seed: int) -> pd.DataFrame:
    """Dev subset used for checkpoint selection: each utt once clean, once with held-out MUSAN/synthetic
    training-type noise (never brown / ESC-50, so test noise stays unseen during model selection)."""
    rng = np.random.default_rng(seed)
    parts = []
    for s, g in valid.groupby("ls_split"):
        parts.append(g.sample(n=min(per_split, len(g)), random_state=seed))
    base = pd.concat(parts).reset_index(drop=True)
    clean = base.assign(noise_type="clean", snr_db=np.nan, cond_seed=0)
    noisy = base.copy()
    noisy["noise_type"] = rng.choice(list(TRAIN_NOISE_TYPES), size=len(noisy))
    noisy["snr_db"] = rng.choice([0.0, 5.0, 10.0, 20.0], size=len(noisy))
    noisy["cond_seed"] = [stable_seed(a, n, s, "val") for a, n, s in zip(noisy.audio_id, noisy.noise_type, noisy.snr_db)]
    out = pd.concat([clean, noisy]).reset_index(drop=True)
    out["noise_part"] = "val"
    return out


def build_test_conditions(test: pd.DataFrame) -> pd.DataFrame:
    conds = [("clean", np.nan)] + [(n, float(s)) for n in TEST_NOISE_TYPES for s in TEST_SNRS]
    parts = []
    for n, s in conds:
        d = test.copy()
        d["noise_type"], d["snr_db"] = n, s
        d["cond_seed"] = [0 if n == "clean" else stable_seed(a, n, s, "test") for a in d.audio_id]
        parts.append(d)
    out = pd.concat(parts).reset_index(drop=True)
    out["noise_part"] = "test"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=min(120, os.cpu_count() or 8))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-train-dur", type=float, default=20.0)
    ap.add_argument("--musan-val-frac", type=float, default=0.10)
    ap.add_argument("--valid-eval-per-split", type=int, default=1000)
    ap.add_argument("--skip-librispeech", action="store_true", help="reuse existing full manifest")
    ap.add_argument("--skip-noise", action="store_true")
    args = ap.parse_args()

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    full_path = MANIFESTS / "librispeech_all.parquet"
    t0 = time.time()
    if args.skip_librispeech and full_path.exists():
        full = pd.read_parquet(full_path)
    else:
        full = build_librispeech(args.workers)
        full.to_parquet(full_path, index=False)
    print(f"[librispeech] {len(full)} utterances, {full.duration.sum() / 3600:.1f} h ({time.time() - t0:.0f}s)")

    # ---- QC + filtering
    qc = {"total_utts": int(len(full)), "decode_failed": int((~full.decode_ok).sum()),
          "empty_text": int((full.n_chars == 0).sum()), "text_changed_by_norm": int(full.text_changed.sum()),
          "non_16k": int((full.original_sample_rate != SAMPLE_RATE).sum()),
          "multi_channel": int((full.channels > 1).sum()), "ctc_infeasible": int((~full.ctc_feasible).sum()),
          "duplicate_ids": int(full.audio_id.duplicated().sum())}
    ok = full.decode_ok & (full.n_chars > 0) & full.ctc_feasible
    train_mask = ok & (full.role == "train") & full.duration.between(args.min_dur, args.max_train_dur)
    qc["train_before_filter"] = int((full.role == "train").sum())
    qc["train_after_filter"] = int(train_mask.sum())
    qc["train_dropped_too_long"] = int(((full.role == "train") & (full.duration > args.max_train_dur)).sum())
    qc["train_dropped_too_short"] = int(((full.role == "train") & (full.duration < args.min_dur)).sum())
    qc["train_hours"] = float(full.loc[train_mask, "duration"].sum() / 3600)

    train = m2_schema(full[train_mask], "train")
    valid = m2_schema(full[ok & (full.role == "valid")], "valid")
    test = m2_schema(full[ok & (full.role == "test")], "test")
    # leakage checks: utterance ids and speakers must be disjoint across splits
    for a, b, na, nb in [(train, valid, "train", "valid"), (train, test, "train", "test"), (valid, test, "valid", "test")]:
        assert not set(a.audio_id) & set(b.audio_id), f"utterance leakage {na}/{nb}"
        qc[f"shared_speakers_{na}_{nb}"] = len(set(a.speaker_id) & set(b.speaker_id))
    for name, d in (("train", train), ("valid", valid), ("test", test)):
        d.to_csv(MANIFESTS / f"{name}.csv", index=False)
        qc[f"{name}_utts"], qc[f"{name}_hours"] = int(len(d)), float(d.duration.sum() / 3600)
    pd.concat([train, valid, test]).to_csv(PROCESSED / "metadata.csv", index=False)

    # ---- noise
    if not args.skip_noise:
        nmeta = build_noise(args.workers, args.seed, args.musan_val_frac)
        nmeta.to_csv(MANIFESTS / "noise_files.csv", index=False)
        qc["noise_roles"] = nmeta.groupby(["corpus", "category", "role"]).size().reset_index().astype(str).values.tolist()

    # ---- fixed evaluation conditions
    valid_eval = build_valid_eval(valid, args.valid_eval_per_split, args.seed)
    valid_eval.to_csv(MANIFESTS / "valid_eval.csv", index=False)
    test_cond = build_test_conditions(test)
    test_cond.to_csv(MANIFESTS / "test_conditions.csv", index=False)
    qc["valid_eval_rows"], qc["test_condition_rows"] = int(len(valid_eval)), int(len(test_cond))

    params = {
        "sample_rate": SAMPLE_RATE,
        "train_noise": {"types": list(TRAIN_NOISE_TYPES), "p_clean": 0.2, "snr_db_uniform": [0, 20],
                        "musan_files": f"90% of MUSAN noise+music (file-level split, seed {args.seed})"},
        "valid_noise": {"types": list(TRAIN_NOISE_TYPES), "snr_db": [0, 5, 10, 20], "musan_files": "held-out 10%"},
        "test_noise": {"types": list(TEST_NOISE_TYPES), "snr_db": list(TEST_SNRS) + ["clean"],
                       "held_out_types": ["brown", "esc50"], "seen_reference_types": ["white", "pink"]},
        "mixing": "noise scaled to P_speech / 10^(SNR/10) over the whole utterance, no clipping",
        "text_normalisation": "upper-case, keep A-Z and apostrophe, collapse whitespace (32-char wav2vec2 vocab)",
        "seed": args.seed,
    }
    (PROCESSED / "augmentation").mkdir(exist_ok=True)
    with open(PROCESSED / "augmentation" / "noise_params.json", "w") as f:
        json.dump(params, f, indent=2)
    with open(MANIFESTS / "qc_summary.json", "w") as f:
        json.dump(qc, f, indent=2, default=str)
    print(json.dumps(qc, indent=2, default=str))
    print(f"[done] {time.time() - t0:.0f}s -> {PROCESSED}")


if __name__ == "__main__":
    main()
