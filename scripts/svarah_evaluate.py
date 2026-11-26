"""Evaluation on Indian-accented English: the Svarah benchmark (AI4Bharat, Interspeech 2023, CC BY 4.0).

A separate evaluation: nothing is retrained and no existing project file is changed. It reuses the project's own
functions by importing them (model inference, seeded noise mixing, LM beam search, ROVER, word-error detector,
CTC forced alignment), so the Indian-accent numbers are directly comparable with the LibriSpeech test grid.

Stages (run all, or pick with --stages):
  prepare   decode the 6,656 Svarah clips to 16 kHz, normalise the transcripts (same normaliser as training;
            digits -> words), keep speaker details (native language, state, gender, age, education)
  infer     every model x every condition on the GPU (same batching / bf16 inference as scripts/evaluate.py);
            greedy hypotheses + CTC log-probabilities
  lm        4-gram LM beam search (alpha 0.8, beta 2.0, beam 50, as on LibriSpeech) for the LM systems
  rover     ROVER of B960_rev_s2 + B960_rev + B460 (lam 0.5): the final system, as on LibriSpeech
  analyse   WER/CER grids, accent gap vs LibriSpeech with bootstrap CIs, speaker-group breakdowns, error types,
            confusions, detector (AUROC, recall@30 %, ECE, review curve), replay window, audio examples

Conditions: clean (as recorded) + ESC-50 at 10 / 5 / 0 dB + white noise at 5 dB, mixed with the project's seeded
apply_condition() (every model hears identical audio).

Usage:
  python scripts/svarah_evaluate.py                       # everything
  python scripts/svarah_evaluate.py --stages analyse      # re-run the analysis only
Inputs : data/raw/svarah/data/test-*.parquet  (downloaded from huggingface.co/datasets/ai4bharat/Svarah)
Outputs: data/processed/svarah/ (audio, log-probs; not in git), experiments/results/svarah/ (per-utterance outputs),
         milestones/M5_evaluation/indian_accent/{tables,figures,audio_examples}/
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from group9_ds_ai_lab.utils.paths import DATA_RAW, EVAL, PROCESSED, REPO, SAMPLE_RATE, setup_env  # noqa: E402

setup_env()
RAW = DATA_RAW / "svarah" / "data"
WORK = PROCESSED / "svarah"
RES = EVAL / "svarah"
OUT = REPO / "milestones" / "M5_evaluation" / "indian_accent"
MODELS = {"ref_base960h": "facebook/wav2vec2-base-960h", "A_clean": "A_clean", "B_noisy": "B_noisy",
          "B2_lr1e-4": "B2_lr1e-4", "B460": "B460", "B960_rev": "B960_rev", "B960_rev_s2": "B960_rev_s2"}
LM_SYSTEMS = ["A_clean", "B_noisy", "B460", "B960_rev", "B960_rev_s2"]
ROVER = ["B960_rev_s2", "B960_rev", "B460"]
CONDITIONS = [("clean", None), ("esc50", 10.0), ("esc50", 5.0), ("esc50", 0.0), ("white", 5.0)]
SEED = 0


def cond_name(n, s):
    return "clean" if n == "clean" else f"{n} {int(s)} dB"


# =============================================================================================== text
ONES = "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN FIFTEEN SIXTEEN " \
       "SEVENTEEN EIGHTEEN NINETEEN".split()
TENS = "_ _ TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY".split()
ORD = {"ONE": "FIRST", "TWO": "SECOND", "THREE": "THIRD", "FIVE": "FIFTH", "EIGHT": "EIGHTH", "NINE": "NINTH",
       "TWELVE": "TWELFTH"}


def num_words(n: int) -> str:
    """Cardinal number in words (0 .. 999,999,999), the way the LibriSpeech transcripts write numbers."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " HUNDRED" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    for div, name in ((10 ** 6, "MILLION"), (1000, "THOUSAND")):
        if n >= div:
            return num_words(n // div) + f" {name}" + ("" if n % div == 0 else " " + num_words(n % div))
    return str(n)


def year_words(n: int) -> str:
    """Years are read as in speech: 1998 -> NINETEEN NINETY EIGHT, 2005 -> TWO THOUSAND FIVE, 2020 -> TWENTY TWENTY."""
    if 2000 <= n <= 2009:
        return num_words(n)
    hi, lo = divmod(n, 100)
    return num_words(hi) + (" HUNDRED" if lo == 0 else (" OH " + ONES[lo] if lo < 10 else " " + num_words(lo)))


def ordinal_words(n: int) -> str:
    w = num_words(n).split()
    last = w[-1]
    w[-1] = ORD.get(last, last[:-1] + "IETH" if last.endswith("Y") else last + "TH")
    return " ".join(w)


def spell_numbers(text: str) -> str:
    t = re.sub(r"(?<=\d),(?=\d{3})", "", text)                       # 10,000 -> 10000
    t = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 TO \2", t)                  # 15-17 -> 15 TO 17
    t = t.replace("%", " PERCENT ").replace("&", " AND ").replace("+", " PLUS ").replace("@", " AT ")
    t = re.sub(r"\$\s*(\d+)", r"\1 DOLLARS", t)
    t = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: ordinal_words(int(m.group(1))), t, flags=re.I)
    t = re.sub(r"(\d+)\.(\d+)", lambda m: num_words(int(m.group(1))) + " POINT " + " ".join(ONES[int(c)] for c in m.group(2)), t)
    t = re.sub(r"\d+", lambda m: year_words(int(m.group())) if len(m.group()) == 4 and 1100 <= int(m.group()) <= 2099
               else num_words(int(m.group())), t)
    return t


def normalise(text: str) -> str:
    from group9_ds_ai_lab.data.text import normalize_text
    return normalize_text(spell_numbers(str(text)))


# =============================================================================================== prepare
def prepare():
    import soundfile as sf

    from group9_ds_ai_lab.data.audio import to_mono_16k
    WORK.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    rows, chunks, off = [], [], [0]
    for f in sorted(RAW.glob("test-*.parquet")):
        t = pq.read_table(f).to_pandas().rename(columns={"age-group": "age_group"})
        for r in t.itertuples(index=False):
            a = r.audio_filepath
            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
            x = to_mono_16k(x, sr)
            chunks.append(np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16))
            off.append(off[-1] + len(x))
            rows.append({"utt_id": f"svarah_{len(rows):05d}", "source_path": a.get("path"), "duration": len(x) / SAMPLE_RATE,
                         "orig_sr": sr, "text_raw": r.text, "text": normalise(r.text),
                         "has_digit": bool(re.search(r"\d", r.text)), "gender": r.gender, "age_group": r.age_group,
                         "primary_language": r.primary_language, "state": r.native_place_state,
                         "district": r.native_place_district, "education": r.highest_qualification,
                         "job_category": r.job_category, "occupation_domain": r.occupation_domain})
    meta = pd.DataFrame(rows)
    np.savez(WORK / "audio.npz", audio=np.concatenate(chunks), offsets=np.array(off, np.int64))
    meta.to_parquet(WORK / "meta.parquet", index=False)
    print(f"prepared {len(meta)} clips, {meta.duration.sum() / 3600:.2f} h, {meta.has_digit.sum()} with digits, "
          f"{(meta.text == '').sum()} empty after normalisation", flush=True)


def load_prepared():
    z = np.load(WORK / "audio.npz")
    return pd.read_parquet(WORK / "meta.parquet"), z["audio"], z["offsets"]


# =============================================================================================== infer
class SvarahDataset:
    """Same interface as group9_ds_ai_lab.data.dataset.ASRDataset (fixed conditions), audio from memory."""

    def __init__(self, meta, audio, offsets, noise, snr):
        self.meta, self.audio, self.off, self.noise, self.snr = meta, audio, offsets, noise, snr
        self.durations = meta.duration.to_numpy()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        from group9_ds_ai_lab.data.dataset import zscore
        from group9_ds_ai_lab.data.noise import apply_condition, stable_seed
        x = self.audio[self.off[i]:self.off[i + 1]].astype(np.float32) / 32767
        if self.noise != "clean":
            x, _ = apply_condition(x, self.noise, self.snr, stable_seed("svarah", self.meta.utt_id.iat[i], self.noise, self.snr))
        return {"input_values": zscore(x), "labels": [], "index": i}


def infer(models):
    import torch
    from transformers import Wav2Vec2ForCTC

    from group9_ds_ai_lab.evaluation.models import model_source
    from group9_ds_ai_lab.inference.decode import run_inference
    meta, audio, off = load_prepared()
    (WORK / "logits").mkdir(parents=True, exist_ok=True)
    for name in models:
        done = all((WORK / "logits" / f"{name}__{cond_name(n, s).replace(' ', '_')}.npz").exists() for n, s in CONDITIONS)
        if done:
            print(name, "already decoded", flush=True)
            continue
        model = Wav2Vec2ForCTC.from_pretrained(model_source(name, "local")).cuda().eval()
        for n, s in CONDITIONS:
            f = WORK / "logits" / f"{name}__{cond_name(n, s).replace(' ', '_')}.npz"
            if f.exists():
                continue
            t0 = time.time()
            res = run_inference(model, SvarahDataset(meta, audio, off, n, s), max_seconds=800.0, num_workers=16,
                                keep_logp=True)
            lps = [r["logp"] for r in res]
            o = np.cumsum([0] + [len(x) for x in lps])
            np.savez(f, logp=np.concatenate(lps), offsets=o)
            pd.DataFrame({"utt_id": meta.utt_id, "hyp": [r["hyp"] for r in res]}).to_parquet(f.with_suffix(".greedy.parquet"))
            print(f"{name} {cond_name(n, s)}: {time.time() - t0:.0f} s", flush=True)
        del model
        torch.cuda.empty_cache()


# =============================================================================================== lm
def lm_decode(systems, workers=96):
    import decode_words as DW
    from pyctcdecode import build_ctcdecoder
    arpa = DATA_RAW / "librispeech_lm" / "4-gram.arpa"
    t0 = time.time()
    DW._G["decoder"] = build_ctcdecoder(DW.labels(), str(arpa), unigrams=DW.arpa_unigrams(arpa), alpha=0.8, beta=2.0)
    DW._G["beam"] = 50
    print(f"4-gram LM loaded in {time.time() - t0:.0f} s", flush=True)
    meta = pd.read_parquet(WORK / "meta.parquet")
    for name in systems:
        out = RES / f"{name}_lm4"
        if (out / "predictions.parquet").exists():
            print(name, "LM already decoded", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        preds, words = [], []
        for n, s in CONDITIONS:
            z = np.load(WORK / "logits" / f"{name}__{cond_name(n, s).replace(' ', '_')}.npz")
            DW._G["logp"], DW._G["off"] = z["logp"], z["offsets"]
            t0 = time.time()
            res = DW.run(list(range(len(meta))), workers)
            print(f"{name} LM {cond_name(n, s)}: {time.time() - t0:.0f} s", flush=True)
            preds.append(meta[["utt_id", "text"]].assign(noise_type=n, snr_db=s, hyp=[h for h, _ in res]))
            for i, (_, ws) in enumerate(res):
                for w in ws:
                    words.append((meta.utt_id.iat[i], n, s, w["word"], w["start"], w["end"], w["conf_min"], w["conf_mean"]))
        p = pd.concat(preds, ignore_index=True)
        p.to_parquet(out / "predictions.parquet", index=False)
        pd.DataFrame(words, columns=["utt_id", "noise_type", "snr_db", "word", "start", "end", "conf_min", "conf_mean"]) \
            .to_parquet(out / "words.parquet", index=False)


def greedy_predictions(name):
    meta = pd.read_parquet(WORK / "meta.parquet")
    parts = []
    for n, s in CONDITIONS:
        g = pd.read_parquet(WORK / "logits" / f"{name}__{cond_name(n, s).replace(' ', '_')}.greedy.parquet")
        parts.append(meta[["utt_id", "text"]].assign(noise_type=n, snr_db=s, hyp=g.hyp.values))
    return pd.concat(parts, ignore_index=True)


# =============================================================================================== rover
def rover():
    from rover import combine
    out = RES / "rover3_final"
    if (out / "predictions.parquet").exists():
        print("ROVER already done", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    sys_ = []
    for r in ROVER:
        p = pd.read_parquet(RES / f"{r}_lm4" / "predictions.parquet")
        w = pd.read_parquet(RES / f"{r}_lm4" / "words.parquet")
        w = w.assign(snr_k=w.snr_db.fillna(-99))  # clean has no SNR (NaN), which cannot be a dict key
        conf = w.groupby(["utt_id", "noise_type", "snr_k"], sort=False).conf_mean.apply(list).to_dict()
        sys_.append((p, [conf.get((a, n, -99 if pd.isna(s) else s), []) for a, n, s in zip(p.utt_id, p.noise_type, p.snr_db)]))
    base = sys_[0][0]
    hyp = [combine([s[0].hyp.iat[i] or "" for s in sys_], [s[1][i] for s in sys_], 0.5) for i in range(len(base))]
    base.drop(columns="hyp").assign(hyp=hyp).to_parquet(out / "predictions.parquet", index=False)


# =============================================================================================== analyse
def system_predictions():
    """name -> (decoding, predictions with hyp). Greedy for all 7 models, 4-gram LM for 5, ROVER final."""
    out = {}
    for m in MODELS:
        out[(m, "greedy")] = greedy_predictions(m)
    for m in LM_SYSTEMS:
        if (RES / f"{m}_lm4" / "predictions.parquet").exists():
            out[(m, "4-gram LM")] = pd.read_parquet(RES / f"{m}_lm4" / "predictions.parquet")
    if (RES / "rover3_final" / "predictions.parquet").exists():
        out[("rover3_final", "4-gram LM + ROVER")] = pd.read_parquet(RES / "rover3_final" / "predictions.parquet")
    for k, d in out.items():
        d["hyp"] = d.hyp.fillna("").astype(str).str.strip()
        d["cond"] = [cond_name(n, s) for n, s in zip(d.noise_type, d.snr_db)]
    return out


def counts(d):
    """Per-utterance H/S/D/I with jiwer (same as scripts/m5_analysis.align)."""
    import m5_analysis as MA
    c, al = MA.align(d.rename(columns={}))
    return c, al


def analyse():
    import jiwer
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import m5_analysis as MA
    from error_detector import ece, features, recall_at, unigram_table
    from group9_ds_ai_lab.utils.plotstyle import GRID, INK, INK2, SERIES, apply_style
    apply_style()
    tab, fig = OUT / "tables", OUT / "figures"
    for p in (tab, fig, OUT / "audio_examples"):
        p.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(WORK / "meta.parquet")
    meta = meta[meta.text.str.len() > 0].reset_index(drop=True)
    keep = set(meta.utt_id)
    preds = system_predictions()
    C = {}
    for k, d in preds.items():
        d = d[d.utt_id.isin(keep)].reset_index(drop=True)
        c, al = MA.align(d)
        C[k] = (d, c, al)

    def pooled(c, m):
        e, n = (c.S + c.D + c.I)[m].sum(), (c.H + c.S + c.D)[m].sum()
        return 100 * e / max(n, 1)

    def cer(d, m):
        from group9_ds_ai_lab.inference.decode import wer_cer
        return 100 * wer_cer(d.text[m].tolist(), d.hyp[m].tolist())[1]

    conds = [cond_name(n, s) for n, s in CONDITIONS]
    nodig = set(meta.utt_id[~meta.has_digit])
    # ---- 1. WER / CER grid
    rows = []
    for (m, dec), (d, c, _) in C.items():
        for cn in conds:
            sel = (d.cond == cn).values
            sel2 = sel & d.utt_id.isin(nodig).values
            rows.append({"system": m, "decoding": dec, "condition": cn, "wer_%": pooled(c, sel), "cer_%": cer(d, sel),
                         "wer_no_digit_utts_%": pooled(c, sel2), "sub_%": 100 * c.S[sel].sum() / (c.H + c.S + c.D)[sel].sum(),
                         "del_%": 100 * c.D[sel].sum() / (c.H + c.S + c.D)[sel].sum(),
                         "ins_%": 100 * c.I[sel].sum() / (c.H + c.S + c.D)[sel].sum(), "n_utts": int(sel.sum())})
    g = pd.DataFrame(rows)
    g.round(3).to_csv(tab / "svarah_wer_cer_grid.csv", index=False)
    for col in ("wer_%", "cer_%"):
        g.pivot_table(index=["decoding", "system"], columns="condition", values=col, sort=False)[conds].round(2) \
            .to_csv(tab / f"svarah_{col[:3]}_pivot.csv")
    # ---- 2. accent gap vs LibriSpeech (same systems, same decoding)
    ls = pd.read_csv(REPO / "milestones" / "M5_evaluation" / "tables" / "g1_wer_cer_all_conditions.csv")
    ls = ls[ls.ls_split.isin(["test.clean", "test.other", "both"])]
    gap = []
    for (m, dec) in C:
        x = ls[(ls.system == m) & (ls.decoding == dec)]
        if not len(x):
            continue
        def lsw(split, nt, snr):
            r = x[(x.ls_split == split) & (x.noise_type == nt) & (x.snr.astype(str) == str(snr))]
            return float(r["wer_%"].iloc[0]) if len(r) else np.nan
        sv = g[(g.system == m) & (g.decoding == dec)].set_index("condition")["wer_%"]
        gap.append({"system": m, "decoding": dec, "LS_test_clean": lsw("test.clean", "clean", "clean"),
                    "LS_test_other": lsw("test.other", "clean", "clean"), "Svarah_clean": sv["clean"],
                    "LS_esc50_5dB": lsw("both", "esc50", 5), "Svarah_esc50_5dB": sv["esc50 5 dB"],
                    "LS_esc50_0dB": lsw("both", "esc50", 0), "Svarah_esc50_0dB": sv["esc50 0 dB"]})
    gap = pd.DataFrame(gap)
    gap["gap_clean_vs_test_clean_x"] = gap.Svarah_clean / gap.LS_test_clean
    gap.round(2).to_csv(tab / "svarah_vs_librispeech.csv", index=False)
    # ---- 3. bootstrap CIs (utterances resampled; all conditions of a clip together)
    rng = np.random.default_rng(SEED)
    comps = [("noise training A -> B (greedy)", ("A_clean", "greedy"), ("B_noisy", "greedy")),
             ("noise training A -> B (LM)", ("A_clean", "4-gram LM"), ("B_noisy", "4-gram LM")),
             ("data: B -> B460 (LM)", ("B_noisy", "4-gram LM"), ("B460", "4-gram LM")),
             ("data + aug.: B460 -> B960_rev (LM)", ("B460", "4-gram LM"), ("B960_rev", "4-gram LM")),
             ("LM: B960_rev greedy -> 4-gram", ("B960_rev", "greedy"), ("B960_rev", "4-gram LM")),
             ("ROVER: B960_rev (LM) -> ROVER", ("B960_rev", "4-gram LM"), ("rover3_final", "4-gram LM + ROVER")),
             ("ours vs published: base-960h -> B960_rev (greedy)", ("ref_base960h", "greedy"), ("B960_rev", "greedy"))]
    W = None
    bt = []
    for lab, kx, ky in comps:
        if kx not in C or ky not in C:
            continue
        dx, cx, _ = C[kx]
        dy, cy, _ = C[ky]
        assert (dx.utt_id.values == dy.utt_id.values).all()
        u = pd.factorize(dx.utt_id)[0]
        nu = u.max() + 1
        if W is None:
            W = rng.multinomial(nu, np.full(nu, 1 / nu), size=1000).astype(np.float32)
        for sub, m in (("clean", dx.cond.eq("clean").values), ("ESC-50 0-5 dB", dx.cond.isin(["esc50 0 dB", "esc50 5 dB"]).values),
                       ("all noisy", dx.cond.ne("clean").values)):
            ex = np.bincount(u[m], (cx.S + cx.D + cx.I).values[m], nu)
            ey = np.bincount(u[m], (cy.S + cy.D + cy.I).values[m], nu)
            nn = np.bincount(u[m], (cx.H + cx.S + cx.D).values[m], nu)
            bx, by = (W @ ex) / (W @ nn), (W @ ey) / (W @ nn)
            rel = 100 * (by - bx) / bx
            bt.append({"comparison": lab, "subset": sub, "wer_X_%": 100 * ex.sum() / nn.sum(), "wer_Y_%": 100 * ey.sum() / nn.sum(),
                       "diff_pts": 100 * (ey.sum() - ex.sum()) / nn.sum(), "diff_ci_low": np.percentile(100 * (by - bx), 2.5),
                       "diff_ci_high": np.percentile(100 * (by - bx), 97.5), "rel_change_%": 100 * (ey.sum() / ex.sum() - 1),
                       "rel_ci_low": np.percentile(rel, 2.5), "rel_ci_high": np.percentile(rel, 97.5)})
    pd.DataFrame(bt).round(3).to_csv(tab / "svarah_bootstrap_ci.csv", index=False)
    # ---- 4. speaker groups (final system), clean and ESC-50 5 dB
    fk = ("rover3_final", "4-gram LM + ROVER") if ("rover3_final", "4-gram LM + ROVER") in C else ("B960_rev", "greedy")
    d, c, al = C[fk]
    dm = d.merge(meta.drop(columns=["text"]), on="utt_id", how="left")
    grp = []
    for col in ("primary_language", "state", "gender", "age_group", "education", "occupation_domain"):
        for cn in ("clean", "esc50 5 dB"):
            sel = (dm.cond == cn).values
            for val, idx in dm[sel].groupby(col).groups.items():
                ii = np.array(list(idx))
                n_spk_utts = len(ii)
                if n_spk_utts < 40:
                    continue
                grp.append({"attribute": col, "group": val, "condition": cn, "n_utts": n_spk_utts,
                            "wer_%": pooled(c, np.isin(np.arange(len(dm)), ii))})
    grp = pd.DataFrame(grp)
    grp.round(2).to_csv(tab / "svarah_wer_by_speaker_group.csv", index=False)
    # duration bins
    dm["dur_bin"] = pd.cut(dm.duration, [0, 2, 4, 8, 16, 99], labels=["< 2 s", "2-4 s", "4-8 s", "8-16 s", "> 16 s"])
    db = []
    for cn in ("clean", "esc50 5 dB"):
        for b, idx in dm[dm.cond == cn].groupby("dur_bin", observed=True).groups.items():
            ii = np.array(list(idx))
            db.append({"condition": cn, "duration": b, "n_utts": len(ii), "wer_%": pooled(c, np.isin(np.arange(len(dm)), ii))})
    pd.DataFrame(db).round(2).to_csv(tab / "svarah_wer_by_duration.csv", index=False)
    # ---- 5. word-level errors: OOV, substitutions, character confusions (final system, clean)
    uni = unigram_table()
    from collections import Counter
    pairs, refw = Counter(), []
    for i, (ref, hyp, chunks) in enumerate(zip(al.references, al.hypotheses, al.alignments)):
        if d.cond.iat[i] != "clean":
            continue
        ok = np.zeros(len(ref), bool)
        for ch in chunks if d.hyp.iat[i] else []:
            if ch.type == "equal":
                ok[ch.ref_start_idx:ch.ref_end_idx] = True
            if ch.type == "substitute":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    pairs[(ref[ch.ref_start_idx + k], hyp[ch.hyp_start_idx + k])] += 1
        refw += list(zip(ref, ok))
    rw = pd.DataFrame(refw, columns=["word", "correct"])
    rw["in_lm"] = rw.word.map(lambda w: w in uni)
    oov = {"ref_words_clean": int(len(rw)), "oov_share_%": 100 * float((~rw.in_lm).mean()),
           "oov_word_error_%": 100 * float(1 - rw[~rw.in_lm].correct.mean()),
           "in_vocab_word_error_%": 100 * float(1 - rw[rw.in_lm].correct.mean()),
           "top_oov_words": Counter(rw.word[~rw.in_lm]).most_common(25)}
    pd.DataFrame([(r, h, n) for (r, h), n in pairs.most_common(40)], columns=["reference", "hypothesis", "count"]) \
        .to_csv(tab / "svarah_top_substitutions.csv", index=False)
    # character confusions inside substituted words
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
    ix = {ch: i for i, ch in enumerate(letters)}
    M = np.zeros((27, 27))
    if pairs:
        rs, hs, ws = zip(*[(r, h, n) for (r, h), n in pairs.items()])
        co = jiwer.process_characters(list(rs), list(hs))
        for a, b, n, chunks in zip(co.references, co.hypotheses, ws, co.alignments):
            for ch in chunks:
                if ch.type == "substitute":
                    for k in range(ch.ref_end_idx - ch.ref_start_idx):
                        ra, hb = a[ch.ref_start_idx + k], b[ch.hyp_start_idx + k]
                        if ra in ix and hb in ix:
                            M[ix[ra], ix[hb]] += n
    off = M.copy()
    np.fill_diagonal(off, 0)
    top_char = [(letters[i], letters[j], int(off[i, j])) for i, j in zip(*np.unravel_index(np.argsort(-off, axis=None)[:15], off.shape))]
    pd.DataFrame(top_char, columns=["reference_char", "recognised_char", "count"]).to_csv(tab / "svarah_top_char_confusions.csv", index=False)
    # ---- 6. error detector (B960_rev + 4-gram LM words, detector trained on LibriSpeech dev, unchanged)
    det_res = {}
    wpath = RES / "B960_rev_lm4" / "words.parquet"
    if wpath.exists():
        import joblib
        from group9_ds_ai_lab.inference.decode import word_error_labels
        from sklearn.metrics import roc_auc_score
        ws = pd.read_parquet(wpath)
        ws = ws[ws.utt_id.isin(keep)].reset_index(drop=True)
        pb = preds[("B960_rev", "4-gram LM")]
        refmap = {(a, n, -99 if pd.isna(s) else s): t for a, n, s, t in zip(pb.utt_id, pb.noise_type, pb.snr_db, pb.text)}
        lab = []
        for (a, n, s), gg in ws.groupby(["utt_id", "noise_type", "snr_db"], dropna=False, sort=False):
            lab += list(zip(gg.index, word_error_labels(refmap[(a, n, -99 if pd.isna(s) else s)], " ".join(gg.word))))
        ws["is_error"] = 0
        ws.loc[[i for i, _ in lab], "is_error"] = [e for _, e in lab]
        f = features(ws.rename(columns={"utt_id": "audio_id"}), uni)
        det = joblib.load(EVAL / "B960_rev_lm4" / "error_detector" / "model.joblib")
        p = det["model"].predict_proba(f[det["features"]].to_numpy(np.float64))[:, 1]
        y = ws.is_error.to_numpy()
        raw = 1 - ws.conf_mean.to_numpy()
        rows = []
        for sub, m in [("all", np.ones(len(y), bool))] + [(cn, (np.array([cond_name(n, s) for n, s in zip(ws.noise_type, ws.snr_db)]) == cn)) for cn in conds]:
            for meth, sc in (("trained detector", p), ("raw confidence", raw)):
                pr = sc[m] >= 0.5
                rows.append({"subset": sub, "method": meth, "n_words": int(m.sum()), "error_rate_%": 100 * y[m].mean(),
                             "auroc": roc_auc_score(y[m], sc[m]), "recall_at_30%": 100 * recall_at(y[m], sc[m]),
                             "ece": ece(y[m], sc[m]) if meth == "trained detector" else np.nan,
                             "precision_at_0.5_%": 100 * (y[m][pr].mean() if pr.any() else np.nan),
                             "recall_at_0.5_%": 100 * (pr & (y[m] == 1)).sum() / max(y[m].sum(), 1)})
        det_res = pd.DataFrame(rows)
        det_res.round(4).to_csv(tab / "svarah_detector.csv", index=False)
        # review-effort curve vs LibriSpeech
        grid = np.linspace(0, 1, 201)

        def curve(yy, sc):
            o = np.argsort(-sc, kind="stable")
            cc = np.concatenate([[0], np.cumsum(yy[o])]) / yy.sum()
            return cc[np.round(grid * len(yy)).astype(int)]
        lsc = pd.read_csv(REPO / "milestones" / "M5_evaluation" / "tables" / "g5_review_effort_curve.csv")
        cur = pd.DataFrame({"share_reviewed": grid, "svarah_detector": curve(y, p), "svarah_raw_conf": curve(y, raw),
                            "librispeech_detector": lsc["trained detector (GBM)"].values})
        cur.round(5).to_csv(tab / "svarah_review_curve.csv", index=False)
        fg, ax = plt.subplots(figsize=(6.4, 4.8))
        ax.plot(100 * grid, 100 * cur.librispeech_detector, color=GRID, lw=2, label="LibriSpeech test grid (detector)")
        ax.plot(100 * grid, 100 * cur.svarah_detector, color=SERIES[1], lw=2.4, label="Svarah, Indian accents (detector)")
        ax.plot(100 * grid, 100 * cur.svarah_raw_conf, color=SERIES[0], lw=1.6, label="Svarah (raw confidence)")
        ax.plot([0, 100], [0, 100], color=INK2, ls="--", lw=1.2, label="random flagging")
        ax.axvline(30, color=GRID, lw=1)
        ax.set_xlabel("share of words reviewed, riskiest first (%)")
        ax.set_ylabel("share of word errors caught (%)")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title("Review effort: Indian accents vs LibriSpeech (B960_rev + 4-gram LM)", loc="left", fontsize=10)
        fg.tight_layout()
        fg.savefig(fig / "svarah_review_curve.png", dpi=150)
        plt.close(fg)
        # reliability (calibration) on Svarah
        bins = np.linspace(0, 1, 11)
        k = np.clip(np.digitize(p, bins) - 1, 0, 9)
        rel = pd.DataFrame({"bin": k, "p": p, "y": y}).groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
        fg, ax = plt.subplots(figsize=(4.8, 4.4))
        ax.plot([0, 1], [0, 1], color=GRID, ls="--")
        ax.plot(rel.pred, rel.obs, marker="o", color=SERIES[1])
        ax.set_xlabel("predicted risk")
        ax.set_ylabel("observed error rate")
        ax.set_title(f"Detector calibration on Svarah (ECE {ece(y, p):.3f})", loc="left", fontsize=10)
        fg.tight_layout()
        fg.savefig(fig / "svarah_reliability.png", dpi=150)
        plt.close(fg)
        # ---- 7. replay window (O4): reference word times by CTC forced alignment on clean B960_rev log-probs
        from replay_window import ctc_force_align

        from group9_ds_ai_lab.data.text import DELIM_ID, encode
        z = np.load(WORK / "logits" / "B960_rev__clean.npz")
        LP, OF = z["logp"], z["offsets"]
        idx_of = {u: i for i, u in enumerate(pd.read_parquet(WORK / "meta.parquet").utt_id)}
        times = {}
        for u, t in zip(meta.utt_id, meta.text):
            i = idx_of[u]
            lp = LP[OF[i]:OF[i + 1]].astype(np.float32)
            labels, owner = [], []
            for wi, w in enumerate(t.split()):
                if wi:
                    labels.append(DELIM_ID)
                    owner.append(-1)
                ids = encode(w)
                labels += ids
                owner += [wi] * len(ids)
            if len(labels) * 1 > len(lp) or not labels:
                continue
            fst, lst = ctc_force_align(lp, labels)
            owner = np.array(owner)
            times[u] = [(fst[owner == wi].min() * 0.02, (lst[owner == wi].max() + 1) * 0.02) for wi in range(len(t.split()))]
        hits, tot = 0, 0
        ws["p"] = p
        for (a, n, s), gg in ws[ws.p >= 0.5].groupby(["utt_id", "noise_type", "snr_db"], dropna=False, sort=False):
            if a not in times:
                continue
            ref = refmap[(a, n, -99 if pd.isna(s) else s)]
            full = ws[(ws.utt_id == a) & (ws.noise_type == n) & ((ws.snr_db == s) if not pd.isna(s) else ws.snr_db.isna())]
            alj = jiwer.process_words(ref, " ".join(full.word)).alignments[0]
            pos = {}
            for ch in alj:
                for kk in range(ch.hyp_end_idx - ch.hyp_start_idx):
                    j = ch.hyp_start_idx + kk
                    pos[j] = ch.ref_start_idx + kk if ch.type in ("equal", "substitute") else max(min(ch.ref_start_idx - 1, len(times[a]) - 1), 0)
            fl = full.reset_index(drop=True)
            for j in np.where(fl.p >= 0.5)[0]:
                rs, re_ = times[a][pos[j]]
                tot += 1
                hits += (fl.start.iat[j] - 1.0 <= rs) and (re_ <= fl.end.iat[j] + 1.0) or \
                        (fl.start.iat[j] - 1.0 <= (rs + re_) / 2 <= fl.end.iat[j] + 1.0)
        det_res_replay = {"flags": tot, "hit_rate_%": 100 * hits / max(tot, 1), "aligned_clips": len(times)}
    else:
        det_res_replay = {}
    # ---- 8. figures: WER per model and condition; groups
    fg, ax = plt.subplots(figsize=(11, 4))
    order = [("ref_base960h", "greedy"), ("A_clean", "greedy"), ("B_noisy", "greedy"), ("B460", "greedy"),
             ("B960_rev", "greedy"), ("B960_rev", "4-gram LM"), ("rover3_final", "4-gram LM + ROVER")]
    order = [o for o in order if o in C]
    x = np.arange(len(conds))
    wdt = 0.8 / len(order)
    for k, (m, dec) in enumerate(order):
        v = g[(g.system == m) & (g.decoding == dec)].set_index("condition")["wer_%"][conds]
        ax.bar(x + (k - len(order) / 2 + 0.5) * wdt, v, width=wdt * 0.92, color=SERIES[k % len(SERIES)],
               label=f"{m}{'' if dec == 'greedy' else ' + ' + dec.replace('4-gram LM + ROVER', 'LM + ROVER').replace('4-gram LM', 'LM')}")
    ax.set_xticks(x, conds)
    ax.set_ylabel("WER (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, ncol=4, loc="upper left")
    ax.set_title("Svarah (Indian-accented English): WER by model and condition", loc="left")
    fg.tight_layout()
    fg.savefig(fig / "svarah_wer_by_model.png", dpi=150)
    plt.close(fg)
    if len(gap):
        gg2 = gap[gap.decoding == "greedy"].set_index("system")
        fg, ax = plt.subplots(figsize=(8.5, 3.8))
        xs = np.arange(len(gg2))
        for k, (col, lab, colr) in enumerate((("LS_test_clean", "LibriSpeech test-clean", GRID), ("LS_test_other", "LibriSpeech test-other", INK2),
                                             ("Svarah_clean", "Svarah (clean)", SERIES[1]), ("Svarah_esc50_5dB", "Svarah + ESC-50 5 dB", SERIES[0]))):
            ax.bar(xs + (k - 1.5) * 0.2, gg2[col], width=0.19, color=colr, label=lab)
        ax.set_xticks(xs, gg2.index, fontsize=8)
        ax.set_ylabel("WER (%), greedy")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8)
        ax.set_title("The accent gap: same models, LibriSpeech vs Indian-accented speech", loc="left", fontsize=10)
        fg.tight_layout()
        fg.savefig(fig / "svarah_accent_gap.png", dpi=150)
        plt.close(fg)
    lang = grp[(grp.attribute == "primary_language") & (grp.condition == "clean")].sort_values("wer_%")
    if len(lang):
        fg, ax = plt.subplots(figsize=(7.5, 0.3 * len(lang) + 1.2))
        ax.barh(np.arange(len(lang)), lang["wer_%"], color=SERIES[1])
        for yy, (v, nn) in enumerate(zip(lang["wer_%"], lang.n_utts)):
            ax.text(v + 0.2, yy, f"{v:.1f} %  (n = {nn})", va="center", fontsize=7.5)
        ax.set_yticks(np.arange(len(lang)), lang.group, fontsize=8)
        ax.set_xlabel("WER (%), final system, clean")
        ax.grid(axis="y", visible=False)
        ax.set_title("WER by the speaker's native language", loc="left", fontsize=10)
        fg.tight_layout()
        fg.savefig(fig / "svarah_wer_by_language.png", dpi=150)
        plt.close(fg)
    # ---- 9. examples (best / worst, clean), with audio
    import soundfile as sf
    _, audio, offs = load_prepared()
    full_meta = pd.read_parquet(WORK / "meta.parquet")
    pos_of = {u: i for i, u in enumerate(full_meta.utt_id)}
    d0, c0, _ = C[fk]
    t = d0.assign(err=(c0.S + c0.D + c0.I).values, n=(c0.H + c0.S + c0.D).values)
    t = t[(t.cond == "clean") & t.n.between(6, 25)]
    pick = pd.concat([t[t.err == 0].sample(min(4, (t.err == 0).sum()), random_state=SEED).assign(kind="correct"),
                      t.assign(r=t.err / t.n).nlargest(4, "r").assign(kind="hardest")])
    ex = []
    for r in pick.itertuples():
        i = pos_of[r.utt_id]
        f = OUT / "audio_examples" / f"{r.kind}_{r.utt_id}.wav"
        sf.write(f, audio[offs[i]:offs[i + 1]], SAMPLE_RATE, subtype="PCM_16")
        mr = full_meta.iloc[i]
        ex.append({"kind": r.kind, "utt_id": r.utt_id, "native_language": mr.primary_language, "state": mr.state,
                   "reference": r.text, "final_system": r.hyp, "errors": int(r.err), "file": f"audio_examples/{f.name}"})
    pd.DataFrame(ex).to_csv(tab / "svarah_examples.csv", index=False)
    summary = {"utterances_scored": int(len(meta)), "hours": float(meta.duration.sum() / 3600), "speaker_languages": int(meta.primary_language.nunique()),
               "states": int(meta.state.nunique()), "utts_with_digits": int(meta.has_digit.sum()), "oov": oov,
               "replay_window": det_res_replay, "final_system": list(fk), "top_char_confusions": top_char[:10]}
    json.dump(summary, open(tab / "svarah_summary.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "oov"}, indent=1, default=str))
    print(g.pivot_table(index=["decoding", "system"], columns="condition", values="wer_%", sort=False)[conds].round(2).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="*", default=["prepare", "infer", "lm", "rover", "analyse"],
                    choices=["prepare", "infer", "lm", "rover", "analyse"])
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args()
    for s in a.stages:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
        if s == "prepare":
            prepare()
        elif s == "infer":
            infer(list(MODELS))
        elif s == "lm":
            lm_decode(LM_SYSTEMS, a.workers)
        elif s == "rover":
            rover()
        else:
            analyse()
        print(f"[{time.strftime('%H:%M:%S')}] {s} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
