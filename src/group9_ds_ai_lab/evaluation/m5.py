"""Milestone 5 evaluation: every analysis as a function, shared by scripts/m5_evaluate.py and
notebooks/M5_evaluation.ipynb, so both produce the same tables and figures.

Stages (each writes CSV tables to <out>/tables, PNG figures to <out>/figures and returns its main results):
  core       the 22 Nov analysis (model comparison, S/D/I, ESC-50 classes, word frequency/length, duration,
             speakers, confusions, LM effect, detector ROC/PR/confusion, qualitative) - scripts/m5_analysis.py
             functions, imported and run unchanged, only redirected to <out>
  g1_grid    WER and CER for every system in every condition (incl. the harder -5 dB / reverb grid)
  g2_break   breakdown SNR per model and noise type (rule fixed in CONFIG before computing)
  g3_acc     how accuracy rises and falls: training curves, model-by-model waterfall, accuracy vs SNR
  g4_boot    paired bootstrap 95 % confidence intervals for the main comparisons (resampling utterances)
  g5_review  review-effort curve: share of errors caught vs share of words reviewed, against baselines
  g6_catch   which kinds of errors the flags catch (numbers, out-of-vocabulary, rare words, function words)
  g7_speed   real-time factor on CPU (4 threads) and GPU, model from local disk or Hugging Face
  g8_audio   noisy test examples rebuilt bit-exactly with the seeded mixing (what the model heard)
  g9_listen  the O4 listening sheet (50 replay clips)
  g12_repro  local vs Hugging Face: weight hashes, and a full-grid greedy re-decode compared word for word
  g11_obj    objective-by-objective table (O1-O6), built from the numbers above
No retraining anywhere; only g7_speed and g12_repro use the GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from group9_ds_ai_lab.evaluation import models as M
from group9_ds_ai_lab.utils.paths import EVAL, HF_HOME, MANIFESTS, REPO, RUNS, SAMPLE_RATE

SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import m5_analysis as MA  # noqa: E402  (the 22 Nov analysis; imported, not modified)
from error_detector import features, unigram_table  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402  (m5_analysis already selected the Agg backend)

from group9_ds_ai_lab.utils.plotstyle import GRID, INK, INK2, SERIES, apply_style  # noqa: E402

# Every choice is fixed here, before any result is computed.
CONFIG = {
    "seed": 0,
    "n_boot": 1000,                # bootstrap resamples (utterances resampled, all conditions of an utterance together)
    "breakdown_factor": 2.0,       # breakdown SNR: WER reaches 2 x the model's own clean WER ...
    "breakdown_abs_wer": 20.0,     # ... and, as a second view, the SNR where WER reaches 20 %
    "detector_run": "B960_rev_lm4",
    "threshold": 0.5,              # app default
    "review_budget": 0.30,         # O3: share of words reviewed
    "budgets": [0.05, 0.10, 0.20, 0.30, 0.50],
    "speed_clips": 6, "speed_threads": 4,
    "repro_models": ["B960_rev"],  # full-grid greedy re-decode from Hugging Face (all 7 with recompute="greedy")
    "audio_examples_per_noise": 2,
}
NOISES = MA.NOISES
SNRS = [20, 10, 5, 0]
GREEDY = dict(MA.GREEDY)                                        # name -> greedy predictions
LM4 = {k: f"{v}/test_predictions.parquet" for k, v in MA.LM.items()}  # name -> 4-gram LM predictions
EXT = {k: f"{k}_ext/predictions.parquet" for k in ("A_clean", "B_noisy", "B2_lr1e-4", "B460", "B960_rev", "B960_rev_s2")}
LABEL = {"ref_base960h": "Reference base-960h", "A_clean": "Model A (clean, 100 h)", "B_noisy": "Model B (noise, 100 h)",
         "B2_lr1e-4": "B2 (lr 1e-4)", "B460": "B460 (460 h)", "B960_rev": "B960_rev (960 h + reverb)",
         "B960_rev_s2": "B960_rev_s2 (seed 7)", "rover3_final": "Final: ROVER of 3"}
COLOR = {"ref_base960h": SERIES[2], "A_clean": SERIES[0], "B_noisy": SERIES[1], "B2_lr1e-4": SERIES[4],
         "B460": SERIES[3], "B960_rev": SERIES[6], "B960_rev_s2": SERIES[5], "rover3_final": SERIES[7]}


class Ctx:
    """Output folders and caches for one run."""

    def __init__(self, out, source="local", recompute="none"):
        self.out = Path(out)
        self.fig, self.tab = self.out / "figures", self.out / "tables"
        self.audio = self.out / "audio_examples"
        for p in (self.fig, self.tab, self.audio):
            p.mkdir(parents=True, exist_ok=True)
        self.source, self.recompute = source, recompute
        self.cache, self.res = {}, {}
        self._uni = None
        apply_style()
        MA.FIG, MA.TAB, MA.OUT = self.fig, self.tab, self.out  # redirect the imported 22 Nov analysis

    @property
    def uni(self):
        if self._uni is None:
            self._uni = unigram_table()
        return self._uni

    def save(self, df, name, index=False):
        df.to_csv(self.tab / name, index=index)
        return df

    def savefig(self, fig, name):
        fig.savefig(self.fig / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return self.fig / name


def environment() -> dict:
    """Hardware / software of this run (printed at the top of the notebook and saved with the tables)."""
    import platform

    import torch
    import transformers
    env = {"python": platform.python_version(), "os": platform.platform(), "cpu": platform.processor() or platform.machine(),
           "cpu_count": os.cpu_count(), "torch": torch.__version__, "transformers": transformers.__version__,
           "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for mod in ("jiwer", "sklearn", "pyctcdecode", "kenlm", "numpy", "pandas"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            env[mod] = "n/a"
    try:
        env["git_commit"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                           capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        env["git_commit"] = "n/a"
    return env


# =============================================================================================== core (22 Nov)
def core(ctx: Ctx) -> dict:
    """The 22 Nov analysis, unchanged (scripts/m5_analysis.py functions), written to ctx.out."""
    cache, uni, summary = ctx.cache, ctx.uni, {}
    comp = MA.model_comparison(cache)
    MA.fig_wer_vs_snr(cache)
    MA.error_types(cache)
    esc, grp = MA.esc50_analysis(cache)
    _, _, summary["oov"] = MA.word_level_analysis(cache, uni)
    summary["duration"] = MA.duration_analysis(cache).to_dict("records")
    summary["speakers_clean"] = MA.speaker_analysis(cache).reset_index().to_dict("records")
    _, summary["top_char_confusions"] = MA.confusions(cache)
    summary["lm_effect"] = MA.lm_effect(cache)
    summary["detector_confusion"] = MA.detector(uni).to_dict("records")
    MA.qualitative(cache)
    summary["esc50_hardest"] = esc.head(8)[["category", "wer_B_noisy_lm4", "wer_rover3_final"]].to_dict("records")
    summary["esc50_easiest"] = esc.tail(5)[["category", "wer_B_noisy_lm4", "wer_rover3_final"]].to_dict("records")
    with open(ctx.tab / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    ctx.res["core"] = {"model_comparison": comp, "summary": summary}
    return ctx.res["core"]


# =============================================================================================== G1 WER + CER grid
def _cond_stats(args):
    system, decoding, path = args
    from group9_ds_ai_lab.inference.decode import wer_cer
    d = pd.read_parquet(EVAL / path)
    d["hyp"] = d.hyp.fillna("").astype(str).str.strip()
    d["text"] = d.text.astype(str).str.strip()
    rows = []
    for (n, s), g in d.groupby(["noise_type", "snr_db"], dropna=False, sort=False):
        for split, gg in [("both", g)] + list(g.groupby("ls_split")):
            w, c = wer_cer(gg.text.tolist(), gg.hyp.tolist())
            rows.append({"system": system, "decoding": decoding, "noise_type": n,
                         "snr": "clean" if pd.isna(s) else int(s), "ls_split": split, "n_utts": len(gg),
                         "n_ref_words": int(gg.text.str.split().str.len().sum()), "wer_%": 100 * w, "cer_%": 100 * c})
    return rows


def g1_grid(ctx: Ctx, workers: int = 24) -> pd.DataFrame:
    jobs = [(k, "greedy", v) for k, v in GREEDY.items()] + [(k, "4-gram LM", v) for k, v in LM4.items()] + \
           [("rover3_final", "4-gram LM + ROVER", "rover3_final/test_predictions.parquet")] + \
           [(k, "greedy (harder grid)", v) for k, v in EXT.items()]
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        rows = [r for part in ex.map(_cond_stats, jobs) for r in part]
    t = pd.DataFrame(rows)
    t["snr_sort"] = t.snr.map(lambda s: 99 if s == "clean" else s)
    t = t.sort_values(["decoding", "system", "noise_type", "snr_sort", "ls_split"], ascending=[True, True, True, False, True])
    t = t.drop(columns="snr_sort").reset_index(drop=True)
    ctx.save(t.round({"wer_%": 4, "cer_%": 4}), "g1_wer_cer_all_conditions.csv")  # full precision kept in memory
    # compact pivot: pooled test-clean + test-other, one row per system, columns = conditions
    both = t[(t.ls_split == "both") & ~t.decoding.str.startswith("greedy (harder")]
    piv = {}
    for m in ("wer_%", "cer_%"):
        p = both.assign(cond=np.where(both.noise_type == "clean", "clean", both.noise_type + " " + both.snr.astype(str)))
        p = p.pivot_table(index=["decoding", "system"], columns="cond", values=m, sort=False)
        order = ["clean"] + [f"{n} {s}" for n in NOISES for s in SNRS]
        piv[m] = ctx.save(p[order].round(2), f"g1_{m[:3]}_grid.csv", index=True)
    _fig_cer(ctx, both)
    ctx.res["g1"] = {"long": t, "wer": piv["wer_%"], "cer": piv["cer_%"]}
    return t


def _fig_cer(ctx, both):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    sel = both[both.decoding == "greedy"]
    for ax, n in zip(axes, NOISES):
        for sysn in ("A_clean", "B_noisy", "B460", "B960_rev"):
            g = sel[sel.system == sysn]
            clean = g[g.noise_type == "clean"]["cer_%"].iloc[0]
            q = g[g.noise_type == n].sort_values("snr", key=lambda s: s.astype(int))
            ax.plot([25] + q.snr.astype(int).tolist()[::-1], [clean] + q["cer_%"].tolist()[::-1], marker="o",
                    color=COLOR[sysn], label=LABEL[sysn])
        ax.set_xticks([0, 5, 10, 15, 20, 25], ["0", "5", "10", "15", "20", "clean"])
        ax.invert_xaxis()
        ax.set_title(MA.NOISE_LABEL[n])
        ax.set_xlabel("SNR (dB)")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("CER (%), greedy decoding")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Character error rate vs noise level (no language model)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    ctx.savefig(fig, "g1_cer_vs_snr.png")


# =============================================================================================== G2 breakdown SNR
def _crossing(points, level):
    """points: [(snr, wer)] from high to low SNR (clean first as +inf). First SNR where WER >= level, linearly
    interpolated between the two measured SNRs around the crossing; None if never reached."""
    for (s1, w1), (s2, w2) in zip(points, points[1:]):
        if w2 >= level:
            if w1 >= level or not np.isfinite(s1):
                return float(s2) if not np.isfinite(s1) else float(s1)
            return float(s1 + (level - w1) * (s2 - s1) / (w2 - w1))
    return None


def g2_breakdown(ctx: Ctx) -> pd.DataFrame:
    t = ctx.res["g1"]["long"] if "g1" in ctx.res else g1_grid(ctx)
    t = t[t.ls_split == "both"]
    rows = []
    for (dec, sysn), g in t[~t.decoding.str.startswith("greedy (harder")].groupby(["decoding", "system"], sort=False):
        clean = g[g.noise_type == "clean"]["wer_%"].iloc[0]
        ext = t[(t.system == sysn) & t.decoding.str.startswith("greedy (harder")] if dec == "greedy" else t.iloc[:0]
        for n in NOISES:
            q = g[g.noise_type == n]
            pts = [(float("inf"), clean)] + [(float(s), float(q[q.snr == s]["wer_%"].iloc[0])) for s in SNRS]
            e = ext[(ext.noise_type == n) & (ext.snr == -5)]
            if len(e):
                pts.append((-5.0, float(e["wer_%"].iloc[0])))
            lo = pts[-1][0]
            rel = _crossing(pts, CONFIG["breakdown_factor"] * clean)
            ab = _crossing(pts, CONFIG["breakdown_abs_wer"])
            rows.append({"decoding": dec, "system": sysn, "noise_type": n, "clean_wer_%": round(clean, 2),
                         "lowest_snr_tested": int(lo), "wer_at_lowest_%": round(pts[-1][1], 2),
                         f"breakdown_snr_{CONFIG['breakdown_factor']:g}x_clean": None if rel is None else round(rel, 1),
                         f"snr_wer_reaches_{CONFIG['breakdown_abs_wer']:g}%": None if ab is None else round(ab, 1)})
    b = ctx.save(pd.DataFrame(rows), "g2_breakdown_snr.csv")
    ctx.res["g2"] = b
    return b


# =============================================================================================== G3 accuracy up / down
def g3_accuracy(ctx: Ctx) -> dict:
    out = {}
    # (a) during training: dev WER at every evaluation step
    logs = []
    for name in ("A_clean", "B_noisy", "B2_lr1e-4", "B460", "B960_rev", "B960_rev_s2"):
        f = RUNS / name / "eval_log.csv"
        if f.exists():
            e = pd.read_csv(f)[["step", "wer_clean", "wer_noisy", "cer_clean", "cer_noisy"]].assign(model=name)
            logs.append(e)
    logs = pd.concat(logs, ignore_index=True)
    for c in ("wer_clean", "wer_noisy", "cer_clean", "cer_noisy"):
        logs[c] = (100 * logs[c]).round(3)
    out["training"] = ctx.save(logs, "g3_dev_wer_during_training.csv")
    best = logs.assign(avg=(logs.wer_clean + logs.wer_noisy) / 2).sort_values("avg").groupby("model").head(1)
    out["best_step"] = ctx.save(best.sort_values("model").drop(columns="avg"), "g3_best_dev_step.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, col, title in ((axes[0], "wer_clean", "dev, clean"), (axes[1], "wer_noisy", "dev, noisy (seen noise types)")):
        for name, g in logs.groupby("model", sort=False):
            ax.plot(g.step / 1000, g[col], marker=".", color=COLOR[name], label=LABEL[name],
                    ls="--" if name == "B960_rev_s2" else "-")
        ax.set_xlabel("training step (thousands)")
        ax.set_ylabel("WER (%), greedy")
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    fig.suptitle("How accuracy improves during fine-tuning (dev set, every 2,000 steps)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    ctx.savefig(fig, "g3_training_curves.png")

    # (b) model by model: what each change bought, on clean and on held-out noise
    t = ctx.res["g1"]["long"] if "g1" in ctx.res else g1_grid(ctx)
    both = t[t.ls_split == "both"]

    def pooled(sysn, dec, noises, snrs):
        g = both[(both.system == sysn) & (both.decoding == dec) & both.noise_type.isin(noises)]
        if snrs is not None:
            g = g[g.snr.isin(snrs)]
        return float((g["wer_%"] * g.n_ref_words).sum() / g.n_ref_words.sum())

    steps = [("Model A (clean, 100 h)", "A_clean", "greedy", "start"),
             ("+ noise-aware training (Model B)", "B_noisy", "greedy", "noise training"),
             ("+ 3.6x data (B460)", "B460", "greedy", "more data"),
             ("+ 2x data, reverb, speed, telephone (B960_rev)", "B960_rev", "greedy", "more data + augmentation"),
             ("+ 4-gram LM", "B960_rev", "4-gram LM", "language model"),
             ("+ ROVER of 3 models (final)", "rover3_final", "4-gram LM + ROVER", "system combination")]
    rows = []
    for lab, sysn, dec, change in steps:
        rows.append({"step": lab, "change": change, "system": sysn, "decoding": dec,
                     "wer_clean_%": pooled(sysn, dec, ["clean"], None),
                     "wer_esc50_0-5dB_%": pooled(sysn, dec, ["esc50"], [0, 5]),
                     "wer_all_noisy_%": pooled(sysn, dec, NOISES, None)})
    w = pd.DataFrame(rows)
    for c in ("wer_clean_%", "wer_esc50_0-5dB_%", "wer_all_noisy_%"):
        w[c.replace("wer", "acc")] = 100 - w[c]
        w[c.replace("wer_", "rel_change_vs_prev_")] = (100 * w[c].pct_change()).round(1)
    out["waterfall"] = ctx.save(w.round(2), "g3_accuracy_model_by_model.csv")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(w))
    for k, (col, lab, colr) in enumerate((("acc_clean_%", "clean", INK2), ("acc_esc50_0-5dB_%", "held-out ESC-50, 0-5 dB", SERIES[1]),
                                          ("acc_all_noisy_%", "all 16 noisy conditions", SERIES[0]))):
        ax.bar(x + (k - 1) * 0.26, w[col], width=0.24, color=colr, label=lab)
        for xi, v in zip(x, w[col]):
            ax.text(xi + (k - 1) * 0.26, v + 0.4, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [s.replace(" (", "\n(") for s in w.step], fontsize=7.5)
    ax.set_ylim(55, 100)
    ax.set_ylabel("word accuracy (100 - WER), %")
    ax.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="x", visible=False)
    ax.set_title("How word accuracy improves, change by change (test set)", loc="left")
    fig.tight_layout()
    ctx.savefig(fig, "g3_accuracy_model_by_model.png")

    # (c) how accuracy falls with noise: word accuracy per SNR, and the drop from clean
    rows = []
    for sysn, dec in (("A_clean", "greedy"), ("B_noisy", "greedy"), ("B960_rev", "greedy"), ("rover3_final", "4-gram LM + ROVER")):
        clean = pooled(sysn, dec, ["clean"], None)
        for n in NOISES:
            for s in SNRS:
                wv = pooled(sysn, dec, [n], [s])
                rows.append({"system": sysn, "decoding": dec, "noise_type": n, "snr": s, "accuracy_%": round(100 - wv, 2),
                             "drop_from_clean_pts": round(wv - clean, 2)})
    acc = ctx.save(pd.DataFrame(rows), "g3_accuracy_vs_snr.csv")
    out["acc_vs_snr"] = acc
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    for ax, n in zip(axes, NOISES):
        for sysn, g in acc[acc.noise_type == n].groupby("system", sort=False):
            clean = 100 - pooled(sysn, g.decoding.iloc[0], ["clean"], None)
            ax.plot([25] + SNRS, [clean] + g.sort_values("snr", ascending=False)["accuracy_%"].tolist(), marker="o",
                    color=COLOR[sysn], label=LABEL[sysn] + (" + LM" if sysn == "rover3_final" else ", greedy"))
        ax.set_xticks([0, 5, 10, 15, 20, 25], ["0", "5", "10", "15", "20", "clean"])
        ax.invert_xaxis()
        ax.set_title(MA.NOISE_LABEL[n])
        ax.set_xlabel("SNR (dB)")
    axes[0].set_ylabel("word accuracy (100 - WER), %")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("How accuracy falls as noise gets louder", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    ctx.savefig(fig, "g3_accuracy_vs_snr.png")

    # (d) hyperparameter sweep (M4, 8,000 steps each)
    sw = pd.read_csv(REPO / "milestones" / "M4_training" / "tables" / "hparam_sweep.csv")
    sw = sw.assign(dev_wer_avg_pct=(100 * sw.dev_wer_avg).round(2)).sort_values("dev_wer_avg_pct")
    out["sweep"] = ctx.save(sw[["run", "change", "dev_wer_avg_pct", "delta_avg_vs_base_pts"]], "g3_hparam_sweep.csv")
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(sw))[::-1]
    ax.barh(y, sw.dev_wer_avg_pct, color=[SERIES[1] if r == "hp_base" else SERIES[0] for r in sw.run], height=0.6)
    for yi, v in zip(y, sw.dev_wer_avg_pct):
        ax.text(v + 0.2, yi, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, sw.change.str.slice(0, 48), fontsize=7.5)
    ax.set_xlabel("dev WER (%), mean of clean and noisy, after 8,000 steps")
    ax.grid(axis="y", visible=False)
    ax.set_title("Hyperparameter sweep (orange = base recipe)", loc="left")
    fig.tight_layout()
    ctx.savefig(fig, "g3_hparam_sweep.png")
    ctx.res["g3"] = out
    return out


# =============================================================================================== G4 bootstrap CIs
SUBSETS = {"clean": (["clean"], None), "all noisy": (NOISES, None), "ESC-50 0-5 dB (held out)": (["esc50"], [0.0, 5.0]),
           "brown 0-5 dB (held out)": (["brown"], [0.0, 5.0]), "white 0-5 dB": (["white"], [0.0, 5.0])}
COMPARISONS = [  # (label, system X, system Y): effect = WER(Y) - WER(X)
    ("noise training: A -> B (greedy)", ("greedy", "A_clean"), ("greedy", "B_noisy")),
    ("noise training: A -> B (4-gram LM)", ("sys", "A_clean_lm4"), ("sys", "B_noisy_lm4")),
    ("learning rate: B (5e-5) -> B2 (1e-4), greedy", ("greedy", "B_noisy"), ("greedy", "B2_lr1e-4")),
    ("data: B -> B460 (LM)", ("sys", "B_noisy_lm4"), ("sys", "B460_lm4")),
    ("data + augmentation: B460 -> B960_rev (LM)", ("sys", "B460_lm4"), ("sys", "B960_rev_lm4")),
    ("language model: B960_rev greedy -> 4-gram LM", ("greedy", "B960_rev"), ("sys", "B960_rev_lm4")),
    ("ROVER: B960_rev (LM) -> ROVER of 3", ("sys", "B960_rev_lm4"), ("sys", "rover3_final")),
    ("seed: B960_rev -> B960_rev_s2 (LM)", ("sys", "B960_rev_lm4"), ("sys", "B960_rev_s2_lm4")),
]


def _errors(ctx, key):
    if key[0] == "greedy":
        if key not in ctx.cache:
            d = MA.load(GREEDY[key[1]])
            c, _ = MA.align(d)
            ctx.cache[key] = (d, c)
        return ctx.cache[key]
    return MA.sys_cache(ctx.cache, key[1])


def g4_bootstrap(ctx: Ctx) -> pd.DataFrame:
    rng = np.random.default_rng(CONFIG["seed"])
    base = None
    rows = []
    for lab, kx, ky in COMPARISONS:
        dx, cx = _errors(ctx, kx)
        dy, cy = _errors(ctx, ky)
        assert (dx.audio_id.values == dy.audio_id.values).all() and (dx.noise_type.values == dy.noise_type.values).all()
        utt = pd.factorize(dx.audio_id)[0]
        n_utt = utt.max() + 1
        if base is None or base[0] != n_utt:  # same resampled utterances for every comparison
            base = (n_utt, rng.multinomial(n_utt, np.full(n_utt, 1 / n_utt), size=CONFIG["n_boot"]).astype(np.float32))
        W = base[1]
        for sub, (noises, snrs) in SUBSETS.items():
            m = dx.noise_type.isin(noises).values & (dx.snr_db.isin(snrs).values if snrs else True)
            u = utt[m]
            ex = np.bincount(u, (cx.S + cx.D + cx.I).values[m], n_utt)
            ey = np.bincount(u, (cy.S + cy.D + cy.I).values[m], n_utt)
            nn = np.bincount(u, (cx.H + cx.S + cx.D).values[m], n_utt)
            wx, wy = ex.sum() / nn.sum(), ey.sum() / nn.sum()
            bx, by = (W @ ex) / (W @ nn), (W @ ey) / (W @ nn)
            diff, rel = 100 * (by - bx), 100 * (by - bx) / bx
            rows.append({"comparison": lab, "subset": sub, "wer_X_%": 100 * wx, "wer_Y_%": 100 * wy,
                         "diff_pts": 100 * (wy - wx), "diff_ci_low": np.percentile(diff, 2.5), "diff_ci_high": np.percentile(diff, 97.5),
                         "rel_change_%": 100 * (wy - wx) / wx, "rel_ci_low": np.percentile(rel, 2.5), "rel_ci_high": np.percentile(rel, 97.5),
                         "share_boot_Y_better": float((by < bx).mean())})
    b = ctx.save(pd.DataFrame(rows).round(3), "g4_bootstrap_ci.csv")
    ctx.res["g4"] = b
    return b


# =============================================================================================== G5 review effort
def _detector_words(ctx):
    if "det_words" not in ctx.cache:
        import joblib
        src = EVAL / CONFIG["detector_run"]
        w = features(pd.read_parquet(src / "test_words.parquet"), ctx.uni)
        det = joblib.load(src / "error_detector" / "model.joblib")
        w["p_gbm"] = det["model"].predict_proba(w[det["features"]].to_numpy(np.float64))[:, 1]
        w["p_raw"] = 1 - w.conf_mean
        ctx.cache["det_words"] = w
    return ctx.cache["det_words"]


def _recall_curve(y, score, grid):
    order = np.argsort(-score, kind="stable")
    caught = np.concatenate([[0], np.cumsum(y[order])]) / y.sum()
    idx = np.round(grid * len(y)).astype(int)
    return caught[idx]


def g5_review(ctx: Ctx) -> dict:
    from sklearn.metrics import f1_score, precision_recall_curve
    w = _detector_words(ctx)
    y = w.is_error.to_numpy().astype(int)
    rng = np.random.default_rng(CONFIG["seed"])
    rand = rng.random(len(y))
    grid = np.linspace(0, 1, 201)
    scores = {"trained detector (GBM)": w.p_gbm.to_numpy(), "raw CTC confidence": w.p_raw.to_numpy(),
              "random flagging": rand, "oracle (perfect ranking)": y + 1e-9 * rand}
    curves = pd.DataFrame({"share_of_words_reviewed": grid, **{k: _recall_curve(y, s, grid) for k, s in scores.items()}})
    ctx.save(curves.round(5), "g5_review_effort_curve.csv")
    rows = []
    for sub, m in (("all conditions", np.ones(len(y), bool)), ("clean", (w.noise_type == "clean").values),
                   ("noisy", (w.noise_type != "clean").values)):
        for k, s in scores.items():
            r = {"subset": sub, "method": k, "error_rate_%": 100 * y[m].mean()}
            for b in CONFIG["budgets"]:
                r[f"errors_caught_reviewing_{int(100 * b)}%_%"] = 100 * float(_recall_curve(y[m], s[m], np.array([b]))[0])
            rows.append(r)
    tab = ctx.save(pd.DataFrame(rows).round(2), "g5_review_effort_budgets.csv")
    f1 = []
    for k in ("trained detector (GBM)", "raw CTC confidence"):
        s = scores[k]
        pr, rc, th = precision_recall_curve(y, s)
        f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
        i = int(np.nanargmax(f[:-1]))
        f1.append({"method": k, "f1_at_0.5": f1_score(y, s >= 0.5), "best_f1": float(f[i]), "threshold_best_f1": float(th[i]),
                   "precision_at_best": float(pr[i]), "recall_at_best": float(rc[i])})
    f1 = ctx.save(pd.DataFrame(f1).round(4), "g5_f1.csv")
    fig, ax = plt.subplots(figsize=(6.4, 5))
    sty = {"trained detector (GBM)": (SERIES[1], "-", 2.4), "raw CTC confidence": (SERIES[0], "-", 1.8),
           "random flagging": (INK2, "--", 1.4), "oracle (perfect ranking)": (SERIES[2], ":", 1.6)}
    for k in scores:
        c, ls, lw = sty[k]
        ax.plot(100 * grid, 100 * curves[k], color=c, ls=ls, lw=lw, label=k)
    b = CONFIG["review_budget"]
    v = 100 * float(curves.loc[np.isclose(curves.share_of_words_reviewed, b), "trained detector (GBM)"].iloc[0])
    ax.axvline(100 * b, color=GRID, lw=1)
    ax.axhline(70, color=GRID, lw=1)
    ax.annotate(f"review {100 * b:.0f} % -> catch {v:.1f} %\n(target: 70 %)", (100 * b, v), (100 * b + 8, v - 22),
                arrowprops={"arrowstyle": "->", "color": INK2}, fontsize=8.5, color=INK)
    ax.set_xlabel("share of words reviewed, riskiest first (%)")
    ax.set_ylabel("share of word errors caught (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Review effort vs errors caught (B960_rev + 4-gram LM, all test words)", loc="left", fontsize=10)
    fig.tight_layout()
    ctx.savefig(fig, "g5_review_effort_curve.png")
    ctx.res["g5"] = {"budgets": tab, "f1": f1}
    return ctx.res["g5"]


# =============================================================================================== G6 what the flags catch
NUMBER_WORDS = set("""ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE TEN ELEVEN TWELVE THIRTEEN FOURTEEN FIFTEEN SIXTEEN
SEVENTEEN EIGHTEEN NINETEEN TWENTY THIRTY FORTY FIFTY SIXTY SEVENTY EIGHTY NINETY HUNDRED THOUSAND MILLION BILLION FIRST
SECOND THIRD FOURTH FIFTH SIXTH SEVENTH EIGHTH NINTH TENTH TWELFTH TWENTIETH HUNDREDTH THOUSANDTH HALF DOZEN""".split())
FUNCTION_WORDS = set("""A AN THE AND OR BUT IF OF TO IN ON AT BY FOR WITH FROM AS INTO THAN THAT THIS THESE THOSE IT ITS
HE HIM HIS SHE HER HERS THEY THEM THEIR WE US OUR YOU YOUR I ME MY IS ARE WAS WERE BE BEEN BEING AM HAVE HAS HAD DO DOES DID
NOT NO SO THEN THERE HERE WHO WHOM WHICH WHAT WHEN WHERE WHY HOW ALL ANY SOME SUCH UPON UP OUT OVER UNDER ABOUT WOULD
SHOULD COULD WILL SHALL MAY MIGHT MUST CAN""".split())


def word_category(word, uni):
    if word in NUMBER_WORDS:
        return "number word"
    lp = uni.get(word)
    if lp is None:
        return "not in LM vocabulary (mostly names)"
    if word in FUNCTION_WORDS:
        return "function word"
    if lp < -5.0:
        return "rare word (log10 p < -5; incl. many names)"
    return "other content word"


def g6_catch(ctx: Ctx) -> pd.DataFrame:
    """For every reference word the system got wrong: was it flagged? Substitutions: the hypothesis word in its
    place is flagged. Deletions: one of the two neighbouring hypothesis words is flagged (its replay covers the gap)."""
    w = _detector_words(ctx)
    run = CONFIG["detector_run"]
    d, _ = MA.sys_cache(ctx.cache, run)
    al = ctx.cache[("al", run)]
    thr30 = float(np.quantile(w.p_gbm, 1 - CONFIG["review_budget"]))
    key = list(zip(w.audio_id, w.noise_type, w.snr_db.fillna(-99)))
    starts = {}
    for i, k in enumerate(key):
        starts.setdefault(k, i)
    p = w.p_gbm.to_numpy()
    uni = ctx.uni
    rows = []
    for i, (ref, chunks) in enumerate(zip(al.references, al.alignments)):
        hyp = d.hyp.iat[i]
        if not hyp:
            continue
        s0 = starts.get((d.audio_id.iat[i], d.noise_type.iat[i], -99 if pd.isna(d.snr_db.iat[i]) else d.snr_db.iat[i]))
        nh = len(hyp.split())
        if s0 is None:
            continue
        ph = p[s0:s0 + nh]
        noisy = d.noise_type.iat[i] != "clean"
        for ch in chunks:
            if ch.type == "substitute":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    rows.append((ref[ch.ref_start_idx + k], "substitution", ph[ch.hyp_start_idx + k], noisy))
            elif ch.type == "delete":
                nb = [ph[j] for j in (ch.hyp_start_idx - 1, ch.hyp_start_idx) if 0 <= j < nh]
                pm = max(nb) if nb else 0.0
                for k in range(ch.ref_start_idx, ch.ref_end_idx):
                    rows.append((ref[k], "deletion", pm, noisy))
    e = pd.DataFrame(rows, columns=["word", "error_type", "p", "noisy"])
    e["category"] = [word_category(x, uni) for x in e.word]
    e["flag_0.5"] = e.p >= CONFIG["threshold"]
    e["flag_30%"] = e.p >= thr30
    g = e.groupby(["error_type", "category"]).agg(n_errors=("p", "size"), caught_at_0_5=("flag_0.5", "mean"),
                                                   caught_in_riskiest_30=("flag_30%", "mean")).reset_index()
    tot = e.groupby("error_type").agg(n_errors=("p", "size"), caught_at_0_5=("flag_0.5", "mean"),
                                      caught_in_riskiest_30=("flag_30%", "mean")).reset_index().assign(category="ALL")
    g = pd.concat([g, tot], ignore_index=True)
    g["share_of_errors_%"] = 100 * g.n_errors / g.error_type.map(tot.set_index("error_type").n_errors)
    for c in ("caught_at_0_5", "caught_in_riskiest_30"):
        g[c] = 100 * g[c]
    g = ctx.save(g.round(2), "g6_flags_by_error_category.csv")
    sub = g[(g.error_type == "substitution") & (g.category != "ALL")].sort_values("caught_in_riskiest_30")
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    yy = np.arange(len(sub))
    ax.barh(yy - 0.18, sub.caught_in_riskiest_30, height=0.34, color=SERIES[1], label="riskiest 30 % of words")
    ax.barh(yy + 0.18, sub.caught_at_0_5, height=0.34, color=SERIES[0], label="threshold 0.5 (app default)")
    for yi, a, b2, n in zip(yy, sub.caught_in_riskiest_30, sub.caught_at_0_5, sub.n_errors):
        ax.text(a + 1, yi - 0.18, f"{a:.0f} %", va="center", fontsize=7.5)
        ax.text(b2 + 1, yi + 0.18, f"{b2:.0f} %  (n = {n:,})", va="center", fontsize=7.5)
    ax.set_yticks(yy, sub.category, fontsize=8)
    ax.set_xlim(0, 110)
    ax.set_xlabel("share of substitution errors that are flagged (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.4, -0.22))
    ax.set_title("Which wrong words get flagged (B960_rev + 4-gram LM, test grid)", loc="left", fontsize=10)
    fig.tight_layout()
    ctx.savefig(fig, "g6_flags_by_error_category.png")
    ctx.res["g6"] = g
    return g


# =============================================================================================== G7 speed
def g7_speed(ctx: Ctx, devices=("cpu", "cuda")) -> pd.DataFrame:
    import jiwer
    import torch
    from benchmark_assistant import long_clips

    from group9_ds_ai_lab.assistant.engine import Assistant
    from group9_ds_ai_lab.data.noise import apply_condition, stable_seed
    clips = long_clips(CONFIG["speed_clips"])
    rows = []
    for dev in devices:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        if ctx.source == "hf":
            M.allow_hub_download()
        t0 = time.time()
        asst = Assistant(model=M.model_source("B960_rev", ctx.source), lm="3gram", threads=CONFIG["speed_threads"], device=dev)
        load = time.time() - t0
        asst.transcribe(clips[0][1][: 5 * SAMPLE_RATE])  # warm-up
        for cid, x, ref, _ in clips:
            for cond, snr in (("clean", None), ("esc50", 5.0), ("white", 5.0)):
                y = x if cond == "clean" else apply_condition(x, cond, snr, stable_seed("bench", cid, cond, snr))[0]
                if dev == "cuda":
                    torch.cuda.synchronize()
                tr = asst.transcribe(y)
                rows.append({"device": "GPU (" + torch.cuda.get_device_name(0) + ")" if dev == "cuda" else f"CPU, {CONFIG['speed_threads']} threads",
                             "model_source": ctx.source, "clip": cid, "condition": cond if cond == "clean" else f"{cond} 5 dB",
                             "duration_s": tr.duration_s, "t_acoustic_s": tr.timings_s.get("acoustic_model"),
                             "t_decoding_s": tr.timings_s.get("decoding"), "t_detector_s": tr.timings_s.get("error_detector"),
                             "rtf": tr.rtf, "wer_%": 100 * jiwer.wer(ref, tr.text), "load_s": load})
        del asst
        if dev == "cuda":
            torch.cuda.empty_cache()
    t = ctx.save(pd.DataFrame(rows).round(4), "g7_speed_runs.csv")
    s = t.groupby(["device", "condition"]).agg(rtf_mean=("rtf", "mean"), rtf_max=("rtf", "max"), seconds_per_2_min=("rtf", lambda r: 120 * r.mean()),
                                                 acoustic_share=("t_acoustic_s", "sum"), wer_mean_pct=("wer_%", "mean")).reset_index()
    tot = t.groupby("device")[["t_acoustic_s", "t_decoding_s", "t_detector_s"]].sum()
    s["acoustic_share"] = s.device.map(tot.t_acoustic_s / tot.sum(axis=1))
    s = ctx.save(s.round(4), "g7_speed_summary.csv")
    ctx.res["g7"] = s
    return s


# =============================================================================================== G8 audio examples
def g8_audio(ctx: Ctx) -> pd.DataFrame:
    """Rebuild noisy test utterances exactly as the model heard them (same seeded mixing as the test grid)."""
    import soundfile as sf

    from group9_ds_ai_lab.data.audio import ArrowAudioReader
    from group9_ds_ai_lab.data.noise import apply_condition
    q = pd.read_csv(ctx.tab / "qualitative_examples.csv") if (ctx.tab / "qualitative_examples.csv").exists() else MA.qualitative(ctx.cache)
    man = pd.read_csv(MANIFESTS / "test_conditions.csv")
    reader = ArrowAudioReader()
    rows = []
    for (nt, kind), g in q.groupby(["noise_type", "kind"], sort=False):
        for r in g.head(1 if kind != "largest improvement over Model B" else CONFIG["audio_examples_per_noise"]).itertuples():
            m = man[(man.audio_id == r.audio_id) & (man.noise_type == nt)]
            m = m[m.snr_db.isna()] if pd.isna(r.snr_db) else m[m.snr_db == r.snr_db]
            mr = m.iloc[0]
            x = reader.load(mr.shard, int(mr.row))
            y, info = apply_condition(x, nt, mr.snr_db, int(mr.cond_seed), part=mr.noise_part)
            peak = max(np.abs(y).max(), 1e-6)
            name = f"{nt}_{'clean' if pd.isna(r.snr_db) else str(int(r.snr_db)) + 'dB'}_{r.audio_id}.wav"
            sf.write(ctx.audio / name, (0.95 * y / peak if peak > 1 else y).astype(np.float32), SAMPLE_RATE)
            rows.append({"file": f"audio_examples/{name}", "kind": kind, "noise_type": nt, "snr_db": r.snr_db,
                         "audio_id": r.audio_id, "noise_clip": Path(info["noise_file"]).name if info["noise_file"] else "",
                         "reference": r.text, "model_B": r.hyp_B, "final_system": r.hyp_final,
                         "errors_B": r.err_B, "errors_final": r.err_final})
    t = ctx.save(pd.DataFrame(rows), "g8_audio_examples.csv")
    ctx.res["g8"] = t
    return t


# =============================================================================================== G9 listening sheet
def g9_listen(ctx: Ctx) -> pd.DataFrame:
    src = REPO / "milestones" / "M5_evaluation" / "replay_check"
    t = pd.read_csv(src / "sample_flags.csv")
    t["clip_path"] = [str((src / "clips" / c).relative_to(REPO)) for c in t["clip"]]
    s = json.load(open(src / "summary.json"))
    ctx.res["g9"] = {"sheet": t, "summary": s}
    return t


# =============================================================================================== G12 reproducibility
def g12_repro(ctx: Ctx, decode=True, models=None) -> dict:
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    M.allow_hub_download()
    rows = []
    for name in [m for m in M.MODELS if m != "ref_base960h"]:
        loc = M.local_path(name)
        hub = M.hub_weights_file(name)
        rows.append({"model": name, "hf_id": M.MODELS[name][0],
                     "local_tensors_sha256": M.tensors_sha256(loc / "model.safetensors") if loc else None,
                     "hf_tensors_sha256": M.tensors_sha256(hub)})
    h = pd.DataFrame(rows)
    h["identical_weights"] = h.local_tensors_sha256 == h.hf_tensors_sha256
    h = ctx.save(h, "g12_weights_local_vs_hf.csv")
    dec = []
    names = models or (list(M.MODELS) if ctx.recompute == "greedy" else CONFIG["repro_models"])
    for name in (names if decode else []):
        tag = f"repro_hf_{name}"
        out = EVAL / tag
        if not (out / "predictions.parquet").exists():
            env = {**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", "HF_HOME": str(HF_HOME)}
            subprocess.run([sys.executable, str(SCRIPTS / "evaluate.py"), "--model", M.MODELS[name][0], "--name", tag],
                           check=True, env=env, cwd=REPO)
        a = pd.read_parquet(EVAL / GREEDY[name]).set_index(["audio_id", "noise_type", "snr_db"])
        b = pd.read_parquet(out / "predictions.parquet").set_index(["audio_id", "noise_type", "snr_db"]).loc[a.index]
        same = (a.hyp.fillna("") == b.hyp.fillna(""))
        sa = pd.read_csv(EVAL / GREEDY[name].replace("predictions.parquet", "summary.csv"))
        sb = pd.read_csv(out / "summary.csv")
        mx = float((sa.wer.to_numpy() - sb.wer.to_numpy()).__abs__().max()) if len(sa) == len(sb) else None
        dec.append({"model": name, "hf_id": M.MODELS[name][0], "utterances": int(len(a)), "identical_hypotheses": int(same.sum()),
                    "identical_%": round(100 * same.mean(), 4), "max_abs_wer_diff_per_condition": mx})
    d = ctx.save(pd.DataFrame(dec), "g12_hf_redecode_vs_saved.csv") if dec else pd.DataFrame()
    ctx.res["g12"] = {"weights": h, "decode": d}
    return ctx.res["g12"]


# =============================================================================================== G11 objectives
def _db(v):
    return "not reached" if v is None or pd.isna(v) else f"{v:g} dB"


def g11_objectives(ctx: Ctx) -> pd.DataFrame:
    tab = ctx.tab
    b = pd.read_csv(tab / "g4_bootstrap_ci.csv")
    o2 = b[(b.comparison == "noise training: A -> B (greedy)")].set_index("subset")
    det = json.load(open(EVAL / CONFIG["detector_run"] / "error_detector" / "metrics.json"))
    gbm = det["gbm"]
    rep = json.load(open(REPO / "milestones" / "M5_evaluation" / "replay_check" / "summary.json"))
    sp = pd.read_csv(tab / "g7_speed_summary.csv") if (tab / "g7_speed_summary.csv").exists() else None
    br = pd.read_csv(tab / "g2_breakdown_snr.csv")
    rp = pd.read_csv(tab / "g12_weights_local_vs_hf.csv") if (tab / "g12_weights_local_vs_hf.csv").exists() else None
    rd = pd.read_csv(tab / "g12_hf_redecode_vs_saved.csv") if (tab / "g12_hf_redecode_vs_saved.csv").exists() else None
    esc = o2.loc["ESC-50 0-5 dB (held out)"]
    cl = o2.loc["clean"]
    bcol = f"breakdown_snr_{CONFIG['breakdown_factor']:g}x_clean"
    fin = br[(br.system == "B960_rev") & (br.decoding == "greedy")].set_index("noise_type")[bcol]
    a_ = br[(br.system == "A_clean") & (br.decoding == "greedy")].set_index("noise_type")[bcol]
    cpu = sp[sp.device.str.startswith("CPU")].rtf_max.max() if sp is not None and (sp.device.str.startswith("CPU")).any() else None
    rows = [
        ("O1", "Characterise ASR degradation under noise", "complete WER/CER grid + WER-vs-SNR plot; breakdown SNR per model",
         f"WER+CER for 8 systems x 17 conditions (+ -5 dB, reverb); breakdown SNR (WER = 2 x clean), Model A vs B960_rev: "
         + ", ".join(f"{n} {_db(a_.get(n))} vs {_db(fin.get(n))}" for n in NOISES), "g1, g2, Fig. 1, report 3.7", "met"),
        ("O2", "Benefit of noise-aware training", ">= 15 % relative WER cut at 0-5 dB (held-out noise); clean <= +2 pts",
         f"ESC-50 0-5 dB: {esc['rel_change_%']:.1f} % "
         f"(95 % CI {esc.rel_ci_low:.1f} to {esc.rel_ci_high:.1f}); clean {cl.diff_pts:+.2f} pts (CI {cl.diff_ci_low:+.2f} to {cl.diff_ci_high:+.2f})",
         "g4, report 3.5", "met" if esc["rel_ci_high"] <= -15 and cl.diff_ci_high <= 2 else "check"),
        ("O3", "Detect error-prone words", "AUROC >= 0.75; >= 70 % of errors in riskiest 30 %; ECE <= 0.10",
         f"AUROC {gbm.get('auroc', float('nan')):.3f}, recall@30 % {100 * gbm.get('recall_at_30pct', float('nan')):.1f} %, ECE {gbm.get('ece', float('nan')):.3f}",
         "g5, Figs. 2-6", "met"),
        ("O4", "Link flags to the right audio", ">= 90 % of 50 sampled flags inside +-1 s",
         f"{100 * rep['by_threshold']['0.5']['hit_rate']:.1f} % of {rep['by_threshold']['0.5']['n_flags']:,} flags (automatic); "
         "50-clip listening sheet in g9", "g9, report 3.4", "met (automatic); listening check by the team"),
        ("O5", "Working MVP", "upload, chunking, transcription, flags, replay, edit, export; RTF <= 1.0 on CPU (<= 2 min)",
         f"worst CPU RTF {cpu:.3f}" if cpu is not None else "see g7", "g7, app", "met" if cpu is not None and cpu <= 1 else "see g7"),
        ("O6", "Reproducibility", "a teammate reproduces the results from the README",
         ("weights identical local vs HF: " + (f"{int(rp.identical_weights.sum())}/{len(rp)}" if rp is not None else "n/a")
          + ("; full-grid re-decode from HF: " + ", ".join(f"{r.model} {r['identical_%']:.2f} % identical" for _, r in rd.iterrows())
             if rd is not None and len(rd) else "")),
         "g12, this notebook", "evidence in place; teammate re-run to confirm"),
    ]
    t = ctx.save(pd.DataFrame(rows, columns=["objective", "name", "target", "result", "evidence", "status"]), "g11_objectives.csv")
    ctx.res["g11"] = t
    return t


# =============================================================================================== driver
STAGES = ["core", "g1_grid", "g2_break", "g3_acc", "g4_boot", "g5_review", "g6_catch", "g8_audio", "g9_listen",
          "g7_speed", "g12_repro", "g11_obj"]
FUNCS = {"core": core, "g1_grid": g1_grid, "g2_break": g2_breakdown, "g3_acc": g3_accuracy, "g4_boot": g4_bootstrap,
         "g5_review": g5_review, "g6_catch": g6_catch, "g7_speed": g7_speed, "g8_audio": g8_audio, "g9_listen": g9_listen,
         "g12_repro": g12_repro, "g11_obj": g11_objectives}


def run_all(out, stages=None, source="local", recompute="none") -> Ctx:
    ctx = Ctx(out, source, recompute)
    env = environment()
    json.dump({"environment": env, "config": CONFIG, "source": source, "recompute": recompute},
              open(ctx.tab / "run_info.json", "w"), indent=2, default=str)
    for s in stages or STAGES:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] {s} ...", flush=True)
        FUNCS[s](ctx)
        print(f"[{time.strftime('%H:%M:%S')}] {s} done in {time.time() - t0:.0f} s", flush=True)
    return ctx
