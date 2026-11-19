"""Figures and tables for the Milestone 4 report (reads experiments/runs and experiments/results only).

Usage: python scripts/make_m4_figures.py
Outputs -> milestones/M4_training/figures/*.png and milestones/M4_training/tables/*.csv
"""
import json
import sys
from pathlib import Path

import jiwer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.utils.paths import EVAL, REPO, RUNS  # noqa: E402
from group9_ds_ai_lab.utils.plotstyle import INK, INK2, MODEL_COLOR, MODEL_LABEL, SERIES, apply_style  # noqa: E402

OUT = REPO / "milestones" / "M4_training"
FIG, TAB = OUT / "figures", OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
apply_style()

SWEEP = [  # (run, change vs hp_base)
    ("hp_base", "reference: lr 5e-5, 640 s/batch, strong masking, p_clean 0.2, SNR 0-20 dB"),
    ("hp_lr1e-4", "learning rate 5e-5 -> 1e-4"),
    ("hp_lr2e-5", "learning rate 5e-5 -> 2e-5"),
    ("hp_mask_light", "masking time 0.4 -> 0.05, feature 0.1 -> 0, layerdrop 0.1 -> 0.05"),
    ("hp_bs320", "batch 640 -> 320 padded seconds"),
    ("hp_pclean0.5", "clean fraction 0.2 -> 0.5"),
    ("hp_snr-5", "SNR range 0-20 -> -5-20 dB"),
]
SWEEP_COLOR = {"hp_base": INK, **{r: SERIES[i] for i, (r, _) in enumerate(SWEEP[1:])}}


def runlogs(name):
    d = RUNS / name
    return pd.read_csv(d / "train_log.csv"), pd.read_csv(d / "eval_log.csv")


# ---------------------------------------------------------------- 1. A vs B training curves
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for m in ("A_clean", "B_noisy"):
    t, e = runlogs(m)
    t = t[t.step > 1]
    sm = t.loss.rolling(20, min_periods=1).mean()
    ax[0].plot(t.step, sm, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
    ax[1].plot(e.step, e.wer_clean * 100, "-o", color=MODEL_COLOR[m], ms=4)
    ax[2].plot(e.step, e.wer_noisy * 100, "-o", color=MODEL_COLOR[m], ms=4)
ax[0].set(title="Training CTC loss (rolling mean)", xlabel="optimizer step", yscale="log")
ax[1].set(title="Dev WER, clean (%)", xlabel="optimizer step", ylim=(0, 40))
ax[2].set(title="Dev WER, held-out MUSAN/synthetic noise (%)", xlabel="optimizer step", ylim=(0, 80))
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
fig.savefig(FIG / "fig1_training_curves_A_vs_B.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 2. hyper-parameter sweep
rows = []
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for r, change in SWEEP:
    t, e = runlogs(r)
    last = e.iloc[-1]
    rows.append({"run": r, "change": change, "steps": int(last.step), "epochs": round(last.step / t[t.epoch == 0].step.max(), 1),
                 "dev_wer_clean": last.wer_clean, "dev_wer_noisy": last.wer_noisy, "dev_wer_avg": last.wer_avg,
                 "dev_cer_clean": last.cer_clean, "dev_cer_noisy": last.cer_noisy,
                 "final_train_loss": t.loss.tail(20).mean(), "max_grad_norm": t.grad_norm.max()})
    lw = 2.6 if r == "hp_base" else 1.8
    for k, col in enumerate(("wer_clean", "wer_noisy")):
        ax[k].plot(e.step, e[col] * 100, "-o", ms=4, lw=lw, color=SWEEP_COLOR[r], label=r)
ax[0].set(title="Dev WER clean (%) - 8k-step sweep", xlabel="optimizer step", ylim=(0, 80))
ax[1].set(title="Dev WER noisy (%) - 8k-step sweep", xlabel="optimizer step", ylim=(0, 80))
ax[1].legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig2_hparam_sweep_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
sweep = pd.DataFrame(rows)
sweep["delta_avg_vs_base_pts"] = (sweep.dev_wer_avg - sweep.dev_wer_avg.iloc[0]) * 100
sweep.round(4).to_csv(TAB / "hparam_sweep.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 3.8))
s = sweep.sort_values("dev_wer_avg")
ax.barh(s.run, s.dev_wer_avg * 100, color=[SWEEP_COLOR[r] for r in s.run], height=0.6)
for y, v in enumerate(s.dev_wer_avg * 100):
    ax.text(v + 0.3, y, f"{v:.1f}", va="center", color=INK2, fontsize=9)
ax.invert_yaxis()
ax.set(title="Mean dev WER (clean + noisy) after 8k steps (%)", xlim=(0, 30))
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIG / "fig3_hparam_sweep_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3. test WER vs SNR (pooled test-clean + test-other)
preds = {m: pd.read_parquet(EVAL / m / "predictions.parquet") for m in ("ref_base960h", "A_clean", "B_noisy")}
grid = []
for m, df in preds.items():
    for (n, snr), g in df.groupby(["noise_type", "snr_db"], dropna=False):
        grid.append({"model": m, "noise_type": n, "snr_db": "clean" if pd.isna(snr) else int(snr),
                     "wer": jiwer.wer(g.text.tolist(), g.hyp.tolist())})
grid = pd.DataFrame(grid)
grid.round(4).to_csv(TAB / "test_wer_grid_pooled.csv", index=False)
fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for k, n in enumerate(("white", "pink", "brown", "esc50")):
    for m in preds:
        g = grid[(grid.model == m) & (grid.noise_type == n)].sort_values("snr_db")
        c = grid[(grid.model == m) & (grid.noise_type == "clean")].wer.iloc[0]
        ax[k].plot(["clean"] + [str(x) for x in g.snr_db[::-1]], [c * 100] + list(g.wer[::-1] * 100), "-o",
                   color=MODEL_COLOR[m], label=MODEL_LABEL[m])
    held = n in ("brown", "esc50")
    ax[k].set(title=f"{n} ({'held-out' if held else 'seen type'})", xlabel="SNR (dB)", ylim=(0, 100))
ax[0].set_ylabel("test WER (%)")
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
fig.savefig(FIG / "fig4_test_wer_vs_snr.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 4. sample outputs
a, b = preds["A_clean"], preds["B_noisy"]
key = ["audio_id", "noise_type", "snr_db"]
ab = a.merge(b[key + ["hyp"]], on=key, suffixes=("_A", "_B"))
ab["wA"] = [jiwer.wer(r, h) for r, h in zip(ab.text, ab.hyp_A)]
ab["wB"] = [jiwer.wer(r, h) for r, h in zip(ab.text, ab.hyp_B)]
ex = []
for n, snr in (("clean", None), ("esc50", 5.0), ("esc50", 0.0), ("white", 5.0), ("brown", 0.0)):
    g = ab[(ab.noise_type == n) & (ab.snr_db.isna() if snr is None else ab.snr_db == snr)]
    g = g[(g.text.str.split().str.len().between(8, 18))]
    g = g.sort_values("audio_id")
    pick = g[(g.wA - g.wB) > 0.2].head(1) if n != "clean" else g[(g.wA > 0) | (g.wB > 0)].head(1)
    ex.append(pick)
ex = pd.concat(ex)[["audio_id", "ls_split", "noise_type", "snr_db", "text", "hyp_A", "hyp_B", "wA", "wB"]]
ex.to_csv(TAB / "sample_outputs.csv", index=False)

w = pd.read_parquet(EVAL / "B_noisy" / "words.parquet")
w0 = w[(w.noise_type == "esc50") & (w.snr_db == 5.0)]
row_ids = w0.groupby("row").filter(lambda g: g.is_error.sum() in (1, 2) and 8 <= len(g) <= 16).row.unique()[:2]
conf_ex = w0[w0.row.isin(row_ids)][["audio_id", "word", "start", "end", "conf_min", "is_error"]]
conf_ex.round(3).to_csv(TAB / "sample_word_confidence.csv", index=False)

summary = {"sweep_best": sweep.sort_values("dev_wer_avg").iloc[0].run,
           "pooled_grid_rows": len(grid)}
print(sweep.round(4).to_string())
print(grid.pivot_table(index=["noise_type", "snr_db"], columns="model", values="wer").round(4).to_string())
print(ex.to_string())
print(conf_ex.round(3).to_string())
print(json.dumps(summary))
