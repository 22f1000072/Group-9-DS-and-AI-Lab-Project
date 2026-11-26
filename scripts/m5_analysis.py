"""Milestone 5 evaluation and error analysis (reads existing test outputs only; no GPU, no retraining).

  1. model comparison   : pooled WER per condition for every model (greedy and 4-gram LM) + ROVER; WER-vs-SNR figure
  2. error types        : substitutions / deletions / insertions per condition (final system and Model B)
  3. error analysis     : WER per ESC-50 sound category (clip recovered from the per-utterance noise seed),
                          by utterance duration, by reference-word frequency and length, top word confusions,
                          character confusion matrix, words the LM fixes vs breaks, per-speaker spread
  4. error detector     : ROC and precision-recall curves, confusion matrices at two operating points,
                          AUROC / recall@30 % per condition (B960_rev + 4-gram LM, GBM detector)
  5. qualitative        : systematic best / worst examples per noise type (final system vs Model B)

Usage : python scripts/m5_analysis.py
Output: milestones/M5_evaluation/figures/*.png, milestones/M5_evaluation/tables/*.csv, tables/summary.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import jiwer
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_detector import features, unigram_table  # noqa: E402
from group9_ds_ai_lab.utils.paths import ESC50_ROOT, EVAL, NOISE_DIR, REPO  # noqa: E402
from group9_ds_ai_lab.utils.plotstyle import GRID, INK, INK2, NOISE_COLOR, SERIES, apply_style  # noqa: E402

OUT = REPO / "milestones" / "M5_evaluation"
FIG, TAB = OUT / "figures", OUT / "tables"
KEY = ["audio_id", "noise_type", "snr_db"]
NOISES = ["white", "pink", "brown", "esc50"]
NOISE_LABEL = {"white": "white (seen)", "pink": "pink (seen)", "brown": "brown (held out)", "esc50": "ESC-50 (held out)"}
# fixed identity colours per system (colour follows the model, never its rank)
SYSTEMS = {  # name -> (predictions file, label, colour)
    "A_clean_lm4": ("A_clean_lm4/test_predictions.parquet", "Model A (clean, 100 h)", SERIES[0]),
    "B_noisy_lm4": ("B_noisy_lm4/test_predictions.parquet", "Model B (noise-aug., 100 h)", SERIES[1]),
    "B460_lm4": ("B460_lm4/test_predictions.parquet", "B460 (460 h)", SERIES[3]),
    "B960_rev_lm4": ("B960_rev_lm4/test_predictions.parquet", "B960_rev (960 h + reverb)", SERIES[6]),
    "rover3_final": ("rover3_final/test_predictions.parquet", "Final: ROVER of 3 models", SERIES[2]),
}
GREEDY = {"ref_base960h": "ref_base960h/predictions.parquet", "A_clean": "A_clean/predictions.parquet",
          "B_noisy": "B_noisy/predictions.parquet", "B2_lr1e-4": "B2_lr1e-4/predictions.parquet",
          "B460": "B460/predictions.parquet", "B960_rev": "B960_rev/predictions.parquet",
          "B960_rev_s2": "B960_rev_s2/predictions.parquet"}
LM = {"A_clean": "A_clean_lm4", "B_noisy": "B_noisy_lm4", "B2_lr1e-4": "B2_lr1e-4_lm4", "B460": "B460_lm4",
      "B960_rev": "B960_rev_lm4", "B960_rev_s2": "B960_rev_s2_lm4"}


def cond(df):
    return np.where(df.noise_type.eq("clean"), "clean",
                    df.noise_type + " " + df.snr_db.fillna(-99).astype(int).astype(str) + " dB")


def load(path):
    d = pd.read_parquet(EVAL / path)
    d["hyp"] = d.hyp.fillna("").astype(str).str.strip()
    d["text"] = d.text.astype(str).str.strip()
    return d.sort_values(KEY, na_position="first").reset_index(drop=True)


def align(df):
    """Per-utterance hits/S/D/I (word level) + the jiwer alignment objects."""
    out = jiwer.process_words(df.text.tolist(), [h if h else "<empty>" for h in df.hyp.tolist()])
    H, S, D, I = (np.zeros(len(df), int) for _ in range(4))
    for i, (al, hw) in enumerate(zip(out.alignments, df.hyp.tolist())):
        if not hw:  # empty hypothesis: everything is a deletion
            D[i] = len(out.references[i])
            continue
        for c in al:
            n_r, n_h = c.ref_end_idx - c.ref_start_idx, c.hyp_end_idx - c.hyp_start_idx
            if c.type == "equal":
                H[i] += n_r
            elif c.type == "substitute":
                S[i] += n_r
            elif c.type == "delete":
                D[i] += n_r
            elif c.type == "insert":
                I[i] += n_h
    return pd.DataFrame({"H": H, "S": S, "D": D, "I": I}), out


def wer_of(c):
    return (c.S.sum() + c.D.sum() + c.I.sum()) / max(c.H.sum() + c.S.sum() + c.D.sum(), 1)


def pooled_table(d, c):
    t = pd.concat([d[["noise_type", "snr_db"]].reset_index(drop=True), c], axis=1)
    t["snr"] = t.snr_db.map(lambda x: "clean" if pd.isna(x) else int(x))
    g = t.groupby(["noise_type", "snr"], sort=False)[["H", "S", "D", "I"]].sum()
    g["N"] = g.H + g.S + g.D
    g["wer"] = (g.S + g.D + g.I) / g.N
    return g.reset_index()


# --------------------------------------------------------------------------------------------- 1. comparison
def model_comparison(cache):
    rows = []
    for name, path in GREEDY.items():
        d = load(path)
        c, _ = align(d)
        cache[("greedy", name)] = (d, c)
        for dec, dd, cc in [("greedy", d, c)] + ([("4-gram LM",) + cache_lm(cache, name)] if name in LM else []):
            p = pooled_table(dd, cc)
            row = {"model": name, "decoding": dec,
                   "test-clean": wer_of(cc[(dd.noise_type == "clean") & (dd.ls_split == "test.clean")].reset_index(drop=True)),
                   "test-other": wer_of(cc[(dd.noise_type == "clean") & (dd.ls_split == "test.other")].reset_index(drop=True)),
                   "noisy pooled": wer_of(cc[dd.noise_type != "clean"])}
            for n in NOISES:
                row[f"{n} 0-5 dB"] = wer_of(cc[(dd.noise_type == n) & dd.snr_db.isin([0.0, 5.0])])
            rows.append(row)
    d, c = sys_cache(cache, "rover3_final")
    row = {"model": "ROVER (B960_rev_s2 + B960_rev + B460)", "decoding": "4-gram LM + ROVER",
           "test-clean": wer_of(c[(d.noise_type == "clean") & (d.ls_split == "test.clean")]),
           "test-other": wer_of(c[(d.noise_type == "clean") & (d.ls_split == "test.other")]),
           "noisy pooled": wer_of(c[d.noise_type != "clean"])}
    for n in NOISES:
        row[f"{n} 0-5 dB"] = wer_of(c[(d.noise_type == n) & d.snr_db.isin([0.0, 5.0])])
    rows.append(row)
    t = pd.DataFrame(rows)
    num = t.columns[2:]
    t[num] = (100 * t[num]).round(2)
    t.to_csv(TAB / "model_comparison.csv", index=False)
    return t


def cache_lm(cache, name):
    d, c = sys_cache(cache, LM[name])
    return d, c


def sys_cache(cache, name):
    if ("sys", name) not in cache:
        path = SYSTEMS[name][0] if name in SYSTEMS else f"{name}/test_predictions.parquet"
        d = load(path)
        c, al = align(d)
        cache[("sys", name)] = (d, c)
        cache[("al", name)] = al
    return cache[("sys", name)]


def fig_wer_vs_snr(cache):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=False)
    for ax, n in zip(axes, NOISES):
        for name, (_, label, col) in SYSTEMS.items():
            d, c = sys_cache(cache, name)
            p = pooled_table(d, c)
            clean = p[p.noise_type == "clean"].wer.iloc[0]
            q = p[p.noise_type == n].sort_values("snr")
            x = [25] + q.snr.tolist()[::-1]
            y = [clean] + q.wer.tolist()[::-1]
            ax.plot(x, [100 * v for v in y], marker="o", color=col, label=label, lw=2 if name != "rover3_final" else 2.6)
        ax.set_title(NOISE_LABEL[n])
        ax.set_xticks([0, 5, 10, 15, 20, 25], ["0", "5", "10", "15", "20", "clean"])
        ax.set_xlabel("SNR (dB)")
        ax.invert_xaxis()
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("WER (%), test-clean + test-other")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Word error rate vs. noise level (4-gram LM decoding)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(FIG / "m5_fig1_wer_vs_snr_all_models.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------- 2. error types
def error_types(cache):
    rows = []
    for name in ("B_noisy_lm4", "B960_rev_lm4", "rover3_final"):
        d, c = sys_cache(cache, name)
        p = pooled_table(d, c)
        for _, r in p.iterrows():
            rows.append({"system": name, "noise_type": r.noise_type, "snr": r.snr, "N_ref_words": int(r.N),
                         "sub_%": 100 * r.S / r.N, "del_%": 100 * r.D / r.N, "ins_%": 100 * r.I / r.N, "wer_%": 100 * r.wer})
    t = pd.DataFrame(rows).round(2)
    t.to_csv(TAB / "error_types_SDI.csv", index=False)
    # figure: stacked S/D/I at 0 dB per noise type, Model B vs final system
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels, xs = [], []
    cols = {"sub_%": SERIES[0], "del_%": SERIES[1], "ins_%": SERIES[3]}
    names = {"sub_%": "substitutions", "del_%": "deletions", "ins_%": "insertions"}
    x = 0
    for n in ["clean"] + NOISES:
        for sysn, short in (("B_noisy_lm4", "B"), ("rover3_final", "Final")):
            r = t[(t.system == sysn) & (t.noise_type == n) & ((t.snr == 0) | (n == "clean"))].iloc[0]
            bottom = 0
            for k in ("sub_%", "del_%", "ins_%"):
                ax.bar(x, r[k], bottom=bottom, color=cols[k], width=0.8, edgecolor="white", linewidth=1,
                       label=names[k] if x == 0 else None)
                bottom += r[k]
            ax.text(x, bottom + 0.6, f"{bottom:.1f}", ha="center", fontsize=8, color=INK2)
            labels.append(f"{short}\n{n if n == 'clean' else n + ' 0dB'}")
            xs.append(x)
            x += 1
        x += 0.6
    ax.set_xticks(xs, labels, fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("% of reference words")
    ax.set_title("Error types: Model B vs final system (clean and 0 dB SNR)", loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "m5_fig2_error_types.png", dpi=150)
    plt.close(fig)
    return t


# --------------------------------------------------------------------------------------------- 3. error analysis
def esc50_category(d):
    """Recover which ESC-50 clip was mixed into each utterance: apply_condition() seeds a Generator with
    cond_seed and the first draw picks the clip index (see data/noise.py)."""
    files = json.load(open(NOISE_DIR / "esc50_test.index.json"))["files"]
    meta = pd.read_csv(ESC50_ROOT / "meta" / "esc50.csv").set_index("filename")
    names = [Path(f).name for f in files]
    cat = {i: meta.loc[n, "category"] for i, n in enumerate(names)}
    k = d.cond_seed.map(lambda s: int(np.random.default_rng(int(s)).integers(len(files))))
    return k.map(cat), k.map(lambda i: names[i])


def check_esc_recovery(d):
    from group9_ds_ai_lab.data.noise import apply_condition
    _, fname = esc50_category(d.head(25))
    for (_, r), f in zip(d.head(25).iterrows(), fname):
        _, info = apply_condition(np.zeros(16000 * 3, np.float32), "esc50", r.snr_db, int(r.cond_seed))
        assert Path(info["noise_file"]).name == f, (info, f)


ESC_GROUP = {  # official ESC-50 major groups (Piczak 2015)
    "Animals": "dog rooster pig cow frog cat hen insects sheep crow",
    "Natural soundscapes & water": "rain sea_waves crackling_fire crickets chirping_birds water_drops wind pouring_water toilet_flush thunderstorm",
    "Human, non-speech": "crying_baby sneezing clapping breathing coughing footsteps laughing brushing_teeth snoring drinking_sipping",
    "Interior / domestic": "door_wood_knock mouse_click keyboard_typing door_wood_creaks can_opening washing_machine vacuum_cleaner clock_alarm clock_tick glass_breaking",
    "Exterior / urban": "helicopter chainsaw siren car_horn engine train church_bells airplane fireworks hand_saw",
}
CAT2GROUP = {c: g for g, cs in ESC_GROUP.items() for c in cs.split()}


def esc50_analysis(cache):
    out = []
    for name in ("B_noisy_lm4", "rover3_final"):
        d, c = sys_cache(cache, name)
        m = (d.noise_type == "esc50") & d.snr_db.isin([0.0, 5.0])
        if name == "B_noisy_lm4":
            check_esc_recovery(d[m].reset_index(drop=True))
        cat, _ = esc50_category(d[m])
        t = pd.concat([cat.rename("category").reset_index(drop=True), c[m].reset_index(drop=True)], axis=1)
        g = t.groupby("category")[["H", "S", "D", "I"]].sum()
        g["n_utts"] = t.groupby("category").size()
        g[f"wer_{name}"] = 100 * (g.S + g.D + g.I) / (g.H + g.S + g.D)
        out.append(g[[f"wer_{name}", "n_utts"]])
    t = out[0].join(out[1].drop(columns="n_utts")).reset_index()
    t["group"] = t.category.map(CAT2GROUP)
    t["rel_reduction_%"] = 100 * (1 - t.wer_rover3_final / t.wer_B_noisy_lm4)
    t = t.sort_values("wer_rover3_final", ascending=False).round(2)
    t.to_csv(TAB / "esc50_per_category.csv", index=False)
    # figure: 50 categories, final system, coloured by major group
    gcol = dict(zip(ESC_GROUP, SERIES[:5]))
    fig, ax = plt.subplots(figsize=(11, 9.5))
    y = np.arange(len(t))[::-1]
    ax.barh(y, t.wer_B_noisy_lm4, color=GRID, height=0.8)
    ax.grid(axis="y", visible=False)
    ax.barh(y, t.wer_rover3_final, color=[gcol[g] for g in t.group], height=0.5)
    ax.set_yticks(y, t.category.str.replace("_", " "), fontsize=7.5)
    ax.set_xlabel("WER (%) at 0 and 5 dB SNR")
    ax.legend(handles=[Patch(color=GRID, label="Model B")] + [Patch(color=c, label=f"Final system: {g}") for g, c in gcol.items()],
              loc="lower right", fontsize=8)
    ax.set_title("Held-out ESC-50 noise: WER per sound category (grey = Model B, colour = final system)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "m5_fig3_esc50_categories.png", dpi=150)
    plt.close(fig)
    grp = t.groupby("group")[["wer_B_noisy_lm4", "wer_rover3_final"]].mean().round(2)
    grp.to_csv(TAB / "esc50_per_group.csv")
    return t, grp


def ref_word_table(cache, name):
    """One row per reference word: correct or not (substituted / deleted)."""
    d, _ = sys_cache(cache, name)
    al = cache[("al", name)]
    rows_w, rows_ok, rows_c = [], [], []
    cnd = cond(d)
    for i, (ref, chunks) in enumerate(zip(al.references, al.alignments)):
        ok = np.zeros(len(ref), bool)
        if d.hyp.iat[i]:
            for ch in chunks:
                if ch.type == "equal":
                    ok[ch.ref_start_idx:ch.ref_end_idx] = True
        rows_w += ref
        rows_ok.append(ok)
        rows_c += [cnd[i]] * len(ref)
    return pd.DataFrame({"word": rows_w, "correct": np.concatenate(rows_ok), "cond": rows_c})


def word_level_analysis(cache, uni):
    t = ref_word_table(cache, "rover3_final")
    t["noisy"] = t.cond != "clean"
    t["logp"] = t.word.map(uni).fillna(-8.0)
    bins = [-99, -6, -5, -4, -3, 0]
    lab = ["rare (< -6)", "-6 to -5", "-5 to -4", "-4 to -3", "frequent (> -3)"]
    t["freq_bin"] = pd.cut(t.logp, bins, labels=lab)
    t["len_bin"] = pd.cut(t.word.str.len(), [0, 2, 4, 6, 8, 99], labels=["1-2", "3-4", "5-6", "7-8", "9+"])
    f = t.groupby(["freq_bin", "noisy"], observed=True).correct.agg(["size", "mean"]).reset_index()
    f["word_error_%"] = 100 * (1 - f["mean"])
    ln = t.groupby(["len_bin", "noisy"], observed=True).correct.agg(["size", "mean"]).reset_index()
    ln["word_error_%"] = 100 * (1 - ln["mean"])
    f.drop(columns="mean").round(2).to_csv(TAB / "error_by_word_frequency.csv", index=False)
    ln.drop(columns="mean").round(2).to_csv(TAB / "error_by_word_length.csv", index=False)
    oov = t[t.logp <= -8.0]
    return f, ln, {"ref_words": len(t), "oov_in_lm_frac": float((t.logp <= -8.0).mean()),
                   "oov_word_error_%": float(100 * (1 - oov.correct.mean())) if len(oov) else None}


def duration_analysis(cache):
    d, c = sys_cache(cache, "rover3_final")
    t = pd.concat([d[["duration", "noise_type"]], c], axis=1)
    t["dur_bin"] = pd.cut(t.duration, [0, 4, 8, 12, 16, 40], labels=["< 4 s", "4-8 s", "8-12 s", "12-16 s", "> 16 s"])
    t["noisy"] = t.noise_type != "clean"
    g = t.groupby(["dur_bin", "noisy"], observed=True)[["H", "S", "D", "I"]].sum()
    g["n_utts"] = t.groupby(["dur_bin", "noisy"], observed=True).size()
    g["wer_%"] = 100 * (g.S + g.D + g.I) / (g.H + g.S + g.D)
    g = g.reset_index().round(2)
    g.to_csv(TAB / "error_by_duration.csv", index=False)
    return g


def speaker_analysis(cache):
    d, c = sys_cache(cache, "rover3_final")
    t = pd.concat([d[["speaker_id", "ls_split", "noise_type"]], c], axis=1)
    t = t[t.noise_type == "clean"]
    g = t.groupby(["ls_split", "speaker_id"])[["H", "S", "D", "I"]].sum()
    g["wer_%"] = 100 * (g.S + g.D + g.I) / (g.H + g.S + g.D)
    s = g.groupby("ls_split")["wer_%"].describe(percentiles=[0.1, 0.5, 0.9]).round(2)
    s.to_csv(TAB / "per_speaker_wer_clean.csv")
    return s


def confusions(cache):
    """Most frequent word substitutions, and the character confusion matrix inside substituted words."""
    d, _ = sys_cache(cache, "rover3_final")
    al = cache[("al", "rover3_final")]
    pairs = Counter()
    for i, (ref, hyp, chunks) in enumerate(zip(al.references, al.hypotheses, al.alignments)):
        if not d.hyp.iat[i]:
            continue
        for ch in chunks:
            if ch.type == "substitute":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    pairs[(ref[ch.ref_start_idx + k], hyp[ch.hyp_start_idx + k])] += 1
    top = pd.DataFrame([(r, h, n) for (r, h), n in pairs.most_common(40)], columns=["reference", "hypothesis", "count"])
    top.to_csv(TAB / "top_word_substitutions.csv", index=False)
    # character confusion matrix over all substituted word pairs (weighted by count)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
    idx = {ch: i for i, ch in enumerate(letters)}
    M = np.zeros((len(letters) + 1, len(letters) + 1))  # last row/col = deletion / insertion
    refs, hyps, w = zip(*[(r, h, n) for (r, h), n in pairs.items()])
    co = jiwer.process_characters(list(refs), list(hyps))
    for a, b, n, chunks in zip(co.references, co.hypotheses, w, co.alignments):
        for ch in chunks:
            if ch.type == "substitute":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    M[idx.get(a[ch.ref_start_idx + k], -1), idx.get(b[ch.hyp_start_idx + k], -1)] += n
            elif ch.type == "delete":
                for k in range(ch.ref_start_idx, ch.ref_end_idx):
                    M[idx.get(a[k], -1), -1] += n
            elif ch.type == "insert":
                for k in range(ch.hyp_start_idx, ch.hyp_end_idx):
                    M[-1, idx.get(b[k], -1)] += n
    labs = letters + ["∅"]
    pd.DataFrame(M.astype(int), index=labs, columns=labs).to_csv(TAB / "char_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    im = ax.imshow(np.log10(M + 1), cmap="Blues")
    ax.set_xticks(range(len(labs)), labs, fontsize=7)
    ax.set_yticks(range(len(labs)), labs, fontsize=7)
    ax.set_xlabel("recognised character (∅ = character inserted)")
    ax.set_ylabel("reference character (∅ = character deleted)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("log10(count + 1)")
    ax.set_title("Character confusions inside substituted words (final system, all conditions)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "m5_fig6_char_confusion.png", dpi=150)
    plt.close(fig)
    off = M[:-1, :-1].copy()
    np.fill_diagonal(off, 0)
    flat = [(labs[i], labs[j], int(off[i, j])) for i, j in zip(*np.unravel_index(np.argsort(-off, axis=None)[:10], off.shape))]
    return top, flat


def lm_effect(cache):
    """Reference words that the 4-gram LM fixes vs breaks (same acoustic model B960_rev)."""
    g = ref_word_table_from(cache, ("greedy", "B960_rev"))
    lmw = ref_word_table(cache, "B960_rev_lm4")
    assert len(g) == len(lmw) and (g.word.values == lmw.word.values).all()
    fixed = (~g.correct) & lmw.correct
    broken = g.correct & (~lmw.correct)
    res = {"ref_words": int(len(g)), "greedy_wrong": int((~g.correct).sum()), "lm_wrong": int((~lmw.correct).sum()),
           "fixed_by_lm": int(fixed.sum()), "broken_by_lm": int(broken.sum())}
    br = Counter(g.word[broken]).most_common(15)
    fx = Counter(g.word[fixed]).most_common(15)
    pd.DataFrame({"fixed_word": [w for w, _ in fx], "fixed_n": [n for _, n in fx],
                  "broken_word": [w for w, _ in br], "broken_n": [n for _, n in br]}).to_csv(TAB / "lm_fixed_vs_broken_words.csv", index=False)
    return res


def ref_word_table_from(cache, key):
    d, _ = cache[key]
    tag = f"{key[0]}_{key[1]}"
    if ("al", tag) not in cache:
        _, al = align(d)
        cache[("al", tag)] = al
        cache[("sys", tag)] = cache[key]
    return ref_word_table(cache, tag)


# --------------------------------------------------------------------------------------------- 4. detector
def detector(uni):
    src = EVAL / "B960_rev_lm4"
    w = features(pd.read_parquet(src / "test_words.parquet"), uni)
    det = joblib.load(src / "error_detector" / "model.joblib")
    p = det["model"].predict_proba(w[det["features"]].to_numpy(np.float64))[:, 1]
    y = w.is_error.to_numpy()
    raw = 1 - w.conf_mean.to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for score, lab, col in ((raw, "raw confidence (1 - conf_mean)", SERIES[0]), (p, "trained detector (GBM)", SERIES[1])):
        fpr, tpr, _ = roc_curve(y, score)
        a1.plot(fpr, tpr, color=col, label=f"{lab}, AUROC {auc(fpr, tpr):.3f}")
        pr, rc, _ = precision_recall_curve(y, score)
        keep = rc > 0.005  # drop the first few points, where ties among the very top scores make precision jump
        a2.plot(rc[keep], pr[keep], color=col, label=f"{lab}, AP {average_precision_score(y, score):.3f}")
    a1.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    a1.set_xlabel("false-positive rate (correct words flagged)")
    a1.set_ylabel("true-positive rate (errors flagged)")
    a1.set_title("ROC — word-level error detection", loc="left")
    a1.legend(loc="lower right", fontsize=8)
    a2.axhline(y.mean(), color=GRID, lw=1, ls="--")
    a2.text(0.98, y.mean() + 0.01, f"base rate {100 * y.mean():.1f} %", ha="right", fontsize=8, color=INK2)
    a2.set_xlabel("recall (share of errors flagged)")
    a2.set_ylabel("precision (flagged words that are errors)")
    a2.set_title("Precision-recall", loc="left")
    a2.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"B960_rev + 4-gram LM, all 17 test conditions ({len(y):,} words)", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "m5_fig4_detector_roc_pr.png", dpi=150)
    plt.close(fig)
    # confusion matrices at two operating points
    thr30 = np.quantile(p, 0.70)
    ops = {"threshold 0.5": 0.5, f"riskiest 30 % (threshold {thr30:.3f})": thr30}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    cm_rows = []
    for ax, (lab, t) in zip(axes, ops.items()):
        pred = p >= t
        cm = np.array([[((y == 0) & ~pred).sum(), ((y == 0) & pred).sum()], [((y == 1) & ~pred).sum(), ((y == 1) & pred).sum()]])
        rown = cm / cm.sum(1, keepdims=True)
        ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({100 * rown[i, j]:.1f} % of row)", ha="center", va="center",
                        color="white" if rown[i, j] > 0.6 else INK, fontsize=9)
        ax.set_xticks([0, 1], ["not flagged", "flagged"])
        ax.set_yticks([0, 1], ["correct word", "wrong word"])
        ax.grid(False)
        ax.set_title(lab, loc="left", fontsize=10)
        cm_rows.append({"operating_point": lab, "threshold": float(t), "TN": int(cm[0, 0]), "FP": int(cm[0, 1]),
                        "FN": int(cm[1, 0]), "TP": int(cm[1, 1]), "precision": cm[1, 1] / max(cm[:, 1].sum(), 1),
                        "recall": cm[1, 1] / max(cm[1].sum(), 1), "flagged_frac": float(pred.mean())})
    fig.suptitle("Error detector confusion matrices (rows normalised)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "m5_fig5_detector_confusion.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(cm_rows).round(4).to_csv(TAB / "detector_confusion.csv", index=False)
    # per-condition heat map of AUROC and recall@30 %
    pc = pd.read_csv(src / "error_detector" / "per_condition.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, (col, lab, target) in zip(axes, (("auroc", "AUROC (target ≥ 0.75)", 0.75), ("recall_at_30pct", "errors in riskiest 30 % (target ≥ 0.70)", 0.70))):
        grid = pc[pc.noise_type != "clean"].pivot(index="noise_type", columns="snr_db", values=col).loc[NOISES]
        grid = grid[sorted(grid.columns, key=lambda s: -int(s))]
        im = ax.imshow(grid.values, cmap="Blues", vmin=0.6, vmax=1.0, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid.values[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if grid.values[i, j] > 0.85 else INK)
        ax.set_xticks(range(grid.shape[1]), [f"{c} dB" for c in grid.columns])
        ax.set_yticks(range(grid.shape[0]), [NOISE_LABEL[n] for n in grid.index])
        ax.grid(False)
        clean = pc[pc.noise_type == "clean"][col].iloc[0]
        ax.set_title(f"{lab}; clean: {clean:.3f}", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "m5_fig7_detector_per_condition.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(cm_rows)


# --------------------------------------------------------------------------------------------- 5. qualitative
def qualitative(cache):
    b, cb = sys_cache(cache, "B_noisy_lm4")
    f, cf = sys_cache(cache, "rover3_final")
    assert (b.audio_id.values == f.audio_id.values).all()
    t = b[KEY + ["text", "duration"]].copy()
    t["hyp_B"], t["hyp_final"] = b.hyp, f.hyp
    n = (cf.H + cf.S + cf.D).clip(lower=1)
    t["err_B"], t["err_final"], t["n_words"] = (cb.S + cb.D + cb.I).values, (cf.S + cf.D + cf.I).values, n.values
    t["gain"] = t.err_B - t.err_final
    pick = []
    for nt in ["clean"] + NOISES:
        g = t[(t.noise_type == nt) & (t.snr_db.isna() | t.snr_db.eq(0.0)) & t.n_words.between(6, 25)]
        pick.append(g.nlargest(3, "gain").assign(kind="largest improvement over Model B"))
        pick.append(g[g.gain < 0].nsmallest(2, "gain").assign(kind="final system worse than Model B"))
        pick.append(g.nlargest(2, "err_final").assign(kind="hardest remaining (final system)"))
    q = pd.concat(pick)
    q.to_csv(TAB / "qualitative_examples.csv", index=False)
    return q


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    apply_style()
    cache = {}
    uni = unigram_table()
    summary = {}
    print("1. model comparison", flush=True)
    comp = model_comparison(cache)
    print(comp.to_string(index=False), flush=True)
    fig_wer_vs_snr(cache)
    print("2. error types", flush=True)
    et = error_types(cache)
    print("3. error analysis", flush=True)
    esc, grp = esc50_analysis(cache)
    print(grp.to_string(), flush=True)
    fq, ln, oov = word_level_analysis(cache, uni)
    summary["oov"] = oov
    summary["duration"] = duration_analysis(cache).to_dict("records")
    summary["speakers_clean"] = speaker_analysis(cache).reset_index().to_dict("records")
    top, chars = confusions(cache)
    summary["top_char_confusions"] = chars
    summary["lm_effect"] = lm_effect(cache)
    print(json.dumps(summary["lm_effect"]), flush=True)
    print("4. detector", flush=True)
    summary["detector_confusion"] = detector(uni).to_dict("records")
    print("5. qualitative", flush=True)
    qualitative(cache)
    summary["esc50_hardest"] = esc.head(8)[["category", "wer_B_noisy_lm4", "wer_rover3_final"]].to_dict("records")
    summary["esc50_easiest"] = esc.tail(5)[["category", "wer_B_noisy_lm4", "wer_rover3_final"]].to_dict("records")
    with open(TAB / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
