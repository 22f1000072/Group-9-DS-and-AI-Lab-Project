"""Shared matplotlib style: reference categorical palette in fixed order, recessive grid/axes."""
import matplotlib
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# fixed identity colours (colour follows the entity, never its rank)
MODEL_COLOR = {"ref_base960h": SERIES[2], "A_clean": SERIES[0], "B_noisy": SERIES[1]}
MODEL_LABEL = {"ref_base960h": "wav2vec2-base-960h (reference, 960 h, not trained by us)",
               "A_clean": "Model A: base + CTC, clean training", "B_noisy": "Model B: base + CTC, noise-augmented"}
NOISE_COLOR = {"clean": INK, "white": SERIES[0], "pink": SERIES[4], "brown": SERIES[3], "esc50": SERIES[1],
               "musan_noise": SERIES[2], "musan_music": SERIES[6]}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 7,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES), "figure.dpi": 110,
    })
