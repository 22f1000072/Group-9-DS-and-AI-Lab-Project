"""Build an augmentation demo kit with the project's own augmentation functions.
Output: docs/augmentation_demo/{audio/*.wav, spectrograms_*.png, augmentation_demo.html, README.md}"""
import base64, io, json, sys
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/home/user1/DSAI/Group-9-DS-and-AI-Lab-Project")
sys.path.insert(0, str(REPO / "src"))
from group9_ds_ai_lab.data.audio import ArrowAudioReader
from group9_ds_ai_lab.data.noise import (make_noise, mix_at_snr, reverberate, sample_rir, bandlimit,
                                         apply_condition, stable_seed)
from scipy.signal import resample_poly
import os; os.chdir(REPO)

OUT = REPO / "docs/augmentation_demo"; (OUT / "audio").mkdir(parents=True, exist_ok=True)
SR = 16000
UID = "374-180298-0065"
df = pd.read_csv("data/processed/manifests/train.csv")
r = df[df.audio_id == UID].iloc[0]
clean = ArrowAudioReader().load(r.shard, int(r.row)).astype(np.float32)
TEXT = r.text

items = []   # (section, file, label, explanation, signal)
def add(section, name, label, expl, x):
    items.append((section, name, label, expl, np.asarray(x, np.float32)))

def snr_of(mix, ref):
    n = mix[: len(ref)] - ref
    return 10 * np.log10(np.mean(ref.astype(np.float64) ** 2) / np.mean(n.astype(np.float64) ** 2))

S0 = "1. Original"
add(S0, "00_clean", "Clean LibriSpeech utterance", f"{UID} ({len(clean)/SR:.2f} s). Text: \"{TEXT}\"", clean)

# ---- noise on its own
S1 = "2. The noise sources on their own"
rng = np.random.default_rng(1)
desc = {"white": "Synthetic white noise: equal energy at every frequency (hiss). Used in training.",
        "pink": "Synthetic pink (1/f) noise: more energy at low frequencies (rain or fan). Used in training.",
        "musan_noise": "MUSAN 'noise': real recorded sounds (traffic, machines, beeps). Training uses 90 % of the files; 10 % are kept for dev.",
        "musan_music": "MUSAN 'music': real music tracks. Same 90/10 file split.",
        "brown": "Synthetic brown (1/f^2) noise: a deep rumble. TEST ONLY, never used in training.",
        "esc50": "ESC-50: 2,000 environmental clips (dogs, rain, sirens, door knocks). TEST ONLY, never used in training."}
for k in ["white", "pink", "musan_noise", "musan_music", "brown", "esc50"]:
    part = "test" if k in ("brown", "esc50") else "train"
    nz, info = make_noise(k, len(clean), rng, part=part)
    nz = nz / (np.sqrt(np.mean(nz.astype(np.float64) ** 2)) + 1e-9) * np.sqrt(np.mean(clean.astype(np.float64) ** 2))
    add(S1, f"noise_{k}", f"Noise only: {k}", desc[k] + (f" File: {info['noise_file']}" if k not in ("white", "pink", "brown") else ""), nz)

# ---- training noise at each SNR
S2 = "3. Training noise mixed at different SNRs (what models B, B2, B460, B960_rev hear)"
for k in ["white", "pink", "musan_noise", "musan_music"]:
    for snr in (20, 10, 5, 0):
        rng = np.random.default_rng(stable_seed("demo", k, snr))
        nz, info = make_noise(k, len(clean), rng, part="train")
        y = mix_at_snr(clean, nz, snr)
        add(S2, f"train_{k}_{snr:02d}dB", f"{k} at {snr} dB SNR",
            f"mix_at_snr(clean, {k}, {snr}). Measured SNR {snr_of(y, clean):.2f} dB. In training the SNR is drawn from U[0, 20] dB, so every value in between also occurs.", y)

# ---- test-only noise (fixed seeds like evaluate.py)
S3 = "4. Held-out test noise (never seen in training)"
for k in ["brown", "esc50"]:
    for snr in (20, 10, 5, 0):
        y, info = apply_condition(clean, k, float(snr), stable_seed(UID, k, snr), part="test")
        add(S3, f"test_{k}_{snr:02d}dB", f"{k} at {snr} dB SNR (test only)",
            f"apply_condition() with a fixed CRC32 seed, exactly as the test sets are built. {desc[k]}", y)

# ---- M5 extras used by B960_rev
S4 = "5. Extra augmentations used by B960_rev / B960_rev_s2"
for sp in (0.9, 1.1):
    y = resample_poly(clean, 10, int(round(10 * sp))).astype(np.float32)
    add(S4, f"speed_{sp}", f"Speed perturbation x{sp}",
        f"Resample 10/{int(round(10*sp))}: {len(y)/SR:.2f} s instead of {len(clean)/SR:.2f} s; pitch moves {'down' if sp < 1 else 'up'}. Applied with p = 0.3 in B960_rev.", y)
rng = np.random.default_rng(3)
add(S4, "reverb_simulated_train", "Reverb, simulated room (training RIR)",
    "Convolution with a simulated room impulse response (OpenSLR 28). Applied with p = 0.3 in B960_rev.", reverberate(clean, sample_rir("train", rng)))
rng = np.random.default_rng(4)
add(S4, "reverb_real_test", "Reverb, real measured room (test only)",
    "Convolution with a REAL measured RIR. Used only in the reverb test condition.", reverberate(clean, sample_rir("test", rng)))
add(S4, "telephone_bandlimit", "Telephone band-limit",
    "16 kHz -> 8 kHz -> 16 kHz: removes everything above ~4 kHz (sounds muffled, like a phone). Applied with p = 0.1 in B960_rev.", bandlimit(clean))

# ---- full chain
S5 = "6. Everything combined: the full B960_rev chain (speed -> reverb -> noise -> telephone)"
x = resample_poly(clean, 10, 11).astype(np.float32)
x = reverberate(x, sample_rir("train", np.random.default_rng(5)))
nz, _ = make_noise("musan_noise", len(x), np.random.default_rng(6), part="train")
x = mix_at_snr(x, nz, 5.0)
add(S5, "chain_full", "Speed x1.1 + simulated reverb + MUSAN noise 5 dB + telephone",
    "Worst-case training example: every augmentation applied at once, in the TrainAugmenter order.", bandlimit(x))

# random draws with B960_rev settings, logging the dice rolls (same logic and order as TrainAugmenter)
cfg = dict(p_clean=0.2, snr=(0.0, 20.0), types=("white", "pink", "musan_noise", "musan_music"), p_speed=0.3, p_reverb=0.3, p_bandlimit=0.1)
for i in range(1, 9):
    rng = np.random.default_rng(1000 + i); x = clean; steps = []
    if rng.random() < cfg["p_speed"]:
        sp = (0.9, 1.1)[int(rng.integers(2))]; x = resample_poly(x, 10, int(round(10 * sp))).astype(np.float32); steps.append(f"speed x{sp}")
    if rng.random() < cfg["p_reverb"]:
        x = reverberate(x, sample_rir("train", rng)); steps.append("reverb")
    if rng.random() < cfg["p_clean"]:
        steps.append("no noise (the 20 % clean case)")
    else:
        k = cfg["types"][int(rng.integers(4))]; snr = float(rng.uniform(*cfg["snr"]))
        nz, _ = make_noise(k, len(x), rng, part="train"); x = mix_at_snr(x, nz, snr); steps.append(f"{k} at {snr:.1f} dB")
    if rng.random() < cfg["p_bandlimit"]:
        x = bandlimit(x); steps.append("telephone")
    add(S5, f"random_draw_{i}", f"Random training draw #{i}: " + " + ".join(steps),
        "One random roll of the B960_rev augmenter (p_speed 0.3, p_reverb 0.3, p_clean 0.2, SNR ~ U[0, 20], p_telephone 0.1). The same utterance gets a different version every time it is loaded.", x)

# ---- write wavs (scale down only if a mix would clip; SNR is unchanged by scaling)
meta = []
for sec, name, label, expl, y in items:
    pk = float(np.max(np.abs(y))); g = 0.98 / pk if pk > 0.98 else 1.0
    sf.write(OUT / "audio" / f"{name}.wav", y * g, SR, subtype="PCM_16")
    meta.append(dict(section=sec, file=f"audio/{name}.wav", label=label, explanation=expl, scaled=g != 1.0))

# ---- spectrograms
def spec(ax, y, title):
    ax.specgram(y, NFFT=512, Fs=SR, noverlap=384, cmap="magma", vmin=-130, vmax=-20)
    ax.set_title(title, fontsize=9); ax.set_ylim(0, 8000); ax.set_xlabel("s", fontsize=7); ax.set_ylabel("Hz", fontsize=7)
    ax.tick_params(labelsize=7)
get = {m[1]: m[4] for m in items}
panels = [("00_clean", "Clean"), ("train_white_05dB", "White 5 dB"), ("train_pink_05dB", "Pink 5 dB"),
          ("train_musan_noise_05dB", "MUSAN noise 5 dB"), ("train_musan_music_05dB", "MUSAN music 5 dB"),
          ("test_brown_05dB", "Brown 5 dB (test only)"), ("test_esc50_05dB", "ESC-50 5 dB (test only)"),
          ("speed_1.1", "Speed x1.1"), ("reverb_simulated_train", "Reverb (simulated room)"),
          ("reverb_real_test", "Reverb (real room, test only)"), ("telephone_bandlimit", "Telephone (cut above 4 kHz)"),
          ("chain_full", "All combined")]
fig, axs = plt.subplots(4, 3, figsize=(13, 11), constrained_layout=True)
for ax, (n, t) in zip(axs.flat, panels): spec(ax, get[n], t)
fig.suptitle(f"Every augmentation on one utterance ({UID})", fontsize=12)
fig.savefig(OUT / "spectrograms_all_augmentations.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(2, 5, figsize=(16, 5.5), constrained_layout=True)
seq = [("00_clean", "Clean")] + [(f"train_white_{s:02d}dB", f"White {s} dB") for s in (20, 10, 5, 0)]
for j, (n, t) in enumerate(seq):
    y = get[n]; tt = np.arange(len(y)) / SR
    axs[0, j].plot(tt, y, lw=0.3, color="#1f4e79"); axs[0, j].set_ylim(-1, 1); axs[0, j].set_title(t, fontsize=10); axs[0, j].tick_params(labelsize=7)
    spec(axs[1, j], y, "")
fig.suptitle("Same speech, louder noise: SNR 20 -> 0 dB (top: waveform, bottom: spectrogram)", fontsize=12)
fig.savefig(OUT / "spectrograms_snr_levels.png", dpi=130); plt.close(fig)

# ---- self-contained HTML (audio embedded)
def b64(p): return base64.b64encode(Path(p).read_bytes()).decode()
html = ["<!doctype html><html><head><meta charset='utf-8'><title>Group 9 augmentation demo</title><style>",
        "body{font-family:Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#222;background:#fff}",
        "h1{margin-bottom:4px}h2{margin-top:32px;border-bottom:2px solid #1f4e79;color:#1f4e79}",
        ".row{display:grid;grid-template-columns:1fr 320px;gap:12px;padding:8px 0;border-bottom:1px solid #eee;align-items:center}",
        ".lab{font-weight:bold}.ex{font-size:13px;color:#555}audio{width:320px}img{max-width:100%;border:1px solid #ddd;margin:8px 0}",
        "</style></head><body><h1>Noise augmentation demo</h1>",
        "<p>Group 9 noise-robust ASR. One LibriSpeech training utterance, processed with the project's own functions in "
        "<code>src/group9_ds_ai_lab/data/noise.py</code> (<code>make_noise</code>, <code>mix_at_snr</code>, <code>reverberate</code>, "
        "<code>bandlimit</code>, <code>apply_condition</code>).</p>",
        f"<p><b>Reference text:</b> {TEXT}</p>",
        "<p><b>SNR</b> = 10·log10(speech power / noise power). 20 dB = noise is quiet; 0 dB = noise as loud as the speech.</p>"]
for png in ("spectrograms_snr_levels.png", "spectrograms_all_augmentations.png"):
    html.append(f"<img src='data:image/png;base64,{b64(OUT / png)}' alt='{png}'>")
cur = None
for m in meta:
    if m["section"] != cur:
        cur = m["section"]; html.append(f"<h2>{cur}</h2>")
    note = " (volume scaled down to avoid clipping; SNR unchanged)" if m["scaled"] else ""
    html.append(f"<div class='row'><div><div class='lab'>{m['label']}</div><div class='ex'>{m['explanation']}{note}</div></div>"
                f"<audio controls preload='none' src='data:audio/wav;base64,{b64(OUT / m['file'])}'></audio></div>")
html.append("</body></html>")
(OUT / "augmentation_demo.html").write_text("\n".join(html))
json.dump(meta, open(OUT / "clips.json", "w"), indent=1)
print(len(meta), "clips")
