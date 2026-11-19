"""Noise generation, noise banks and SNR mixing.

Noise roles (no leakage between training and test noise):
  training   : white, pink (synthetic) + MUSAN noise + MUSAN music (90 % of files)
  validation : same noise *types*, held-out 10 % of MUSAN files
  test only  : brown (synthetic) + ESC-50 (all 2,000 clips, resampled to 16 kHz)
Room reverberation (M5): simulated RIRs (OpenSLR 28) for training/validation, REAL measured RIRs test only.
"""
import json
import zlib

import numpy as np
from scipy.signal import fftconvolve, resample_poly

from group9_ds_ai_lab.utils.paths import NOISE_DIR

SYNTHETIC = ("white", "pink", "brown")
TRAIN_NOISE_TYPES = ("white", "pink", "musan_noise", "musan_music")
TEST_NOISE_TYPES = ("white", "pink", "brown", "esc50")  # white/pink = "seen type" reference
TEST_SNRS = (20, 10, 5, 0)


def stable_seed(*parts) -> int:
    """Deterministic 32-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def colored_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    """White / pink (1/f) / brown (1/f^2) noise, zero DC, unit RMS (same as M2/M3 notebook)."""
    if n <= 0:
        return np.zeros(0, np.float32)
    white = rng.standard_normal(n)
    if kind == "white":
        x = white
    elif kind in ("pink", "brown"):
        alpha = 1.0 if kind == "pink" else 2.0
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(n)
        scale = np.zeros_like(f)
        scale[1:] = f[1:] ** (-alpha / 2.0)
        x = np.fft.irfft(spec * scale, n=n)
    else:
        raise ValueError(kind)
    rms = np.sqrt(np.mean(x ** 2))
    return (x / rms if rms > 0 else x).astype(np.float32)


class NoiseBank:
    """Concatenated noise clips stored as one float16 .npy (memory-mapped) + an index."""

    def __init__(self, name: str):
        self.name = name
        self.audio = np.load(NOISE_DIR / f"{name}.npy", mmap_mode="r")
        with open(NOISE_DIR / f"{name}.index.json") as f:
            idx = json.load(f)
        self.files = idx["files"]
        self.starts = np.asarray(idx["starts"], np.int64)
        self.lengths = np.asarray(idx["lengths"], np.int64)

    def __len__(self):
        return len(self.files)

    def sample(self, n: int, rng: np.random.Generator):
        """Random clip, random offset, tiled to length n. Returns (noise, info)."""
        k = int(rng.integers(len(self.files)))
        s, L = int(self.starts[k]), int(self.lengths[k])
        clip = np.asarray(self.audio[s: s + L], dtype=np.float32)
        off = int(rng.integers(L)) if L > n else 0
        if L >= n:
            seg = clip[off: off + n]
            if len(seg) < n:  # wrap around
                seg = np.concatenate([seg, clip[: n - len(seg)]])
        else:
            seg = np.tile(clip, int(np.ceil(n / L)))[:n]
        return seg, {"noise_file": self.files[k], "noise_offset": off}


_BANKS: dict = {}


def get_bank(name: str) -> NoiseBank:
    if name not in _BANKS:
        _BANKS[name] = NoiseBank(name)
    return _BANKS[name]


def make_noise(kind: str, n: int, rng: np.random.Generator, part: str = "train"):
    """kind in white|pink|brown|musan_noise|musan_music|esc50; part selects MUSAN train/val files."""
    if kind in SYNTHETIC:
        return colored_noise(n, kind, rng), {"noise_file": kind, "noise_offset": 0}
    if kind in ("musan_noise", "musan_music"):
        return get_bank(f"{kind}_{part}").sample(n, rng)
    if kind == "esc50":
        return get_bank("esc50_test").sample(n, rng)
    raise ValueError(kind)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """clean + noise scaled so 10*log10(P_clean/P_noise) == snr_db. No clipping (keeps SNR exact)."""
    pc = float(np.mean(clean.astype(np.float64) ** 2))
    pn = float(np.mean(noise.astype(np.float64) ** 2))
    if pc < 1e-12 or pn < 1e-12:
        return clean.astype(np.float32, copy=True)
    g = np.sqrt(pc / (pn * 10.0 ** (snr_db / 10.0)))
    return (clean + g * noise).astype(np.float32)


def reverberate(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with a room impulse response, aligned on the direct-path peak (timestamps are not shifted),
    output rescaled to the input RMS."""
    rir = np.asarray(rir, dtype=np.float32)
    peak = int(np.argmax(np.abs(rir)))
    rir = rir[peak:] / (np.sqrt(np.sum(rir[peak:].astype(np.float64) ** 2)) + 1e-9)
    y = fftconvolve(x, rir)[: len(x)]
    g = np.sqrt((np.mean(x.astype(np.float64) ** 2) + 1e-12) / (np.mean(y.astype(np.float64) ** 2) + 1e-12))
    return (y * g).astype(np.float32)


def sample_rir(part: str, rng: np.random.Generator) -> np.ndarray:
    """part: train | val (simulated rooms) | test (real measured RIRs)."""
    bank = get_bank("rir_real_test" if part == "test" else f"rir_sim_{part}")
    k = int(rng.integers(len(bank)))
    s, L = int(bank.starts[k]), int(bank.lengths[k])
    return np.asarray(bank.audio[s: s + L], dtype=np.float32)


def bandlimit(x: np.ndarray) -> np.ndarray:
    """Telephone-like channel: 16 kHz -> 8 kHz -> 16 kHz (removes content above ~4 kHz)."""
    return resample_poly(resample_poly(x, 1, 2), 2, 1)[: len(x)].astype(np.float32)


def apply_condition(clean: np.ndarray, noise_type: str, snr_db, seed: int, part: str = "test", reverb: bool = False):
    """Deterministic corruption for a fixed (val/test) condition: optional reverb first, then additive noise."""
    rng = np.random.default_rng(int(seed))
    x = reverberate(clean, sample_rir(part, rng)) if reverb else clean
    if noise_type == "clean" or snr_db is None or (isinstance(snr_db, float) and np.isnan(snr_db)):
        return x, {"noise_file": "", "noise_offset": 0}
    noise, info = make_noise(noise_type, len(x), rng, part=part)
    return mix_at_snr(x, noise, float(snr_db)), info


class TrainAugmenter:
    """On-the-fly training augmentation: p_clean clean, else random type and SNR ~ U(lo, hi).
    Optional (M5, all off by default): speed perturbation, simulated-room reverb, telephone band-limiting.
    Order: speed -> reverb -> additive noise -> band-limit."""

    def __init__(self, p_clean=0.2, snr_range=(0.0, 20.0), types=TRAIN_NOISE_TYPES,
                 p_speed=0.0, speeds=(0.9, 1.1), p_reverb=0.0, p_bandlimit=0.0):
        self.p_clean, self.snr_range, self.types = p_clean, snr_range, tuple(types)
        self.p_speed, self.speeds, self.p_reverb, self.p_bandlimit = p_speed, tuple(speeds), p_reverb, p_bandlimit

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.p_speed and rng.random() < self.p_speed:
            sp = self.speeds[int(rng.integers(len(self.speeds)))]
            up, down = (10, int(round(10 * sp)))  # speed 1.1 -> 10/11 samples (shorter, higher pitch)
            x = resample_poly(x, up, down).astype(np.float32)
        if self.p_reverb and rng.random() < self.p_reverb:
            x = reverberate(x, sample_rir("train", rng))
        x = self._noise(x, rng)
        if self.p_bandlimit and rng.random() < self.p_bandlimit:
            x = bandlimit(x)
        return x

    def _noise(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() < self.p_clean:
            return x
        kind = self.types[int(rng.integers(len(self.types)))]
        snr = float(rng.uniform(*self.snr_range))
        noise, _ = make_noise(kind, len(x), rng, part="train")
        return mix_at_snr(x, noise, snr)
