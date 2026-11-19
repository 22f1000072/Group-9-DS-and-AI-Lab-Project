"""Audio I/O.

LibriSpeech is read directly from the Hugging Face arrow shards already on disk
(each row stores the original FLAC bytes). Decoding with soundfile avoids the
torchcodec dependency of `datasets>=4` and avoids duplicating 960 h of audio.
"""
import io

import numpy as np
import pyarrow as pa
import soundfile as sf
from scipy.signal import resample_poly

from group9_ds_ai_lab.utils.paths import SAMPLE_RATE, absp


def to_mono_16k(x: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != target_sr:
        g = np.gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, int(sr) // g).astype(np.float32)
    return x


def decode_flac_bytes(b: bytes) -> tuple[np.ndarray, int, int]:
    """Returns (mono float32 waveform at native rate, sample_rate, n_channels)."""
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr, x.shape[1]


def open_arrow(path: str) -> pa.Table:
    """Zero-copy memory-mapped read of an HF arrow (IPC stream) shard."""
    return pa.ipc.open_stream(pa.memory_map(absp(path))).read_all()


class ArrowAudioReader:
    """Lazy, per-process random access to LibriSpeech rows: (shard, row) -> 16 kHz mono."""

    def __init__(self):
        self._cols = {}

    def _col(self, shard: str):
        c = self._cols.get(shard)
        if c is None:
            # keep the memory-mapped ChunkedArray (zero-copy); combine_chunks() would copy ~0.5 GB per shard
            # into every DataLoader worker (OOM risk on 460 h / 960 h)
            c = open_arrow(shard).column("audio")
            self._cols[shard] = c
        return c

    def load(self, shard: str, row: int) -> np.ndarray:
        b = self._col(shard)[int(row)]["bytes"].as_py()
        x, sr, _ = decode_flac_bytes(b)
        return to_mono_16k(x, sr)


def frame_stats(x: np.ndarray, sr: int = SAMPLE_RATE, hop_s: float = 0.01, rel_db: float = 40.0) -> dict:
    """Cheap signal-quality statistics used by preprocessing QC and EDA."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if n else 0.0
    out = {
        "rms_dbfs": 20 * np.log10(rms + 1e-12),
        "peak": float(np.abs(x).max()) if n else 0.0,
        "clip_frac": float(np.mean(np.abs(x) >= 0.999)) if n else 0.0,
        "dc_offset": float(x.mean()) if n else 0.0,
    }
    hop = int(sr * hop_s)
    m = n // hop
    if m >= 3:
        e = (x[: m * hop].astype(np.float64).reshape(m, hop) ** 2).mean(1)
        e_db = 10 * np.log10(e + 1e-12)
        active = e_db > (e_db.max() - rel_db)
        idx = np.flatnonzero(active)
        out["lead_sil_s"] = float(idx[0] * hop_s)
        out["trail_sil_s"] = float((m - 1 - idx[-1]) * hop_s)
        out["active_ratio"] = float(active.mean())
        # crude per-file SNR estimate: loud-frame energy vs quiet-frame energy
        lo, hi = np.percentile(e_db, [10, 90])
        out["est_snr_db"] = float(hi - lo)
    else:
        out.update(lead_sil_s=0.0, trail_sil_s=0.0, active_ratio=0.0, est_snr_db=0.0)
    return out
