"""Central path definitions for the project (inputs under data/, outputs under data/processed and experiments/)."""
import glob
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # src/group9_ds_ai_lab/utils/paths.py -> repo root

# Inputs (already downloaded, read-only)
DATA_RAW = REPO / "data" / "raw"
HF_HOME = REPO / "data" / "hf_home"
LS_ROOT = DATA_RAW / "librispeech" / "openslr___librispeech_asr" / "all" / "0.0.0"
MUSAN_ROOT = DATA_RAW / "musan" / "musan"
ESC50_ROOT = DATA_RAW / "esc50" / "ESC-50-master"

# Outputs
PROCESSED = REPO / "data" / "processed"
MANIFESTS = PROCESSED / "manifests"
NOISE_DIR = PROCESSED / "noise"
EXPERIMENTS = REPO / "experiments"
RUNS = EXPERIMENTS / "runs"          # checkpoints + training curves per model
EVAL = EXPERIMENTS / "results"       # test-grid evaluation per model
EDA = EXPERIMENTS / "eda"
LOGS = EXPERIMENTS / "logs"
CONFIGS = REPO / "configs"

LS_SPLITS = [
    "train.clean.100", "train.clean.360", "train.other.500",
    "validation.clean", "validation.other", "test.clean", "test.other",
]

SAMPLE_RATE = 16000


def setup_env() -> None:
    """Point Hugging Face at the project cache and force offline mode."""
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def librispeech_shards(split: str) -> list[str]:
    """Arrow shard files for one LibriSpeech split (absolute paths)."""
    snap = sorted(glob.glob(str(LS_ROOT / "*")))
    snap = [s for s in snap if os.path.isdir(s)][0]
    single = glob.glob(f"{snap}/librispeech_asr-{split}.arrow")
    multi = glob.glob(f"{snap}/librispeech_asr-{split}-*-of-*.arrow")
    files = sorted(single + multi)
    if not files:
        raise FileNotFoundError(f"No arrow shards for split {split} under {snap}")
    return files


def rel(p) -> str:
    """Path relative to the repo (portable in manifests)."""
    return os.path.relpath(str(p), str(REPO))


def absp(p) -> str:
    return str(p) if os.path.isabs(str(p)) else str(REPO / p)
