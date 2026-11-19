"""Download datasets and models for Noise-Robust Speech Recognition project.

Supports four modes:
1. Stream-complete: Stream full dataset (no local storage, infinite iterable)
2. Stream-subset: Stream a small subset (no local storage, limited samples)
3. Download-subset: Download a small representative subset for EDA (local cache)
4. Download-full: Download full datasets (large, 50-100GB+)

Everything that gets stored locally goes inside the project's data/ folder
(data/raw/<dataset>, data/hf_home for models and HF metadata, data/tmp for
temporary files), so nothing is written to ~/.cache or the RAM-backed /tmp.

Usage:
    # Stream complete dataset (no download, for inspection only)
    python scripts/download_data.py --mode stream-complete --datasets librispeech

    # Stream a small subset (no download, limited samples for quick check)
    python scripts/download_data.py --mode stream-subset --subset-size 100 --datasets librispeech

    # Download small subset for EDA (local cache, ~2-5 GB)
    python scripts/download_data.py --mode download-subset --subset-size 500

    # Download full datasets (Milestone 3+, ~50-100 GB)
    python scripts/download_data.py --mode download-full

    # Download everything including MUSAN, ESC-50 and the wav2vec2 models
    python scripts/download_data.py --mode download-full --datasets librispeech musan esc50 --models

    # Only the models
    python scripts/download_data.py --models-only
"""

import argparse
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Keep every cache and temp file inside the project's data/ folder.
# setdefault() means anything you already exported in the shell still wins.
# This must happen before `datasets` / `huggingface_hub` are imported.
# ---------------------------------------------------------------------------
DATA_ROOT = project_root / "data"
RAW_DIR = DATA_ROOT / "raw"
for _d in (RAW_DIR, DATA_ROOT / "hf_home", DATA_ROOT / "tmp"):
    _d.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(DATA_ROOT / "hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", str(RAW_DIR / "librispeech"))
os.environ.setdefault("TMPDIR", str(DATA_ROOT / "tmp"))

# Correct HuggingFace dataset identifiers
LIBRISPEECH_REPO = "openslr/librispeech_asr"
LIBRISPEECH_CONFIG = "all"

# Valid split names for openslr/librispeech_asr
LIBRISPEECH_SPLITS = {
    "train.clean.100": "train.clean.100",
    "train.clean.360": "train.clean.360",
    "train.other.500": "train.other.500",
    "validation.clean": "validation.clean",
    "validation.other": "validation.other",
    "test.clean": "test.clean",
    "test.other": "test.other",
}

# Direct download links
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"

# Models used in the project: wav2vec2-base (fine-tuned) and wav2vec2-base-960h (reference)
MODEL_IDS = [
    "facebook/wav2vec2-base",       # pretrained backbone to fine-tune
    "facebook/wav2vec2-base-960h",  # LibriSpeech fine-tuned, useful as a baseline + vocab
]


def _download_file(url: str, dest: Path) -> None:
    """Download a URL to dest with resume support (like wget -c)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")

    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # range not satisfiable -> file already complete
            print(f"   Already downloaded: {dest.name}")
            return
        raise

    resumed = resp.status == 206
    total = resp.headers.get("Content-Length")
    total = int(total) + (existing if resumed else 0) if total else None
    mode = "ab" if resumed else "wb"
    done = existing if resumed else 0

    with open(dest, mode) as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r   {dest.name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()


def stream_librispeech_complete() -> None:
    """Stream the complete LibriSpeech dataset (no local storage)."""
    try:
        from datasets import load_dataset

        print("Streaming COMPLETE LibriSpeech dataset (no local storage)...")
        print("   This creates an iterable dataset - data fetched on-demand from HF Hub.")

        train_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split="train.clean.100",
            streaming=True,
        )
        print(f"   train.clean.100: {train_ds}")

        val_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split="validation.clean",
            streaming=True,
        )
        print(f"   validation.clean: {val_ds}")

        test_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split="test.clean",
            streaming=True,
        )
        print(f"   test.clean: {test_ds}")

        print("\nPublic dataset URL: https://huggingface.co/datasets/openslr/librispeech_asr")
        print("Official source: https://github.com/LibriSpeech/librispeech-lm-data")

    except ImportError:
        print("HuggingFace datasets not installed.")
        print("   Install: pip install datasets")


def stream_librispeech_subset(subset_size: int = 100) -> None:
    """Stream a small subset of LibriSpeech (no local storage, limited samples).

    Uses streaming mode with .take() to limit samples without downloading.
    """
    try:
        from datasets import load_dataset

        print(f"Streaming LibriSpeech SUBSET ({subset_size} samples, no local storage)...")

        # Stream full dataset then take first N samples
        train_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split="train.clean.100",
            streaming=True,
        ).take(subset_size)
        print(f"   train.clean.100 (first {subset_size}): {train_ds}")

        val_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split="validation.clean",
            streaming=True,
        ).take(subset_size)
        print(f"   validation.clean (first {subset_size}): {val_ds}")

        test_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split="test.clean",
            streaming=True,
        ).take(subset_size)
        print(f"   test.clean (first {subset_size}): {test_ds}")

        print("\nPublic dataset URL: https://huggingface.co/datasets/openslr/librispeech_asr")
        print("Official source: https://github.com/LibriSpeech/librispeech-lm-data")

    except ImportError:
        print("HuggingFace datasets not installed.")
        print("   Install: pip install datasets")


def download_librispeech_subset(data_dir: Path, subset_size: int = 500) -> None:
    """Download a small representative subset of LibriSpeech for EDA.

    Downloads only `subset_size` samples from train-clean-100 plus validation
    and test splits. This avoids downloading the full 960h (~50-100GB+).
    Files are cached in data/raw/librispeech by the HuggingFace datasets library.
    """
    try:
        from datasets import load_dataset

        cache = data_dir / "librispeech"
        print(f"Downloading LibriSpeech SUBSET ({subset_size} samples for EDA)...")
        print(f"   Files cached in: {cache}")

        os.makedirs(cache, exist_ok=True)

        # Load small subsets for EDA
        train_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split=f"train.clean.100[:{subset_size}]",
            cache_dir=str(cache),
        )
        print(f"   train-clean-100 subset: {len(train_ds)} samples")

        val_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split="validation.clean",
            cache_dir=str(cache),
        )
        print(f"   dev-clean: {len(val_ds)} samples")

        test_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split="test.clean",
            cache_dir=str(cache),
        )
        print(f"   test-clean: {len(test_ds)} samples")

        print("\nLibriSpeech subset cached locally (~2-5 GB)")
        print("   Public dataset URL: https://huggingface.co/datasets/openslr/librispeech_asr")
        print("   Official source: https://github.com/LibriSpeech/librispeech-lm-data")

    except ImportError:
        print("HuggingFace datasets not installed.")
        print("   Install: pip install datasets")
        print("\nManual download alternative:")
        print("   Visit: https://github.com/LibriSpeech/librispeech-lm-data")
        print("   Download train-clean-100.tar.gz and extract to data/raw/librispeech/")


def download_librispeech_full(data_dir: Path) -> None:
    """Download full LibriSpeech dataset (50-100GB+). Only for Milestone 3+."""
    try:
        from datasets import load_dataset

        cache = data_dir / "librispeech"
        os.makedirs(cache, exist_ok=True)

        print("Downloading FULL LibriSpeech dataset (960h, ~50-100GB+)...")
        print(f"   Files cached in: {cache}")
        print("   This may take several hours depending on your connection.")
        print("   Re-running the same command resumes and skips finished splits.")

        splits = ["train.clean.100", "train.clean.360", "train.other.500",
                  "validation.clean", "validation.other", "test.clean", "test.other"]

        for split in splits:
            try:
                ds = load_dataset(
                    LIBRISPEECH_REPO,
                    LIBRISPEECH_CONFIG,
                    split=split,
                    cache_dir=str(cache),
                )
                print(f"   {split}: {len(ds)} samples")
            except Exception as e:
                print(f"   {split}: ERROR - {e}")

    except ImportError:
        print("HuggingFace datasets not installed.")
        print("   Install: pip install datasets")


def download_musan(data_dir: Path) -> None:
    """Download MUSAN dataset (noise augmentation source, ~11 GB)."""
    target = data_dir / "musan"
    os.makedirs(target, exist_ok=True)

    if (target / "musan").exists():
        print(f"MUSAN already extracted at: {target / 'musan'}")
        return

    archive = target / "musan.tar.gz"
    print(f"Downloading MUSAN (~11 GB) to {archive} ...")
    try:
        _download_file(MUSAN_URL, archive)
        print("   Extracting...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(target)
        archive.unlink()
        print(f"MUSAN ready at: {target / 'musan'}")
    except Exception as e:
        print(f"   MUSAN download failed: {e}")
        print("   Manual: wget -c https://www.openslr.org/resources/17/musan.tar.gz")
        print("   Then extract into data/raw/musan/")


def download_esc50(data_dir: Path) -> None:
    """Download ESC-50 dataset (held-out test noise, ~600 MB)."""
    target = data_dir / "esc50"
    os.makedirs(target, exist_ok=True)

    if (target / "ESC-50-master").exists():
        print(f"ESC-50 already extracted at: {target / 'ESC-50-master'}")
        return

    archive = target / "esc50.zip"
    print(f"Downloading ESC-50 (~600 MB) to {archive} ...")
    try:
        _download_file(ESC50_URL, archive)
        print("   Extracting...")
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        archive.unlink()
        print(f"ESC-50 ready at: {target / 'ESC-50-master'}")
    except Exception as e:
        print(f"   ESC-50 download failed: {e}")
        print("   Manual: https://github.com/karolpiczak/ESC-50 (download zip)")
        print("   Then extract into data/raw/esc50/")


def download_models() -> None:
    """Download the wav2vec2 models into data/hf_home (HF hub cache)."""
    try:
        from huggingface_hub import snapshot_download

        print("Downloading models (cached in data/hf_home/hub)...")
        for repo_id in MODEL_IDS:
            try:
                path = snapshot_download(
                    repo_id,
                    # skip TensorFlow / Flax weights, we only need PyTorch
                    ignore_patterns=["*.msgpack", "*.h5", "tf_model*", "flax_model*"],
                )
                print(f"   {repo_id}: {path}")
            except Exception as e:
                print(f"   {repo_id}: ERROR - {e}")

    except ImportError:
        print("huggingface_hub not installed.")
        print("   Install: pip install huggingface_hub")


def create_gitignore(data_dir: Path, project_root: Path) -> None:
    """Add data directories to .gitignore."""
    gitignore_path = project_root / ".gitignore"

    gitignore_entries = [
        "# Data directories (large, git-ignored)\n",
        "data/raw/*\n",
        "data/processed/*/*\n",
        "data/hf_home/\n",
        "data/tmp/\n",
        "\n",
        "# Checkpoints and logs\n",
        "experiments/runs/*\n",
        "*.ckpt\n",
        "*.pth\n",
        "\n",
        "# Environment files\n",
        ".env\n",
    ]

    if gitignore_path.exists():
        content = gitignore_path.read_text()
        for entry in gitignore_entries:
            if entry.strip() not in content:
                with open(gitignore_path, "a") as f:
                    f.write(entry)
    else:
        gitignore_path.write_text("# Git ignore for DSAI Lab Project\n" + "".join(gitignore_entries))


def main():
    parser = argparse.ArgumentParser(description="Download datasets for speech recognition project")
    parser.add_argument(
        "--mode",
        choices=["stream-complete", "stream-subset", "download-subset", "download-full"],
        default="stream-subset",
        help="""Mode:
          stream-complete  : Stream full dataset (no local storage)
          stream-subset    : Stream small subset (no local storage, limited samples)
          download-subset  : Download small subset for EDA (local cache)
          download-full    : Download full datasets (50-100GB+, Milestone 3+)""",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["librispeech", "musan", "esc50"],
        default=["librispeech"],
        help="Datasets to download/stream (default: librispeech)"
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
        help="Number of samples for stream-subset or download-subset (default: 100)"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Also download the wav2vec2 models after the datasets",
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Only download the wav2vec2 models and exit",
    )

    args = parser.parse_args()

    data_dir = RAW_DIR

    print("=" * 60)
    print("DSAI Lab Project - Dataset Downloader")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"TMPDIR: {os.environ['TMPDIR']}")
    if args.models_only:
        print("Mode: models only")
    else:
        print(f"Mode: {args.mode}")
        print(f"Datasets: {', '.join(args.datasets)}")
        if args.mode in ("stream-subset", "download-subset"):
            print(f"Subset size: {args.subset_size}")
    print("=" * 60)

    # Create gitignore entries
    create_gitignore(data_dir, project_root)

    if args.models_only:
        download_models()
        print("\nModel download complete!")
        return

    if args.mode == "stream-complete":
        print("\nStreaming COMPLETE dataset mode selected")
        for dataset in args.datasets:
            if dataset == "librispeech":
                stream_librispeech_complete()
            elif dataset == "musan":
                print("MUSAN: Streaming not supported, use download mode")
            elif dataset == "esc50":
                print("ESC-50: Streaming not supported, use download mode")
        print("\nStreaming complete! Data is fetched on-demand from HF Hub.")

    elif args.mode == "stream-subset":
        print(f"\nStreaming SUBSET mode selected ({args.subset_size} samples)")
        for dataset in args.datasets:
            if dataset == "librispeech":
                stream_librispeech_subset(args.subset_size)
            elif dataset == "musan":
                print("MUSAN: Streaming not supported, use download mode")
            elif dataset == "esc50":
                print("ESC-50: Streaming not supported, use download mode")
        print("\nStreaming subset complete! Data is fetched on-demand from HF Hub.")

    elif args.mode == "download-subset":
        print("\nDownloading SUBSET mode selected (for EDA)")
        for dataset in args.datasets:
            if dataset == "librispeech":
                download_librispeech_subset(data_dir, args.subset_size)
            elif dataset == "musan":
                download_musan(data_dir)
            elif dataset == "esc50":
                download_esc50(data_dir)
        print("\nSubset download complete!")
        print("   Next step: Run EDA: python scripts/eda_analysis.py")
        print("   Then preprocess: python scripts/preprocess.py --limit 500")

    elif args.mode == "download-full":
        print("\nDownloading FULL dataset mode selected (50-100GB+)")
        for dataset in args.datasets:
            if dataset == "librispeech":
                download_librispeech_full(data_dir)
            elif dataset == "musan":
                download_musan(data_dir)
            elif dataset == "esc50":
                download_esc50(data_dir)
        print("\nFull download complete!")
        print("   Next step: Run preprocessing: python scripts/preprocess.py")

    if args.models:
        print()
        download_models()


if __name__ == "__main__":
    main()
