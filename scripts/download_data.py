"""Download datasets for Noise-Robust Speech Recognition project.

Supports four modes:
1. Stream-complete: Stream full dataset (no local storage, infinite iterable)
2. Stream-subset: Stream a small subset (no local storage, limited samples)
3. Download-subset: Download a small representative subset for EDA (local cache)
4. Download-full: Download full datasets (large, 50-100GB+)

Usage:
    # Stream complete dataset (no download, for inspection only)
    python scripts/download_data.py --mode stream-complete --datasets librispeech

    # Stream a small subset (no download, limited samples for quick check)
    python scripts/download_data.py --mode stream-subset --subset-size 100 --datasets librispeech

    # Download small subset for EDA (local cache, ~2-5 GB)
    python scripts/download_data.py --mode download-subset --subset-size 500

    # Download full datasets (Milestone 3+, ~50-100 GB)
    python scripts/download_data.py --mode download-full

    # Specify datasets
    python scripts/download_data.py --mode download-subset --datasets librispeech musan esc50 --subset-size 500
"""

import argparse
import os
import sys
from pathlib import Path

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

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
    Files are cached locally by HuggingFace datasets library.
    """
    try:
        from datasets import load_dataset

        print(f"Downloading LibriSpeech SUBSET ({subset_size} samples for EDA)...")
        print("   Files cached locally by HuggingFace datasets library.")

        os.makedirs(data_dir / "librispeech", exist_ok=True)

        # Load small subsets for EDA
        train_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split=f"train.clean.100[:{subset_size}]"
        )
        print(f"   train-clean-100 subset: {len(train_ds)} samples")

        val_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split="validation.clean"
        )
        print(f"   dev-clean: {len(val_ds)} samples")

        test_ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split="test.clean"
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

        print("Downloading FULL LibriSpeech dataset (960h, ~50-100GB+)...")
        print("   This may take several hours depending on your connection.")
        print("   Only run this for Milestone 3+ model training.")

        splits = ["train.clean.100", "train.clean.360", "train.other.500",
                  "validation.clean", "validation.other", "test.clean", "test.other"]

        for split in splits:
            try:
                ds = load_dataset(
                    LIBRISPEECH_REPO,
                    LIBRISPEECH_CONFIG,
                    split=split
                )
                print(f"   {split}: {len(ds)} samples")
            except Exception as e:
                print(f"   {split}: ERROR - {e}")

    except ImportError:
        print("HuggingFace datasets not installed.")
        print("   Install: pip install datasets")


def download_musan(data_dir: Path) -> None:
    """Download MUSAN dataset (noise augmentation source)."""
    os.makedirs(data_dir / "musan", exist_ok=True)
    print("MUSAN dataset directory created at: data/raw/musan/")
    print("  Manual download required:")
    print("  1. Visit: https://github.com/keithito/musan")
    print("  2. Download music.tar.gz, speech.tar.gz, noise.tar.gz")
    print("  3. Extract to data/raw/musan/")
    print("  Or see README for wget commands")


def download_esc50(data_dir: Path) -> None:
    """Download ESC-50 dataset (held-out test noise)."""
    os.makedirs(data_dir / "esc50", exist_ok=True)
    print("ESC-50 dataset directory created at: data/raw/esc50/")
    print("  Manual download required:")
    print("  1. Visit: https://github.com/karolp/esc50")
    print("  2. Download esc50.zip")
    print("  3. Extract to data/raw/esc50/")


def create_gitignore(data_dir: Path, project_root: Path) -> None:
    """Add data directories to .gitignore."""
    gitignore_path = project_root / ".gitignore"

    gitignore_entries = [
        "# Data directories (large, git-ignored)\n",
        "data/raw/*\n",
        "data/processed/*/*\n",
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

    args = parser.parse_args()

    data_dir = Path("data/raw")
    project_root = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print("DSAI Lab Project - Dataset Downloader")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Mode: {args.mode}")
    print(f"Datasets: {', '.join(args.datasets)}")
    if args.mode in ("stream-subset", "download-subset"):
        print(f"Subset size: {args.subset_size}")
    print("=" * 60)

    # Create gitignore entries
    create_gitignore(data_dir, project_root)

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


if __name__ == "__main__":
    main()