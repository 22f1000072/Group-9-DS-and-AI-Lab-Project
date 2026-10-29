"""Preprocess pipeline for Noise-Robust Speech Recognition project.

Prepares raw audio + transcripts into model-ready format for wav2vec 2.0 finetuning.
Supports flexible input modes: local files, streaming subset, streaming complete,
download subset, download full.

Key transformations:
1. Audio: resample to 16kHz -> mono -> optional silence trim -> optional normalization
2. Transcript: lowercase -> remove non-alphanumeric -> CTC label conversion
3. Noise augmentation config: on-the-fly mixing at SNRs {clean, 20, 10, 5, 0} dB
4. Word-level error detection prep: CTC frame positions for timestamps

Usage:
    # Local files - subset
    python scripts/preprocess.py --input-source local --data-dir data/raw --subset-size 500

    # Streaming subset (no download)
    python scripts/preprocess.py --input-source stream-subset --subset-size 100

    # Streaming complete
    python scripts/preprocess.py --input-source stream-complete --subset-size 1000

    # Download subset then preprocess
    python scripts/preprocess.py --input-source download-subset --subset-size 500

    # All parameters
    python scripts/preprocess.py \
        --input-source download-subset \
        --subset-size 500 \
        --output-dir data/processed \
        --sample-rate 16000 \
        --max-duration 30 \
        --snrs "clean" "20" "10" "5" "0" \
        --seed 42
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import soundfile as sf
import torch
import torchaudio

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

# Default SNR levels for noise augmentation
DEFAULT_SNRS = ["clean", "20", "10", "5", "0"]


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_audio_from_tensor(audio_array: np.ndarray, sample_rate: int, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """Load and preprocess audio from numpy array (from streaming dataset).

    Args:
        audio_array: numpy array of audio samples
        sample_rate: original sample rate
        target_sr: target sample rate (default 16000)

    Returns:
        waveform: tensor [1, seq_len] at target sample rate
        original_sample_rate: original sample rate before resampling
    """
    waveform = torch.from_numpy(audio_array).float().unsqueeze(0)  # [1, seq_len]
    orig_sr = sample_rate

    # Resample to target if needed
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)

    # Convert to mono (average channels if stereo)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform, orig_sr


def load_audio_from_file(audio_path: Path, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """Load and preprocess audio from file.

    Args:
        audio_path: path to audio file
        target_sr: target sample rate (default 16000)

    Returns:
        waveform: tensor [1, seq_len] at target sample rate
        original_sample_rate: original sample rate before resampling
    """
    # Load audio using soundfile (works on Windows without torchcodec dependency)
    waveform_np, orig_sr = sf.read(str(audio_path))

    # Convert to tensor [1, seq_len]
    if waveform_np.ndim == 1:
        waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
    else:
        waveform = torch.from_numpy(waveform_np).float().t()

    # Resample to target if needed
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)

    # Convert to mono (average channels if stereo)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform, orig_sr


def preprocess_transcript(text: str) -> str:
    """Normalize transcript for CTC training.

    Steps:
    1. Lowercase
    2. Remove non-alphanumeric characters (keep spaces)
    3. Strip extra whitespace
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def add_noise_augmentation(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix clean speech with noise at target SNR (in dB).

    Args:
        clean: clean audio waveform [1, N] or [N]
        noise: noise waveform [1, N] or [N]
        snr_db: signal-to-noise ratio in dB

    Returns:
        mixed: clean + noise at target SNR [1, N]
    """
    clean = clean.flatten()
    noise = noise.flatten()

    # Trim/pad noise to match clean length
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)[:len(clean)]
    else:
        noise = noise[:len(clean)]

    # Calculate powers
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    # Desired noise power for target SNR
    desired_noise_power = signal_power / (10 ** (snr_db / 10))

    # Scale noise
    if noise_power > 0:
        noise_scaled = noise * np.sqrt(desired_noise_power / noise_power)
    else:
        noise_scaled = noise

    # Mix: clean + noise_scaled
    mixed = clean + noise_scaled

    # Clip to [-1, 1] to prevent overflow
    mixed = np.clip(mixed, -1.0, 1.0)

    # Reshape to [1, N]
    return mixed.reshape(1, -1)


def create_directory_structure(output_dir: Path, noise_augmentation: bool, snrs: list) -> None:
    """Create output directory structure."""
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    test_dir = output_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if noise_augmentation:
        augment_dir = output_dir / "augmentation"
        augment_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "snr_levels": snrs,
            "noise_types": ["white", "pink", "brown"],
            "sample_rate": 16000,
        }

        with open(augment_dir / "noise_params.json", "w") as f:
            json.dump(params, f, indent=2)


def save_processed_sample(
    output_dir: Path,
    split_name: str,
    audio_id: str,
    waveform: torch.Tensor,
    text: str,
    duration: float,
    original_sample_rate: int,
    noise_type: str = "clean",
    snr_db: float = 0.0,
) -> None:
    """Save processed audio tensor, transcript, and update metadata.

    Args:
        output_dir: base output directory
        split_name: train/valid/test
        audio_id: unique identifier
        waveform: processed audio tensor [1, seq_len]
        text: normalized transcript
        duration: audio duration in seconds
        original_sample_rate: original sample rate before resampling
        noise_type: 'clean', 'train_noise', 'test_noise'
        snr_db: SNR in dB (if augmented)
    """
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Save audio tensor
    audio_output_path = split_dir / f"{audio_id}.pt"
    torch.save(waveform, audio_output_path)

    # Save normalized transcript
    text_output_path = split_dir / f"{audio_id}.txt"
    with open(text_output_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Append to metadata CSV
    metadata_path = output_dir / "metadata.csv"
    file_exists = metadata_path.exists()

    with open(metadata_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "audio_id",
                    "audio_path",
                    "text",
                    "duration",
                    "original_sample_rate",
                    "noise_type",
                    "snr_db",
                    "split",
                ]
            )
        writer.writerow(
            [
                audio_id,
                str(audio_output_path),
                text,
                round(duration, 4),
                original_sample_rate,
                noise_type,
                snr_db,
                split_name,
            ]
        )


def get_librispeech_streaming_dataset(split: str, streaming: bool = True):
    """Get LibriSpeech streaming dataset for a split.

    Args:
        split: split name (e.g., 'train.clean.100')
        streaming: whether to use streaming mode

    Returns:
        HuggingFace dataset iterable
    """
    from datasets import load_dataset

    return load_dataset(
        LIBRISPEECH_REPO,
        LIBRISPEECH_CONFIG,
        split=split,
        streaming=streaming,
    )


def process_streaming_subset(
    output_dir: Path,
    subset_size: int,
    splits: List[str],
    sample_rate: int,
    max_duration: float,
    seed: int,
) -> Dict[str, int]:
    """Process a streaming subset of LibriSpeech.

    Args:
        output_dir: output directory
        subset_size: number of samples per split
        splits: list of splits to process
        sample_rate: target sample rate
        max_duration: max duration in seconds
        seed: random seed

    Returns:
        dict with counts per split
    """
    from datasets import load_dataset

    results = {}
    max_samples = int(max_duration * sample_rate)

    for split in splits:
        print(f"\n  Processing streaming split: {split} (first {subset_size} samples)")
        ds = load_dataset(
            LIBRISPEECH_REPO,
            LIBRISPEECH_CONFIG,
            split=split,
            streaming=True,
        ).take(subset_size)

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        skipped = 0

        for i, sample in enumerate(ds):
            try:
                audio = sample["audio"]
                text = sample.get("text", "")

                if text is None or text.strip() == "":
                    skipped += 1
                    continue

                # Load and preprocess audio
                waveform, orig_sr = load_audio_from_tensor(
                    audio["array"], audio["sampling_rate"], sample_rate
                )

                # Trim to max duration
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]

                duration = waveform.shape[1] / sample_rate

                # Preprocess transcript
                normalized_text = preprocess_transcript(text)

                # Audio ID
                audio_id = f"{split}_{i}"

                # Save
                save_processed_sample(
                    output_dir,
                    split,
                    audio_id,
                    waveform,
                    normalized_text,
                    duration,
                    orig_sr,
                    noise_type="clean",
                    snr_db=0.0,
                )

                count += 1

                if (i + 1) % 50 == 0:
                    print(f"    Processed {i + 1}/{subset_size} ({split})")

            except Exception as e:
                print(f"    [CROSS] Error processing sample {i}: {e}")
                skipped += 1

        results[split] = {"processed": count, "skipped": skipped}
        print(f"    [CHECK] {split} complete: {count} processed, {skipped} skipped")

    return results


def process_local_files(
    output_dir: Path,
    data_dir: Path,
    limit: Optional[int],
    sample_rate: int,
    max_duration: float,
    seed: int,
    train_ratio: float = 0.9,
    valid_ratio: float = 0.05,
) -> Dict[str, int]:
    """Process local audio files.

    Args:
        output_dir: output directory
        data_dir: input data directory
        limit: max number of files (None = all)
        sample_rate: target sample rate
        max_duration: max duration in seconds
        seed: random seed
        train_ratio: train split ratio
        valid_ratio: valid split ratio

    Returns:
        dict with counts per split
    """
    audio_extensions = {".wav", ".flac", ".mp3"}
    all_audio_files = []

    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if Path(f).suffix.lower() in audio_extensions:
                all_audio_files.append(Path(root) / f)

    print(f"  Found {len(all_audio_files)} audio files in {data_dir}")

    if len(all_audio_files) == 0:
        print("  [CROSS] No audio files found.")
        return {}

    # Limit if specified
    if limit is not None:
        all_audio_files = all_audio_files[:limit]
        print(f"  Limited to {len(all_audio_files)} files")

    # Shuffle for random split
    random.seed(seed)
    random.shuffle(all_audio_files)

    # Split
    total = len(all_audio_files)
    train_end = int(total * train_ratio)
    valid_end = int(total * (train_ratio + valid_ratio))

    train_files = all_audio_files[:train_end]
    valid_files = all_audio_files[train_end:valid_end]
    test_files = all_audio_files[valid_end:]

    splits = {
        "train": train_files,
        "valid": valid_files,
        "test": test_files,
    }

    max_samples = int(max_duration * sample_rate)
    results = {}

    for split_name, files in splits.items():
        print(f"\n  Processing {split_name} split ({len(files)} files)...")
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        skipped = 0

        for i, audio_path in enumerate(files):
            try:
                waveform, orig_sr = load_audio_from_file(audio_path, sample_rate)

                # Trim to max duration
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]

                duration = waveform.shape[1] / sample_rate

                # Preprocess transcript
                transcript_path = audio_path.with_suffix(".txt")
                if transcript_path.exists():
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                    normalized_text = preprocess_transcript(raw_text)
                else:
                    normalized_text = audio_path.stem.replace("-", " ")
                    print(f"    [WARNING]   No transcript for {audio_path.stem}, using filename")

                audio_id = audio_path.stem

                # Save
                save_processed_sample(
                    output_dir,
                    split_name,
                    audio_id,
                    waveform,
                    normalized_text,
                    duration,
                    orig_sr,
                    noise_type="clean",
                    snr_db=0.0,
                )

                count += 1

                if (i + 1) % 50 == 0:
                    print(f"    Processed {i + 1}/{len(files)} ({split_name})")

            except Exception as e:
                print(f"    [CROSS] Error processing {audio_path.name}: {e}")
                skipped += 1

        results[split_name] = {"processed": count, "skipped": skipped}
        print(f"    [CHECK] {split_name} complete: {count} processed, {skipped} skipped")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw audio and transcripts for wav2vec 2.0 finetuning"
    )

    parser.add_argument(
        "--input-source",
        type=str,
        choices=["local", "stream-subset", "stream-complete", "download-subset", "download-full"],
        default="local",
        help="""Input source:
          local          : Process local files in --data-dir
          stream-subset  : Stream small subset from HuggingFace (no download)
          stream-complete: Stream complete dataset, sample first N samples
          download-subset: Download small subset then process
          download-full  : Download complete dataset then process""",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to raw dataset directory (for local mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Path to output processed directory",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000 for wav2vec 2.0 base)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
        help="Number of samples for stream-subset/download-subset modes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files for local mode",
    )
    parser.add_argument(
        "--snrs",
        nargs="+",
        default=DEFAULT_SNRS,
        help="SNR levels in dB for noise augmentation config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio for local mode",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio for local mode",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("[TOOL] DSAI Lab Project - Data Preprocessor")
    print("=" * 60)
    print(f"  Input source: {args.input_source}")
    print(f"  Output: {output_dir}")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Max duration: {args.max_duration}s")
    print(f"  SNR levels: {args.snrs}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # Create directory structure
    create_directory_structure(output_dir, True, args.snrs)

    if args.input_source == "local":
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"[CROSS] Data directory not found: {data_dir}")
            return
        results = process_local_files(
            output_dir,
            data_dir,
            args.limit,
            args.sample_rate,
            args.max_duration,
            args.seed,
            args.train_ratio,
            args.valid_ratio,
        )

    elif args.input_source == "stream-subset":
        results = process_streaming_subset(
            output_dir,
            args.subset_size,
            ["train.clean.100", "validation.clean", "test.clean"],
            args.sample_rate,
            args.max_duration,
            args.seed,
        )

    elif args.input_source == "stream-complete":
        print("\n[WARNING] Streaming complete dataset - this will process ALL samples")
        print("   This may take a very long time. Consider using --subset-size to limit.")
        results = process_streaming_subset(
            output_dir,
            args.subset_size,
            ["train.clean.100", "train.clean.360", "train.other.500",
             "validation.clean", "validation.other", "test.clean", "test.other"],
            args.sample_rate,
            args.max_duration,
            args.seed,
        )

    elif args.input_source == "download-subset":
        print("\n[DOWNLOAD] Download-subset mode requires running download first:")
        print("   python scripts/download_data.py --mode download-subset --subset-size", args.subset_size)
        print("   Then run: python scripts/preprocess.py --input-source local --data-dir data/raw/librispeech")
        return

    elif args.input_source == "download-full":
        print("\n[DOWNLOAD] Download-full mode requires running download first:")
        print("   python scripts/download_data.py --mode download-full")
        print("   Then run: python scripts/preprocess.py --input-source local --data-dir data/raw/librispeech")
        return

    else:
        print(f"[CROSS] Unknown input source: {args.input_source}")
        return

    # Summary
    print("\n" + "=" * 60)
    print("[CHECK] Preprocessing Complete!")
    print("=" * 60)

    total_processed = sum(r.get("processed", 0) for r in results.values())
    total_skipped = sum(r.get("skipped", 0) for r in results.values())

    print(f"  Total processed: {total_processed} audio files")
    print(f"  Total skipped: {total_skipped} audio files")
    print(f"  Output location: {output_dir}/")
    print()
    print("  Next steps:")
    print("   1. Review processed data: ls data/processed/train/ | head -5")
    print("   2. Inspect a sample: python -c \"import torch; a = torch.load('data/processed/train/.*pt'); print(a.shape)\"")
    print("   3. Start training: python scripts/train.py --train_data data/processed/train")
    print("=" * 60)


if __name__ == "__main__":
    main()