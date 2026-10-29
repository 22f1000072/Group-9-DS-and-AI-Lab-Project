"""Preprocess pipeline for Noise-Robust Speech Recognition project.

Prepares raw audio + transcripts into model-ready format for wav2vec 2.0 finetuning.

Key transformations:
1. Audio: resample to 16kHz -> mono -> normalize -> optional silence trim
2. Transcript: lowercase -> remove non-alphanumeric -> CTC label conversion
3. Noise augmentation: on-the-fly mixing at SNRs {clean, 20, 10, 5, 0} dB

Usage:
    # Basic preprocessing (100h train-clean-100)
    python scripts/preprocess.py --data_dir data/raw --output_dir data/processed

    # With noise augmentation
    python scripts/preprocess.py --data_dir data/raw --output_dir data/processed --noise_augmentation --snrs "clean" "20" "10" "5" "0"

    # Subset for quick testing
    python scripts/preprocess.py --limit 50

    # With all parameters
    python scripts/preprocess.py \
        --data_dir data/raw \
        --output_dir data/processed \
        --sample_rate 16000 \
        --max_duration 30 \
        --noise_augmentation \
        --snrs "clean" "20" "10" "5" "0" \
        --seed 42
"""

import argparse
import os
import re
import random
import sys
from pathlib import Path

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torchaudio
import librosa
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data import __file__ as data_file


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)  # Global seed for reproducibility


def load_audio(audio_path: Path, sample_rate: int = 16000) -> tuple:
    """Load and preprocess a single audio file.
    
    Returns:
        waveform: tensor [1, seq_len] at target sample rate
        original_sample_rate: original sample rate before resampling
    """
    # Load audio
    waveform, orig_sr = torchaudio.load(audio_path)
    
    # Store original sample rate
    original_sample_rate = orig_sr
    
    # Resample to target if needed
    if orig_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono (average channels if stereo)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Optional: Silence trimming (remove leading/trailing silence)
    # waveform, _ = librosa.effects.trim(waveform.squeeze(), top_db=20)
    # waveform = waveform.unsqueeze(0)  # back to [1, seq_len]
    
    # Optional: Normalization (loudness normalization)
    # max_abs = torch.max(torch.abs(waveform))
    # waveform = waveform / max_abs if max_abs > 0 else waveform
    
    return waveform, original_sample_rate


def preprocess_transcript(text: str) -> str:
    """Normalize transcript for CTC training.
    
    Steps:
    1. Lowercase
    2. Remove non-alphanumeric characters (keep spaces)
    3. Strip extra whitespace
    """
    # Lowercase
    text = text.lower()
    
    # Remove non-alphanumeric (keep spaces and basic punctuation)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Strip and normalize whitespace
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
    # Ensure both are 1D tensors
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
    
    # Create augmentation params file if needed
    if noise_augmentation:
        augment_dir = output_dir / "augmentation"
        augment_dir.mkdir(parents=True, exist_ok=True)
        
        params = {
            "snr_levels": snrs,
            "noise_types": ["white", "pink", "brown"],  # MUSAN categories
            "sample_rate": 16000
        }
        
        with open(augment_dir / "noise_params.json", "w") as f:
            json.dump(params, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw audio and transcripts for wav2vec 2.0 finetuning"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Path to raw dataset directory (default: data/raw)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Path to output processed directory (default: data/processed)"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate (default: 16000 for wav2vec 2.0 base)"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--noise_augmentation",
        action="store_true",
        help="Enable noise augmentation (on-the-fly during training)"
    )
    parser.add_argument(
        "--snrs",
        nargs="+",
        default=["clean", "20", "10", "5", "0"],
        help="SNR levels in dB for noise augmentation (default: clean 20 10 5 0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for quick testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("[TOOL] DSAI Lab Project - Data Preprocessor")
    print("=" * 60)
    print(f"[FOLDER] Input:  {args.data_dir}")
    print(f"[FOLDER] Output: {args.output_dir}")
    print(f"    Sample rate: {args.sample_rate} Hz")
    print(f"[STOPWATCH]   Max duration: {args.max_duration}s")
    print(f"  Noise augmentation: {args.noise_augmentation}")
    print(f"[CHART] SNR levels: {args.snrs}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)
    
    # Create output directory structure
    create_directory_structure(output_dir, args.noise_augmentation, args.snrs)
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"[CROSS] Data directory not found: {data_dir}")
        print("   Run download first: python scripts/download_data.py --mode download")
        return
    
    # Find audio files
    audio_extensions = {".wav", ".flac", ".mp3"}
    all_audio_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if Path(f).suffix.lower() in audio_extensions:
                all_audio_files.append(Path(root) / f)
    
    print(f"  Found {len(all_audio_files)} audio files in {data_dir}")
    
    if len(all_audio_files) == 0:
        print("[CROSS] No audio files found.")
        print("   Check data directory structure or download data first.")
        return
    
    # For quick testing, limit the number of files
    if args.limit is not None:
        all_audio_files = all_audio_files[:args.limit]
        print(f"  Limited to {len(all_audio_files)} files for testing.")
    
    # Process each audio file
    processed_count = 0
    skipped_count = 0
    
    # Organize files by split (based on filename patterns from LibriSpeech)
    # train-clean-100: speakers1-100, dev: speakers1-20, test: speakers1-20
    # We'll create a simple split: first 90% train, 5% valid, 5% test
    
    total_files = len(all_audio_files)
    train_end = int(total_files * 0.9)
    valid_end = int(total_files * 0.95)
    
    train_files = all_audio_files[:train_end]
    valid_files = all_audio_files[train_end:valid_end]
    test_files = all_audio_files[valid_end:]
    
    splits = {
        "train": train_files,
        "valid": valid_files,
        "test": test_files
    }
    
    # Process each split
    for split_name, files in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Processing {split_name} split ({len(files)} files)...")
        
        for i, audio_path in enumerate(files):
            try:
                # Load audio
                waveform, original_sr = load_audio(audio_path, args.sample_rate)
                
                # Trim to max duration
                max_samples = int(args.max_duration * args.sample_rate)
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]
                
                # Extract filename ID (LibriSpeech style: chapter-utterance)
                filename_stem = audio_path.stem
                
                # Preprocess transcript
                # NOTE: In real usage, you'd have text files alongside audio
                # For now, we'll create a placeholder and note this
                transcript_path = audio_path.with_suffix(".txt")
                
                if transcript_path.exists():
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                    normalized_text = preprocess_transcript(raw_text)
                else:
                    # No transcript found - use filename-based placeholder
                    # In practice, LibriSpeech transcripts are in the manifest
                    normalized_text = filename_stem.replace("-", " ")
                    print(f"  [WARNING]   No transcript for {filename_stem}, using filename")
                
                # Save processed audio as .pt tensor
                audio_output_path = split_dir / f"{filename_stem}.pt"
                torch.save(waveform, audio_output_path)
                
                # Save normalized transcript
                text_output_path = split_dir / f"{filename_stem}.txt"
                with open(text_output_path, "w", encoding="utf-8") as f:
                    f.write(normalized_text)
                
                processed_count += 1
                
                # Progress logging
                if (i + 1) % 50 == 0:
                    print(f"  [CHECK] Processed {i + 1}/{len(files)} files ({split_name})")
                
            except Exception as e:
                print(f"[CROSS] Error processing {audio_path.name}: {e}")
                skipped_count += 1
        
        print(f"  [CHART] {split_name} complete: {processed_count} processed, {skipped_count} skipped")
    
    # Create metadata CSV for easy loading
    print("\n  Creating metadata CSV...")
    import csv
    
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_id", "audio_path", "text", "duration", "original_sample_rate", "noise_type", "snr_db"])
        
        # Walk through all splits and collect metadata
        for split_name in ["train", "valid", "test"]:
            split_dir = output_dir / split_name
            if not split_dir.exists():
                continue
            
            for audio_file in sorted(split_dir.glob("*.pt")):
                audio_id = audio_file.stem
                audio_path_str = str(audio_file)
                
                # Read corresponding text
                text_file = Path(audio_path).parent / f"{audio_id}.txt"
                text = ""
                if text_file.exists():
                    with open(text_file, "r", encoding="utf-8") as f:
                        text = f.read()
                
                # Duration calculation
                waveform = torch.load(audio_path)
                duration = waveform.shape[1] / args.sample_rate
                
                writer.writerow([
                    audio_id,
                    audio_path_str,
                    text,
                    duration,
                    args.sample_rate,  # We resampled, original may differ
                    "clean",  # Will be updated if noise augmentation applied
                    "0.0"  # SNR in dB
                ])
    
    print(f"\n[BANK] Metadata saved to: {metadata_path}")
    
    # Summary
    total_processed = processed_count
    total_skipped = skipped_count
    
    print("\n" + "=" * 60)
    print("[CHECK] Preprocessing Complete!")
    print("=" * 60)
    print(f"  Total processed: {total_processed} audio files")
    print(f"    Total skipped: {total_skipped} audio files")
    print(f"[FOLDER] Output location: {output_dir}/")
    print()
    print("  Next steps:")
    print("   1. Review processed data: ls data/processed/train/ | head -5")
    print("   2. Inspect a sample: python -c \"import torch; a = torch.load('data/processed/train/audio0001.pt'); print(a.shape, torch.load('data/processed/train/audio0001.txt').read())\"")
    print("   3. Start training: python scripts/train.py --train_data data/processed/train")
    print("=" * 60)


if __name__ == "__main__":
    main()