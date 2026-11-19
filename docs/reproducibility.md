# Reproducibility

## Environment

All versions are pinned in `requirements.txt` (the exact environment used for every M4/M5 run). Main ones:

- **Python:** 3.14.4
- **PyTorch:** 2.14.1+cu130 (CUDA 13.0), 1x NVIDIA L40S
- **transformers:** 5.23.0 (tokenizers 0.23.3, accelerate 1.16.0)
- **numpy / pandas / pyarrow:** 2.5.5 / 3.1.1 / 26.0.1
- **soundfile / librosa:** 0.14.0 / 1.0.1
- **jiwer / scikit-learn / scipy:** 4.0.0 / 1.9.1 / 1.18.1
- **pyctcdecode / kenlm:** 0.5.0 / 0.3.0 (beam search with a 4-gram LM)

Install:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps pyctcdecode==0.5.0
```

`pyctcdecode` declares `numpy<2` but works with numpy 2.5.5, so it must be installed with `--no-deps`; otherwise pip refuses to resolve (or downgrades numpy and breaks torch/pandas). `kenlm` is built from source (C++ compiler, CMake, Boost; on Python 3.14 regenerate `python/kenlm.cpp` with Cython first).

The conda file below is the original Milestone 1 plan and is kept for reference only; use `requirements.txt`.

### Conda environment file (`environment.yml`)

```yaml
name: dsai-lab-project
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.3.0=py3.10_cuda121_cudnn8.9.5
  - torchvision
  - torchaudio
  - pip
  - pip:
    - transformers==4.41.0
    - datasets==2.18.0
    - gradio==5.0.0
    - jiwer==1.3.0
    - librosa==0.10.1
    - soundfile==0.12.0
```

## Data

### Dataset versions and hashes

- **LibriSpeech:** Version 0.9.1 (2015)
  - `train-clean-100`: Manifest hash `sha256:...`
  - `train-clean-360`: Manifest hash `sha256:...`
  - `train-other-500`: Manifest hash `sha256:...`
  - `dev-clean`: Manifest hash `sha256:...`
  - `dev-other`: Manifest hash `sha256:...`
  - `test-clean`: Manifest hash `sha256:...`
  - `test-other`: Manifest hash `sha256:...`

- **MUSAN:** Download date October 2026; verify file counts after extraction
  - Expected: `music/` ~2,000 files, `speech/` ~2,000 files, `noise/` ~6,000 files

- **ESC-50:** Download date October 2026; `esc50.csv` manifest included

### Data preparation steps (documented in `scripts/preprocess.py`)

1. **Download step:** Run `python scripts/download_data.py --datasets librispeech musan esc50`
2. **Preprocessing step:** Run `python scripts/preprocess.py --data_dir data/raw --output_dir data/processed`
3. **Preprocessing actions:**
   - Resample all audio to 16 kHz using `torchaudio.transforms.Resample`
   - Convert stereo to mono: `torch.mean(audio, dim=0)`
   - Normalize: optional `torchaudio.functional.highpass_biquad` or loudness normalization
   - CTC label conversion: map characters to vocab indices, add blank token
   - Save as `.pt` tensors + `.txt` transcripts

### Data splits created

| Split | Source | Audio hours | Purpose |
| --- | --- | --- | --- |
| `train` | `train-clean-100` / `train-clean-360` / `train-other-500` | 100/360/500 | Model training |
| `valid` | `dev-clean` / `dev-other` | 2.5/3.5 | Hyperparameter tuning, early stopping |
| `test` | `test-clean` / `test-other` | 5/5 | Final evaluation (held-out noise) |
| `train_noise` | Train splits + MUSAN noise + white/pink noise | Varies | Noise-augmented training |
| `test_noise` | Test splits + brown noise + ESC-50 | Varies | Hold-out noise evaluation |

## Commands

### Full pipeline from scratch

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate dsai-lab-project

# 2. Download datasets
python scripts/download_data.py --datasets librispeech musan esc50

# 3. Preprocess data (creates data/processed/ with .pt + .txt files)
python scripts/preprocess.py \
  --data_dir data/raw \
  --output_dir data/processed \
  --sample_rate 16000 \
  --max_duration 30 \
  --noise_augmentation \
  --snrs "clean" "20" "10" "5" "0"

# 4. Train model
python scripts/train.py \
  --config configs/training.yaml \
  --train_data data/processed/train \
  --valid_data data/processed/valid \
  --output_dir experiments/run_1 \
  --seed 42

# 5. Evaluate model
python scripts/evaluate.py \
  --model experiments/run_1/best.ckpt \
  --data data/processed/test \
  --snrs "clean" "20" "10" "5" "0" \
  --output_dir experiments/run_1/eval

# 6. Launch Gradio interface
python scripts/gradio_demo.py \
  --model experiments/run_1/best.ckpt \
  --data_dir data/processed/test
```

## Seeds and hardware

### Random seeds

All experiments use fixed random seeds for reproducibility:

- **Python seed:** `random.seed(42)` at script startup
- **PyTorch seed:** `torch.manual_seed(42)` before model initialization
- **NumPy seed:** `np.random.seed(42)` before any numpy operations
- **CUDA seed (if available):** `torch.cuda.manual_seed_all(42)` when using GPU

Seed is set in:
- `scripts/train.py` line 15
- `scripts/preprocess.py` line 12
- `scripts/evaluate.py` line 10

### Hardware requirements

| Component | Minimum | Recommended |
| --- | --- | --- |
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 16 GB | 32+ GB |
| **GPU** | Not required (CPU mode) | 8GB+ VRAM (CUDA) |
| **Storage** | 50 GB free | 200+ GB free (for LibriSpeech 960h) |

### CPU inference timing

- **wav2vec 2.0 base:** RTF (Real-Time Factor) ≈ 1.0–1.5x on CPU
- **Full pipeline (preprocess + transcribe + flags):** RTF ≈ 2.0–3.0x on CPU
- **Gradio demo:** Runs locally on `http://localhost:7860`

### Reproducibility checklist (for final submission)

- [ ] All random seeds documented and set at script startup
- [x] Environment captured in `requirements.txt` with pinned versions
- [ ] Dataset versions and download dates recorded in `data/README.md`
- [ ] Preprocessing parameters (sample rate, SNR levels, augmentation config) documented
- [ ] Train/valid/test split strategy documented (no data leakage across splits)
- [ ] All training commands logged in `experiments/EXPERIMENT_LOG.md`
- [ ] Checkpoints saved to external artifact store (not tracked in Git); links recorded in experiment log
- [ ] Random seed (42) used for all experiments; results reported with seed used
