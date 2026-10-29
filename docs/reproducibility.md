# Reproducibility

## Environment

- **Python version:** 3.10.x (tested with 3.10.12)
- **PyTorch version:** 2.3.0+cu121 (PyTorch 2.3 with CUDA 12.1)
- **Transformers version:** 4.41.0
- **datasets version:** 2.18.0
- **torchaudio version:** 2.3.0
- **librosa version:** 0.10.1
- **Gradio version:** 5.0.0
- **evaluate (JiWER) version:** 1.3.0
- **Key dependencies** (from `requirements.txt`):
  - `torch>=2.3.0`
  - `transformers>=4.41.0`
  - `torchaudio>=2.3.0`
  - `librosa>=0.10.1`
  - `gradio>=5.0.0`
  - `jiwer>=1.3.0`
  - `numpy>=1.24.0`
  - `soundfile>=0.12.0`

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
- [ ] Environment captured via `environment.yml` or `requirements.txt` with pinned versions
- [ ] Dataset versions and download dates recorded in `data/README.md`
- [ ] Preprocessing parameters (sample rate, SNR levels, augmentation config) documented
- [ ] Train/valid/test split strategy documented (no data leakage across splits)
- [ ] All training commands logged in `experiments/EXPERIMENT_LOG.md`
- [ ] Checkpoints saved to external artifact store (not tracked in Git); links recorded in experiment log
- [ ] Random seed (42) used for all experiments; results reported with seed used
