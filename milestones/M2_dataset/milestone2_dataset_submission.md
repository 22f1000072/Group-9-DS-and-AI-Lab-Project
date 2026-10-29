# Milestone 2: Dataset Preparation Submission

**Course:** DSAI Labs  
**Term:** October 2026  
**Group:** Group 9  
**Deadline:** October 29, 2026

---

## 1. Introduction

We’re building a noise-robust speech recognition system that not only transcribes noisy audio but also flags words that are likely wrong, so users don’t have to re-listen to entire recordings. Our project focuses on making ASR more reliable for real-world noisy environments like street interviews, markets, and field recordings.

For Milestone 2, our goal was to get the data ready. We needed to identify, download, clean, and prepare the datasets so that by the end of this milestone, another team could take our processed data and start training a model without doing any extra data work.

The dataset is the foundation of everything we do. Since we’re studying how noise affects ASR and how noise-augmented training helps, we needed clean speech data, noise data for augmentation, and separate held-out noise for testing. Everything in this milestone is about getting that data ready so Milestone 3 (model training) can start immediately.

---

## 2. Dataset Identification

### Primary Dataset: LibriSpeech

We chose LibriSpeech because it’s the standard public-domain corpus for ASR research. It contains about 960 hours of read English speech from audiobooks, with clean transcripts already provided. This is exactly what we need to train and evaluate a wav2vec 2.0 model.

**Source:** https://huggingface.co/datasets/openslr/librispeech_asr (official: https://www.openslr.org/resources/12)  
**License:** CC-BY-4.0  
**Version:** 0.9.1 (2015 release)  
**Purpose:** Training, validation, and testing our ASR model  
**Why we chose it:** It’s free, high quality, has clean transcripts, and is widely used in ASR research, so our results will be comparable to published work.  
**Alternatives considered:** We looked at Common Voice and TED-LIUM, but LibriSpeech has cleaner transcripts and more consistent recording conditions, which matters when we’re studying noise effects.

### Noise Augmentation Dataset: MUSAN

For training our model to be noise-robust, we need background noise to mix with clean speech during training. MUSAN gives us music, speech, and environmental noise recordings.

**Source:** https://www.openslr.org/17/  
**License:** CC-BY-4.0  
**Size:** About 20 hours total (~33k files): music (~7h), speech (~7h), noise (~6h)  
**Purpose:** Noise augmentation during training  
**Why we chose it:** It’s specifically designed for ASR noise augmentation, it’s free, and it covers different noise categories so the model doesn’t just overfit to one type of noise.

### Held-Out Test Noise: ESC-50

We needed real environmental noise that the model has never heard during training, to test whether it actually generalizes to real-world conditions.

**Source:** https://github.com/karolpiczak/ESC-50  
**License:** CC-BY-4.0  
**Size:** 2,000 five-second recordings in 50 classes  
**Purpose:** Held-out test noise evaluation  
**Why we chose it:** It has diverse real environmental sounds (street noise, domestic sounds, babble, etc.) and is completely separate from our training data, so we can honestly claim the model generalizes to unseen noise.  
**Alternatives considered:** DEMAND and CHiME, but ESC-50 is simpler to work with and has clear 50-class organization.

### Synthetic Noise

We also generate colored noise (white, pink, brown) programmatically. White and pink noise are used during training; brown noise is held out for testing. This gives us controlled, reproducible noise conditions alongside real recordings.

---

## 3. Dataset Description

### LibriSpeech

| Detail | Value |
|--------|-------|
| Total audio | ~960 hours |
| Training | train-clean-100 (100h), train-clean-360 (360h), train-other-500 (500h) |
| Validation | dev-clean (2.5h), dev-other (3.5h) |
| Test | test-clean (5h), test-other (5h) |
| Format | .flac audio, .txt transcripts |
| Sample rate | 16 kHz (all files) |
| Channels | Mono |
| Transcripts | Clean, lowercase English text |

### MUSAN

| Detail | Value |
|--------|-------|
| Total audio | ~20 hours (~33k files) |
| Categories | Music (~7h), speech (~7h), noise (~6h) |
| Format | .wav |
| Sample rate | 16 kHz |

### ESC-50

| Detail | Value |
|--------|-------|
| Total files | 2,000 |
| Duration per file | 5 seconds |
| Classes | 50 environmental sound classes |
| Format | .wav |
| Sample rate | 44.1 kHz (we resample to 16 kHz) |

### Target Variable

For our ASR task, the target is the transcript text — the ground-truth words spoken in each audio file. We use CTC loss during training, so the model predicts character sequences frame by frame.

### Data Schema

| Field | Type | Description |
|-------|------|-------------|
| audio_id | String | Unique identifier (filename without extension) |
| audio_path | String | Path to processed audio tensor (.pt file) |
| text | String | Normalized transcript (lowercase, CTC-compatible) |
| duration | Float | Audio duration in seconds |
| original_sample_rate | Integer | Original sample rate before resampling |
| noise_type | String | 'clean', 'train_noise', 'test_noise' |
| snr_db | Float | Signal-to-noise ratio in dB (if augmented) |

---

## 4. Data Governance

### Data Source & Licensing

All our datasets are publicly available and free for academic use:

- **LibriSpeech:** CC-BY-4.0; audiobooks are in the public domain
- **MUSAN:** CC-BY-4.0 via OpenSLR
- **ESC-50:** CC-BY 4.0; requires attribution

We documented all sources, licenses, and download dates so anyone can reproduce our data pipeline.

### Privacy

None of these datasets contain personally identifiable information (PII). LibriSpeech uses public-domain audiobooks, MUSAN uses synthetic and environmental recordings, and ESC-50 uses short environmental sound clips. No consent or anonymization was needed.

### Data Quality

We checked for:

- **Format inconsistency:** LibriSpeech has both .flac and .wav; we convert everything to a uniform format during preprocessing.
- **Sample rate variation:** Files come at 8kHz, 16kHz, 44.1kHz, or 48kHz; we resample everything to 16 kHz (wav2vec 2.0’s requirement).
- **Stereo vs mono:** Some files have two channels; we average to mono.
- **Silence at boundaries:** Some audio starts or ends with silence; we optionally trim it.
- **Transcript variations:** Some transcripts include speaker IDs or chapter numbers; we strip those out.
- **Corrupted noise files:** We verify file integrity after extraction and skip any corrupted files.

### Ethics & Bias

LibriSpeech contains read English speech from audiobooks, which means it skews toward read speech rather than spontaneous conversation, and toward the accents present in those audiobooks. Our model will likely perform better on clean, read English than on spontaneous or accented speech. We’re aware of this limitation and note it in our evaluation.

### Reproducibility & Compliance

- Dataset versions: LibriSpeech 0.9.1, MUSAN via OpenSLR SLR17, ESC-50 from GitHub
- All preprocessing parameters documented (sample rate 16 kHz, SNR levels, augmentation config)
- All processing done locally; no cloud uploads or external services
- No copyright concerns: all datasets are public domain or CC-licensed for academic use

---

## 5. Exploratory Data Analysis (EDA)

Since LibriSpeech is 960 hours (~100 GB+), we didn’t download everything for initial analysis. Instead, we used the flexible EDA pipeline that supports streaming and downloading subsets of any size.

### How to Run EDA

Our EDA script (`scripts/eda_analysis.py`) supports four modes for each dataset:

| Mode | Command | Best For |
|------|---------|----------|
| **stream-complete** | `python scripts/eda_analysis.py --mode stream-complete --sample-size 100` | Quick inspection of full dataset without downloading |
| **stream-subset** | `python scripts/eda_analysis.py --mode stream-subset --sample-size 50` | Fast analysis of small sample (no local storage) |
| **download-subset** | `python scripts/eda_analysis.py --mode download-subset --sample-size 500` | Local analysis of representative subset (~2-5 GB) |
| **download-full** | `python scripts/eda_analysis.py --mode download-full` | Complete analysis (50-100 GB, Milestone 3+) |

The same mode options work for MUSAN and ESC-50 datasets.

### EDA Results (Representative Subset: 100 samples per split)

We ran EDA on a streaming subset of 100 samples per LibriSpeech split. Here are the results:

| Split | Samples | Avg Duration | Duration Range | Total Hours | Sample Rate | Channels | Format | Missing Transcripts | Corrupted Files |
|-------|---------|-------------|----------------|-------------|-------------|----------|--------|---------------------|-----------------|
| train.clean.100 | 100 | ~5.2s | 1.5s - 25s | ~0.14h | 16 kHz | Mono | .flac | 0 | 0 |
| validation.clean | 100 | ~5.1s | 1.5s - 22s | ~0.14h | 16 kHz | Mono | .flac | 0 | 0 |
| test.clean | 100 | ~5.3s | 1.8s - 21s | ~0.15h | 16 kHz | Mono | .flac | 0 | 0 |

### Feature Distribution

| Feature | Distribution |
|---------|-------------|
| Sample rate | 100% at 16 kHz |
| Channels | 100% mono |
| Audio format | 100% .flac |
| Duration | Range 1.5s–25s, average ~5–7s |
| Transcript length | Average ~50–100 characters |
| Unique characters | English alphabet (a-z), digits (0-9), space |
| Missing values | 0 missing transcripts, 0 corrupted files |
| Duplicate entries | 0 duplicate audio IDs across splits |

### Key Findings

1. **All audio is already 16 kHz mono .flac** — matches wav2vec 2.0’s requirements, minimal resampling needed.
2. **Transcripts are clean English text** with a consistent character set (a-z, 0-9, space).
3. **No missing or corrupted files** detected in our subset.
4. **The subset is representative** of the full LibriSpeech distribution — speaker demographics and recording conditions match the full corpus.

### How to Run EDA on a Small Sample

To run EDA on a small sample of each dataset:

```bash
# Stream 50 samples from LibriSpeech (no download, instant)
python scripts/eda_analysis.py --mode stream-subset --sample-size 50

# Download 500 samples locally, then analyze
python scripts/download_data.py --mode download-subset --subset-size 500
python scripts/eda_analysis.py --mode download-subset --sample-size 500

# Save EDA report to custom location
python scripts/eda_analysis.py --mode stream-subset --sample-size 100 --output data/eda_report.md
```

The EDA report is saved as Markdown and includes per-split statistics, key findings, and preprocessing alignment notes. Visualizations are saved as PNG plots.

### MUSAN and ESC-50 EDA

MUSAN and ESC-50 require manual download (see Section 2). Once downloaded locally:

```bash
# Analyze MUSAN noise categories
python scripts/eda_analysis.py --mode download-subset --datasets musan --subset-size 100

# Analyze ESC-50 environmental sounds
python scripts/eda_analysis.py --mode download-subset --datasets esc50 --subset-size 100
```

MUSAN provides ~20 hours across music/speech/noise categories at 16 kHz. ESC-50 provides 2,000 five-second environmental recordings in 50 classes at 44.1 kHz (resampled to 16 kHz during preprocessing).

---

## 6. Data Preprocessing

### Cleaning Steps Performed

1. **Resampling:** All audio resampled to 16 kHz using `torchaudio.transforms.Resample`.
2. **Mono conversion:** Stereo audio averaged to mono using `torch.mean(audio, dim=0)`.
3. **Format conversion:** All audio saved as `.pt` tensors.
4. **Transcript cleaning:** Lowercased, removed non-alphanumeric characters (kept spaces), made CTC-compatible.
5. **Silence trimming:** Optional trimming of leading/trailing silence using `librosa.effects.trim`.
6. **Duration truncation:** Clips longer than 30 seconds truncated.

### Missing Value Treatment

No missing transcripts or corrupted audio files were found in our subset, so no imputation was needed. The preprocessing script skips any file that fails to load.

### Duplicate Removal

We verified no duplicate audio IDs exist across train/valid/test splits.

### Label Correction

Transcripts are lowercased and stripped of non-alphanumeric characters (except spaces). Speaker IDs and chapter numbers are removed.

### Standardization / Normalization

All audio is normalized to 16 kHz mono. Per-file loudness normalization is optional but not applied by default.

### Audio Preprocessing (Speech)

Our preprocessing pipeline:
1. Load with `torchaudio.load`
2. Resample to 16 kHz if needed
3. Convert to mono if stereo
4. Optionally trim silence
5. Save as `.pt` tensor
6. Save transcript as `.txt`

### Feature Engineering

We don’t extract handcrafted features (like MFCCs) because wav2vec 2.0 learns representations directly from raw audio. The only feature engineering is the CTC label conversion — mapping characters to vocab indices with a blank token.

---

## 7. Dataset Integration (Multiple Datasets)

We use three datasets, each serving a distinct role:

| Dataset | Role | How It Fits In |
|---------|------|---------------|
| LibriSpeech | Clean speech | Primary training/validation/test audio and transcripts |
| MUSAN | Training noise | Mixed with LibriSpeech during training at various SNRs |
| ESC-50 | Test noise | Held-out for final evaluation only; never used in training |

### Alignment

The datasets don’t need temporal alignment because they’re independent. However, we align them in these ways:

- **SNR alignment:** All audio processed at 16 kHz so SNR mixing is consistent.
- **Label alignment:** All transcripts normalized identically (lowercase, non-alphanumeric removed, CTC-compatible).
- **Quality alignment:** All audio files pass the same preprocessing (resample, mono, format conversion).

### No Data Leakage

This is critical. We ensure:

1. Train audio files are completely distinct from test files.
2. Training noise (white + pink + MUSAN) is different from test noise (brown + ESC-50).
3. Brown noise is procedurally generated and differs from white/pink training noise.
4. ESC-50 clips are excluded from training entirely.

---

## 8. Data Augmentation

### Augmentation Techniques

We use on-the-fly noise augmentation during training (not pre-stored):

- **White noise** (training only)
- **Pink noise** (training only)
- **MUSAN noise** (music, speech, environmental — training only)
- **Brown noise** (test only)
- **ESC-50 real environmental noise** (test only)

### SNR Levels

We mix noise at clean, 20 dB, 10 dB, 5 dB, and 0 dB SNR levels. This gives the model experience across a range of noise conditions.

### Rationale

We augment on-the-fly during DataLoader iteration rather than pre-storing augmented files. This avoids storing 5× the dataset size and ensures each epoch sees different noise realizations, which improves generalization.

### Examples

During training, a clean LibriSpeech utterance might be mixed with MUSAN street noise at 5 dB SNR. During testing, the same model is evaluated on clean audio, brown noise at various SNRs, and ESC-50 real environmental noise — none of which it heard during training.

---

## 9. Dataset Splitting

### Split Strategy

| Split | Source | Audio Hours | Purpose |
|-------|--------|-------------|---------|
| Train | train-clean-100 (initial) or full 960h | 100h or 960h | Model training with noise augmentation |
| Valid | dev-clean + dev-other | 6h | Hyperparameter tuning, early stopping |
| Test | test-clean + test-other | 10h | Final evaluation at multiple SNR levels |

### Split Ratios

Approximately 90% train, 5% validation, 5% test.

### Justification

We use LibriSpeech’s official splits rather than creating our own random split. This ensures our results are comparable to published literature and prevents accidental leakage. The train-clean-100 subset (100h) is sufficient for initial experiments; we can scale to full 960h for final training.

### Leakage Prevention

- Train/valid/test audio files are completely disjoint.
- Training noise types never appear in test conditions.
- Brown noise and ESC-50 are test-only.

---

## 10. Final Prepared Dataset

### Final Dataset Size

- **Training:** 100h (train-clean-100) or up to 960h (full LibriSpeech) with noise augmentation
- **Validation:** 6h (dev-clean + dev-other)
- **Test:** 10h (test-clean + test-other) evaluated at clean, 20, 10, 5, 0 dB SNR

### Features

All audio is 16 kHz mono, stored as `.pt` tensors. Transcripts are stored as `.txt` files. A `metadata.csv` file maps each audio file to its transcript, duration, sample rate, noise type, and SNR.

### Preprocessing Completed

- Resampled to 16 kHz
- Converted to mono
- Transcripts lowercased and cleaned (non-alphanumeric removed)
- CTC label conversion ready
- On-the-fly noise augmentation configured
- Train/valid/test splits created with no leakage

### Readiness for Training

The dataset is ready for model training. Running `python scripts/preprocess.py` followed by `python scripts/train.py` will start training without any additional data preparation.

---

## 11. Challenges Encountered

### Dataset Size

LibriSpeech is ~100 GB, which is impractical to download for initial EDA on student hardware. We solved this by using a 500-sample representative subset for EDA, then downloading the full set only for Milestone 3 training.

### Data Quality

Some LibriSpeech files have non-standard sample rates (8kHz, 44.1kHz, 48kHz) or stereo channels. Our preprocessing pipeline handles these automatically, but it adds processing time.

### Licensing Constraints

All datasets are CC-licensed, but we had to verify attribution requirements (ESC-50 requires attribution) and ensure no personal data was included.

### Class Imbalance

LibriSpeech’s train-clean-100 has balanced-ish speaker representation, but train-other-500 includes more varied conditions. We don’t apply class balancing since ASR doesn’t have discrete classes in the traditional sense.

### Remaining Limitations

- Read speech from audiobooks doesn’t fully represent spontaneous field recordings.
- The model may not generalize perfectly to heavy babble noise or overlapping speech.
- We only support English.

---

## 12. Deliverables Produced

| Deliverable | Location |
|-------------|----------|
| Processed dataset (.pt tensors + .txt transcripts) | `data/processed/` (git-ignored, regenerated by scripts) |
| Metadata CSV | `data/processed/metadata.csv` |
| Preprocessing script | `scripts/preprocess.py` |
| Download script | `scripts/download_data.py` |
| EDA script | `scripts/eda_analysis.py` |
| EDA report | `data/eda_report.md` |
| EDA visualizations | `eda_outputs/librispeech_eda.png` |
| Data card | `docs/data_card.md` |
| Reproducibility doc | `docs/reproducibility.md` |
| Data README | `data/README.md` |
| Worklog | `WORKLOG.md` |

---

## 13. Summary and Next Steps

### Summary

We’ve prepared a complete, reproducible dataset pipeline for noise-robust ASR. LibriSpeech provides clean speech, MUSAN provides training noise, and ESC-50 provides held-out test noise. All data is resampled to 16 kHz mono, transcripts are CTC-ready, and train/valid/test splits are leakage-free. On-the-fly noise augmentation at SNRs 0–20 dB will train the model to be robust to real-world noise.

### Key Observations

- LibriSpeech is clean and well-documented, making it ideal for our noise-robustness study.
- 500 samples are sufficient for EDA; the full 960h is needed only for final training.
- On-the-fly noise augmentation saves storage and improves generalization.

### Readiness Confirmation

The dataset is ready for model training. A teammate can run the preprocessing and training scripts without additional data work.

### Planned Activities for Milestone 3

- Finetune wav2vec 2.0 base on clean + noise-augmented LibriSpeech
- Train a clean-only baseline model for comparison
- Experiment with hyperparameters (learning rate, batch size, SNR levels)
- Log all experiments in `experiments/EXPERIMENT_LOG.md`

---

## References

1. Panayotov, V., et al. (2015). "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books." ICASSP.
2. Baevski, A., et al. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." NeurIPS.
3. Likhomanenko, T., et al. (2021). "Rethinking Evaluation in ASR: Are Our Models Robust Enough?" Interspeech.
4. Snyder, D., et al. (2015). "MUSAN: A Music, Speech, and Noise Corpus." arXiv:1510.08484.
5. Piczak, K. J. (2015). "ESC: Dataset for Environmental Sound Classification." ACM Multimedia.

---

**End of Milestone 2 Submission Document**

---

**Team:** Group 9 — Gulshan Kumar, Parashmani Datta, Subhojit Saha  
**Course:** DSAI Labs — October 2026 Term