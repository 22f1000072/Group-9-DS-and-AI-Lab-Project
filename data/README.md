# Dataset Instructions

## Source and license

- **Primary dataset: LibriSpeech**
  - **Source:** https://huggingface.co/datasets/openslr/librispeech_asr
  - **Official source:** https://www.openslr.org/resources/12
  - **Version:** 0.9.1 (train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other, test-clean, test-other)
  - **Access date:** October 2026
  - **License:** CC-BY-4.0 for language model data; audiobooks are public domain
  - **Citation:** Panayotov et al., "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books," ICASSP 2015.

- **Noise augmentation datasets:**
  - **MUSAN:** https://www.openslr.org/17/
    - **License:** CC-BY-4.0
    - **Contents:** Music (~7 hours), Speech (~7 hours), Noise (~6 hours)
    - **Use:** Noise augmentation for training (music, speech, environmental noise)
  - **ESC-50:** https://github.com/karolpiczak/ESC-50
    - **License:** CC-BY 4.0
    - **Contents:** 2,000 five-second environmental recordings in 50 classes
    - **Use:** Held-out test noise evaluation (brown noise variants)

- **Synthetic noise:** White, pink, and brown noise generated programmatically at target SNR levels (0, 5, 10, 20 dB)

## Download and preparation

### Reproducible download pipeline (`scripts/download_data.py`)

1. **LibriSpeech** (960 hours total):
   - `train-clean-100` (~100 hours) - used for initial experiments
   - `train-clean-360` (~360 hours) - full training set
   - `train-other-500` (~500 hours) - additional clean speech
   - `dev-clean` + `dev-other` - validation set
   - `test-clean` + `test-other` - test set (held-out, noise types never seen in training)

2. **MUSAN** for noise augmentation:
   - Download `music.tar.gz`, `speech.tar.gz`, `noise.tar.gz`
   - Extract to `data/raw/musan/`

3. **ESC-50** for test noise:
   - Download `esc50.zip`
   - Extract to `data/raw/esc50/`

### Preprocessing pipeline (`scripts/preprocess.py`)

1. **Audio preprocessing:**
   - Resample all audio to **16 kHz** (wav2vec 2.0 base requirement)
   - Convert to **mono channel** (average stereo channels)
   - Normalize audio volume (optional: per-file loudness normalization)
   - Trim silence at head/tail (optional: 200ms threshold)

2. **Transcript preprocessing:**
   - Lowercase all text
   - Remove non-alphanumeric characters (keep spaces)
   - Tokenize using wav2vec 2.0 vocab (60 wordpieces + blank token)
   - Convert to CTC label format: `[char_id_1, char_id_2, ..., blank, ...]`

3. **Noise augmentation** (applied on-the-fly in DataLoader):
   - Mix clean speech with noise at SNR levels: **clean, 20 dB, 10 dB, 5 dB, 0 dB**
   - Training noise: white/pink noise (seen during training)
   - Test noise: brown noise + ESC-50 real environmental noise (held-out)

4. **Directory structure after preprocessing:**
   ```
   data/processed/
       train/              # training pairs (audio .pt + text .txt)
           *.audio.pt      # tensor [1, seq_len] - resampled/mono normalized
           *.text.txt      # normalized transcript
       valid/              # validation pairs
       test/               # test pairs (clean + noise-augmented)
   data/augmentation/
       noise_params.json   # SNR levels, noise types used
   ```

## Schema

| Field | Type | Description |
| --- | --- | --- |
| `audio_id` | String | Unique identifier (filename without extension) |
| `audio_path` | String | Path to processed audio tensor (.pt file) |
| `text` | String | Normalized transcript (lowercase, CTC-compatible) |
| `duration` | Float | Audio duration in seconds |
| `original_sample_rate` | Integer | Original sample rate (before resampling to 16kHz) |
| `noise_type` | String | `'clean'`, `'train_noise'`, `'test_noise'` |
| `snr_db` | Float | Signal-to-noise ratio in dB (if augmented) |

## Quality and privacy

### Known quality issues

1. **File format inconsistency:** LibriSpeech provides `.flac` and `.wav` formats; ensure uniform format after preprocessing
2. **Sample rate variation:** Original recordings vary (8kHz, 16kHz, 44.1kHz, 48kHz); all must be resampled to 16kHz
3. **Silence variation:** Some audio files begin/end with silence; consider trimming for consistent input lengths
4. **Transcript variations:** Some transcripts contain speaker IDs, chapter numbers; strip these during preprocessing

### Privacy

- **No personal data:** LibriSpeech contains public domain audiobooks; no personally identifiable information
- **Noise data:** MUSAN and ESC-50 are also public domain/CC-licensed
- **No audio uploads:** All processing done locally; no cloud services required
- **Data keeping:** Keep raw data outside Git (add to `.gitignore`); only track processed tensors and metadata

### Recommended `.gitignore` entries

```gitignore
# Raw datasets (large, git-ignored)
data/raw/*

# Processed tensors (regeneratable)
data/processed/*/*

# Checkpoints and logs
experiments/runs/*
*.ckpt
*.pth

# Environment files
.env
```
