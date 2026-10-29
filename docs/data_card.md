# Data Card

## Dataset overview

- **Primary dataset: LibriSpeech**
  - **Type:** Read English speech, 960 hours of audiobooks
  - **Purpose:** Speech recognition training, validation, and testing
  - **Version:** 0.9.1 (2015 release)
  - **License:** CC-BY-4.0 for language model data; audiobooks in public domain
  - **Related papers:** Panayotov et al., ICASSP 2015

- **Noise augmentation sources:**
  - **MUSAN:** Music, speech, and noise corpora (CC-0 license)
  - **ESC-50:** Environmental sound classification (CC-BY 4.0)
  - **Synthetic noise:** White, pink, and brown noise generated at target SNR levels

- **Overall project goal:** Build a noise-robust ASR system with word-level error detection and a transcription assistant UI

## Collection and composition

### LibriSpeech composition

| Subset | Hours | Utterances | Speakers | Characteristics |
| --- | --- | --- | --- | --- |
| `train-clean-100` | ~100h | ~28k | ~857 | Clean, read speech |
| `train-clean-360` | ~360h | ~101k | ~2.5k | Clean, read speech |
| `train-other-500` | ~500h | ~136k | ~6.3k | Clean, varied speakers/conditions |
| `dev-clean` | ~2.5h | ~7.6k | ~173 | Clean, used for validation |
| `dev-other` | ~3.5h | ~10.3k | ~233 | Clean, varied conditions |
| `test-clean` | ~5h | ~1.5k | ~357 | Clean, held-out evaluation |
| `test-other` | ~5h | ~1.5k | ~357 | Clean, held-out evaluation |

### Noise augmentation splits

- **Training noise:** White noise, pink noise (SNR: 0, 5, 10, 20 dB) — **seen during training**
- **Test noise:** Brown noise (SNR: 0, 5, 10, 20 dB) + ESC-50 real environmental — **held-out, never seen during training**
- **Clean baseline:** No noise added — **baseline comparison**

### Data splits strategy

- **Train:** `train-clean-100` (initial) or `train-clean-360` + `train-other-500` (full 960h)
- **Validation:** `dev-clean` + `dev-other`
- **Test:** `test-clean` + `test-other` (evaluated at multiple SNR levels: clean, 20dB, 10dB, 5dB, 0dB)

## Quality and preprocessing

### Preprocessing steps applied

1. **Audio resampling:** All audio resampled from original rates (8/16/44.1/48 kHz) to **16 kHz** target
2. **Channel conversion:** Stereo audio averaged to **mono** channel
3. **Noise augmentation:** On-the-fly mixing at specified SNR levels during DataLoader iteration
4. **Transcript normalization:** Lowercasing, removal of non-alphanumeric characters (except spaces)
5. **CTC labeling:** Conversion to character-level labels with blank token for model input
6. **Silence trimming (optional):** 200ms threshold at head/tail of utterances

### Quality checks performed

- Verified all audio files are valid `.flac` or `.wav` format
- Confirmed consistent sample rates after resampling (all 16 kHz)
- Checked transcript-label alignment: each character maps to a valid audio frame
- Verified no duplicate audio IDs across train/valid/test splits
- Ensured training noise types are distinct from test noise types (no data leakage)

### Known limitations

- **Genre restriction:** Only read English speech (audiobooks); no spontaneous speech, no dialects/accents beyond General American
- **Channel restriction:** Single-channel (mono) recordings only
- **Vocabulary size:** Limited to wordpieces in wav2vec 2.0 vocab (~60 wordpieces + blank)
- **Noise types:** Only stationary and quasi-stationary noise tested; non-stationary babble noise is out of scope

## Ethics, privacy, and limitations

### Ethics statement

- **Public domain data:** LibriSpeech audiobooks are in the public domain; no copyright concerns
- **No individual harm:** No personally identifiable information in the dataset
- **Academic use:** Dataset used for research on noise-robust ASR, consistent with original publication intent
- **Reproducibility:** All data sources and versions documented for reproducibility

### Privacy

- **No personal data:** LibriSpeech contains public domain audiobook recordings; no user consent needed
- **Noise data:** MUSAN and ESC-50 are also public domain or CC-licensed
- **No audio uploads:** All processing occurs on local machine; no cloud services required
- **Data minimization:** Only necessary features (audio + transcript) kept; discarding raw waveforms after preprocessing

### Known limitations

1. **Genre narrowness:** Read speech from audiobooks does not represent full range of real-world speech styles
2. **Language restriction:** English only; multilingual capabilities not addressed
3. **Environmental scope:** Clean studio + controlled noise; highly complex real-world environments not covered
4. **Model scope:** wav2vec 2.0 base size; larger models or different architectures not evaluated

### Safety and social impact

- **Positive:** Enables reliable transcription of field recordings for journalism, research, and accessibility
- **Potential bias:** Models may perform better on clean speech or specific accents present in LibriSpeech
- **Misuse prevention:** Transcription assistant designed for verification, not for generating unverified transcripts
