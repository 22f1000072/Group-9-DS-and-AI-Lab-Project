# Problem Statement

## Approved Problem

**Project Title:** Noise-Robust Speech Recognition with Word-Level Error Detection

**Problem:** Automatic Speech Recognition (ASR) systems degrade noticeably on noisy recordings such as street interviews, market conversations, vehicle recordings, and field surveys. A bigger issue is that modern recognizers write fluent text even when they are wrong — a mis-heard number or name reads just as confidently as a correct one. These "silent errors" mean users cannot tell where mistakes are without listening to the entire recording again, which wipes out most of the time ASR was supposed to save.

This project studies how noise type and noise level affect a pretrained recognizer, trains a noise-robust version using noise-augmented finetuning, and builds a transcription assistant that flags words most likely to be wrong, links each flag to its exact position in the original audio, and lets users replay and correct only those parts.

## Scope

### In Scope
- English speech recognition using wav2vec 2.0 base model finetuned with CTC
- Noise-robust training using on-the-fly noise augmentation (white noise, pink noise, MUSAN noise corpora)
- Evaluation on held-out noise types (brown noise, ESC-50 real environmental sounds) at multiple SNR levels (clean, 20 dB, 10 dB, 5 dB, 0 dB)
- Word-level error detection using CTC posterior confidence scores (no extra model training)
- Word-level timestamps derived from CTC frame positions for audio replay
- A Gradio-based review interface for flagged words with audio replay and in-place correction
- Dataset preparation pipeline: download, preprocess, split, and evaluate LibriSpeech, MUSAN, and ESC-50

### Out of Scope
- Training a new ASR or enhancement model from scratch (only finetuning pretrained wavvec 2.0 base)
- Speech enhancement or denoising front ends
- Whisper or any second recognizer
- Babble noise and additional noise datasets beyond MUSAN and ESC-50
- Multi-channel recordings, overlapping speech, speaker diarization
- Punctuation and language-model decoding
- Real-time streaming, mobile apps, languages other than English
- Certified transcripts for legal or medical use

## Stakeholders

### Primary Stakeholders

1. **Journalists** — Need reliable quotes from street and event recordings with a quick way to verify them
2. **Researchers in qualitative work, social science, and linguistics** — Record interviews outside studios and need to know which transcript parts can be trusted
3. **Survey and field-data teams** (NGOs, public health, market research) — Handle many noisy recordings on small budgets

### Secondary Stakeholders

- **Professional transcribers and reviewers** — Can spend time on risky sections first rather than listening to everything
- **Accessibility and archive users** — Work with poor-quality recordings
- **ASR researchers** — Interested in noise-augmented training and word-level confidence under noise
- **Course instructors and TAs** — Evaluate the work as part of the DSAI Lab course

## Objectives and Success Criteria

| # | Objective | Measure | Target |
|---|-----------|---------|--------|
| O1 | Characterize ASR degradation under noise | Complete WER/CER grid and WER-vs-SNR plot | Identify SNR where each model breaks down |
| O2 | Show benefit of noise-aware training | Relative WER reduction (noise-trained vs clean-trained) | At least 15% relative WER reduction averaged over 0 and 5 dB, with clean-audio WER worsening by no more than 2 points absolute |
| O3 | Detect error-prone words | AUROC, precision, recall, ECE | AUROC ≥ 0.75, at least 70% of word errors captured within riskiest 30% of words, ECE ≤ 0.10 |
| O4 | Link flags to the right audio | Replay-window hit rate | At least 90% of 50 sampled flags have target word inside replay window (±1s) |
| O5 | Deliver working MVP | Upload, chunking, transcription, flags, replay, edit, export | RTF ≤ 1.0 for clips up to 2 minutes on CPU |
| O6 | Reproducibility | Results table reproducible from README | A teammate can reproduce results from documentation |

## Constraints and Assumptions

### Constraints
- **English speech only** — Single-channel recordings, offline processing
- **No training from scratch** — Only finetuning pretrained wav2vec 2.0 base; no larger models
- **No speech enhancement front end** — Noise-augmented training only, no denoising preprocessing
- **CPU-friendly** — Must run on CPU with RTF ≤ 1.0 for clips up to 2 minutes
- **Datasets are public domain or CC-licensed** — LibriSpeech (CC-BY-4.0), MUSAN (CC-BY-4.0 via OpenSLR), ESC-50 (CC-BY-4.0)

### Assumptions
- LibriSpeech read speech is representative enough for initial noise-robustness evaluation
- CTC frame posteriors provide a usable confidence signal for word-level error detection
- 100 hours of training data (train-clean-100) is sufficient for initial experiments; full 960h can be used for final results
- On-the-fly noise augmentation during DataLoader iteration avoids storing augmented datasets
- Training noise (white/pink + MUSAN) is distinct from test noise (brown + ESC-50) to prevent data leakage

## Dataset Summary

| Dataset | Source | Size | Role |
|---------|--------|------|------|
| LibriSpeech | https://huggingface.co/datasets/openslr/librispeech_asr | 960h (train/dev/test) | Primary training + validation + test data |
| MUSAN | https://www.openslr.org/17/ | ~20h (~33k files) | Noise augmentation for training |
| ESC-50 | https://github.com/karolpiczak/ESC-50 | 2,000 files (50 classes) | Held-out test noise evaluation |

## References

1. Panayotov, V., et al. (2015). "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books." ICASSP.
2. Baevski, A., et al. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." NeurIPS.
3. Likhomanenko, T., et al. (2021). "Rethinking Evaluation in ASR: Are Our Models Robust Enough?" Interspeech.
4. Snyder, D., et al. (2015). "MUSAN: A Music, Speech, and Noise Corpus." arXiv:1510.08484.
5. Piczak, K. J. (2015). "ESC: Dataset for Environmental Sound Classification." ACM Multimedia.
