**Noise-Aware Speech Transcription**

**with Word-Level Verification**

*When noise speaks louder than words, can AI still understand us?*

**GROUP 9 Milestone 1**

**Course:** DSAI Labs

**Team:** Group 9

**Term:** October 2026

# 1. Problem Definition

**Problem statement:** Speech recognition (ASR) gets noticeably worse on noisy recordings, the kind people actually make in street interviews, markets, vehicles and field surveys. A bigger issue is that modern recognizers write fluent text even when they have it wrong. A mis-heard number or name reads just as confidently as a correct one, so nobody can tell where the mistakes are. The only safe option is to listen to the whole recording again, and that wipes out most of the time ASR was supposed to save. This project looks at how different noise types and levels affect a pretrained recognizer, trains a version that copes better with noise, and builds a transcription assistant on top of it. The assistant marks the words most likely to be wrong, links each mark to the exact spot in the original audio, and lets the user replay and fix only those parts.

## 1.1 The product: a transcription assistant

| **Element**      | **Description**                                                                                                          |
|------------------|--------------------------------------------------------------------------------------------------------------------------|
| Input            | A noisy audio recording (WAV or MP3), a few minutes long                                                                 |
| Core model       | A wav2vec 2.0 (base) model fine-tuned with a CTC head and noise augmentation                                             |
| Output           | A transcript with risky words highlighted, word-level timestamps, a short reliability summary, and export to TXT or JSON |
| Core interaction | Click a highlighted word, hear about 2 seconds of the **original** audio around it, and correct the text in place        |
| Key promise      | The user checks a small share of the words instead of the whole recording and still catches most of the errors           |

**Example.** A journalist records a market interview in which someone says *"prices went up fifteen percent."* In the noise, the model writes *"fifty percent"* and gives no warning. The assistant highlights *"fifty"* because the model was unsure about that word. The journalist clicks it, listens to the original audio and fixes it in a few seconds.

## 1.2 Stakeholders

**Primary stakeholders**

- **Journalists** need reliable quotes from street and event recordings and a quick way to check them.

- **Researchers in qualitative work, social science and linguistics** often record interviews outside a studio and need to know which parts of a transcript can be trusted.

- **Survey and field-data teams** (NGOs, public health, market research) handle many noisy recordings on small budgets.

**Secondary stakeholders**

- Professional transcribers and reviewers, who can spend their time on the risky sections first.

- Accessibility and archive users who work with poor-quality recordings.

- ASR researchers interested in noise-augmented training and word-level confidence under noise.

- Course instructors and TAs who will evaluate the work.

## 1.3 Scope

**In scope**

- English speech, single-channel recordings, offline processing.

- Fine-tuning wav2vec 2.0 base with and without noise augmentation, and testing on LibriSpeech test-clean and test-other at clean, 20, 10, 5 and 0 dB using noise the model never saw in training.

- Word-level confidence and timestamps taken from the CTC output of the same model, with no extra model training.

- A calibrated score for flagging likely errors, evaluated on held-out test noise.

- A web review interface built with Gradio, run locally or on a free CPU hosting tier.

**Explicitly out of scope**

- Training a new ASR or enhancement model from scratch, or using any larger model.

- Speech enhancement or denoising front ends. They are covered in the literature review and listed only as possible future work.

- Whisper or any second recognizer, babble noise and additional noise datasets (future work).

- Multi-channel recordings, overlapping speech, speaker diarization, punctuation and language-model decoding.

- Real-time streaming, mobile apps and languages other than English.

- Certified transcripts for legal or medical use.

# 2. Why the Problem Matters and Where It Arises

A transcript is usually the first step before analysis, reporting or a decision, and most of its cost is human time.

- **Journalism.** A wrong number or name in a quote can do real damage. Silent errors are the worst kind, because they look correct.

- **Research and fieldwork.** Interviews and surveys are rarely recorded in quiet rooms. Typing everything out by hand is slow and expensive, and an imperfect ASR transcript still needs a full manual check.

- **Public-interest data collection.** Surveys in markets, villages and streets produce a lot of noisy audio, and there is seldom money to transcribe it.

- **Trust in ASR output.** Koenecke et al. studied Whisper-style systems in 2023 and found that about 1% of transcriptions contained whole phrases that nobody had spoken. Without some signal of uncertainty, errors like these go unnoticed.

There are several reasons why the task is hard.

- Noise types differ a lot. A steady fan hum is fairly easy to ignore. Babble, meaning other people talking, overlaps with the target speech in both frequency and rhythm, so it is much harder.

- Quiet consonants get masked first, which makes similar words easy to confuse ("fifteen" and "fifty"). The recognizer then falls back on whatever sounds plausible.

- People raise their voice and change how they speak when it is noisy (the Lombard effect), so real noisy speech is different from clean speech with noise added afterwards.

- Pretrained recognizers are often trained on clean read speech and lose accuracy when conditions change \[8\].

# 3. How the Problem Is Currently Addressed

## 3.1 Early approaches

The first ASR systems combined hidden Markov models with Gaussian mixture acoustic models, a pronunciation dictionary and an n-gram language model. Each part was designed by hand. Noise was usually dealt with by signal-processing front ends such as spectral subtraction or by features that adapt to noise. The Aurora framework made evaluation more consistent by adding noise to clean speech at fixed SNRs, and we follow the same idea. These pipelines were hard to tune, and they got worse whenever the noise at test time differed from the noise assumed when they were designed.

## 3.2 Deep-learning ASR

- **CTC (Connectionist Temporal Classification)** lets a network predict one character or a blank symbol for every audio frame. Repeats are then merged and blanks removed to give the text, so no frame-level alignment is needed. Because each frame is predicted on its own, mistakes tend to be phonetic and look odd. The model also gives a probability and a time for every character it emits, which is what we use later.

- **wav2vec 2.0 and HuBERT ** are pretrained on unlabeled speech and then fine-tuned on a small amount of labeled data with CTC. For the Base model, wav2vec 2.0 reports (without a language model) about 24.5 / 29.7% WER on LibriSpeech test-clean / test-other with 1 hour of labeled data, 11.1 / 17.6% with 10 hours and 6.1 / 13.3% with 100 hours. These numbers are also reproduced in. With a 4-gram language model the 100-hour Base result drops to about 3.4 / 8.0%. WavLM  builds on the same recipe and adds denoising-style pretraining on mixed and noisy speech.

- **Whisper ** is an encoder-decoder model trained on 680,000 hours of weakly supervised, multilingual web audio. Whisper large-v2 reaches about 2.7% WER on LibriSpeech test-clean, close to a supervised wav2vec 2.0 Large model. Its real advantage shows on other datasets, where the authors report around 55% lower error on average across 12 sets, which points to much better robustness outside the training distribution. It decodes one token at a time and produces fluent text, but this can also lead to hallucinated text, and it does not give calibrated word-level confidence. WhisperX  adds word-level timestamps by forced alignment with a phoneme model, but it does nothing about reliability.

- **Hugging Face Transformers ** provides Wav2Vec2ForCTC, which we will fine-tune for this study.

## 3.3 Robustness through noise-aware training

Training on noise-augmented data is a common and inexpensive way to make a model more robust. Likhomanenko et al found that models trained on one dataset often transfer poorly to others, and that adding reverberation and noise during training improves generalization across domains. SpecAugment showed that masking blocks of time and frequency in the spectrogram during training also helps. Speech Robust Bench collects many kinds of corruption in one benchmark and reports, among other things, that larger models tend to be more robust than smaller ones. The Aurora tradition fixes the noise type and SNR so that results can be compared, and later work points out that the noise used for testing should be different from the noise used for training, otherwise robustness looks better than it is . For a student project, adding noise on the fly during fine-tuning is realistic, while switching to a much larger model is not. That is why this study asks how far noise augmentation alone can take a base-size model.

## 3.4 Speech enhancement

| **Approach**                | **Examples**                                                      | **Strengths**                        | **Weaknesses**                                                       |
|-----------------------------|-------------------------------------------------------------------|--------------------------------------|----------------------------------------------------------------------|
| Classical signal processing | Spectral subtraction , spectral gating (noisereduce) | Fast, needs no training, runs on CPU | Weak on non-stationary noise such as babble; can add "musical noise" |
| Lightweight neural          | RNNoise , DeepFilterNet                               | Small and fast enough for real time  | Quality depends on noise type; may remove parts of the speech        |
| Waveform-domain neural      | Demucs denoiser                                            | Good perceptual quality              | Heavier compute; tied to the noise it was trained on                 |

Enhancement is normally judged with perceptual metrics (PESQ, STOI, SI-SDR) and not with recognition accuracy. Iwamoto et al. \[10\] split enhancement errors into a noise part and an artifact part. They found that the artifact part, which is distortion introduced by the enhancement itself, can hurt ASR more than the leftover noise does. This is why a denoising front end does not always help a recognizer.

## 3.5 Confidence estimation and calibration

Word-level confidence in speech recognition has been studied for a long time. Jiang surveys the main options: posterior probabilities, acoustic and language-model scores, and learned classifiers that decide whether a recognized word is right. For CTC models the frame posteriors already carry a useful signal. The probability of each emitted character, combined over a word, gives a cheap confidence score that needs no extra training. Attention-based sequence-to-sequence models are less convenient, since their output probabilities say less, and Li et al. train a separate module to estimate confidence for them. Raw probabilities from neural networks are also known to be overconfident. Guo et al. show that modern networks are poorly calibrated and that simple fixes on held-out data, such as temperature scaling or logistic and isotonic fits, repair much of the problem. Confidence scores have been used to improve accuracy as well, for example in ROVER, which combines several recognizers by voting with word confidences. The last ingredient is timing. Word boundaries can be read from CTC frame positions or obtained by forced alignment, as in the Montreal Forced Aligner. Our design puts these three ideas together (CTC confidence, post-hoc calibration and word timing) in one review tool aimed at noisy recordings.

## 3.6 Existing transcription tools

| **Tool**                          | **Type**                             | **Strength**                                        | **Limitation for our problem**                                                                                                                   |
|-----------------------------------|--------------------------------------|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| Otter.ai, Trint, Descript, Rev    | Commercial transcription and editing | Polished editing, collaboration, audio-text linking | Closed models; little information on noise robustness; limited or no uncertainty highlighting; subscription cost; audio is uploaded to the cloud |
| Whisper, faster-whisper, WhisperX | Open source                          | Strong accuracy, word timestamps                    | No review interface; no calibrated word-level uncertainty; no analysis of noise effects                                                          |
| Google, Azure, AWS speech APIs    | Cloud services                       | Scalable; some word-level confidence                | Black box; per-hour cost; audio leaves the user's machine; confidence not tuned for noise                                                        |
| Audacity, ELAN, oTranscribe       | Manual tools                         | Fine manual control                                 | No automatic way to find where to look                                                                                                           |

## 3.7 Benchmarks and datasets

| **Dataset**                               | **What it is**                                                                                    | **Role in our project**                                                      |
|-------------------------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| LibriSpeech                         | About 1,000 hours of read English audiobooks with transcripts; standard clean and other test sets | Clean speech with ground truth for training, validation and testing          |
| Environmental sound clips (ESC-50 ) | 2,000 five-second environmental recordings in 50 classes                                          | Real noise, split 80% for training and 20% for testing                       |
| Synthetic noise                           | White, pink and brown noise generated at a target SNR                                             | White and pink for training, brown for testing (controlled stationary noise) |
| MUSAN , DEMAND                | Music, speech and noise corpora; recordings of real environments                                  | Possible future noise and babble sources (not used)                          |
| CHiME-6                            | Real, noisy, multi-speaker dinner-party conversations                                             | Reference for how hard real conditions are (not used directly)               |

## 3.8 Common practice and gaps in reporting

Many studies report one WER on a clean benchmark, or on a single noise type. Likhomanenko et al. argue that testing on a single dataset overstates robustness. Papers on enhancement report perceptual scores more often than recognition accuracy, and papers on confidence estimation rarely check **how much human review effort** the scores actually save. That last question is the one a user of a transcription tool cares about most.

# 4. Metrics and How Success Is Defined

| **Purpose**          | **Metric**                                                                 | **Notes**                                                                                                     |
|----------------------|----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| ASR accuracy         | **Word Error Rate (WER)** = (S + D + I) / N, and CER                       | Computed at corpus level (total errors over total reference words), with one text normalizer for every system |
| Robustness           | WER versus SNR curve per noise type; breakdown SNR                         | SNR levels: clean, 20, 10, 5, 0 dB                                                                            |
| Training benefit     | Relative WER reduction (noise-trained vs clean-trained)                    | (WER_clean_trained - WER_noise_trained) / WER_clean_trained                                                   |
| Error detection      | AUROC, precision, recall, F1; positive class = incorrectly recognized word | Word labels come from aligning the hypothesis to the reference                                                |
| Calibration          | Expected Calibration Error, Brier score, reliability diagram         | Calibrate on validation data, report on test data                                                             |
| Review efficiency    | **Error recall versus fraction of words reviewed**                         | The metric that matches the product goal                                                                      |
| Timestamp usefulness | Replay-window hit rate on a sample of flags                                | Manual check of 50 flags                                                                                      |
| Efficiency           | Real-time factor (processing time / audio duration)                        | Reported on CPU and on a free GPU                                                                             |

WER counts every error the same, so "a" for "the" costs as much as "fifty" for "fifteen". To make up for this we also report which kinds of errors the flags catch, such as numbers and names, in an error-analysis section.

**Reference points.** Published wav2vec 2.0 Base results use far more fine-tuning than a student project can afford, and often a language model. Whisper large-v2 reaches about 2.7% WER on test-clean. Our own WER will be higher because we use a small fine-tuning set, a base-size model and greedy decoding without a language model. The point of our experiments is the **relative comparison** between clean-trained and noise-trained models and the **shape of the degradation as SNR drops**. We are not trying to match published absolute numbers.

# 5. Where Current Solutions Fall Short, and the Opportunity

1.  **No visible uncertainty.** Fluent recognizers hide their mistakes, and commercial tools give little or no reliable sign of where a transcript is likely to be wrong. Users end up listening to everything.

2.  **Confidence is rarely calibrated or tested on noise.** Raw scores are overconfident, and few papers measure how much review effort they save.

3.  **Robustness is under-reported.** Results are usually a single number on clean data. Practitioners have no per-noise, per-SNR picture on **held-out** noise for a small model they can train themselves.

4.  **Enhancement is applied by default,** even though its artifacts can hurt recognition. We rely on noise-augmented training instead of a denoising front end.

5.  **Cost and privacy.** The strongest tools are either closed and paid, or large models that need a GPU. Cloud services also require uploading sensitive interviews.

6.  **No easy path back to the audio.** Most tools show text, and getting from a doubtful word to the exact moment in the recording takes effort.

**Opportunity.** A small, open pipeline that runs on a CPU can cover the part that matters for verification: recognition that copes with noise, word-level flags that are calibrated and tested on unseen noise, and a review interface tied to the **original** audio.

**How our approach differs**

| **Dimension**    | **Typical existing approach**    | **Our assistant**                                                                                     |
|------------------|----------------------------------|-------------------------------------------------------------------------------------------------------|
| Noise handling   | None, or unconditional denoising | Noise-augmented training evaluated on held-out noise                                                  |
| Uncertainty      | None, or raw confidence          | Calibrated CTC word confidence that needs no extra training, checked with AUROC, ECE and error recall |
| Output           | Flat transcript                  | Transcript with flagged words and timestamps linked to the original audio                             |
| Verification     | Listen to the whole recording    | Review only the flagged words with surrounding context                                                |
| Evaluation       | WER only                         | WER and CER, error recall versus review budget, and calibration                                       |
| Cost and privacy | Cloud or large GPU models        | Runs locally on CPU; open weights                                                                     |

# 6. Nature of the Contribution

Our contribution is mostly about robustness evaluation and workflow, and we do not propose a new architecture. It has three parts. First, a controlled and reproducible comparison of clean-trained and noise-trained wav2vec 2.0 across SNR levels on held-out noise. Second, a word-level error detector built on CTC posteriors that needs no extra training, is calibrated, and is judged by how much review effort it saves. Third, a working review tool that puts these results in front of a user. We are not claiming state-of-the-art WER.

# 7. Evidence That Will Demonstrate the Approach Works

- **Quantitative:** the WER and CER grid and WER-vs-SNR plots for both models, the relative WER reduction from noise training, and AUROC, precision, recall and ECE for the word-level risk score.

- **Product metric:** a curve showing *"if the user reviews X% of the words, they catch Y% of the errors."*

- **Generalization:** every result uses noise that was not seen in training (brown noise and the held-out real clips).

- **Baselines for flagging:** random flagging at the same review budget, and a simple rule on uncalibrated raw confidence. The calibrated score has to beat both.

- **Qualitative:** side-by-side examples of reference, prediction and flags, including failures such as numbers, short words and heavy noise.

- **Timestamps:** replay-window hit rate on 50 sampled flags.

- **App check:** a short recorded demo of upload, flags, replay and editing on a few clips, including a noisy recording we make ourselves.

- **Reproducibility:** fixed seeds and splits, one text normalization, public code and published weights.

**Initial targets**

| **\#** | **Objective**                            | **Target**                                                                                                                       |
|--------|------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| O1     | Characterize ASR degradation under noise | Complete WER/CER grid and WER-vs-SNR plot; identify the SNR at which each model breaks down                                      |
| O2     | Show the benefit of noise-aware training | At least 15% relative WER reduction averaged over 0 and 5 dB, with clean-audio WER worsening by no more than 2 points absolute   |
| O3     | Detect error-prone words                 | AUROC of at least 0.75; at least 70% of word errors captured within the riskiest 30% of words; ECE of at most 0.10               |
| O4     | Link flags to the right audio            | At least 90% of 50 sampled flags have the target word inside the replay window (word time plus or minus 1 s)                     |
| O5     | Deliver a working MVP                    | Upload, chunking, transcription, flags, replay, edit and export all working; RTF of at most 1.0 for clips up to 2 minutes on CPU |
| O6     | Reproducibility                          | A teammate can reproduce the results table from the README                                                                       |

# References

1.  Graves, A., Fernández, S., Gomez, F., Schmidhuber, J. Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks. ICML, 2006.

2.  Baevski, A., Zhou, H., Mohamed, A., Auli, M. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. NeurIPS, 2020. arXiv:2006.11477.

3.  Hsu, W.-N., Bolte, B., Tsai, Y.-H. H., Lakhotia, K., Salakhutdinov, R., Mohamed, A. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2021. arXiv:2106.07447.

4. Chen, S. et al. WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. IEEE Journal of Selected Topics in Signal Processing, 16(6), 2022. arXiv:2110.13900.

5. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I. Robust Speech Recognition via Large-Scale Weak Supervision. ICML, 2023. arXiv:2212.04356.

6. Bain, M., Huh, J., Han, T., Zisserman, A. WhisperX: Time-Accurate Speech Transcription of Long-Form Audio. Interspeech, 2023. arXiv:2303.00747.

6. Panayotov, V., Chen, G., Povey, D., Khudanpur, S. Librispeech: An ASR Corpus Based on Public Domain Audio Books. ICASSP, 2015.

7. Likhomanenko, T., Xu, Q., Pratap, V., Tomasello, P., Kahn, J., Avidov, G., Collobert, R., Synnaeve, G. Rethinking Evaluation in ASR: Are Our Models Robust Enough? Interspeech, 2021. arXiv:2010.11745.

8. Shah, M. A. et al. Speech Robust Bench: A Robustness Benchmark For Speech Recognition. arXiv:2403.07937, 2024.

9. Iwamoto, K., Ochiai, T., Delcroix, M., Ikeshita, R., Sato, H., Araki, S., Katagiri, S. How Bad Are Artifacts?: Analyzing the Impact of Speech Enhancement Errors on ASR. Interspeech, 2022. arXiv:2201.06685.

10. Boll, S. F. Suppression of Acoustic Noise in Speech Using Spectral Subtraction. IEEE Transactions on Acoustics, Speech, and Signal Processing, 27(2), 1979.

11. Piczak, K. J. ESC: Dataset for Environmental Sound Classification. ACM Multimedia, 2015.

12. Koenecke, A., Choi, A. S. G., Mei, K., Schellmann, H., Sloane, M. Careless Whisper: Speech-to-Text Hallucination Harms. ACM FAccT, 2024. DOI: 10.1145/3630106.3658996. arXiv:2402.08021.

13. Hirsch, H.-G., Pearce, D. The Aurora Experimental Framework for the Performance Evaluation of Speech Recognition Systems under Noisy Conditions. ISCA ITRW ASR2000, 2000.

14. Park, D. S. et al. SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. Interspeech, 2019. arXiv:1904.08779.

15. Valin, J.-M. A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement. IEEE MMSP, 2018. arXiv:1709.08243.

16. Schröter, H., Escalante-B., A. N., Rosenkranz, T., Maier, A. DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio Based on Deep Filtering. ICASSP, 2022. arXiv:2110.05588.

17. Défossez, A., Synnaeve, G., Adi, Y. Real Time Speech Enhancement in the Waveform Domain. Interspeech, 2020. arXiv:2006.12847.

18. Sainburg, T. noisereduce: Noise reduction in Python using spectral gating. Software, https://github.com/timsainb/noisereduce.

19. Wolf, T. et al. Transformers: State-of-the-Art Natural Language Processing. EMNLP (System Demonstrations), 2020.

20. Jiang, H. Confidence Measures for Speech Recognition: A Survey. Speech Communication, 45(4), 2005.

21. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. On Calibration of Modern Neural Networks. ICML, 2017. arXiv:1706.04599.

22. Li, Q. et al. Confidence Estimation for Attention-Based Sequence-to-Sequence Models for Speech Recognition. ICASSP, 2021. arXiv:2010.11428.

23. Fiscus, J. G. A Post-Processing System to Yield Reduced Word Error Rates: Recognizer Output Voting Error Reduction (ROVER). IEEE ASRU Workshop, 1997.

24. McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., Sonderegger, M. Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi. Interspeech, 2017.

25. Snyder, D., Chen, G., Povey, D. MUSAN: A Music, Speech, and Noise Corpus. arXiv:1510.08484, 2015.

26. Thiemann, J., Ito, N., Vincent, E. The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND). Proceedings of Meetings on Acoustics, 2013.

27. Watanabe, S. et al. CHiME-6 Challenge: Tackling Multispeaker Speech Recognition for Unsegmented Recordings. CHiME Workshop, 2020. arXiv:2004.09249.

28. Abid, A., Abdalla, A., Abid, A., Khan, D., Alfozan, A., Zou, J. Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild. arXiv:1906.02569, 2019.

**Tools used:** Hugging Face Transformers and Datasets, evaluate and JiWER (WER and CER), librosa, PyTorch, Gradio \[29\], and a free cloud GPU (Kaggle or Colab).

## Work Tracking

Individual contributions are recorded in WORKLOG.md and updated with every milestone submission.
