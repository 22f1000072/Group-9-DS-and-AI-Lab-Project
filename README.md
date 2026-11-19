# Group-9-DS-and-AI-Lab-Project
Noise-Robust Speech Recognition and Error Detection
An Intelligent Audio Transcription and Verification Assistant
To build a transcription assistant that flags the words most likely to be wrong, links each flag to its exact position in the original audio, and lets users replay and correct only those parts.
Automatic Speech Recognition (ASR) loses accuracy on noisy recordings such as street interviews, markets, vehicles and field surveys. Modern recognizers also produce fluent text even when they are wrong, so errors are silent: a wrong number or name looks as confident as a correct one. Users therefore re-listen to the whole recording to verify it, which removes most of the time ASR was meant to save. This project studies how noise type and level affect a pretrained recognizer, trains a noise-robust version, and builds , a transcription assistant that flags the words most likely to be wrong, links each flag to its exact position in the original audio, and lets users replay and correct only those parts.

## Team

Gulshan Kumar 22f1000072

Parashmani Datta 21f2000987

Subhajit Saha 21f1003945

## Project

- Problem statement: [problem_statement/problem_statement.md](problem_statement/problem_statement.md)
- Final report: [final/final_report.md](final/final_report.md)
- Demo: [final/demo_link.md](final/demo_link.md)

## Setup

Tested with Python 3.14. Create and activate a virtual environment (ideally outside the repository), then install the pinned dependencies in two steps:

```bash
python -m venv ../venvs/group9-asr
source ../venvs/group9-asr/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps pyctcdecode==0.5.0   # see the note at the top of requirements.txt
```

Copy `.env.example` to `.env` and fill in only the values required by the project. Do not commit `.env` or private credentials.

## Run

All long jobs are resumable (finished stages are skipped, interrupted training resumes); run them inside `tmux`/`screen`.

```bash
G9_VENV=../venvs/group9-asr source scripts/env.sh       # activates the venv, sets PYTHONPATH and the HF cache (data/hf_home)
python scripts/download_data2.py --mode download-full --datasets librispeech musan esc50 --models
python scripts/preprocess.py                             # QC, manifests, noise banks -> data/processed/
bash scripts/pipeline/run_main.sh                        # reference model + Model B + Model A: train and evaluate
bash scripts/pipeline/run_sweep.sh                       # hyper-parameter sweep (7 x 8k steps)
python scripts/preprocess_m5.py                          # 460 h / 960 h manifests + room-impulse-response banks
bash scripts/pipeline/run_followup.sh                    # B460 -> B960_rev -> B960_rev_s2, LM decoding, error detector
python scripts/rover.py --runs B960_rev_s2_lm4 B960_rev_lm4 B460_lm4 --set test --name rover3_final --lam 0.5
```

Single steps, extra downloads (room impulse responses, 4-gram LM) and run times are listed in
[the Milestone 4 report, Section 6](milestones/M4_training/Milestone4_Group9_Training_Report.md).

## Trained models

Weights are not stored in git; all trained models are public on Hugging Face (load with `Wav2Vec2ForCTC.from_pretrained`):

| Model | Hugging Face | test-clean / test-other WER (4-gram LM) |
|---|---|---|
| **B960_rev** (best single model) | [/kumargulshan/wav2vec2-base-ls960-noise-robust](https://huggingface.co/kumargulshan/wav2vec2-base-ls960-noise-robust) | 2.97 / 7.01 % |
| B960_rev_s2 (seed 7) | [/kumargulshan/wav2vec2-base-ls960-noise-robust-seed7](https://huggingface.co/kumargulshan/wav2vec2-base-ls960-noise-robust-seed7) | 2.97 / 7.06 % |
| B460 | [/kumargulshan/wav2vec2-base-ls460-noise-aug](https://huggingface.co/kumargulshan/wav2vec2-base-ls460-noise-aug) | 2.97 / 7.58 % |
| Model B (100 h, noise-augmented) | [/kumargulshan/wav2vec2-base-ls100-noise-aug](https://huggingface.co/kumargulshan/wav2vec2-base-ls100-noise-aug) | 3.70 / 8.85 % |
| B2 (100 h, lr 1e-4) | [/kumargulshan/wav2vec2-base-ls100-noise-aug-lr1e-4](https://huggingface.co/kumargulshan/wav2vec2-base-ls100-noise-aug-lr1e-4) | 3.67 / 9.18 % |
| Model A (100 h, clean baseline) | [/kumargulshan/wav2vec2-base-ls100-clean](https://huggingface.co/kumargulshan/wav2vec2-base-ls100-clean) | 3.69 / 9.04 % |

Final system (ROVER of B960_rev_s2, B960_rev and B460 with the LM): **2.89 % / 6.92 %** on test-clean / test-other and
**11.42 %** on held-out ESC-50 noise at 0–5 dB SNR.

## Repository guide

- `src/group9_ds_ai_lab/`: reusable project code
- `scripts/`: command-line entry points; `scripts/pipeline/`: resumable run queues; `scripts/env.sh`: shell environment
- `configs/`: training configurations (`train_<run>.yaml`, one per trained model)
- `data/`: local datasets; only `data/README.md` is tracked
- `milestones/`: milestone reports and supporting assets
- `experiments/`: EDA, training curves (`runs/<run>/train_log.csv`, `eval_log.csv`), test metrics (`results/`); checkpoints and per-utterance predictions are not tracked
- `notebooks/`: presentation notebooks (`M4_01`–`M4_04`) and earlier exploration
- `deployment/`: demo application and deployment instructions
- `docs/`: architecture, data, model, and reproducibility documentation

## Milestones

| Milestone | Report |
|---|---|
| M1 – Problem definition and literature review | [Milestone1_Group9_Submit.md](milestones/M1_problem_literature/Milestone1_Group9_Submit.md) |
| M2 – Dataset preparation | [milestone2_dataset_submission.md](milestones/M2_dataset/milestone2_dataset_submission.md) |
| M3 – Model architecture | [Milestone 3 Report.md](<milestones/M3_architecture/Milestone 3 Report.md>) |
| M4 – Model training and experimentation | [Milestone4_Group9_Training_Report.md](milestones/M4_training/Milestone4_Group9_Training_Report.md) ([PDF](milestones/M4_training/Milestone4_Group9_Training_Report.pdf)) |
| M5 – Evaluation and analysis | in progress |
| M6 – Deployment | planned |
