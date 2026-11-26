# Worklog

Record contributions at each milestone or project week.

| Date / week | Member | Contribution | Evidence / link |
| --- | --- | --- | --- |
| 15/10/2026 | Gulshan Kumar | Problem statement submission | milestones/M1_problem_literature/ |
| 16/10/2026 | Parashmani Datta | Literature and reference collection | milestones/M1_problem_literature/report.md |
| 16/10/2026 | Subhojit Saha | Literature compilation | milestones/M1_problem_literature/report.md |
| 16/10–20/10/2026 | All members | Literature review | milestones/M1_problem_literature/ |
| 19/10/2026 | Gulshan Kumar | Repository creation | Git repository initialized |
| 20/10/2026 | Gulshan Kumar | Draft formation | README.md, project structure |
| 20/10/2026 | All members | Discussion | Team meeting |
| 21/10/2026 | Gulshan Kumar, Parashmani Datta | Draft finalisation | problem_statement/problem_statement.md |
| 22/10/2026 | Gulshan Kumar | Milestone 1 submission | milestones/M1_problem_literature/Milestone1_Group9_Submit.md |
| 24/10/2026 | Gulshan Kumar | Data download pipeline implemented | scripts/download_data.py |
| 25/10/2026 | Parashmani Datta | Preprocessing pipeline skeleton | scripts/preprocess.py |
| 25/10/2026 | Subhojit Saha | Data card documentation | docs/data_card.md |
| 26/10/2026 | Subhojit Saha | Reproducibility documentation | docs/reproducibility.md |
| 27/10/2026 | All members | Data structure validation | data/README.md |
| 28/10/2026 | All members | Pipeline testing & verification | scripts/eda_analysis.py, scripts/preprocess.py |
| 29/10/2026 | Gulshan Kumar | Milestone 2 dataset identification & sources | milestones/M2_dataset/milestone2_dataset_submission.md Section 2 |
| 29/10/2026 | Parashmani Datta | Milestone 2 dataset quality & EDA analysis | milestones/M2_dataset/milestone2_dataset_submission.md Section 2.5-2.6, scripts/eda_analysis.py |
| 29/10/2026 | Subhojit Saha | Milestone 2 splits, synthetic data & reproducibility | milestones/M2_dataset/milestone2_dataset_submission.md Section 3-7 |
| 16/11–17/11/2026 | Gulshan Kumar | Full-data preprocessing (QC of 292k utterances, speaker-disjoint splits, MUSAN/ESC-50 noise banks, fixed val/test noise conditions) | scripts/preprocess.py, data/processed/manifests/qc_summary.json |
| 17/11/2026 | Gulshan Kumar | EDA on the complete data (LibriSpeech 982 h, MUSAN, ESC-50) | scripts/eda_full.py, experiments/eda/, notebooks/M4_01_eda_complete_data.ipynb |
| 17/11/2026 | Gulshan Kumar | Training of Model B (noise-augmented) and Model A (clean), 36k steps each on L40S GPU; test evaluation of A, B and reference model | scripts/train.py, scripts/evaluate.py, configs/train_*.yaml, experiments/runs/, experiments/results/ |
| 17/11–18/11/2026 | Gulshan Kumar | Hyperparameter sweep (7 runs × 8k steps: learning rate, masking, batch size, clean fraction, SNR range) | scripts/pipeline/run_sweep.sh, experiments/runs/hp_*/, milestones/M4_training/tables/hparam_sweep.csv |
| 18/11/2026 | Gulshan Kumar | Milestone 4 report, figures and presentation notebooks | milestones/M4_training/, notebooks/M4_0*.ipynb |
| 18/11–19/11/2026 | Parashmani Datta | Milestone 4 report review and code review (training, evaluation and sweep scripts) | milestones/M4_training/Milestone4_Group9_Training_Report.md, scripts/train.py, scripts/evaluate.py |
| 18/11–19/11/2026 | Subhojit Saha | Milestone 4 report review and code review (training, evaluation and sweep scripts) | milestones/M4_training/Milestone4_Group9_Training_Report.md, scripts/train.py, scripts/evaluate.py |
| 19/11/2026 | All members | Review and sign-off of Milestone 4 report | milestones/M4_training/Milestone4_Group9_Training_Report.md |
| 18/11–22/11/2026 | Gulshan Kumar | Milestone 5 test-grid evaluation: LM decoding, ROVER, word-error detector, replay-window check, error analysis and first report draft | scripts/m5_analysis.py, scripts/replay_window.py, milestones/M5_evaluation/ |
| 22/11–25/11/2026 | Gulshan Kumar | Single evaluation entry point (shared code, script and notebook giving identical tables); WER/CER grid, bootstrap CIs, breakdown SNR, review effort, CPU/GPU speed; Hugging Face reproducibility check | src/group9_ds_ai_lab/evaluation/, scripts/m5_evaluate.py, notebooks/M5_evaluation.ipynb, milestones/M5_evaluation/tables/g*.csv |
| 24/11–26/11/2026 | Gulshan Kumar | Indian-accent evaluation on Svarah (separate script and report), added as Section 8 of the M5 report | scripts/svarah_evaluate.py, milestones/M5_evaluation/indian_accent/ |
| 26/11/2026 | Gulshan Kumar | Milestone 5 report: literature comparison, objectives table, contributions | milestones/M5_evaluation/Milestone5_Group9_Evaluation_Report.md |
| 26/11/2026 | Parashmani Datta | O4 listening check of the 50 sampled replay clips (46/50 heard); Milestone 5 report review and code review | milestones/M5_evaluation/replay_check/sample_flags_PD.csv, src/group9_ds_ai_lab/evaluation/, scripts/m5_evaluate.py, scripts/svarah_evaluate.py |
| 26/11/2026 | Subhojit Saha | Milestone 5 report review and code review | milestones/M5_evaluation/Milestone5_Group9_Evaluation_Report.md, src/group9_ds_ai_lab/evaluation/, scripts/m5_evaluate.py, scripts/svarah_evaluate.py |
| 26/11/2026 | All members | Review and sign-off of Milestone 5 report | milestones/M5_evaluation/Milestone5_Group9_Evaluation_Report.md |

---
**Acknowledgement:**
| Name | Roll No. | Signature |
|------|----------|-----------|
| Gulshan Kumar | 22f1000072 | G.K. |
| Parashmani Datta | 21f2000987 | P.D. |
| Subhojit Saha | 21f1003945 | S.S. |

**Acknowledgement:**

All milestone submissions have been completed jointly by Group 9 members listed above.
This work is part of the DSAI Lab course (October 2026 term).
Milestone 1 submission was completed on October 22, 2026.
Milestone 2 submission is completed on October 29, 2026.
Milestone 4 submission is completed on November 19, 2026.
Milestone 5 submission is completed on November 26, 2026.