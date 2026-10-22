# Group-9-DS-and-AI-Lab-Project

To build a transcription assistant that flags the words most likely to be wrong, links each flag to its exact position in the original audio, and lets users replay and correct only those parts.
Automatic Speech Recognition (ASR) loses accuracy on noisy recordings such as street interviews, markets, vehicles and field surveys. Modern recognizers also produce fluent text even when they are wrong, so errors are silent: a wrong number or name looks as confident as a correct one. Users therefore re-listen to the whole recording to verify it, which removes most of the time ASR was meant to save. This project studies how noise type and level affect a pretrained recognizer, trains a noise-robust version, and builds , a transcription assistant that flags the words most likely to be wrong, links each flag to its exact position in the original audio, and lets users replay and correct only those parts.

## Team

Gulshan Kumar 22f1000072

Parashmani Datta 21f2000987

Subhajit Saha 21f1003945

Dabgar Milav JayeshKumar 21f1005510

## Project

- Problem statement: [problem_statement/problem_statement.md](problem_statement/problem_statement.md)
- Final report: [final/final_report.md](final/final_report.md)
- Demo: [final/demo_link.md](final/demo_link.md)

## Setup

Requires Python 3.10 or later. Create and activate a virtual environment, then install the pinned project dependencies as they are added:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in only the values required by the project. Do not commit `.env` or private credentials.

## Run

TODO: Document the project's end-to-end run command after its pipeline is implemented.

## Repository guide

- `src/group9_ds_ai_lab/`: reusable project code
- `scripts/`: command-line entry points
- `configs/`: data, model, training, and evaluation configuration
- `data/`: local datasets; only `data/README.md` is tracked
- `milestones/`: milestone reports and supporting assets
- `experiments/`: run logs and metrics; checkpoints are not tracked
- `notebooks/`: exploration and prototyping
- `deployment/`: demo application and deployment instructions
- `docs/`: architecture, data, model, and reproducibility documentation

## Milestones

See [milestones/](milestones/) for submissions. Add links to the final report and presentation here when available.
