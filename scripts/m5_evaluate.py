"""Milestone 5: run the complete evaluation from the command line (the notebook notebooks/M5_evaluation.ipynb
runs exactly the same functions from group9_ds_ai_lab.evaluation.m5, so both give the same tables).

Usage:
  python scripts/m5_evaluate.py                                    # all stages, models from the server
  python scripts/m5_evaluate.py --source hf                        # models downloaded from Hugging Face
  python scripts/m5_evaluate.py --stages g1_grid g5_review         # selected stages
  python scripts/m5_evaluate.py --recompute greedy                 # also re-decode the full test grid (GPU) for all models
  python scripts/m5_evaluate.py --out /tmp/m5_check                # write somewhere else (e.g. to compare two runs)

Stages: core g1_grid g2_break g3_acc g4_boot g5_review g6_catch g8_audio g9_listen g7_speed g12_repro g11_obj
Outputs: <out>/tables/*.csv, <out>/figures/*.png, <out>/audio_examples/*.wav (default out: milestones/M5_evaluation)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.evaluation import m5  # noqa: E402
from group9_ds_ai_lab.utils.paths import REPO, setup_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "milestones" / "M5_evaluation"))
    ap.add_argument("--source", choices=["local", "hf"], default="local", help="where the models are loaded from")
    ap.add_argument("--recompute", choices=["none", "greedy"], default="none",
                    help="greedy: re-decode the full test grid on the GPU for every model (about 4 min per model)")
    ap.add_argument("--stages", nargs="*", default=None, choices=m5.STAGES)
    args = ap.parse_args()
    setup_env()
    m5.run_all(args.out, args.stages, args.source, args.recompute)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
