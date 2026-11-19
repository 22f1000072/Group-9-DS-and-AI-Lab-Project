#!/usr/bin/env bash
# Milestone 4 hyperparameter / regularisation sweep on the Model B recipe (8k steps each, dev-set selection only).
# Safe to re-run: finished runs are skipped, interrupted ones resume. Results: experiments/runs/hp_*/eval_log.csv
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
LOG=experiments/logs; RUNS=experiments/runs
stamp() { echo "[$(date '+%F %T')] $*"; }
STEPS=8000
COMMON="max_steps=$STEPS eval_every=2000 save_every=2000"
RUNS_DEF=(
  "hp_base|"
  "hp_lr1e-4|lr=1.0e-4"
  "hp_lr2e-5|lr=2.0e-5"
  "hp_mask_light|mask_time_prob=0.05 mask_feature_prob=0.0 layerdrop=0.05"
  "hp_bs320|max_batch_seconds=320"
  "hp_pclean0.5|p_clean=0.5"
  "hp_snr-5|snr_min=-5.0"
)
for def in "${RUNS_DEF[@]}"; do
  name=${def%%|*}; over=${def#*|}
  st="$RUNS/$name/last/trainer_state.pt"; extra=""
  if [ -f "$st" ]; then
    step=$(python -c "import torch;print(torch.load('$st',weights_only=False)['step'])")
    if [ "$step" -ge "$STEPS" ]; then stamp "$name already finished"; continue; fi
    extra="--resume"
  fi
  stamp "$name start $over $extra"
  python scripts/train.py --config configs/train_B_noisy.yaml $extra --set name=$name $COMMON $over >> "$LOG/train_$name.log" 2>&1
  stamp "$name exit $?"
done
stamp "SWEEP DONE"
