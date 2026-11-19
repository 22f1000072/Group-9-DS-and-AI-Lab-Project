#!/usr/bin/env bash
# Full GPU pipeline, safe to re-run at any time (finished stages are skipped, interrupted training resumes):
#   1. reference eval of facebook/wav2vec2-base-960h on the test grid
#   2. train + eval Model B (noise-augmented)   <- main result first
#   3. train + eval Model A (clean baseline)
# Long-running: start it inside tmux/screen
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
LOG=experiments/logs; RUNS=experiments/runs; EVAL=experiments/results; mkdir -p $LOG
stamp() { echo "[$(date '+%F %T')] $*"; }

evaluate() {  # name model_path
  if [ -f "$EVAL/$1/metrics.json" ]; then stamp "eval $1 already done"; return; fi
  stamp "eval $1 start"; python scripts/evaluate.py --model "$2" --name "$1" >> "$LOG/eval_$1.log" 2>&1
  stamp "eval $1 exit $?"
}

train() {  # config name
  local st="$RUNS/$1/last/trainer_state.pt" max
  max=$(python -c "import yaml;print(yaml.safe_load(open('configs/train_$1.yaml'))['max_steps'])")
  if [ -f "$st" ]; then
    local step; step=$(python -c "import torch;print(torch.load('$st',weights_only=False)['step'])")
    if [ "$step" -ge "$max" ]; then stamp "train $1 already finished ($step steps)"; return; fi
    stamp "train $1 resume from step $step"
    python scripts/train.py --config configs/train_$1.yaml --resume >> "$LOG/train_$1.log" 2>&1
  else
    stamp "train $1 start"
    python scripts/train.py --config configs/train_$1.yaml >> "$LOG/train_$1.log" 2>&1
  fi
  stamp "train $1 exit $?"
}

evaluate ref_base960h facebook/wav2vec2-base-960h
train B_noisy && evaluate B_noisy "$RUNS/B_noisy/best"
train A_clean && evaluate A_clean "$RUNS/A_clean/best"
stamp "PIPELINE DONE"
