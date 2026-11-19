#!/usr/bin/env bash
# Follow-up runs (Section 5 of the M4 report): B460 -> B960_rev -> B960_rev_s2. Safe to re-run: finished steps are skipped.
#   1. (optional) wait for an already running B460 training (PID $1); train or resume B460
#   2. B460       : test + extended eval, dump logits, LM decode + error detector (CPU, background)
#   3. B960_rev   : continue from B460 on 960 h + noise/reverb/speed/telephone, 40k steps -> same evals
#   4. B960_rev_s2: same with seed 7 (second ensemble member)                               -> same evals
# Usage: bash scripts/pipeline/run_followup.sh [pid_of_running_B460_training]
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
LOG=experiments/logs; RUNS=experiments/runs; EVAL=experiments/results
stamp() { echo "[$(date '+%F %T')] $*"; }

train() {  # config-stem
  local cfg="configs/train_$1.yaml" name max st step
  name=$(python -c "import yaml;print(yaml.safe_load(open('$cfg'))['name'])")
  max=$(python -c "import yaml;print(yaml.safe_load(open('$cfg'))['max_steps'])")
  st="$RUNS/$name/last/trainer_state.pt"
  if [ -f "$st" ]; then
    step=$(python -c "import torch;print(torch.load('$st',weights_only=False)['step'])")
    if [ "$step" -ge "$max" ]; then stamp "train $name already finished ($step steps)"; return 0; fi
    stamp "train $name resume from step $step"
    python scripts/train.py --config "$cfg" --resume >> "$LOG/train_$name.log" 2>&1
  else
    stamp "train $name start"
    python scripts/train.py --config "$cfg" >> "$LOG/train_$name.log" 2>&1
  fi
  local rc=$?; stamp "train $name exit $rc"; return $rc
}

evaluate() {  # name model_path [manifest]
  local man="${3:-test_conditions.csv}"
  if [ -f "$EVAL/$1/metrics.json" ]; then stamp "eval $1 already done"; return; fi
  stamp "eval $1 start ($man)"
  python scripts/evaluate.py --model "$2" --name "$1" --manifest "$man" >> "$LOG/eval_$1.log" 2>&1
  stamp "eval $1 exit $?"
}

dump() {  # name  (GPU is free at this point -> default batch size)
  for s in dev test; do
    if [ -f "data/processed/logits/$1/$s.npz" ]; then stamp "logits $1/$s already done"; continue; fi
    stamp "logits $1/$s start"
    python scripts/dump_logits.py --model "$RUNS/$1/best" --name "$1" --set "$s" --workers 24 >> "$LOG/logits_$1.log" 2>&1
    stamp "logits $1/$s exit $?"
  done
}

lm_detect() {  # name  (CPU only; runs in the background while the next model trains)
  {
    for s in dev test; do
      [ -f "$EVAL/${1}_lm4/${s}_metrics.json" ] || \
        python scripts/decode_words.py --name "$1" --set "$s" --decoder lm --lm 4gram --alpha 0.8 --beta 2.0 --workers 48
    done
    python scripts/error_detector.py --run "${1}_lm4"
  } >> "$LOG/lm_$1.log" 2>&1
  stamp "lm+detector $1 exit $?"
}

after_train() {  # name
  evaluate "$1" "$RUNS/$1/best"
  evaluate "${1}_ext" "$RUNS/$1/best" test_conditions_ext.csv
  dump "$1"
  lm_detect "$1" &
}

PID=${1:-}
if [ -n "$PID" ]; then
  stamp "waiting for B460 (pid $PID)"
  while kill -0 "$PID" 2>/dev/null; do sleep 60; done
fi
stamp "day 2 start"
train B460 && after_train B460
[ -f "$RUNS/B460/best/model.safetensors" ] || { stamp "no B460 weights - stop"; exit 1; }
train B960_rev && after_train B960_rev
train B960_rev_s2 && after_train B960_rev_s2
wait
stamp "M5 DAY 2 DONE"
