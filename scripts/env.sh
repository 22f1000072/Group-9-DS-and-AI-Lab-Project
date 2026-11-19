# Shell environment for the pipeline scripts:   source scripts/env.sh
# Keep the virtual environment OUTSIDE the repository, e.g.
#   python -m venv ~/venvs/group9-asr && source ~/venvs/group9-asr/bin/activate
#   pip install -r requirements.txt && pip install --no-deps pyctcdecode==0.5.0
# then either activate it yourself, or set G9_VENV=/path/to/venv before sourcing this file.
if [ -n "${G9_VENV:-}" ]; then source "$G9_VENV/bin/activate"; fi
export G9_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$G9_REPO/data/hf_home}"          # model/dataset cache stays inside data/ (git-ignored)
# work offline once the wav2vec2 checkpoints are cached (scripts/download_data.py --models-only)
if [ -d "$HF_HOME/hub/models--facebook--wav2vec2-base" ]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
export PYTHONPATH="$G9_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
