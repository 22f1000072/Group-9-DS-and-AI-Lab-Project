"""The evaluated models: where each one is on the server and on Hugging Face, and how to load it.

Every model can be loaded from either source:
  source="local" : experiments/runs/<name>/best (the checkpoints trained on the group server)
  source="hf"    : the public copy on the Hugging Face Hub (same weights; checked by weights_sha256 / G12)
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from group9_ds_ai_lab.utils.paths import RUNS

# name -> (Hugging Face id, training data, role in the evaluation)
MODELS = {
    "ref_base960h": ("facebook/wav2vec2-base-960h", "960 h clean (published, not trained by us)",
                     "reference: validates the evaluation code"),
    "A_clean": ("kumargulshan/wav2vec2-base-ls100-clean", "100 h clean", "Model A: baseline without noise training"),
    "B_noisy": ("kumargulshan/wav2vec2-base-ls100-noise-aug", "100 h + noise", "Model B: noise-aware training (O2)"),
    "B2_lr1e-4": ("kumargulshan/wav2vec2-base-ls100-noise-aug-lr1e-4", "100 h + noise, lr 1e-4", "hyperparameter comparison"),
    "B460": ("kumargulshan/wav2vec2-base-ls460-noise-aug", "460 h + noise", "effect of data size"),
    "B960_rev": ("kumargulshan/wav2vec2-base-ls960-noise-robust", "960 h + noise, reverb, speed, telephone",
                 "best single model; used in the assistant"),
    "B960_rev_s2": ("kumargulshan/wav2vec2-base-ls960-noise-robust-seed7", "as B960_rev, seed 7",
                    "seed variance; ROVER member"),
}


def local_path(name: str) -> Path | None:
    p = RUNS / name / "best"
    return p if (p / "model.safetensors").exists() else None


def model_source(name: str, source: str = "local") -> str:
    """Path or Hugging Face id to pass to from_pretrained(). The reference model only exists on the Hub."""
    hf_id = MODELS[name][0]
    if source == "hf" or name == "ref_base960h":
        return hf_id
    p = local_path(name)
    if p is None:
        raise FileNotFoundError(f"{name}: no local checkpoint at {RUNS / name / 'best'}; use source='hf'")
    return str(p)


def allow_hub_download():
    """setup_env() puts Hugging Face offline for the training/evaluation scripts; downloading needs it online."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    try:  # huggingface_hub reads the variable once, at import time
        import huggingface_hub.constants as C
        C.HF_HUB_OFFLINE = False
    except ImportError:
        pass


def hub_weights_file(name: str) -> Path:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(MODELS[name][0], "model.safetensors"))


def load_model(name: str, source: str = "local", device: str = "cpu"):
    from transformers import Wav2Vec2ForCTC
    if source == "hf":
        allow_hub_download()
    return Wav2Vec2ForCTC.from_pretrained(model_source(name, source)).eval().to(device)


def sha256(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def tensors_sha256(path) -> str:
    """Hash of the tensor values only (names, dtypes, shapes, bytes), so two files with the same weights but
    different metadata headers (e.g. saved by different library versions) still compare equal."""
    from safetensors import safe_open
    h = hashlib.sha256()
    with safe_open(str(path), "np") as f:
        for k in sorted(f.keys()):
            t = f.get_tensor(k)
            h.update(f"{k}|{t.dtype}|{t.shape}|".encode())
            h.update(t.tobytes())
    return h.hexdigest()
