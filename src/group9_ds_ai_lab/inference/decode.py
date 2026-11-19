"""Batched inference, greedy CTC decoding with word confidences/timestamps, and WER helpers."""
import jiwer
import numpy as np
import torch
from torch.utils.data import DataLoader

from group9_ds_ai_lab.data.dataset import DurationBatchSampler, collate_ctc
from group9_ds_ai_lab.data.text import BLANK_ID, DELIM_ID, ID2CH

FRAME_SEC = 0.02  # wav2vec2 conv stack stride = 320 samples @ 16 kHz
SPECIAL = {1, 2, 3}  # <s>, </s>, <unk>


def greedy_words(logp: np.ndarray):
    """logp: [T, V] log-probabilities of ONE utterance (padding frames removed).
    Returns transcript and word list with confidence + CTC timestamps.

    Word confidence (used later for error flagging, O3):
      conf_min  = min over the word's emitting frames of max-posterior
      conf_mean = mean of the same
    """
    ids = logp.argmax(-1)
    pmax = np.exp(logp.max(-1))
    words, chars, confs, start, end, prev = [], [], [], None, None, -1

    def flush():
        if chars:
            words.append({"word": "".join(chars), "start": round(start * FRAME_SEC, 3),
                          "end": round((end + 1) * FRAME_SEC, 3),
                          "conf_min": float(np.min(confs)), "conf_mean": float(np.mean(confs))})

    for t, i in enumerate(ids):
        i = int(i)
        if i == BLANK_ID or i in SPECIAL:
            prev = i
            continue
        if i == DELIM_ID:
            if prev != DELIM_ID:
                flush()
                chars, confs, start = [], [], None
            prev = i
            continue
        if i != prev:
            chars.append(ID2CH[i])
            if start is None:
                start = t
        confs.append(pmax[t])
        end = t
        prev = i
    flush()
    return " ".join(w["word"] for w in words), words


@torch.no_grad()
def run_inference(model, dataset, max_seconds=400.0, num_workers=16, device="cuda", keep_words=True, desc=None,
                  keep_logp=False):
    """Returns list (aligned with dataset rows) of dicts: hyp, words (+ logp [T, V] float16 if keep_logp)."""
    model.eval()
    sampler = DurationBatchSampler(dataset.durations, max_seconds, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_ctc, num_workers=num_workers,
                        pin_memory=True, persistent_workers=False, prefetch_factor=4 if num_workers else None)
    out = [None] * len(dataset)
    for batch in loader:
        x = batch["input_values"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).logits
        logp = torch.log_softmax(logits.float(), dim=-1)
        flen = model._get_feat_extract_output_lengths(batch["input_lengths"]).tolist()
        logp = logp.cpu().numpy()
        for k, idx in enumerate(batch["index"].tolist()):
            hyp, words = greedy_words(logp[k, : flen[k]])
            out[idx] = {"hyp": hyp, "words": words if keep_words else None}
            if keep_logp:
                out[idx]["logp"] = logp[k, : flen[k]].astype(np.float16)
    return out


def wer_cer(refs, hyps):
    refs = [r if r.strip() else "<empty>" for r in refs]
    hyps = [h if h.strip() else "" for h in hyps]
    return float(jiwer.wer(refs, hyps)), float(jiwer.cer(refs, hyps))


def word_error_labels(ref: str, hyp: str):
    """1 for each hypothesis word that is wrong (substitution or insertion), else 0."""
    hw = hyp.split()
    if not hw:
        return []
    if not ref.strip():
        return [1] * len(hw)
    lab = [0] * len(hw)
    al = jiwer.process_words(ref, hyp).alignments[0]
    for ch in al:
        if ch.type in ("substitute", "insert"):
            for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                lab[j] = 1
    return lab
