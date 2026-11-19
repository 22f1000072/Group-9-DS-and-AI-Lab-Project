"""PyTorch dataset, duration-bucketed batch sampler and CTC collator."""
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from group9_ds_ai_lab.data.audio import ArrowAudioReader
from group9_ds_ai_lab.data.noise import TrainAugmenter, apply_condition
from group9_ds_ai_lab.data.text import encode


def zscore(x: np.ndarray) -> np.ndarray:
    """Wav2Vec2FeatureExtractor normalisation (do_normalize=True)."""
    return ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)


class ASRDataset(Dataset):
    """Rows need: audio_id, shard, row, text, duration.
    Optional fixed corruption columns: noise_type, snr_db, cond_seed, noise_part.
    `augmenter` (training only) applies random on-the-fly noise."""

    def __init__(self, df: pd.DataFrame, augmenter: TrainAugmenter | None = None, fixed_conditions: bool = False):
        self.df = df.reset_index(drop=True)
        self.augmenter = augmenter
        self.fixed = fixed_conditions
        self.reader = ArrowAudioReader()
        self.durations = self.df["duration"].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.reader.load(r["shard"], r["row"])
        if self.fixed:
            rev = r.get("reverb", False)
            x, _ = apply_condition(x, r["noise_type"], r["snr_db"], int(r["cond_seed"]), part=r.get("noise_part", "test"),
                                   reverb=bool(rev) and not (isinstance(rev, float) and np.isnan(rev)))
        elif self.augmenter is not None:
            # torch seeds every worker differently per epoch; combine with index -> reproducible given the global seed
            rng = np.random.default_rng([torch.initial_seed() % (2 ** 63), i])
            x = self.augmenter(x, rng)
        return {"input_values": zscore(x), "labels": encode(r["text"]), "index": i}


def collate_ctc(batch):
    """Zero-pad audio (wav2vec2-base uses group-norm -> no attention mask), pad labels with -100."""
    n = max(len(b["input_values"]) for b in batch)
    L = max(len(b["labels"]) for b in batch)
    x = torch.zeros(len(batch), n, dtype=torch.float32)
    y = torch.full((len(batch), L), -100, dtype=torch.long)
    lens = torch.zeros(len(batch), dtype=torch.long)
    for k, b in enumerate(batch):
        a = torch.from_numpy(b["input_values"])
        x[k, : len(a)] = a
        lens[k] = len(a)
        y[k, : len(b["labels"])] = torch.tensor(b["labels"], dtype=torch.long)
    return {"input_values": x, "labels": y, "input_lengths": lens,
            "index": torch.tensor([b["index"] for b in batch])}


class DurationBatchSampler(Sampler):
    """Batches whose *padded* audio length (max_dur * batch_size) stays under max_seconds.
    Training: shuffle -> sort inside mega-buckets -> shuffle batches. Eval: sorted, deterministic."""

    def __init__(self, durations, max_seconds: float, shuffle: bool, seed: int = 42, bucket_mult: int = 100, max_batch: int = 64):
        self.d = np.asarray(durations, dtype=np.float64)
        self.max_seconds, self.shuffle, self.seed = max_seconds, shuffle, seed
        self.bucket_mult, self.max_batch = bucket_mult, max_batch
        self.epoch = 0
        self._batches = self._make()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._batches = self._make()

    def _pack(self, order):
        batches, cur, cur_max = [], [], 0.0
        for i in order:
            m = max(cur_max, self.d[i])
            if cur and (m * (len(cur) + 1) > self.max_seconds or len(cur) >= self.max_batch):
                batches.append(cur)
                cur, m = [], self.d[i]
            cur.append(int(i))
            cur_max = m
        if cur:
            batches.append(cur)
        return batches

    def _make(self):
        if not self.shuffle:
            return self._pack(np.argsort(-self.d, kind="stable"))
        rng = np.random.default_rng(self.seed + self.epoch)
        perm = rng.permutation(len(self.d))
        approx_bs = max(1, int(self.max_seconds / max(np.median(self.d), 1e-3)))
        chunk = approx_bs * self.bucket_mult
        batches = []
        for s in range(0, len(perm), chunk):
            part = perm[s: s + chunk]
            batches += self._pack(part[np.argsort(-self.d[part], kind="stable")])
        return [batches[j] for j in rng.permutation(len(batches))]

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)

    def audio_seconds(self):
        return float(self.d.sum())


def num_batches_per_epoch(durations, max_seconds):
    return math.ceil(len(DurationBatchSampler(durations, max_seconds, shuffle=True)))
