"""Objective O5 benchmark: real-time factor (RTF = processing time / audio duration) of the transcription assistant
on CPU for long recordings (target: RTF <= 1.0 for clips up to 2 minutes), and a check that long-audio chunking
does not hurt accuracy.

Long-form test clips: consecutive LibriSpeech test-other utterances of one chapter concatenated (0.3 s silence in
between) up to ~2 minutes; each clip in three versions: clean, ESC-50 noise at 5 dB, white noise at 5 dB
(noise mixed with the same seeded function as the test grid). Only test data is used.

Usage : CUDA_VISIBLE_DEVICES= python scripts/benchmark_assistant.py [--threads 4] [--clips 6]
Output: milestones/M5_evaluation/tables/assistant_benchmark.csv, assistant_benchmark.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.assistant.engine import Assistant  # noqa: E402
from group9_ds_ai_lab.data.audio import ArrowAudioReader  # noqa: E402
from group9_ds_ai_lab.data.noise import apply_condition, stable_seed  # noqa: E402
from group9_ds_ai_lab.utils.paths import MANIFESTS, REPO, SAMPLE_RATE  # noqa: E402

TAB = REPO / "milestones" / "M5_evaluation" / "tables"


def long_clips(n_clips: int, target_s: float = 120.0):
    m = pd.read_csv(MANIFESTS / "test_conditions.csv")
    m = m[(m.noise_type == "clean") & (m.ls_split == "test.other")]
    reader, clips = ArrowAudioReader(), []
    for (spk, ch), g in m.groupby(["speaker_id", "chapter_id"]):
        g = g.sort_values("audio_id")
        if g.duration.sum() < target_s:
            continue
        parts, texts, dur = [], [], 0.0
        for r in g.itertuples():
            if dur + r.duration > target_s + 5:
                break
            parts += [reader.load(r.shard, r.row), np.zeros(int(0.3 * SAMPLE_RATE), np.float32)]
            texts.append(r.text.strip())
            dur += r.duration + 0.3
        clips.append((f"{spk}-{ch}", np.concatenate(parts), " ".join(texts), [len(texts)]))
        if len(clips) == n_clips:
            break
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--clips", type=int, default=6)
    args = ap.parse_args()
    clips = long_clips(args.clips)
    print(f"{len(clips)} clips, durations (s):", [round(len(x) / SAMPLE_RATE, 1) for _, x, _, _ in clips], flush=True)
    rows, meta = [], {"threads": args.threads}
    for quant in (False, True):
        t0 = time.time()
        asst = Assistant(lm="3gram", threads=args.threads, quantize=quant, device="cpu")
        meta[f"load_time_s_{'int8' if quant else 'fp32'}"] = round(time.time() - t0, 2)
        asst.transcribe(clips[0][1][: 5 * SAMPLE_RATE])  # warm-up
        for cid, x, ref, _ in clips:
            for cond, snr in (("clean", None), ("esc50", 5.0), ("white", 5.0)):
                y = x if cond == "clean" else apply_condition(x, cond, snr, stable_seed("bench", cid, cond, snr))[0]
                tr = asst.transcribe(y)
                rows.append({"model": "int8" if quant else "fp32", "clip": cid, "condition": cond if cond == "clean" else f"{cond} 5 dB",
                             "duration_s": tr.duration_s, **{f"t_{k}": v for k, v in tr.timings_s.items()},
                             "rtf": tr.rtf, "wer": jiwer.wer(ref, tr.text) if tr.text else 1.0,
                             "n_words": len(tr.words), "n_flagged": sum(w.flagged for w in tr.words)})
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "assistant_benchmark.csv", index=False)
    s = t.groupby(["model", "condition"]).agg(rtf_mean=("rtf", "mean"), rtf_max=("rtf", "max"), wer_mean=("wer", "mean"),
                                               dur_mean=("duration_s", "mean")).round(4)
    meta["summary"] = s.reset_index().to_dict("records")
    meta["target_O5"] = "RTF <= 1.0 for clips up to 2 minutes on CPU"
    with open(TAB / "assistant_benchmark.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(s.to_string())


if __name__ == "__main__":
    main()
