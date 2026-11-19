"""Fine-tune wav2vec 2.0 with CTC (clean-only or with on-the-fly noise augmentation).

Usage:
  python scripts/train.py --config configs/train_A_clean.yaml [--smoke] [--resume]

Outputs -> experiments/runs/<name>/ : config.json, train_log.csv, eval_log.csv,
           best/ (lowest dev WER, HF format), last/ (+ trainer_state.pt for --resume)
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from group9_ds_ai_lab.utils.paths import MANIFESTS, RUNS, setup_env  # noqa: E402

setup_env()
from transformers import Wav2Vec2ForCTC, get_linear_schedule_with_warmup  # noqa: E402

from group9_ds_ai_lab.data.dataset import ASRDataset, DurationBatchSampler, collate_ctc  # noqa: E402
from group9_ds_ai_lab.inference.decode import run_inference, wer_cer  # noqa: E402
from group9_ds_ai_lab.data.noise import TrainAugmenter  # noqa: E402

DEFAULTS = dict(
    name="run", init_model="facebook/wav2vec2-base", augment=False, p_clean=0.2, snr_min=0.0, snr_max=20.0,
    max_steps=20000, lr=1e-4, warmup_ratio=0.1, weight_decay=0.005, max_grad_norm=1.0,
    max_batch_seconds=320.0, grad_accum=1, eval_every=1000, save_every=1000, log_every=50,
    num_workers=24, seed=42, mask_time_prob=0.05, mask_time_length=10, mask_feature_prob=0.0,
    mask_feature_length=64, layerdrop=0.05,
    attention_dropout=0.1, hidden_dropout=0.1, feat_proj_dropout=0.0, final_dropout=0.0,
    freeze_feature_encoder=True, eval_max_seconds=600.0, max_train_utts=None,
    # M5 additions (defaults reproduce the M4 runs exactly)
    train_manifest="train.csv", init_from=None, p_speed=0.0, speeds=[0.9, 1.1], p_reverb=0.0, p_bandlimit=0.0,
)


def load_config(path, overrides):
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def build_model(cfg):
    model = Wav2Vec2ForCTC.from_pretrained(
        cfg["init_from"] or cfg["init_model"], vocab_size=32, pad_token_id=0, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        mask_time_prob=cfg["mask_time_prob"], mask_time_length=cfg["mask_time_length"],
        mask_feature_prob=cfg["mask_feature_prob"], mask_feature_length=cfg["mask_feature_length"], layerdrop=cfg["layerdrop"],
        attention_dropout=cfg["attention_dropout"], hidden_dropout=cfg["hidden_dropout"],
        feat_proj_dropout=cfg["feat_proj_dropout"], final_dropout=cfg["final_dropout"],
    )
    if cfg["freeze_feature_encoder"]:
        model.freeze_feature_encoder()
    return model


def ctc_loss(model, logits, labels, input_lengths):
    """CTC with the *true* (unpadded) frame counts; mean over batch of per-utterance loss / target length."""
    logp = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # [T, B, V]
    in_lens = model._get_feat_extract_output_lengths(input_lengths).clamp(max=logp.shape[0])
    mask = labels >= 0
    tgt_lens = mask.sum(-1)
    targets = labels.masked_select(mask)
    return F.ctc_loss(logp, targets, in_lens, tgt_lens, blank=0, reduction="mean", zero_infinity=True)


def evaluate_dev(model, dev_ds, dev_df, cfg):
    t0 = time.time()
    res = run_inference(model, dev_ds, max_seconds=cfg["eval_max_seconds"], num_workers=16, keep_words=False)
    hyps = [r["hyp"] for r in res]
    out = {}
    for tag, m in (("clean", dev_df.noise_type == "clean"), ("noisy", dev_df.noise_type != "clean")):
        idx = np.flatnonzero(m.to_numpy())
        w, c = wer_cer([dev_df.text.iloc[i] for i in idx], [hyps[i] for i in idx])
        out[f"wer_{tag}"], out[f"cer_{tag}"] = w, c
    out["wer_avg"] = (out["wer_clean"] + out["wer_noisy"]) / 2
    out["empty_hyp_frac"] = float(np.mean([h == "" for h in hyps]))
    out["eval_sec"] = time.time() - t0
    out["example_ref"], out["example_hyp"] = dev_df.text.iloc[0][:120], hyps[0][:120]
    model.train()
    return out


def save(model, path, extra=None):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    if extra is not None:
        torch.save(extra, path / "trainer_state.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=False)
    ap.add_argument("--set", nargs="*", help="override config: key=value")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check memory/throughput")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg.update(name=cfg["name"] + "_smoke", max_steps=cfg.get("smoke_steps", 60), eval_every=10 ** 9,
                   save_every=10 ** 9, log_every=10, max_train_utts=cfg.get("max_train_utts") or 4000)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run_dir = RUNS / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg, indent=2), flush=True)

    # ---------------- data
    train_df = pd.read_csv(MANIFESTS / cfg["train_manifest"])
    if cfg["max_train_utts"]:
        train_df = train_df.sample(n=min(int(cfg["max_train_utts"]), len(train_df)), random_state=cfg["seed"])
    dev_df = pd.read_csv(MANIFESTS / "valid_eval.csv")
    if args.smoke:
        dev_df = dev_df.groupby("noise_type", group_keys=False).head(40)
    aug = TrainAugmenter(cfg["p_clean"], (cfg["snr_min"], cfg["snr_max"]), p_speed=cfg["p_speed"],
                         speeds=cfg["speeds"], p_reverb=cfg["p_reverb"], p_bandlimit=cfg["p_bandlimit"]) \
        if cfg["augment"] else None
    train_ds = ASRDataset(train_df, augmenter=aug)
    dev_ds = ASRDataset(dev_df, fixed_conditions=True)
    sampler = DurationBatchSampler(train_ds.durations, cfg["max_batch_seconds"], shuffle=True, seed=cfg["seed"])
    loader_kw = dict(collate_fn=collate_ctc, num_workers=cfg["num_workers"], pin_memory=True,
                     persistent_workers=False, prefetch_factor=4)
    steps_per_epoch = len(sampler) / cfg["grad_accum"]
    print(f"train utts {len(train_df)} ({train_df.duration.sum() / 3600:.1f} h), {len(sampler)} batches/epoch, "
          f"~{cfg['max_steps'] / steps_per_epoch:.1f} epochs; dev rows {len(dev_df)}", flush=True)

    # ---------------- model / optim
    model = build_model(cfg).cuda()
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or "norm" in n.lower() or n.endswith(".bias") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-8,
                            fused=True)
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * cfg["max_steps"]), cfg["max_steps"])
    step, epoch, best = 0, 0, math.inf
    if args.resume and (run_dir / "last" / "trainer_state.pt").exists():
        st = torch.load(run_dir / "last" / "trainer_state.pt", weights_only=False)
        model.load_state_dict(Wav2Vec2ForCTC.from_pretrained(run_dir / "last").state_dict())
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, epoch, best = st["step"], st["epoch"] + 1, st["best"]
        torch.manual_seed(cfg["seed"] + step)
        print(f"resumed at step {step}, epoch {epoch}, best {best:.4f}", flush=True)

    tlog = open(run_dir / "train_log.csv", "a", newline="")
    tw = csv.writer(tlog)
    if step == 0:
        tw.writerow(["step", "epoch", "loss", "lr", "grad_norm", "sec_per_step", "audio_sec_per_sec", "max_mem_gb"])
    elog_path = run_dir / "eval_log.csv"

    model.train()
    t_last, audio_acc, loss_acc, n_acc = time.time(), 0.0, 0.0, 0
    done = step >= cfg["max_steps"]
    while not done:
        sampler.set_epoch(epoch)
        loader = DataLoader(train_ds, batch_sampler=sampler, **loader_kw)
        micro = 0
        for batch in loader:
            x = batch["input_values"].cuda(non_blocking=True)
            y = batch["labels"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x).logits
            loss = ctc_loss(model, logits, y, batch["input_lengths"].cuda())
            (loss / cfg["grad_accum"]).backward()
            audio_acc += batch["input_lengths"].sum().item() / 16000
            loss_acc += loss.item()
            n_acc += 1
            micro += 1
            if micro % cfg["grad_accum"]:
                continue
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            if torch.isfinite(gn):
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            if step % cfg["log_every"] == 0 or step == 1:
                dt = time.time() - t_last
                k = cfg["log_every"] if step > 1 else 1
                row = [step, epoch, loss_acc / n_acc, sched.get_last_lr()[0], float(gn), dt / k, audio_acc / dt,
                       torch.cuda.max_memory_allocated() / 2 ** 30]
                tw.writerow(row)
                tlog.flush()
                print(f"step {step:6d} ep {epoch} loss {row[2]:.4f} lr {row[3]:.2e} gn {row[4]:.2f} "
                      f"{row[5]:.3f}s/step {row[6]:.0f} audio-s/s mem {row[7]:.1f}GB", flush=True)
                t_last, audio_acc, loss_acc, n_acc = time.time(), 0.0, 0.0, 0

            if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
                ev = evaluate_dev(model, dev_ds, dev_df, cfg)
                ev["step"] = step
                new = not elog_path.exists()
                pd.DataFrame([ev]).to_csv(elog_path, mode="a", header=new, index=False)
                print(f"[eval {step}] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()}), flush=True)
                if ev["wer_avg"] < best:
                    best = ev["wer_avg"]
                    save(model, run_dir / "best")
                    with open(run_dir / "best" / "dev_metrics.json", "w") as f:
                        json.dump(ev, f, indent=2)
                t_last = time.time()
            if step % cfg["save_every"] == 0 or step == cfg["max_steps"]:
                save(model, run_dir / "last", {"opt": opt.state_dict(), "sched": sched.state_dict(),
                                               "step": step, "epoch": epoch, "best": best})
                t_last = time.time()
            if step >= cfg["max_steps"]:
                done = True
                break
        epoch += 1
    tlog.close()
    print(f"[done] best dev wer_avg {best:.4f} -> {run_dir}", flush=True)


if __name__ == "__main__":
    main()
