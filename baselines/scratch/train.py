"""Train A2 (Conformer CTC) or A3 (Conformer RNN-T) from scratch on our train split.

- Character vocabulary from the normalized train transcripts (id 0 = blank).
- All train and dev audio is read once into memory (about 3 GB for 25 h), so epochs are fast.
- Batches hold clips of similar length, up to --batch-seconds of (padded) audio.
- AdamW, linear warmup then cosine decay. After the first epoch the number of epochs is fitted
  to --max-hours, so the schedule ends inside one Kaggle session.
- Dev WER after every epoch; the best epoch is kept in <out>/best.pt. <out>/last.pt allows
  resuming: run the same command again (or pass --resume-from <previous Kaggle output folder>).

Run from the repo root:
    python baselines/scratch/train.py --model ctc  --out-dir /kaggle/working/a2_ctc
    python baselines/scratch/train.py --model rnnt --out-dir /kaggle/working/a3_rnnt
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.asr_data import (SEED, clip_seconds, env_info, load_audio, normalize, read_split,  # noqa: E402
                             set_seed, update_run_log, wer_cer)
from model import build_model  # noqa: E402

SYSTEMS = {"ctc": "A2_conformer-ctc", "rnnt": "A3_conformer-rnnt"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["ctc", "rnnt"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--clean-dir", default=None)
    # model size (defaults: about 19M parameters)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--kernel", type=int, default=31)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pred-dim", type=int, default=320, help="RNN-T prediction network size")
    ap.add_argument("--joint-dim", type=int, default=320, help="RNN-T joiner size")
    ap.add_argument("--aux-ctc", type=float, default=0.3, help="RNN-T only: weight of the extra CTC loss")
    # training
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-steps", type=int, default=1500)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-seconds", type=float, default=None, help="padded audio per batch (default: ctc 400, rnnt 200)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-hours", type=float, default=11.0)
    ap.add_argument("--no-fit-to-time", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--eval-batch-seconds", type=float, default=400)
    ap.add_argument("--max-train-seconds", type=float, default=30.0)
    ap.add_argument("--num-workers", type=int, default=4, help="threads for reading audio")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-dev", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="very small model, ONLY for testing the code")
    return ap.parse_args()


# ---------------------------------------------------------------- data


def read_audio_all(rows: list[dict], workers: int) -> list[np.ndarray]:
    """Read every clip once and keep it as int16 (half the memory of float32)."""
    def one(r):
        return (np.clip(load_audio(r), -1, 1) * 32767).astype(np.int16)

    with ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(one, rows, chunksize=64))


def seconds_batches(lengths: list[float], max_seconds: float, seed: int | None) -> list[list[int]]:
    """Batches of similar-length clips; padded length x clips <= max_seconds. seed=None: no shuffle."""
    idx = list(range(len(lengths)))
    rng = random.Random(seed)
    if seed is not None:
        rng.shuffle(idx)
        groups = [sorted(idx[s:s + 2000], key=lambda i: -lengths[i]) for s in range(0, len(idx), 2000)]
    else:
        groups = [sorted(idx, key=lambda i: -lengths[i])]
    batches = []
    for g in groups:
        cur = []
        for i in g:
            longest = lengths[cur[0]] if cur else lengths[i]
            if cur and longest * (len(cur) + 1) > max_seconds:
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
    if seed is not None:
        rng.shuffle(batches)
    return batches


def make_batch(ids, audio, targets, device):
    wavs = [audio[i] for i in ids]
    n = max(len(w) for w in wavs)
    x = torch.zeros(len(ids), n)
    for k, w in enumerate(wavs):
        x[k, :len(w)] = torch.from_numpy(w.astype(np.float32) / 32767)
    wav_lens = torch.tensor([len(w) for w in wavs])
    batch = {"wav": x.to(device), "wav_lens": wav_lens.to(device)}
    if targets is not None:
        tl = torch.tensor([len(targets[i]) for i in ids])
        t = torch.zeros(len(ids), max(1, int(tl.max())), dtype=torch.long)
        for k, i in enumerate(ids):
            t[k, :len(targets[i])] = torch.tensor(targets[i], dtype=torch.long)
        batch.update(targets=t.to(device), target_lens=tl.to(device))
    return batch


def decode_all(model, audio, lengths, vocab, batch_seconds, device, fp16) -> list[str]:
    model.eval()
    hyps = [""] * len(audio)
    for ids in seconds_batches(lengths, batch_seconds, None):
        b = make_batch(ids, audio, None, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            out = model.decode(b["wav"], b["wav_lens"])
        for i, seq in zip(ids, out):
            hyps[i] = "".join(vocab[t] for t in seq).strip()
    return hyps


# ---------------------------------------------------------------- main


def main():
    args = parse_args()
    set_seed(SEED)
    system = SYSTEMS[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"
    if args.tiny:
        args.d_model, args.layers, args.heads, args.ffn, args.kernel = 64, 2, 2, 128, 15
        args.pred_dim = args.joint_dim = 64
    batch_seconds = args.batch_seconds or (400.0 if args.model == "ctc" else 200.0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume_from and not (out / "last.pt").exists():
        shutil.copytree(args.resume_from, out, dirs_exist_ok=True)

    # ---- data
    train_rows = read_split("train", args.clean_dir, args.limit_train)
    dev_rows = read_split("dev", args.clean_dir, args.limit_dev)
    train_rows = [r for r in train_rows if clip_seconds(r) <= args.max_train_seconds]
    texts = [normalize(r["transcript"]) for r in train_rows]
    keep = [i for i, t in enumerate(texts) if t]
    train_rows, texts = [train_rows[i] for i in keep], [texts[i] for i in keep]
    vocab_path = out / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    else:
        vocab = ["<blank>"] + sorted(set("".join(texts)))
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    tok = {c: i for i, c in enumerate(vocab)}
    targets = [[tok[c] for c in t if c in tok] for t in texts]
    print(f"vocabulary {len(vocab)}: {''.join(vocab[1:])!r}")
    t = time.time()
    train_audio = read_audio_all(train_rows, args.num_workers)
    dev_audio = read_audio_all(dev_rows, args.num_workers)
    train_len = [len(a) / 16000 for a in train_audio]
    dev_len = [len(a) / 16000 for a in dev_audio]
    print(f"read {len(train_rows)} train clips ({sum(train_len) / 3600:.2f} h) and {len(dev_rows)} dev clips "
          f"in {time.time() - t:.0f}s")

    # ---- model
    cfg = {k: getattr(args, k) for k in ("d_model", "layers", "heads", "ffn", "kernel", "dropout",
                                         "pred_dim", "joint_dim", "aux_ctc")}
    model = build_model(args.model, len(vocab), cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    steps_per_epoch = len(seconds_batches(train_len, batch_seconds, SEED))
    state = {"epoch": 0, "step": 0, "best_wer": None, "best_epoch": None, "epochs": args.epochs,
             "fitted_to_time": False, "train_seconds": 0.0, "history": []}

    def lr_at(step):
        total = state["epochs"] * steps_per_epoch
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        p = min(1.0, (step - args.warmup_steps) / max(1, total - args.warmup_steps))
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda") if fp16 else None
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if scaler and ck["scaler"]:
            scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        random.setstate(ck["py_rng"])
        torch.set_rng_state(ck["torch_rng"])
        print(f"Resumed after epoch {state['epoch']}")
    print(f"{system}: {n_params / 1e6:.1f}M parameters, {steps_per_epoch} steps per epoch, "
          f"up to {state['epochs']} epochs, device {device}")

    # ---- training
    t0 = time.time()
    prev_seconds = state["train_seconds"]
    budget = args.max_hours * 3600
    while state["epoch"] < state["epochs"]:
        ep_start = time.time()
        model.train()
        losses = []
        for ids in seconds_batches(train_len, batch_seconds, SEED + state["epoch"]):
            b = make_batch(ids, train_audio, targets, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = model.loss(b["wav"], b["wav_lens"], b["targets"], b["target_lens"])
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            state["step"] += 1
            losses.append(loss.item())
        state["epoch"] += 1

        hyps = decode_all(model, dev_audio, dev_len, vocab, args.eval_batch_seconds, device, fp16)
        wer, cer = wer_cer([r["transcript"] for r in dev_rows], hyps)
        better = state["best_wer"] is None or wer < state["best_wer"]
        if better:
            state["best_wer"], state["best_epoch"] = wer, state["epoch"]
            torch.save({"model": model.state_dict(), "kind": args.model, "cfg": cfg, "vocab": vocab,
                        "epoch": state["epoch"], "dev_wer": wer, "dev_cer": cer}, out / "best.pt")
        ep_time = time.time() - ep_start
        entry = {"epoch": state["epoch"], "step": state["step"], "loss": round(float(np.mean(losses or [float('nan')])), 4),
                 "dev_wer": round(wer, 4), "dev_cer": round(cer, 4), "best": better, "lr": sched.get_last_lr()[0],
                 "epoch_minutes": round(ep_time / 60, 2), "example": hyps[0] if hyps else ""}
        state["history"].append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

        elapsed = time.time() - t0
        if not args.no_fit_to_time and not state["fitted_to_time"]:
            fit = state["epoch"] + int((budget * 0.95 - elapsed) / ep_time)
            state["fitted_to_time"] = True
            if fit < state["epochs"]:
                print(f"{ep_time / 60:.1f} min per epoch: epochs reduced from {state['epochs']} to {fit} to fit {args.max_hours} h")
                state["epochs"] = max(fit, state["epoch"])
        state["train_seconds"] = prev_seconds + elapsed
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None, "state": state,
                    "py_rng": random.getstate(), "torch_rng": torch.get_rng_state()}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(out / "last.pt")
        if elapsed + ep_time > budget * 0.97 and state["epoch"] < state["epochs"]:
            print("Stopped for time. Run the same command again to continue.")
            break

    finished = state["epoch"] >= state["epochs"]
    update_run_log(system, {
        "system": system, "model": f"Conformer {args.model.upper()} trained from scratch (torchaudio.models.Conformer)",
        "parameters": n_params, "seed": SEED,
        "train_data": {"clips": len(train_rows), "hours": round(sum(train_len) / 3600, 3)},
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("out_dir", "resume_from", "clean_dir")}
        | {"batch_seconds": batch_seconds, "epochs_planned": args.epochs, "epochs_used": state["epochs"],
           "steps_per_epoch": steps_per_epoch, "optimizer": "AdamW (0.9, 0.98)",
           "scheduler": "linear warmup + cosine to 1% of lr", "features": "80 log-mel, 25 ms window, 10 ms hop, per-clip CMVN",
           "specaugment": "2 freq masks (27), 5 time masks (5% of length)", "subsampling": "4x (2 conv layers)",
           "vocabulary": vocab, "fp16": fp16},
        "training": {"finished": finished, "epochs_done": state["epoch"], "best_epoch": state["best_epoch"],
                     "best_dev_wer_quick": state["best_wer"], "train_hours": round(state["train_seconds"] / 3600, 2),
                     "history": state["history"]},
        "chosen_checkpoint": f"epoch {state['best_epoch']} (lowest dev WER)",
        "env": env_info(),
    })
    print("Training finished." if finished else "")


if __name__ == "__main__":
    main()
