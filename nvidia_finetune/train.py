"""C4: fine-tune the NVIDIA Uzbek FastConformer (the A8/A9 model) on our USC train split.

Start checkpoint: nvidia/stt_uz_fastconformer_hybrid_large_pc at a fixed revision (the same model
as A8/A9). It is a hybrid model: one FastConformer encoder with a transducer (RNN-T) head and a
CTC head; both heads are trained together, with the model's own loss weights. The model's own
SentencePiece tokenizer is kept (no new vocabulary).

- Train: data/clean/train.csv (trimmed audio through the manifest offset/duration fields), text normalized
  with the team normalizer (data/normalize.py), like C2.
- Dev: a fixed random subset of data/clean/dev.csv (seed 42) is scored after every epoch; the epoch
  with the lowest dev WER (transducer head) is kept and saved as <out-dir>/best.nemo.
- Our transcripts use the ASCII apostrophe; for training it is written the way the start model writes it
  (oʻ/gʻ with U+02BB, the glottal stop with ’), or in the one form the tokenizer knows. The scoring
  normalizer maps every form back to ', so this does not change WER/CER.

Run from the repo root (Kaggle, one T4):
    python nvidia_finetune/train.py --out-dir /kaggle/working/c4_run --max-hours 9.5
Then:
    python nvidia_finetune/predict.py --model /kaggle/working/c4_run/best.nemo --split dev
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "baselines" / "nvidia_fastconformer")]
from common.asr_data import (SEED, clip_seconds, env_info, normalize, read_split,  # noqa: E402
                             resolve_path, set_seed, update_run_log)
from nemo_utils import MODEL_ID, count_params, load_model, transcribe_rows  # noqa: E402

SYSTEM_PREFIX = "C4"
APOSTROPHE_FORMS = ["'", "ʻ", "ʼ", "‘", "’"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--revision", default=None, help="start model commit hash; default = latest (recorded)")
    ap.add_argument("--start-nemo", default=None, help="local .nemo file to start from instead (tests only)")
    ap.add_argument("--clean-dir", default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch-size x grad-accum")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-hours", type=float, default=9.5, help="training stops after this time (best epoch kept)")
    ap.add_argument("--val-limit", type=int, default=1000, help="dev clips scored after each epoch (fixed sample)")
    ap.add_argument("--max-train-seconds", type=float, default=30.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--precision", default="32-true",
                    help="Lightning precision. 32-true: with 16-mixed the NVIDIA model gives only <unk> on a T4")
    ap.add_argument("--max-start-wer", type=float, default=0.5,
                    help="stop if the start model's dev-check WER is above this (A8 is about 0.14): something is wrong")
    ap.add_argument("--limit-train", type=int, default=None, help="only the first N train clips (quick check)")
    ap.add_argument("--no-start-check", action="store_true",
                    help="skip scoring the start model on the dev check before training")
    ap.add_argument("--no-save", action="store_true", help="no checkpoint and no best.nemo (learning-rate search)")
    ap.add_argument("--memory-probe", action="store_true",
                    help="one training step on the longest train clips and one dev check on the longest dev clips, "
                         "to see if this --batch-size fits in GPU memory (no model written)")
    return ap.parse_args()


def lightning_for(model):
    """The Lightning package this NeMo version builds its models on (lightning.pytorch or pytorch_lightning)."""
    try:
        import lightning.pytorch as pl

        if isinstance(model, pl.LightningModule):
            return pl
    except ImportError:
        pass
    import pytorch_lightning as pl

    return pl


def _roundtrip(tokenizer, text: str) -> bool:
    return tokenizer.ids_to_text(tokenizer.text_to_ids(text)).strip() == text


def pick_apostrophe(tokenizer) -> str:
    """First apostrophe form that the tokenizer can write and read back unchanged."""
    for a in APOSTROPHE_FORMS:
        if _roundtrip(tokenizer, f"o{a}zbek tilida so{a}z"):
            return a
    return "'"


def apostrophe_rule(tokenizer):
    """Returns (function, description) that puts our ASCII apostrophe in the form the start model writes.

    Uzbek spelling: oʻ / gʻ use U+02BB, the glottal stop (maʼno) uses another sign; the NVIDIA model writes
    'ʻ' and '’'. If the tokenizer knows both, we use them the same way, so the model has less to relearn.
    """
    if _roundtrip(tokenizer, "oʻzbek soʻz") and _roundtrip(tokenizer, "ma’no"):
        def fix(t: str) -> str:
            return re.sub(r"(?<=[og])'", "ʻ", t).replace("'", "’")
        return fix, "o'/g' -> oʻ/gʻ (U+02BB), other ' -> ’ (U+2019)"
    a = pick_apostrophe(tokenizer)
    return (lambda t: t.replace("'", a)), f"' -> {a!r}"


def unknown_chars(tokenizer, texts: list[str]) -> list[str]:
    chars = sorted({c for t in texts for c in t if c != " "})
    return [c for c in chars if tokenizer.ids_to_text(tokenizer.text_to_ids(f"a{c}a")).strip() != f"a{c}a"]


def write_manifest(path: Path, rows: list[dict], texts: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r, t in zip(rows, texts):
            start, end = float(r["start_s"]), float(r["end_s"])
            f.write(json.dumps({"audio_filepath": str(resolve_path(r["path"]).resolve()), "offset": start,
                                "duration": round(end - start, 3), "text": t}, ensure_ascii=False) + "\n")


def ds_config(manifest: Path, batch_size: int, shuffle: bool, workers: int, max_s: float) -> dict:
    return {"manifest_filepath": str(manifest), "sample_rate": 16000, "batch_size": batch_size,
            "shuffle": shuffle, "num_workers": workers, "pin_memory": True, "max_duration": max_s,
            "min_duration": 0.1, "trim_silence": False, "use_start_end_token": False, "is_tarred": False,
            "tarred_audio_filepaths": None, "use_lhotse": False}


def diagnose(model, dev_rows: list[dict]) -> None:
    """Printed when the start check fails: audio as NeMo's loader reads it vs as A8 reads it, and A8's output."""
    import numpy as np
    import torch

    from common.asr_data import load_audio

    try:
        sig, sig_len = next(iter(model._validation_dl))[:2]
        for i, r in enumerate(dev_rows[:3]):
            x = sig[i, : int(sig_len[i])].float()
            a = load_audio(r)
            print(f"{r['id']}: NeMo loader {len(x)} samples, max {x.abs().max():.3f}, mean {x.abs().mean():.4f} | "
                  f"A8 loader {len(a)} samples, max {np.abs(a).max():.3f}, mean {np.abs(a).mean():.4f}")
        model.eval()
        with torch.no_grad():
            print("transcribe() as in A8:", transcribe_rows(model, dev_rows[:3], batch_size=3, progress=False))
    except Exception as err:  # the diagnosis must not hide the real message
        print(f"diagnosis failed: {err!r}")


def main():
    args = parse_args()
    set_seed(SEED)
    import torch
    from omegaconf import OmegaConf, open_dict

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- model (same loader and revision handling as A8/A9)
    if args.start_nemo:
        from nemo.collections.asr.models import ASRModel

        model, sha = ASRModel.restore_from(args.start_nemo, map_location=torch.device("cpu")).eval(), None
    else:
        model, sha = load_model(args.revision, device="cpu")
    start_model = args.start_nemo or MODEL_ID
    pl = lightning_for(model)
    tok = model.tokenizer
    fix_apostrophe, apostrophe = apostrophe_rule(tok)
    print(f"Loaded {start_model} @ {sha}; apostrophes for the tokenizer: {apostrophe}")

    # ---- data
    train_rows = [r for r in read_split("train", args.clean_dir, args.limit_train)
                  if clip_seconds(r) <= args.max_train_seconds]
    dev_rows = read_split("dev", args.clean_dir)
    if args.memory_probe:
        # Every batch is padded to its longest clip and its longest text, so the largest possible batch holds the
        # longest clip and the longest text. Two optimizer steps (the second one also holds the optimizer's state,
        # as in training), no shuffling, and that largest batch comes last.
        def worst_order(rows: list[dict], n: int) -> list[dict]:
            longest = sorted(rows, key=clip_seconds, reverse=True)[:n]
            most_text = max(rows, key=lambda r: len(r["transcript"]))
            rest = sorted((r for r in longest if r is not most_text), key=clip_seconds)
            return rest[len(rest) - (n - 1):] + [most_text]

        train_rows = worst_order(train_rows, 2 * args.batch_size * args.grad_accum)
        dev_rows = worst_order(dev_rows, args.batch_size)
        args.epochs, args.no_save, args.no_start_check = 1, True, True
    elif args.val_limit and len(dev_rows) > args.val_limit:
        dev_rows = sorted(random.Random(SEED).sample(dev_rows, args.val_limit), key=lambda r: r["id"])

    def target(transcript: str) -> str:   # team-normalized text, apostrophes in the tokenizer's form
        return fix_apostrophe(normalize(transcript))

    train_rows = [r for r in train_rows if target(r["transcript"]).strip()]
    train_texts = [target(r["transcript"]) for r in train_rows]
    dev_texts = [target(r["transcript"]) for r in dev_rows]
    bad = unknown_chars(tok, train_texts)
    if bad:
        print(f"WARNING: the tokenizer cannot write these characters: {bad}")
    write_manifest(out / "train_manifest.jsonl", train_rows, train_texts)
    write_manifest(out / "dev_manifest.jsonl", dev_rows, dev_texts)
    hours = sum(map(clip_seconds, train_rows)) / 3600
    print(f"train {len(train_rows)} clips ({hours:.2f} h), dev check {len(dev_rows)} clips")

    # ---- trainer, data loaders, optimizer
    best_cb = pl.callbacks.ModelCheckpoint(dirpath=str(out / "ckpt"), filename="best", monitor="val_wer",
                                           mode="min", save_top_k=1, save_weights_only=True)
    log_path = out / "train_log.jsonl"
    log_path.unlink(missing_ok=True)
    t0 = time.time()

    class EpochLog(pl.Callback):
        """One line per dev check in train_log.jsonl: dev WER of both heads, loss, time per step."""

        WARM = 10   # the first steps of an epoch include start-up work, so the speed is measured after them

        def __init__(self):
            self.warm_time = self.val_start = self.train_seconds = None
            self.step_at_start = self.warm_steps = 0

        def on_train_epoch_start(self, trainer, pl_module):
            self.step_at_start, self.warm_time = trainer.global_step, None

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if self.warm_time is None and trainer.global_step - self.step_at_start >= self.WARM:
                self.warm_time, self.warm_steps = time.time(), trainer.global_step

        def on_validation_start(self, trainer, pl_module):
            self.val_start = time.time()
            # validation runs at the end of each training epoch, so this is the epoch's training time
            self.train_seconds = time.time() - self.warm_time if self.warm_time else None

        # on_validation_end runs after the model has logged this epoch's val_wer
        # (callbacks' on_validation_epoch_end would still see the previous epoch)
        def on_validation_end(self, trainer, pl_module):
            m = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith(("val_", "train_loss"))}
            fitting = trainer.state.fn == "fit"
            steps = trainer.global_step - self.warm_steps
            rec = {"epoch": trainer.current_epoch + 1 if fitting else "start", "step": trainer.global_step,
                   "minutes": round((time.time() - t0) / 60, 1),
                   "val_minutes": round((time.time() - self.val_start) / 60, 2)}
            if fitting and self.train_seconds and steps:
                rec["sec_per_step"] = round(self.train_seconds / steps, 3)
            if not fitting:
                m = {k: v for k, v in m.items() if k.startswith("val_")}
            rec.update(m)
            if torch.cuda.is_available():   # highest GPU memory use so far
                rec["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print("epoch log:", rec, flush=True)

    trainer = pl.Trainer(devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         precision=args.precision if torch.cuda.is_available() else "32",
                         max_epochs=args.epochs, max_time=timedelta(hours=args.max_hours),
                         accumulate_grad_batches=args.grad_accum, gradient_clip_val=1.0,
                         check_val_every_n_epoch=1, num_sanity_val_steps=0, log_every_n_steps=50,
                         logger=False, enable_progress_bar=False,
                         callbacks=([] if args.no_save else [best_cb]) + [EpochLog()],
                         enable_checkpointing=not args.no_save)
    model.set_trainer(trainer)
    with open_dict(model.cfg):
        model.cfg.train_ds = OmegaConf.create(ds_config(out / "train_manifest.jsonl", args.batch_size,
                                                        not args.memory_probe,
                                                        args.num_workers, args.max_train_seconds))
        model.cfg.validation_ds = OmegaConf.create(ds_config(out / "dev_manifest.jsonl", args.batch_size, False,
                                                             args.num_workers, 60.0))
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    optim = {"name": "adamw", "lr": args.lr, "betas": [0.9, 0.98], "weight_decay": args.weight_decay,
             "sched": {"name": "CosineAnnealing", "warmup_steps": args.warmup_steps, "min_lr": args.min_lr}}
    with open_dict(model.cfg):
        model.cfg.optim = OmegaConf.create(optim)
    model.setup_optimization(model.cfg.optim)

    # ---- dev check of the start model (= A8 on this dev sample), train, keep the epoch with the lowest dev WER
    if not args.no_start_check:
        trainer.validate(model, verbose=False)
        start_wer = float(trainer.callback_metrics.get("val_wer", float("nan")))
        if not start_wer <= args.max_start_wer:
            diagnose(model, dev_rows)
            raise SystemExit(f"The start model scores WER {start_wer:.2f} on the dev check, but as A8 the same model "
                             "gets about 0.14. Something is wrong (audio loading or number precision): stopping "
                             "before training.")
    # the loaded model is in eval mode (dropout and SpecAugment off, and cuDNN cannot train the LSTM in eval
    # mode); Lightning keeps the mode it finds, so switch to training mode here
    model.train()
    trainer.fit(model)
    train_hours = round((time.time() - t0) / 3600, 2)
    if args.memory_probe:
        gpu = f", GPU memory used {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if torch.cuda.is_available() else ""
        print(f"MEMORY OK: batch {args.batch_size} on the longest clips ({max(map(clip_seconds, train_rows)):.1f} s "
              f"train, {max(map(clip_seconds, dev_rows)):.1f} s dev){gpu}", flush=True)
        return
    if args.no_save:
        print(f"Done after {train_hours} h (--no-save: no model written)")
        return
    best_path = best_cb.best_model_path
    if best_path:
        state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state)
    best_score = float(best_cb.best_model_score) if best_cb.best_model_score is not None else None
    model.save_to(str(out / "best.nemo"))
    print(f"Saved {out / 'best.nemo'} (dev-check WER {best_score}) after {train_hours} h")

    log = [json.loads(line) for line in open(log_path, encoding="utf-8")] if log_path.exists() else []
    start = next((e for e in log if e["epoch"] == "start"), {})
    epochs = [e for e in log if e["epoch"] != "start"]
    best_epoch = min(epochs, key=lambda e: e.get("val_wer", 9e9))["epoch"] if epochs else None
    info = {
        "start_model": start_model, "start_revision": sha, "parameters": count_params(model), "seed": SEED,
        "train_data": {"clips": len(train_rows), "hours": round(hours, 3)},
        "hyperparameters": {**{k: v for k, v in vars(args).items() if k not in ("out_dir",)},
                            "optimizer": "AdamW (0.9, 0.98)", "scheduler": "warmup + cosine annealing",
                            "tokenizer": "the start model's own SentencePiece tokenizer (unchanged)",
                            "apostrophe_in_targets": apostrophe, "heads": "transducer + CTC trained together"},
        "training": {"epochs_done": len(epochs), "best_epoch": best_epoch, "best_dev_check_wer": best_score,
                     "start_model_dev_check_wer": start.get("val_wer"),
                     "start_model_dev_check_wer_ctc": start.get("val_wer_ctc"),
                     "dev_check_clips": len(dev_rows), "train_hours": train_hours},
        "env": env_info(),
    }
    (out / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for head in ("rnnt", "ctc"):
        update_run_log(f"{SYSTEM_PREFIX}{'a' if head == 'rnnt' else 'b'}_fastconformer-usc-ft-{head}",
                       {"training": info})


if __name__ == "__main__":
    main()
