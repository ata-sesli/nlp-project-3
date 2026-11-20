"""C2: fine-tune facebook/wav2vec2-xls-r-300m with a CTC head on our 25 h USC train split.

- Character vocabulary from the normalized train transcripts ('|' = word separator); when training further
  from our own fine-tuned model (C2long), that model's vocabulary is kept.
- Feature encoder (CNN) frozen, gradient checkpointing, fp16 on GPU.
- Dev WER every --eval-every steps; the best dev checkpoint is kept in <out>/best.
- A full checkpoint (<out>/last) is saved regularly, so a stopped run continues where it stopped:
  just run the same command again (or pass --resume-from <folder of a previous Kaggle output>).
- --max-hours: the planned number of steps is fitted to this time budget after the speed is
  measured, so the learning-rate schedule ends inside one Kaggle session (12 h limit).

Run from the repo root:
    python xlsr_ctc/train.py --out-dir /kaggle/working/xlsr_run
Quick check (a few minutes):
    python xlsr_ctc/train.py --out-dir /tmp/xlsr_check --limit-train 200 --limit-dev 50 --max-steps 30 --eval-every 15
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.asr_data import (SEED, clip_seconds, env_info, hf_revision, read_split,  # noqa: E402
                             set_seed, update_run_log, wer_cer)
from xlsr_lib import (BASE_MODEL, SYSTEM, ClipDataset, Collator, build_vocab,  # noqa: E402
                      length_grouped_batches, lr_lambda_factory, make_processor, save_json,
                      target_text, transcribe)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resume-from", default=None, help="copy a previous run folder (e.g. /kaggle/input/...) into --out-dir first")
    ap.add_argument("--clean-dir", default=None)
    ap.add_argument("--model-id", default=BASE_MODEL)
    ap.add_argument("--revision", default=None, help="commit hash of the start checkpoint; default = latest (recorded)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8, help="clips per forward pass")
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch-size x grad-accum")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-steps", type=int, default=None, help="overrides --epochs")
    ap.add_argument("--max-hours", type=float, default=11.0, help="time budget for this session")
    ap.add_argument("--no-fit-to-time", action="store_true", help="do not shorten the schedule to fit --max-hours")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--max-train-seconds", type=float, default=30.0, help="safety filter (rule 3 already removes longer clips)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--mask-time-prob", type=float, default=0.05)
    ap.add_argument("--no-grad-checkpointing", action="store_true")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-dev", type=int, default=None)
    ap.add_argument("--tiny", action="store_true", help="small random model, ONLY for testing the code on CPU")
    return ap.parse_args()


def load_model(args, processor, device, sha, resume_dir: Path | None):
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    overrides = dict(attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, layerdrop=0.0,
                     mask_time_prob=args.mask_time_prob, ctc_loss_reduction="mean", ctc_zero_infinity=True,
                     pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer))
    if resume_dir is not None:
        model = Wav2Vec2ForCTC.from_pretrained(resume_dir)
    elif args.tiny:
        cfg = Wav2Vec2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                             conv_dim=(32,) * 7, feat_extract_norm="layer", do_stable_layer_norm=True,
                             num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4, mask_time_length=2,
                             **overrides)
        model = Wav2Vec2ForCTC(cfg)
    else:
        model = Wav2Vec2ForCTC.from_pretrained(args.model_id, revision=sha, **overrides)
    if not args.tiny:  # the tiny test model has a random CNN, so it must learn it
        model.freeze_feature_encoder()
    if not args.no_grad_checkpointing and device.startswith("cuda"):
        model.gradient_checkpointing_enable()
    return model.to(device)


def model_vocab(model_id: str, sha: str | None) -> Path | None:
    """vocab.json of a model already fine-tuned with this code (C2 -> C2long), else None (base model)."""
    if Path(model_id).is_dir():
        p = Path(model_id) / "vocab.json"
        return p if p.exists() else None
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(model_id, "vocab.json", revision=sha))
    except Exception:   # the XLS-R base model has no vocab.json
        return None


def save_checkpoint(out: Path, model, opt, sched, scaler, state: dict) -> None:
    tmp = out / "last_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "model")
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state()}, tmp / "state.pt")
    save_json(tmp / "trainer_state.json", state)
    shutil.rmtree(out / "last", ignore_errors=True)
    tmp.rename(out / "last")


def main():
    args = parse_args()
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume_from and not (out / "last").exists():
        print(f"Copying {args.resume_from} -> {out}")
        shutil.copytree(args.resume_from, out, dirs_exist_ok=True)
    resume = (out / "last" / "trainer_state.json").exists()

    # ---- data
    train_rows = read_split("train", args.clean_dir, args.limit_train)
    dev_rows = read_split("dev", args.clean_dir, args.limit_dev)
    too_long = [r for r in train_rows if clip_seconds(r) > args.max_train_seconds]
    train_rows = [r for r in train_rows if clip_seconds(r) <= args.max_train_seconds]
    texts = [target_text(r["transcript"]) for r in train_rows]
    keep = [i for i, t in enumerate(texts) if t]
    dropped_empty = len(train_rows) - len(keep)
    train_rows = [train_rows[i] for i in keep]
    texts = [texts[i] for i in keep]
    print(f"train {len(train_rows)} clips ({sum(map(clip_seconds, train_rows)) / 3600:.2f} h), dev {len(dev_rows)} clips; "
          f"skipped {len(too_long)} over {args.max_train_seconds}s and {dropped_empty} with empty text")

    # ---- vocabulary / processor (reused on resume so ids never change)
    sha = None if args.tiny else hf_revision(args.model_id, args.revision)
    proc_dir = out / "processor"
    if not (proc_dir / "vocab.json").exists():
        proc_dir.mkdir(parents=True, exist_ok=True)
        start_vocab = model_vocab(args.model_id, sha)
        if start_vocab:   # training further from our own fine-tuned model: keep its letter ids
            print(f"Using the vocabulary of {args.model_id}")
            shutil.copy(start_vocab, proc_dir / "vocab.json")
        else:
            save_json(proc_dir / "vocab.json", build_vocab(texts))
    processor = make_processor(proc_dir / "vocab.json")
    print(f"vocabulary ({len(processor.tokenizer)}): {''.join(sorted(processor.tokenizer.get_vocab()))}")
    label_ids = [processor.tokenizer(t).input_ids for t in texts]
    unk = processor.tokenizer.unk_token_id
    assert not any(unk in ids for ids in label_ids), "train text has characters outside the vocabulary"

    # ---- model, optimizer, schedule
    model = load_model(args, processor, device, sha, (out / "last" / "model") if resume else None)
    n_params = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    lengths = [clip_seconds(r) for r in train_rows]
    batches_per_epoch = len(length_grouped_batches(lengths, args.batch_size, SEED))
    steps_per_epoch = math.ceil(batches_per_epoch / args.grad_accum)
    planned = args.max_steps or args.epochs * steps_per_epoch
    state = {"step": 0, "epoch": 0, "batch_in_epoch": 0, "best_wer": None, "best_step": None,
             "warmup_steps": args.warmup_steps, "total_steps": planned, "planned_steps": planned,
             "fitted_to_time": False, "train_seconds": 0.0, "model_revision": sha}
    sched_state = {"warmup_steps": args.warmup_steps, "total_steps": planned}
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(sched_state))
    scaler = torch.amp.GradScaler("cuda") if fp16 else None
    if resume:
        st = torch.load(out / "last" / "state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        if scaler and st["scaler"]:
            scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["np_rng"])
        state = json.loads((out / "last" / "trainer_state.json").read_text())
        sched_state.update(warmup_steps=state["warmup_steps"], total_steps=state["total_steps"])
        print(f"Resumed at step {state['step']} (epoch {state['epoch']}, batch {state['batch_in_epoch']})")

    hparams = {k: v for k, v in vars(args).items() if k not in ("out_dir", "resume_from", "clean_dir")}
    print(f"{n_params / 1e6:.1f}M parameters, {steps_per_epoch} steps per epoch, planned {state['total_steps']} steps")

    def evaluate(step: int) -> float:
        t = time.time()
        hyps = transcribe(model, processor, dev_rows, args.eval_batch_size, device, fp16, args.num_workers)
        wer, cer = wer_cer([r["transcript"] for r in dev_rows], hyps)
        model.train()
        better = state["best_wer"] is None or wer < state["best_wer"]
        if better:
            state["best_wer"], state["best_step"] = wer, step
            shutil.rmtree(out / "best", ignore_errors=True)
            model.save_pretrained(out / "best")
            processor.save_pretrained(out / "best")
            save_json(out / "best" / "best.json", {"step": step, "dev_wer": wer, "dev_cer": cer})
        log({"step": step, "dev_wer": round(wer, 4), "dev_cer": round(cer, 4), "best": better,
             "eval_seconds": round(time.time() - t, 1), "example": hyps[0] if hyps else ""})
        return wer

    def log(entry: dict) -> None:
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        with open(out / "train_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---- training loop
    dataset = ClipDataset(train_rows, label_ids)
    collate = Collator(processor)
    t0 = time.time()
    session_start_step = state["step"]
    prev_seconds = state["train_seconds"]
    budget = args.max_hours * 3600
    stop_for_time = False
    model.train()
    running = []
    while state["step"] < state["total_steps"] and not stop_for_time:
        epoch = state["epoch"]
        batches = length_grouped_batches(lengths, args.batch_size, SEED + epoch)
        skip = state["batch_in_epoch"]
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches[skip:], collate_fn=collate,
                                             num_workers=args.num_workers, persistent_workers=False)
        for bi, batch in enumerate(loader, start=skip):
            with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                loss = model(batch["input_values"].to(device), attention_mask=batch["attention_mask"].to(device),
                             labels=batch["labels"].to(device)).loss
            running.append(loss.item())
            (scaler.scale(loss / args.grad_accum) if scaler else loss / args.grad_accum).backward()
            last_in_epoch = bi == len(batches) - 1
            if (bi + 1) % args.grad_accum and not last_in_epoch:
                continue
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            state["step"] += 1
            state["batch_in_epoch"] = bi + 1
            step = state["step"]
            elapsed = time.time() - t0

            # Fit the schedule to the time budget once the speed is known (first session only).
            if (not args.no_fit_to_time and not state["fitted_to_time"] and session_start_step == 0
                    and step == min(50, state["total_steps"])):
                sec_per_step = elapsed / step
                usable = budget * 0.85 - elapsed  # 15% kept for dev evaluation and saving
                fit = step + int(usable / sec_per_step)
                state["fitted_to_time"] = True
                if fit < state["total_steps"]:
                    log({"note": f"{sec_per_step:.2f}s/step: schedule shortened from {state['total_steps']} to {fit} steps to fit {args.max_hours} h"})
                    state["total_steps"] = sched_state["total_steps"] = max(fit, state["warmup_steps"] + 1)
                else:
                    log({"note": f"{sec_per_step:.2f}s/step: all {state['total_steps']} steps fit in {args.max_hours} h"})

            if step % 50 == 0 or step == 1:
                log({"step": step, "epoch": epoch, "loss": round(float(np.mean(running)), 4),
                     "lr": sched.get_last_lr()[0], "minutes": round(elapsed / 60, 1)})
                running = []
            done = step >= state["total_steps"]
            stop_for_time = elapsed > budget * 0.95
            if step % args.eval_every == 0 or done or stop_for_time:
                evaluate(step)
            if step % args.save_every == 0 or done or stop_for_time:
                state["train_seconds"] = prev_seconds + (time.time() - t0)
                save_checkpoint(out, model, opt, sched, scaler, state)
            if done or stop_for_time:
                break
        else:
            state["epoch"] += 1
            state["batch_in_epoch"] = 0

    finished = state["step"] >= state["total_steps"]
    print("Training finished." if finished else "Stopped for time. Run the same command again to continue.")

    # ---- run log (benchmark/results/C2_xlsr-300m-ctc.json)
    update_run_log(SYSTEM, {
        "system": SYSTEM, "model_id": args.model_id, "revision": sha, "parameters": n_params,
        "trainable_parameters": sum(p.numel() for p in trainable), "seed": SEED,
        "train_data": {"clips": len(train_rows), "hours": round(sum(lengths) / 3600, 3),
                       "skipped_too_long": len(too_long), "skipped_empty_text": dropped_empty},
        "hyperparameters": hparams | {"effective_batch": args.batch_size * args.grad_accum,
                                      "steps_per_epoch": steps_per_epoch, "planned_steps": state["planned_steps"],
                                      "total_steps": state["total_steps"], "scheduler": "linear warmup + linear decay",
                                      "optimizer": "AdamW", "fp16": fp16, "feature_encoder": "frozen",
                                      "dropouts": "attention/hidden/feat_proj 0.0, layerdrop 0.0"},
        "training": {"finished": finished, "steps_done": state["step"], "best_step": state["best_step"],
                     "best_dev_wer_quick": state["best_wer"], "train_hours": round(state["train_seconds"] / 3600, 2)},
        "chosen_checkpoint": f"step {state['best_step']} (lowest dev WER)",
        "env": env_info(),
    })


if __name__ == "__main__":
    main()
