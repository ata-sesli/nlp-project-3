"""C1: fine-tune openai/whisper-small on our 25-hour USC train split.

Starting settings (roadmap): lr 1e-5, 500 warmup steps, mixed precision,
effective batch 32, dev WER every 500 steps, the checkpoint with the best dev
WER is kept. Seed 42. Checkpoints are saved every --save-steps into
--output-dir (put it on Google Drive in Colab) so a dropped session can resume
with --resume.

Text targets are the clean CSV transcripts as they are. Dev WER during training
uses the shared normalizer (whisper_common.normalize).

Writes
    <output-dir>/checkpoint-*/          Trainer checkpoints (not in git)
    <output-dir>/best/                  best dev-WER model + processor
    benchmark/results/C1_whisper-small-ft.json   hyperparameters, best step, dev WER

Then make the dev (and, after the freeze, test) predictions with
baselines/whisper_zero_shot/transcribe.py --model <output-dir>/best.

Usage:
    python whisper_finetune/train.py --output-dir /content/drive/MyDrive/c1_whisper
    python whisper_finetune/train.py --output-dir ... --resume       # continue after a disconnect
    python whisper_finetune/train.py --output-dir ... --smoke        # 60 steps on 512 clips, pipeline check
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baselines" / "whisper_zero_shot"))
import whisper_common as wc  # noqa: E402

SYSTEM = "C1_whisper-small-ft"
MAX_LABEL_TOKENS = 448  # Whisper decoder limit


class WhisperDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict], processor: WhisperProcessor):
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        feats = self.processor.feature_extractor(
            wc.load_audio(row), sampling_rate=wc.SAMPLE_RATE).input_features[0]
        labels = self.processor.tokenizer(row["transcript"]).input_ids
        return {"input_features": feats, "labels": labels[:MAX_LABEL_TOKENS]}


@dataclass
class Collator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        # The model prepends the decoder start token itself; drop it from the labels.
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=4000, help="~5 epochs of 25 h at batch 32")
    p.add_argument("--batch-size", type=int, default=32, help="per device")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500, help="must be a multiple of --eval-steps")
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--spec-augment", action="store_true", help="SpecAugment time/feature masking")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --output-dir")
    p.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    args = p.parse_args()
    if args.smoke:
        args.max_steps, args.eval_steps, args.save_steps, args.warmup_steps = 60, 30, 30, 10
    set_seed(wc.SEED)

    revision = wc.hub_revision(args.model)
    processor = WhisperProcessor.from_pretrained(args.model, revision=revision)
    processor.tokenizer.set_prefix_tokens(language="uzbek", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, revision=revision)
    model.generation_config.language = "uzbek"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_feature_prob = 0.05

    train_rows = [r for r in wc.load_split("train") if r["end_s"] - r["start_s"] <= 30.0]
    dev_rows = wc.load_split("dev")
    if args.smoke:
        train_rows, dev_rows = train_rows[:512], dev_rows[:128]
    print(f"train {len(train_rows)} clips, {sum(r['end_s'] - r['start_s'] for r in train_rows) / 3600:.2f} h; "
          f"dev {len(dev_rows)} clips")

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        hyps = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        s = wc.provisional_scores(refs, hyps)
        return {"wer": s["wer"], "cer": s["cer"]}

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=args.workers,
        remove_unused_columns=False,
        seed=wc.SEED,
        data_seed=wc.SEED,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=WhisperDataset(train_rows, processor),
        eval_dataset=WhisperDataset(dev_rows, processor),
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    best_dir = args.output_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    state = trainer.state
    history = [h for h in state.log_history if "eval_wer" in h]
    (args.output_dir / "eval_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if args.smoke:
        print(f"smoke run done; best dev WER {state.best_metric:.2%} (not a result)")
        return
    log_path = wc.write_run_log(SYSTEM, "train", {
        "system": SYSTEM,
        "base_model_id": args.model,
        "base_revision": revision,
        "parameters": sum(p.numel() for p in model.parameters()),
        "hyperparameters": {
            "learning_rate": args.lr, "warmup_steps": args.warmup_steps, "max_steps": args.max_steps,
            "per_device_batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum * max(torch.cuda.device_count(), 1),
            "precision": "bf16" if bf16 else "fp16", "scheduler": "linear",
            "spec_augment": args.spec_augment, "eval_every_steps": args.eval_steps,
        },
        "seed": wc.SEED,
        **wc.environment(),
        "run": {
            "train_clips": len(train_rows),
            "train_hours": round(sum(r["end_s"] - r["start_s"] for r in train_rows) / 3600, 3),
            "selected_checkpoint": Path(state.best_model_checkpoint).name if state.best_model_checkpoint else None,
            "best_dev_wer_during_training": state.best_metric,
            "dev_wer_history": [{"step": h["step"], "wer": h["eval_wer"], "cer": h["eval_cer"]} for h in history],
        },
    })
    print(f"best checkpoint {state.best_model_checkpoint}, dev WER {state.best_metric:.2%}")
    print(f"saved {best_dir}\nwrote {log_path}")


if __name__ == "__main__":
    main()
