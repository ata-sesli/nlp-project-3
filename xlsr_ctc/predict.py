"""C2: transcribe dev / test with the best fine-tuned XLS-R checkpoint.

Writes:
  benchmark/predictions/C2_xlsr-300m-ctc__<split>.jsonl
  benchmark/results/C2_xlsr-300m-ctc.json   (adds RTF and quick WER per split)
  <logits-dir>/<split>/<id>.npy             log-probabilities for the KenLM decoder (C3, Elif)
  <logits-dir>/labels.json                  token list in id order ('' = CTC blank, ' ' = word separator)

Run from the repo root:
    python xlsr_ctc/predict.py --model /kaggle/working/xlsr_run/best --split dev --logits-dir /kaggle/working/logits
    python xlsr_ctc/predict.py --model <hf-user>/<repo> --revision <hash> --split test --allow-test --logits-dir ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.asr_data import (RTFTimer, check_test_allowed, env_info, read_split, set_seed,  # noqa: E402
                             update_run_log, wer_cer, write_predictions)
from xlsr_lib import SYSTEM, ctc_labels, save_json, transcribe  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="folder (e.g. <out>/best) or Hugging Face repo id")
    ap.add_argument("--revision", default=None, help="commit hash when --model is a Hugging Face repo")
    ap.add_argument("--split", nargs="+", default=["dev"], choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true", help="needed for test sets (after the freeze)")
    ap.add_argument("--logits-dir", default=None, help="save log-probabilities here (for C3)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--clean-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    for split in args.split:
        check_test_allowed(split, args.allow_test)
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(args.model, revision=args.revision)
    model = Wav2Vec2ForCTC.from_pretrained(args.model, revision=args.revision).to(device)
    best = Path(args.model) / "best.json"
    chosen = json.loads(best.read_text()) if best.exists() else None

    if args.logits_dir:
        Path(args.logits_dir).mkdir(parents=True, exist_ok=True)
        save_json(Path(args.logits_dir) / "labels.json", ctc_labels(processor))

    for split in args.split:
        rows = read_split(split, args.clean_dir, args.limit)
        timer = RTFTimer()
        logits_dir = Path(args.logits_dir) / split if args.logits_dir else None
        hyps = transcribe(model, processor, rows, args.batch_size, device, True, args.num_workers, timer, logits_dir)
        path = write_predictions(SYSTEM, split, [r["id"] for r in rows], hyps)
        wer, cer = wer_cer([r["transcript"] for r in rows], hyps)
        print(f"{split}: {len(rows)} clips  WER {wer:.4f}  CER {cer:.4f}  RTF {timer.rtf:.4f}  -> {path}")
        update_run_log(SYSTEM, {
            "decoder": "CTC greedy (no LM)",
            "prediction_model": {"source": args.model, "revision": args.revision, "chosen": chosen},
            "runs": {split: {"clips": len(rows), "wer_quick": round(wer, 4), "cer_quick": round(cer, 4),
                             "rtf": round(timer.rtf, 5), "audio_hours": round(timer.audio / 3600, 3),
                             "gpu": env_info().get("gpu"), "limit": args.limit}},
            "notes": {"wer_quick": "normalized WER from common/asr_data.py; the official numbers come from benchmark/evaluate.py"},
        })


if __name__ == "__main__":
    main()
