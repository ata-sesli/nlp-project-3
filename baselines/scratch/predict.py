"""A2 / A3: transcribe dev / test with the best epoch (best.pt).

Writes benchmark/predictions/<system>__<split>.jsonl and adds RTF + quick WER to
benchmark/results/<system>.json.

Run from the repo root:
    python baselines/scratch/predict.py --ckpt /kaggle/working/a2_ctc/best.pt --split dev
    python baselines/scratch/predict.py --ckpt .../best.pt --split test --allow-test   # after the freeze
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.asr_data import (RTFTimer, check_test_allowed, env_info, read_split, set_seed,  # noqa: E402
                             update_run_log, wer_cer, write_predictions)
from model import build_model  # noqa: E402
from train import SYSTEMS, decode_all, read_audio_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", nargs="+", default=["dev"], choices=["dev", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--clean-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-seconds", type=float, default=400)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    for split in args.split:
        check_test_allowed(split, args.allow_test)
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["kind"], len(ck["vocab"]), ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    system = SYSTEMS[ck["kind"]]
    for split in args.split:
        rows = read_split(split, args.clean_dir, args.limit)
        audio = read_audio_all(rows, args.num_workers)
        lengths = [len(a) / 16000 for a in audio]
        timer = RTFTimer()
        timer.start()
        hyps = decode_all(model, audio, lengths, ck["vocab"], args.batch_seconds, device, device == "cuda")
        timer.stop(sum(lengths))
        path = write_predictions(system, split, [r["id"] for r in rows], hyps)
        wer, cer = wer_cer([r["transcript"] for r in rows], hyps)
        print(f"{system} {split}: {len(rows)} clips  WER {wer:.4f}  CER {cer:.4f}  RTF {timer.rtf:.4f}  -> {path}")
        update_run_log(system, {"decoder": "greedy", "chosen_epoch": ck["epoch"],
                                "runs": {split: {"clips": len(rows), "wer_quick": round(wer, 4),
                                                 "cer_quick": round(cer, 4), "rtf": round(timer.rtf, 5),
                                                 "audio_hours": round(timer.audio / 3600, 3),
                                                 "gpu": env_info().get("gpu"), "limit": args.limit}},
                                "notes": {"wer_quick": "normalized WER from common/asr_data.py; the official numbers come from benchmark/evaluate.py"}})


if __name__ == "__main__":
    main()
