"""A8 / A9: NVIDIA Uzbek FastConformer, zero-shot, on dev / test.

A8 = transducer head (the model's default), A9 = CTC head of the same model.
Writes benchmark/predictions/<system>__<split>.jsonl and benchmark/results/<system>.json.

Examples (run from the repo root):
    python baselines/nvidia_fastconformer/run_nvidia.py --split dev
    python baselines/nvidia_fastconformer/run_nvidia.py --split test --allow-test   # only after the freeze
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.asr_data import (RTFTimer, check_test_allowed, env_info, read_split,  # noqa: E402
                             set_seed, update_run_log, wer_cer, write_predictions)
from nemo_utils import MODEL_ID, count_params, load_model, set_head, transcribe_rows  # noqa: E402

SYSTEMS = {"rnnt": "A8_nvidia-fastconformer-rnnt", "ctc": "A9_nvidia-fastconformer-ctc"}

# Model card: trained on Common Voice, UzbekVoice and FLEURS; USC is not in its training data.
DATA_NOTE = "model card: trained on Common Voice, UzbekVoice and FLEURS; USC is not in its training data"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", default=["dev"], choices=["dev", "test"])
    ap.add_argument("--heads", nargs="+", default=["rnnt", "ctc"], choices=["rnnt", "ctc"])
    ap.add_argument("--revision", default=None, help="model commit hash; default = latest, recorded in the log")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--clean-dir", default=None, help="folder with <split>.csv (default data/clean)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (quick check)")
    ap.add_argument("--allow-test", action="store_true", help="needed for test sets (after the freeze)")
    args = ap.parse_args()

    for split in args.split:
        check_test_allowed(split, args.allow_test)
    set_seed()
    model, sha = load_model(args.revision)
    n_params = count_params(model)
    print(f"Loaded {MODEL_ID} @ {sha}, {n_params / 1e6:.1f}M parameters")

    for head in args.heads:
        set_head(model, head)
        system = SYSTEMS[head]
        for split in args.split:
            rows = read_split(split, args.clean_dir, args.limit)
            print(f"[{system}] {split}: {len(rows)} clips")
            timer = RTFTimer()
            hyps = transcribe_rows(model, rows, batch_size=args.batch_size, timer=timer)
            out = write_predictions(system, split, [r["id"] for r in rows], hyps)
            wer, cer = wer_cer([r["transcript"] for r in rows], hyps)
            print(f"[{system}] {split}: WER {wer:.4f}  CER {cer:.4f}  RTF {timer.rtf:.4f}  -> {out}")
            info = {
                "system": system,
                "model_id": MODEL_ID,
                "revision": sha,
                "decoder": "transducer greedy (model default)" if head == "rnnt" else "CTC greedy",
                "training": "none (zero-shot)",
                "parameters": n_params,
                "seed": 42,
                "batch_size": args.batch_size,
                "env": env_info(),
                "runs": {split: {"clips": len(rows), "wer_quick": round(wer, 4), "cer_quick": round(cer, 4),
                                 "rtf": round(timer.rtf, 5), "audio_hours": round(timer.audio / 3600, 3),
                                 "limit": args.limit}},
                "notes": {"training_data": DATA_NOTE,
                          "wer_quick": "normalized WER from common/asr_data.py; the official numbers come from benchmark/evaluate.py"},
            }
            update_run_log(system, info)


if __name__ == "__main__":
    main()
