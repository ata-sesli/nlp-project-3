"""C4a / C4b: our fine-tuned FastConformer (nvidia_finetune/train.py) on dev / test.

C4a = transducer head, C4b = CTC head of the same fine-tuned model (as A8 / A9 for the start model).
Writes benchmark/predictions/<system>__<split>.jsonl and benchmark/results/<system>.json.

Run from the repo root:
    python nvidia_finetune/predict.py --model /kaggle/working/c4_run/best.nemo --split dev
    python nvidia_finetune/predict.py --model <hf-user>/stt-uz-fastconformer-usc --split test --allow-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "baselines" / "nvidia_fastconformer")]
from common.asr_data import (RTFTimer, check_test_allowed, env_info, hf_revision, read_split,  # noqa: E402
                             set_seed, update_run_log, wer_cer, write_predictions)
from nemo_utils import count_params, set_head, transcribe_rows  # noqa: E402

SYSTEMS = {"rnnt": "C4a_fastconformer-usc-ft-rnnt", "ctc": "C4b_fastconformer-usc-ft-ctc"}


def load_finetuned(model: str, revision: str | None):
    """A local .nemo file, or a Hugging Face repo that holds one. Returns (model, source, revision)."""
    import torch
    from nemo.collections.asr.models import ASRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Path(model).exists():
        m = ASRModel.restore_from(model, map_location=device)
        return m.eval(), str(model), None
    from huggingface_hub import hf_hub_download, list_repo_files

    sha = hf_revision(model, revision)
    name = [f for f in list_repo_files(model, revision=sha) if f.endswith(".nemo")][0]
    m = ASRModel.restore_from(hf_hub_download(model, name, revision=sha), map_location=device)
    return m.eval(), model, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="best.nemo from train.py, or a Hugging Face repo id")
    ap.add_argument("--revision", default=None, help="commit hash when --model is a Hugging Face repo")
    ap.add_argument("--split", nargs="+", default=["dev"], choices=["dev", "test"])
    ap.add_argument("--heads", nargs="+", default=["rnnt", "ctc"], choices=["rnnt", "ctc"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--clean-dir", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-test", action="store_true", help="needed for test sets (after the freeze)")
    args = ap.parse_args()

    for split in args.split:
        check_test_allowed(split, args.allow_test)
    set_seed()
    model, source, sha = load_finetuned(args.model, args.revision)
    n_params = count_params(model)
    for head in args.heads:
        set_head(model, head)
        system = SYSTEMS[head]
        for split in args.split:
            rows = read_split(split, args.clean_dir, args.limit)
            timer = RTFTimer()
            hyps = transcribe_rows(model, rows, batch_size=args.batch_size, timer=timer)
            out = write_predictions(system, split, [r["id"] for r in rows], hyps)
            wer, cer = wer_cer([r["transcript"] for r in rows], hyps)
            print(f"[{system}] {split}: WER {wer:.4f}  CER {cer:.4f}  RTF {timer.rtf:.4f}  -> {out}")
            update_run_log(system, {
                "system": system,
                "model": "NVIDIA Uzbek FastConformer hybrid, fine-tuned by us on USC train (nvidia_finetune/)",
                "prediction_model": {"source": source, "revision": sha},
                "decoder": "transducer greedy" if head == "rnnt" else "CTC greedy",
                "parameters": n_params, "seed": 42, "env": env_info(),
                "runs": {split: {"clips": len(rows), "wer_quick": round(wer, 4), "cer_quick": round(cer, 4),
                                 "rtf": round(timer.rtf, 5), "audio_hours": round(timer.audio / 3600, 3),
                                 "limit": args.limit}},
                "notes": {"wer_quick": "normalized WER from common/asr_data.py; the official numbers come from "
                                       "benchmark/evaluate.py"},
            })


if __name__ == "__main__":
    main()
