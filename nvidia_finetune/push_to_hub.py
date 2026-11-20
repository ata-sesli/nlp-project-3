"""Upload the fine-tuned FastConformer (C4) to your Hugging Face account, with a model card.

The token is NEVER written in code: on Kaggle add the HF_TOKEN secret (a "write" token).

Run from the repo root (again after the test run, so the card shows the test scores too):
    python nvidia_finetune/push_to_hub.py --nemo <run>/best.nemo --repo <hf-user>/stt-uz-fastconformer-usc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.asr_data import RESULTS_DIR, update_run_log  # noqa: E402

START = "nvidia/stt_uz_fastconformer_hybrid_large_pc"
SYSTEMS = {"transducer head (C4a)": "C4a_fastconformer-usc-ft-rnnt", "CTC head (C4b)": "C4b_fastconformer-usc-ft-ctc"}
# the start model as it is (our baselines A8/A9), scored the same way, for comparison
BASELINES = {"start model, transducer head (A8)": "A8_nvidia-fastconformer-rnnt",
             "start model, CTC head (A9)": "A9_nvidia-fastconformer-ctc"}

CARD = """---
language: uz
base_model: {start}
datasets:
- issai/Uzbek_Speech_Corpus
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
tags:
- nemo
- fastconformer
- uzbek
---

# NVIDIA Uzbek FastConformer, fine-tuned on the Uzbek Speech Corpus

Fine-tuned from [`{start}`](https://huggingface.co/{start}) (revision `{start_rev}`) on a
speaker-based subset of the Uzbek Speech Corpus (USC, ISSAI) train split ({hours} h after our cleaning steps).
Both heads of the hybrid model (transducer and CTC) were trained together; the start model's
tokenizer is unchanged. Part of a university course project (COE025, Project 3). Code: {code_url}

Training: {clips} clips; learning rate {lr} (chosen on dev); best epoch {best_epoch} of {epochs}
by WER on a fixed sample of {dev_clips} dev clips.
Output text is lowercase with no punctuation (like our training text).
License: see the start model's card.

## Results (greedy decoding, no language model)

| Model | Set | WER | CER |
|---|---|---|---|
{rows}

WER/CER are on normalized text. Official numbers: `benchmark/results/scores.csv` in the code repo.

## Usage

```python
import nemo.collections.asr as nemo_asr
from huggingface_hub import hf_hub_download

model = nemo_asr.models.ASRModel.restore_from(hf_hub_download("{repo_id}", "{nemo_name}"))
print(model.transcribe(["clip.wav"])[0])          # transducer head (default)
model.change_decoding_strategy(decoder_type="ctc")
print(model.transcribe(["clip.wav"])[0])          # CTC head
```
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--code-url", default="")
    ap.add_argument("--dry-run", action="store_true", help="only print the model card (quick check)")
    args = ap.parse_args()

    rows, train = [], {}
    for split in ("dev", "test"):
        for label, system in {**SYSTEMS, **BASELINES}.items():
            path = RESULTS_DIR / f"{system}.json"
            log = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if system in SYSTEMS.values():
                train = log.get("training", train)
            r = log.get("runs", {}).get(split)
            if r:
                rows.append(f"| {label} | USC {split} | {r['wer_quick']:.2%} | {r['cer_quick']:.2%} |")
    t = train.get("training", {})
    nemo_name = "stt_uz_fastconformer_usc_ft.nemo"
    card = CARD.format(start=START, start_rev=train.get("start_revision"), code_url=args.code_url,
                       clips=train.get("train_data", {}).get("clips"),
                       hours=round(train.get("train_data", {}).get("hours") or 0, 1),
                       best_epoch=t.get("best_epoch"), epochs=t.get("epochs_done"),
                       lr=train.get("hyperparameters", {}).get("lr"), dev_clips=t.get("dev_check_clips"),
                       rows="\n".join(rows) or "| | | | |", repo_id=args.repo, nemo_name=nemo_name)
    if args.dry_run:
        print(card)
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, exist_ok=True)
    api.upload_file(path_or_fileobj=args.nemo, path_in_repo=nemo_name, repo_id=args.repo)
    info = api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=args.repo)
    sha = getattr(info, "oid", None) or api.model_info(args.repo).sha
    print(f"Uploaded to https://huggingface.co/{args.repo} (revision {sha})")
    for system in SYSTEMS.values():
        update_run_log(system, {"huggingface": {"repo": args.repo, "revision": sha, "file": nemo_name}})


if __name__ == "__main__":
    main()
