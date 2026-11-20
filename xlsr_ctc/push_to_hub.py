"""Upload the best C2 checkpoint to your Hugging Face account, with a model card.

The token is NEVER written in code. On Kaggle: Add-ons > Secrets > add HF_TOKEN (a "write" token),
then the notebook reads it. Elsewhere: run `huggingface-cli login` or set the HF_TOKEN variable.

Run from the repo root:
    python xlsr_ctc/push_to_hub.py --model-dir /kaggle/working/xlsr_run/best --repo <hf-user>/xls-r-300m-uzbek-usc
Run it again after the test run: the card then also shows the test scores from the run log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.asr_data import RESULTS_DIR  # noqa: E402
from xlsr_lib import BASE_MODEL, SYSTEM  # noqa: E402

CARD = """---
language: uz
license: apache-2.0
base_model: {base}
datasets:
- issai/Uzbek_Speech_Corpus
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
tags:
- ctc
- wav2vec2
- uzbek
---

# XLS-R 300M fine-tuned for Uzbek speech recognition (CTC)

Fine-tuned from [`{base}`](https://huggingface.co/{base}) (revision `{base_rev}`) with a character
CTC head, on a 25-hour, speaker-based subset of the Uzbek Speech Corpus (USC, ISSAI) train split.
Part of a university course project (COE025, Project 3). Code: {repo_url}

## Data

- Train: {train_clips} clips, {train_hours} h, whole speakers chosen from the official USC train split (seed 42), after the team's cleaning steps.
- Checkpoint chosen on the official USC dev split (step {best_step}).
- Test: official USC test (speakers not in train).
- Text is lowercase Latin Uzbek with one apostrophe form (`o'`, `g'`, `'`) and no punctuation.

## Results (greedy decoding, no language model)

| Set | WER | CER |
|---|---|---|
{rows}

WER/CER are on normalized text (lowercase, no punctuation). Official numbers: `benchmark/results/scores.csv` in the code repo.

## Training settings

{hparams}

## Usage

```python
import soundfile as sf, torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

repo = "{repo_id}"
processor = Wav2Vec2Processor.from_pretrained(repo)
model = Wav2Vec2ForCTC.from_pretrained(repo).eval()

audio, sr = sf.read("clip.wav")  # 16 kHz mono
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    ids = model(**inputs).logits.argmax(-1)
print(processor.batch_decode(ids)[0])
```

## Data licenses

USC: MIT on the Hugging Face card, CC BY 4.0 in the paper (https://arxiv.org/abs/2107.14419).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--repo", required=True, help="<hf-user>/<model-name>")
    ap.add_argument("--code-url", default="(GitHub link of the team repo)")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import HfApi

    log_path = RESULTS_DIR / f"{SYSTEM}.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    runs = log.get("runs", {})
    names = {"dev": "USC dev", "test": "USC test"}
    rows = "\n".join(f"| {names[s]} | {runs[s]['wer_quick']:.3f} | {runs[s]['cer_quick']:.3f} |"
                     for s in ("dev", "test") if s in runs) or "| (not run yet) | | |"
    hp = log.get("hyperparameters", {})
    keep = ("lr", "warmup_steps", "batch_size", "grad_accum", "effective_batch", "total_steps", "optimizer",
            "scheduler", "fp16", "feature_encoder", "mask_time_prob", "dropouts")
    hparams = "\n".join(f"- {k}: {hp[k]}" for k in keep if k in hp) + "\n- seed: 42"
    td = log.get("train_data", {})
    card = CARD.format(base=log.get("model_id", BASE_MODEL), base_rev=log.get("revision"),
                       repo_url=args.code_url, train_clips=td.get("clips"), train_hours=td.get("hours"),
                       best_step=log.get("training", {}).get("best_step"), rows=rows, hparams=hparams,
                       repo_id=args.repo)
    Path(args.model_dir, "README.md").write_text(card, encoding="utf-8")

    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    commit = api.upload_folder(repo_id=args.repo, folder_path=args.model_dir,
                               commit_message="Upload C2 XLS-R CTC model")
    sha = getattr(commit, "oid", None)
    print(f"Uploaded: https://huggingface.co/{args.repo}  revision {sha}")
    from common.asr_data import update_run_log

    update_run_log(SYSTEM, {"huggingface": {"repo": args.repo, "revision": sha}})


if __name__ == "__main__":
    main()
