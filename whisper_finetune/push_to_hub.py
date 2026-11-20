"""Upload the C1 model (whisper_finetune/train.py's <output-dir>/best) to Hugging Face
with a model card, and print the commit hash to put in the README and run log.

The token is never written in code: log in first with `huggingface-cli login`,
or in Colab store it as the secret HF_TOKEN (the notebook reads it into the
HF_TOKEN environment variable).

Scores in the card come from benchmark/results/C1_whisper-small-ft.json: the dev
WER from training, and the test WER/CER that transcribe.py records for the
single test run after the freeze (rerun this script then to update the card).
Check that they match benchmark/results/scores.csv from evaluate.py.

Usage:
    python whisper_finetune/push_to_hub.py --model-dir /content/drive/MyDrive/c1_whisper/best \
        --repo-id <hf-user>/whisper-small-uzbek-usc25h
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
RUN_LOG = ROOT / "benchmark" / "results" / "C1_whisper-small-ft.json"
GITHUB = "https://github.com/ata-sesli/nlp-project-3"

CARD = """---
language: uz
license: apache-2.0
base_model: openai/whisper-small
datasets:
- issai/Uzbek_Speech_Corpus
metrics:
- wer
- cer
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- uzbek
---

# Whisper-small fine-tuned on 25 h of Uzbek (USC)

`openai/whisper-small` (revision `{base_revision}`) fine-tuned on a 25-hour,
speaker-disjoint subset of the [Uzbek Speech Corpus]({usc}) (Latin script).
Course project COE025 Project 3; code, data splits and evaluation: {github}.

## Training

| | |
|---|---|
| Train data | {train_hours} h, {train_clips} clips of USC train (speakers not in USC dev/test, seed 42) |
| Hyperparameters | {hp} |
| Selected checkpoint | {ckpt} (best dev WER) |
| Hardware | {hardware} |

## Results (normalized text: lowercase, no punctuation, apostrophes kept)

| Set | WER | CER |
|---|---|---|
| USC dev (during training) | {dev_wer} | |
| USC test | {test_wer} | {test_cer} |

## Usage

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="{repo_id}")
print(asr("clip.wav", generate_kwargs={{"language": "uzbek", "task": "transcribe"}})["text"])
```

## Limitations

Trained and evaluated only on USC read speech; clips longer than 30 s were not
used in training.
"""


def pct(x) -> str:
    return f"{x:.2%}" if isinstance(x, (int, float)) else "n/a"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    log = json.loads(RUN_LOG.read_text(encoding="utf-8"))
    train = log["runs"]["train"]
    test = log["runs"].get("test", {}).get("provisional_scores", {})
    card = CARD.format(
        base_revision=log["base_revision"], usc="https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus",
        github=GITHUB, train_hours=train["train_hours"], train_clips=train["train_clips"],
        hp=", ".join(f"{k} {v}" for k, v in log["hyperparameters"].items()),
        ckpt=train["selected_checkpoint"], hardware=log["hardware"],
        dev_wer=pct(train["best_dev_wer_during_training"]),
        test_wer=pct(test.get("wer")), test_cer=pct(test.get("cer")), repo_id=args.repo_id,
    )
    (args.model_dir / "README.md").write_text(card, encoding="utf-8")

    api = HfApi()
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    commit = api.upload_folder(repo_id=args.repo_id, folder_path=str(args.model_dir),
                               commit_message="Upload C1 whisper-small Uzbek fine-tune")
    print(f"uploaded https://huggingface.co/{args.repo_id}  revision {commit.oid}")


if __name__ == "__main__":
    main()
