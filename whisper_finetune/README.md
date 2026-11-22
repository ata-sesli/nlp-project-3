# C1: whisper-small fine-tune (our contribution)

`openai/whisper-small` fine-tuned on our 25-hour train split
(`data/clean/train.csv`, clips up to 30 s). Starting settings from the plan:
lr 1e-5, 500 warmup steps, linear decay, effective batch 32, bf16 on A100
(fp16 on GPUs without bf16), 4000 steps, dev WER every 500 steps, best dev-WER
checkpoint kept, seed 42. All choices are made on dev.

| File | Contents |
|------|----------|
| `train.py` | Fine-tuning with the Hugging Face `Seq2SeqTrainer`; writes checkpoints to `--output-dir`, the best model to `<output-dir>/best`, and the run log `benchmark/results/C1_whisper-small-ft.json` |
| `push_to_hub.py` | Uploads the best model with a model card to Hugging Face and prints its revision |
| `colab_whisper.ipynb` | Colab notebook for A4, A5 and C1: sparse clone, install, dev runs, training, upload, the single test run |
| `requirements-whisper.txt` | Pinned packages for A4, A5 and C1 (on top of Colab's preinstalled PyTorch, whose version is in the run log) |

Predictions for dev and test are made with
`baselines/whisper_zero_shot/transcribe.py --model <hub id> --system C1_whisper-small-ft`,
the same inference code as the zero-shot baselines.

Model weights are not in git. The model is on Hugging Face:
[`akifkulll/whisper-small-uzbek-usc25h`](https://huggingface.co/akifkulll/whisper-small-uzbek-usc25h),
revision `b38781ed83b508cbd44ce5fdcefa488efe849d29`. This revision made all dev and test predictions. The
later revision `fdd98ccc585456041474fc6cb20c8daca48e34e8` only updates the model card (test results); the
weights file is the same.

## Dev results (all 3,783 dev clips, normalized text, greedy decoding)

Training ran on a Colab A100 40 GB (about 49 minutes). Dev WER after every 500 steps:

| Step | 500 | 1000 | 1500 | 2000 | 2500 | **3000** | 3500 | 4000 |
|---|---|---|---|---|---|---|---|---|
| Dev WER | 49.4% | 40.8% | 37.2% | 36.0% | 34.9% | **33.85%** | 34.3% | 34.0% |

The best checkpoint (step 3000) is the uploaded model. Downloaded from Hugging Face and run again with
`transcribe.py`, it gives the numbers below.

| Row | Model | WER | CER |
|---|---|---:|---:|
| A4 | `openai/whisper-small`, zero-shot | 136.47% | 67.47% |
| A5 | `openai/whisper-large-v3`, zero-shot | 96.31% | 34.28% |
| **C1** | `openai/whisper-small` fine-tuned on our train split | **33.83%** | **9.99%** |

These are quick scores with the normalizer copy in `whisper_common.py`; the official numbers come from
`benchmark/evaluate.py`.

## Test results (all 3,836 test clips, one run after the freeze, Colab T4)

| Row | Model | Dev WER | Test WER | Test CER | Test RTF (T4) |
|---|---|---:|---:|---:|---:|
| A4 | `openai/whisper-small`, zero-shot | 136.47% | 119.87% | 53.19% | 0.0379 |
| A5 | `openai/whisper-large-v3`, zero-shot | 96.31% | 93.67% | 32.70% | 0.0850 |
| **C1** | `openai/whisper-small` fine-tuned | **33.83%** | **35.56%** | **9.91%** | **0.0201** |

C1's test WER is only 1.7 points above dev and its CER is the same (9.99% -> 9.91%), so the step-3000 choice
on dev did not overfit dev. Test clips are longer on average (4.1 s vs 3.8 s), and C1's WER hardly changes with
clip length. C1 is also the fastest of the three on the same T4: about 2x faster than zero-shot small, which
loops on some clips.
