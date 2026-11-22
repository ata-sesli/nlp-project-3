# C4: NVIDIA Uzbek FastConformer, fine-tuned on our USC train split (our contribution)

Start checkpoint: [`nvidia/stt_uz_fastconformer_hybrid_large_pc`](https://huggingface.co/nvidia/stt_uz_fastconformer_hybrid_large_pc), the same model that is used as-is for the baselines A8/A9 (exact revision in the run log). Here it is trained further on our 25 h train split, so A8/A9 vs C4 shows what our training adds.

The model has one encoder and two decoders, so C4 gives two rows:

| Row | Head | Decoding |
|---|---|---|
| C4a | Transducer (RNN-T) | greedy |
| C4b | CTC | greedy |

## Why C4 was added (it was not in the first plan)

The plan had three contributions: C1 (Whisper-small fine-tuned), C2 (XLS-R 300M + CTC) and C3 (C2 + a word language
model). After the first training run, C2 reached 34.05% dev WER, while the ready NVIDIA model (A8), with no training on
our data, already reached 14.25%. So our planned fine-tune was far worse than a ready baseline.

The reason: XLS-R was pre-trained on speech without any transcripts, so with our 24 hours it had to learn Uzbek
speech recognition almost from the start. The NVIDIA model was already trained for Uzbek (Common Voice, UzbekVoice and
FLEURS), but never on USC. So we fine-tuned the NVIDIA model itself on our USC train split, to adapt it to USC's
speakers, recording conditions and text style. C2 was still improving when its run ended, so we also trained it 20
more epochs (C2long).

Problems we hit while setting up C4, and what we did:

- With 16-bit (fp16) numbers on the T4, the model produced invalid values and wrote only `<unk>`. We train in
  full 32-bit precision instead, and training stops at the start if the start model scores badly on dev.
- In 32-bit, a batch of 16 of the longest clips may not fit in the T4's 15 GB, so a short memory test on the longest
  clips runs first and picks a smaller batch if needed (16 fit, using 11.4 GB).
- We did not keep the default learning rate: four values were compared on dev for one epoch each, and the best one was used.

## Settings

| Setting | Value |
|---|---|
| Training | both heads together, with the model's own loss weights |
| Tokenizer | the start model's SentencePiece tokenizer, unchanged; our apostrophe is written in the form the tokenizer knows |
| Text | `data/normalize.py` (lowercase, no punctuation), so the fine-tuned model also writes lowercase without punctuation |
| Audio | `data/clean/train.csv`; silence trimming through `offset`/`duration` in the NeMo manifest |
| Learning rate | chosen on dev: 5e-5, 1e-4, 2e-4 and 4e-4 each train 1 epoch from the start model and are scored on the same 1,000 dev clips; the best one is used (`benchmark/results/C4_lr_search.json`) |
| Optimizer | AdamW (0.9, 0.98), weight decay 1e-3, warmup 500 steps, then cosine to 1e-6 |
| Batch | 16 clips x 2 accumulation steps = 32. Before training, `--memory-probe` runs one step on the longest clips; if 16 clips do not fit in the T4's memory, the notebook uses 8 x 4 (or 4 x 8), so the total stays 32 |
| Other | full fp32 precision (with fp16 the start model wrote only `<unk>` on a T4), gradient clipping 1.0, the model's own SpecAugment, seed 42 |
| Checkpoint choice | dev WER (transducer head) on a fixed sample of 1,000 dev clips (seed 42) after every epoch; the best epoch is kept. The start model is scored on the same clips before training, so the gain is visible in the log. If that start score is above 50% WER (A8 gets about 14%), training stops before it begins, because something is broken |
| Length | as many epochs as fit in the Kaggle time (from the speed measured in the learning-rate search), at most 30; `--max-hours` stops it in any case |

## Results

Official scores from `benchmark/evaluate.py`: all 3,783 dev and 3,836 test clips, normalized text, greedy decoding.
A8/A9 are the same model before our training. Training and the choices below used dev only; test was run once after
the freeze (notebook step `test`).

| Row | Model | Dev WER | Dev CER | Test WER | Test CER |
|---|---|---:|---:|---:|---:|
| A8 | start model, transducer head | 14.25% | 5.01% | 12.07% | 3.65% |
| **C4a** | fine-tuned, transducer head | **11.60%** | 3.93% | **11.38%** | 3.41% |
| A9 | start model, CTC head | 15.95% | 5.21% | 13.08% | 3.61% |
| **C4b** | fine-tuned, CTC head | 12.46% | **3.78%** | 12.25% | **3.31%** |

The gain is smaller on test (about 0.7 to 0.8 points) than on dev (about 2.6 to 3.5 points), because the learning rate
and the best epoch were picked on dev.

Learning-rate search (1 epoch each, WER of the transducer head on the 1,000-clip dev check):

| Learning rate | 5e-5 | 1e-4 | 2e-4 | 4e-4 |
|---|---|---|---|---|
| Dev-check WER | 14.08% | 12.58% | 12.37% | **11.47%** |

- 4e-4 was the best value and was used for the full run: 27 epochs (6.6 h, about 1 s per step, batch 16 x 2, 11.4 GB of GPU memory). The best epoch was 21 (dev-check WER 11.04%).
- 4e-4 is the largest value we tried, so a larger one might be slightly better; there was no Kaggle time left to test it.
- Most of the gain comes in the first epoch (12.2% after epoch 1); the remaining epochs add about 1 point.
- `start_model_dev_check_wer` in the run log (43%) is NeMo's own WER on raw text during training. The start model writes capital letters and punctuation and our dev text has neither, so that number is not comparable. On normalized text the start model is A8 (14.25% on dev).

## Files

- `train.py`: writes the NeMo manifests, trains, keeps the best epoch, saves `<out-dir>/best.nemo` and `train_log.jsonl`.
- `predict.py`: C4a and C4b on `dev` / `test` (test needs `--allow-test`), from `best.nemo` or from Hugging Face.
- `push_to_hub.py`: uploads `best.nemo` with a model card to Hugging Face. The token comes from Kaggle Secrets or `huggingface-cli login`, never from code.
- `requirements.txt`: pinned NeMo packages (same versions as A8/A9). `numba-cuda` runs the RNN-T loss on the GPU; the Kaggle notebook installs it only when numba cannot run CUDA kernels yet.

## Run (from the repo root)

```bash
pip install -r nvidia_finetune/requirements.txt
python nvidia_finetune/train.py --out-dir runs/c4
python nvidia_finetune/predict.py --model runs/c4/best.nemo --split dev
python nvidia_finetune/predict.py --model runs/c4/best.nemo --split test --allow-test
python nvidia_finetune/push_to_hub.py --nemo runs/c4/best.nemo --repo <hf-user>/<name>
```

Outputs: `benchmark/predictions/C4a_fastconformer-usc-ft-rnnt__<split>.jsonl`, `benchmark/predictions/C4b_fastconformer-usc-ft-ctc__<split>.jsonl`, `benchmark/results/C4a_*.json`, `benchmark/results/C4b_*.json` (settings, revisions, best epoch, RTF, package versions), `benchmark/results/C4_fastconformer-usc-ft_train_log.jsonl`. Official WER/CER come from `benchmark/evaluate.py`. Model weights are not committed (Hugging Face link in the README).
