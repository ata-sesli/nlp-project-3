# Enes Özbek

- Student ID: 2309011083
- GitHub: [Enes-Ozbek](https://github.com/Enes-Ozbek)
- Hugging Face: [Waxech](https://huggingface.co/Waxech)

My parts of Project 3 (Uzbek speech recognition on USC): data cleaning, the NVIDIA FastConformer baselines (A8, A9),
the two models trained from scratch (A2, A3), and our fine-tuned models C2, C2long and C4. All training ran on
Kaggle (Tesla T4) with `notebooks/kaggle_enes_models.ipynb`.

## 1. Data cleaning (`data/`)

- Cleaning rules 2-10 in `data/rules.py` and `data/clean.py` (rule 1, duplicate audio, is Ata's):
  broken file, too long (over 30 s, train only), Cyrillic look-alike letters, apostrophe forms, long silence
  (trimmed, not removed), non-Uzbek Latin letters, text/audio length mismatch, wrong transcript (rule 8) and
  noise/clipping (rule 9). Limits come from the train data itself (Tukey fences), see `data/clean/cleaning_limits.json`.
- Rule 8: the NVIDIA CTC model (A9) transcribed all 27,658 train clips; 475 clips had a CER over 0.5. I listened to a
  random sample of 50 of them: 32 really had a wrong transcript, so 448 train clips were removed.
- By-ear checks: 147 clips in total (97 for rule 9, 50 for rule 8), answers in `data/clean/manual_check.csv`.
  Rules 8 and 9 remove train clips only after this check. My rules never remove dev or test clips; they only flag them
  (long silences are trimmed).
- `data/stats.py`: speakers, clips and hours per split, rule counts, and dev/test sentences that also appear in train
  (`data/clean/stats.md`, `data/clean/test_in_train.csv`). Tests: `tests/test_rules.py`.

| Split | Speakers | Clips | Hours |
|---|---:|---:|---:|
| train | 638 | 27,144 | 23.92 |
| dev | 41 | 3,783 | 3.95 |
| test | 38 | 3,836 | 4.40 |

## 2. Baselines

| Row | Model | Folder | What I did |
|---|---|---|---|
| A8 | `nvidia/stt_uz_fastconformer_hybrid_large_pc` (rev `cd32f63`), transducer head | `baselines/nvidia_fastconformer/` | ran the ready model, no training |
| A9 | same model, CTC head | `baselines/nvidia_fastconformer/` | ran the ready model, no training |
| A2 | Conformer + CTC from scratch (20.1M parameters) | `baselines/scratch/` | trained 100 epochs (2.5 h) |
| A3 | Conformer + RNN-T from scratch (21.1M parameters) | `baselines/scratch/` | trained 100 epochs (3.4 h) |

A2 and A3 share one encoder code (`baselines/scratch/`).

## 3. Our fine-tuned models (contribution)

| Row | Model | Folder | What I did |
|---|---|---|---|
| C2 | `facebook/wav2vec2-xls-r-300m` + CTC (letters) | `xlsr_ctc/` | fine-tuned on our train split (5.2 h); model on Hugging Face [`Waxech/xls-r-300m-uzbek-usc`](https://huggingface.co/Waxech/xls-r-300m-uzbek-usc); dev logits for Elif's language model (C3) |
| C2long | C2 trained further | `xlsr_ctc/` | 20 more epochs from C2 (6.6 h) |
| C4a / C4b | the A8/A9 NVIDIA model fine-tuned on our train split (both heads) | `nvidia_finetune/` | learning-rate search on dev (5e-5, 1e-4, 2e-4, 4e-4), then 27 epochs with 4e-4 (6.6 h), best epoch kept; model on Hugging Face [`Waxech/stt-uz-fastconformer-usc`](https://huggingface.co/Waxech/stt-uz-fastconformer-usc) |

Why C4 (added after the plan): after the first run, our planned model C2 got 34.05% dev WER, while the ready NVIDIA
model (A8) already got 14.25% without any training on our data. XLS-R was pre-trained without transcripts, so 24 hours
were not enough for it; the NVIDIA model already knew Uzbek but had never seen USC. So I fine-tuned the NVIDIA model on
our USC train split. On the way I fixed two problems: fp16 made the model output only `<unk>` on the T4 (so training
uses fp32), and I added a GPU memory test on the longest clips and a learning-rate search on dev. Details:
`nvidia_finetune/README.md`.

## 4. Results (official scores from `benchmark/evaluate.py`)

All 3,783 dev clips and all 3,836 test clips, normalized text, greedy decoding, no language model. Learning rates and
best epochs were chosen on dev only; the test set was run once, after the freeze (notebook step `test`).

| Type | Row | Model | Dev WER | Dev CER | Test WER | Test CER |
|---|---|---|---:|---:|---:|---:|
| Baseline | A8 | NVIDIA FastConformer, ready, transducer head | 14.25% | 5.01% | 12.07% | 3.65% |
| Baseline | A9 | NVIDIA FastConformer, ready, CTC head | 15.95% | 5.21% | 13.08% | 3.61% |
| Baseline | A3 | Conformer RNN-T from scratch | 40.46% | 15.77% | 42.19% | 16.24% |
| Baseline | A2 | Conformer CTC from scratch | 47.25% | 13.41% | 50.17% | 13.97% |
| Ours | C4a | NVIDIA FastConformer fine-tuned, transducer head | **11.60%** | 3.93% | **11.38%** | 3.41% |
| Ours | C4b | NVIDIA FastConformer fine-tuned, CTC head | 12.46% | **3.78%** | 12.25% | **3.31%** |
| Ours | C2long | XLS-R 300M, trained longer | 32.57% | 8.05% | 32.96% | 7.78% |
| Ours | C2 | XLS-R 300M fine-tuned | 34.05% | 8.46% | 35.40% | 8.45% |

Fine-tuning the NVIDIA model on our data lowers test WER from 12.07% to 11.38% (transducer head) and from 13.08% to
12.25% (CTC head). The gain is smaller on test than on dev (14.25% to 11.60%), because the learning rate and the best
epoch were picked on dev, so the dev numbers are a little optimistic. The test numbers are the fair ones.

## 5. Shared code

- `common/asr_data.py`: reading the clean splits, audio loading, quick WER/CER, run logs in `benchmark/results/`.
- `common/pin_requirements.py`: writes the exact package versions of each Kaggle run into the `requirements.txt` files.
- `common/make_fake_data.py`: small fake dataset for the quick checks.
- `notebooks/kaggle_enes_models.ipynb`: every Kaggle run (`STEP` = smoke, rule8, train_night, train_day, improve, test).
- Tests: `tests/test_asr_models.py`, `tests/test_nvidia_finetune.py`, `tests/test_nvidia_utils.py`, `tests/test_rules.py`.
- Predictions and run logs: `benchmark/predictions/` and `benchmark/results/` for A2, A3, A8, A9, C2, C2long, C4a, C4b.
