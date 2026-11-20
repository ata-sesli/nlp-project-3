# C2: XLS-R 300M + CTC, fine-tuned on our USC train split (our contribution)

Start checkpoint: [`facebook/wav2vec2-xls-r-300m`](https://huggingface.co/facebook/wav2vec2-xls-r-300m) (exact revision in the run log). A new CTC output layer is added and the model is fine-tuned on our 25 h train split.

## Settings

| Setting | Value |
|---|---|
| Output units | characters of the normalized train transcripts, `\|` = word separator, `[PAD]` = CTC blank |
| Text | `data/normalize.py` (lowercase, one apostrophe form, no punctuation) |
| Frozen | CNN feature encoder |
| Optimizer | AdamW, lr 3e-4, linear warmup 500 steps, then linear decay |
| Batch | 8 clips x 4 accumulation steps = 32, clips of similar length together |
| Other | fp16, gradient checkpointing, SpecAugment-style time masking 0.05, dropouts 0, seed 42 |
| Checkpoint choice | dev WER every 1000 steps; the lowest dev WER is kept |
| Time limit | `--max-hours` (default 11): after 50 steps the schedule is shortened to fit, so it ends inside one Kaggle session |

## Files

- `train.py`: training, dev evaluation, best checkpoint, full checkpoint for resuming (`--resume-from`).
- `predict.py`: greedy decoding for `dev` / `test` (test needs `--allow-test`); saves log-probabilities per clip for the KenLM decoder (C3, `lm/`).
- `push_to_hub.py`: uploads the best checkpoint with a model card to Hugging Face. The token comes from Kaggle Secrets or `huggingface-cli login`, never from code.
- `xlsr_lib.py`: vocabulary, data loading, batching, decoding.

## C2long: C2 trained further

After the first run (15 epochs, dev WER still going down at the end), C2 is trained 20 more epochs from its own
checkpoint on Hugging Face: lr 1e-4, warmup 300 steps, same data and settings, same letter vocabulary (taken from the
checkpoint). Its results have their own names (`C2long_xlsr-300m-ctc`), set with the `XLSR_SYSTEM` variable:

```bash
XLSR_SYSTEM=C2long_xlsr-300m-ctc python xlsr_ctc/train.py --out-dir runs/xlsr_long --model-id <hf-user>/<C2 repo> --lr 1e-4 --warmup-steps 300 --epochs 20
XLSR_SYSTEM=C2long_xlsr-300m-ctc python xlsr_ctc/predict.py --model runs/xlsr_long/best --split dev
```

Dev results (all 3,783 dev clips, normalized text, greedy decoding): C2 34.05% WER / 8.46% CER, C2long 32.58% WER /
8.05% CER (all 20 epochs done in 6.6 h on one T4, best checkpoint at step 16,000 of 16,199). The extra training helps a
little, but C2long stays far behind the NVIDIA model (A8 14.26%, C4a 11.60%).

## Run (from the repo root)

```bash
python xlsr_ctc/train.py --out-dir runs/xlsr
python xlsr_ctc/predict.py --model runs/xlsr/best --split dev --logits-dir runs/logits
python xlsr_ctc/predict.py --model runs/xlsr/best --split test --allow-test --logits-dir runs/logits
python xlsr_ctc/push_to_hub.py --model-dir runs/xlsr/best --repo <hf-user>/<name>
```

Logits format for C3: `<logits-dir>/<split>/<clip id>.npy`, float16 log-probabilities, shape (frames, vocabulary), 20 ms per frame; `<logits-dir>/labels.json` is the token list in id order (`""` = blank, `" "` = word separator), ready for `pyctcdecode.build_ctcdecoder`.

Outputs: `benchmark/predictions/C2_xlsr-300m-ctc__<split>.jsonl`, `benchmark/results/C2_xlsr-300m-ctc.json` (all settings, revision, chosen step, dev WER, RTF, package versions), `benchmark/results/C2_xlsr-300m-ctc_train_log.jsonl` (loss and dev WER during training). Model weights and logits are not committed (Hugging Face link in the README).
