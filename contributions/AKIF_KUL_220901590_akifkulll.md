# Akif Kul

- Student ID: 220901590
- GitHub: [akifkulll](https://github.com/akifkulll)
- Hugging Face: [akifkulll](https://huggingface.co/akifkulll)

## Summary

I built four of the twelve systems in our benchmark, from code to trained model to predictions:

| Row | Type | Model | Folder |
|---|---|---|---|
| A4 | Baseline (ready model) | `openai/whisper-small`, zero-shot | `baselines/whisper_zero_shot/` |
| A5 | Baseline (ready model) | `openai/whisper-large-v3`, zero-shot | `baselines/whisper_zero_shot/` |
| A1 | Baseline (trained by us) | MFCC + GMM-HMM in Kaldi | `baselines/gmm_hmm/` |
| C1 | Our contribution | `openai/whisper-small` fine-tuned on our train split | `whisper_finetune/` |

Every script, notebook and README in these three folders is mine, together with the A1, A4, A5 and C1 files in
`benchmark/predictions/` and `benchmark/results/`. Whisper ran on a Colab A100 40 GB and Kaldi on a Colab CPU
runtime (8 cores); the single test run used a Colab T4 for Whisper. The data cleaning (`data/`) is Ata's and Enes's work; I checked their clean splits before training.

## 1. Whisper zero-shot baselines (A4, A5)

What I wrote:

- `baselines/whisper_zero_shot/transcribe.py`: one inference script for A4, A5 and C1. It reads `data/clean/<split>.csv`,
  cuts each clip to its `start_s`/`end_s` (the cleaning step's silence trimming), sorts clips by length so batches
  waste less padding, and decodes with `language="uzbek"`, `task="transcribe"`, greedy search, fp16. It times only the
  GPU work (with `torch.cuda.synchronize`) to compute RTF, writes `benchmark/predictions/<system>__<split>.jsonl`
  and a run log with the model's Hugging Face revision, parameter count, decoding settings, RTF, hardware and
  package versions.
- `baselines/whisper_zero_shot/whisper_common.py`: shared helpers for A4, A5 and C1: split loading, audio loading,
  the team normalizer (with a copy of the plan's rules until `data/normalize.py` is merged), quick WER/CER,
  output formats, and a test guard: `--split test` is refused unless `--after-freeze` is given, so test cannot be
  touched by accident before the freeze.

What I found: both models mostly hear the Uzbek words but do not know the Uzbek Latin spelling. They write k for q,
drop the apostrophe in o' and g', and use Turkish ş and ç. Large-v3 halves the CER of small but joins words
("farqi bor" -> "farqibar"), so its WER stays above 90%. A4's WER is above 100% because a few clips loop on a
repeated phrase up to the 225-token limit, which adds many inserted words.

## 2. Whisper fine-tune (C1, our contribution)

What I wrote:

- `whisper_finetune/train.py`: fine-tuning with the Hugging Face `Seq2SeqTrainer`. Log-mel features are computed on the
  fly from the trimmed audio; the data collator pads features and labels, masks label padding with -100 and removes
  the decoder start token that the model adds itself. The Uzbek/transcribe prefix tokens are fixed in both the
  tokenizer and the generation config, so training and evaluation decode the same way. Dev WER is computed with
  generation every 500 steps and the best checkpoint is kept (`load_best_model_at_end`, metric WER). A `--smoke`
  mode (60 steps on 512 clips) checks the whole pipeline in a few minutes before the real run.
- `whisper_finetune/push_to_hub.py`: uploads the best model with a model card (data, settings, dev results, usage
  example) built from the run log, and prints the revision.
- `whisper_finetune/colab_whisper.ipynb`: the full Colab run for A4, A5 and C1: sparse clone of the repo, pinned
  install, dev runs, smoke test, training (checkpoints saved to Google Drive so a disconnect does not lose the run),
  upload, and the single test run.

Settings (from the plan, not tuned on test): lr 1e-5, 500 warmup steps, linear decay, effective batch 32, bf16,
4000 steps, seed 42.

Results: dev WER falls from 136% (zero-shot) to 49.4% after only 500 steps. The model already heard the speech and
mainly had to learn the spelling. The best checkpoint is step 3000 (33.85% dev WER); after that dev WER stops
improving, so the later checkpoints were not used. Training took about 49 minutes. The model is on Hugging Face as
[`akifkulll/whisper-small-uzbek-usc25h`](https://huggingface.co/akifkulll/whisper-small-uzbek-usc25h), revision
`b38781ed83b508cbd44ce5fdcefa488efe849d29`. I downloaded it again from the Hub and re-ran dev to check the upload
(33.83% WER, the same model). The fine-tuned model is also 2.6x faster than zero-shot small (RTF 0.0145 -> 0.0056),
because it stops looping.

## 3. Kaldi GMM-HMM baseline (A1)

What I wrote:

- `baselines/gmm_hmm/prepare_data.py`: Kaldi data directories from `data/clean/*.csv` (`wav.scp` decodes FLAC on the
  fly, `segments` applies the silence trimming, speaker maps), the lexicon and the LM training text. The lexicon is
  grapheme-based, because Uzbek Latin is written much as it is spoken and there is no free Uzbek phoneme lexicon.
  o', g', sh, ch, ng and the apostrophe (tutuq belgisi) are single units, e.g. o'zbek -> o' z b e k.
- `baselines/gmm_hmm/lm_arpa.py`: an interpolated Kneser-Ney word 3-gram LM written as ARPA, trained on the train
  transcripts only. I wrote it because KenLM is not available on conda-forge.
- `baselines/gmm_hmm/run.sh`: the recipe in 10 stages: data, lang, LM, MFCC + CMVN, monophone (5,000 shortest
  utterances), tri1 (deltas), tri2b (LDA+MLLT), tri3b (SAT/fMLLR), graph + decoding, scoring. `--stage` and
  `--stop-stage` let a run continue after a disconnect. It refuses to decode test without `--after-freeze true`, and
  `--eval-only` runs only the stages that test needs.
- `baselines/gmm_hmm/score.py`: chooses the LM weight and word insertion penalty on dev, writes predictions and a run
  log (Gaussians, Kaldi and recipe versions, CPU, RTF). Test reuses the dev choice from `best_params.json`.
- `baselines/gmm_hmm/colab_gmm_hmm.ipynb`: installs Kaldi from conda-forge (no compiling), runs the stages, and saves
  the trained models to Google Drive.

Kaldi is a conda-forge build (`kaldi=5.5.1172`), and the recipe scripts (`steps/`, `utils/`) come from the Kaldi
GitHub repo at a recorded commit.

Results: the tri3b model has 2,000 states and 15,015 Gaussians. 0.9% of train clips failed to align, and the
likelihood rose in every pass. My first LM weight grid (7-17 x penalty 0.0-1.0) put the best value on its edge
(LM weight 17, penalty 1.0). I widened it to 7-22 x 0.0-2.0 (80 pairs), and the best pair, 16 / 1.5, came out inside
the grid: 53.61% dev WER. RTF on the Colab CPU with 8 parallel jobs is 0.178.

## 4. Dev results (all 3,783 dev clips, normalized text)

| Type | Row | Model | WER | CER |
|---|---|---|---:|---:|
| Baseline | A4 | Whisper-small, zero-shot | 136.47% | 67.47% |
| Baseline | A5 | Whisper-large-v3, zero-shot | 96.31% | 34.28% |
| Baseline | A1 | Kaldi GMM-HMM (tri3b) | 53.61% | 24.66% |
| Ours | C1 | Whisper-small fine-tuned | **33.83%** | **9.99%** |

These are quick scores with the normalizer copy in `whisper_common.py`; the official numbers come from
`benchmark/evaluate.py`.

What the numbers show:

- **Language data matters more than model size.** Whisper-small fine-tuned on 24 hours beats the 6x larger
  large-v3 by 62 WER points.
- **A classic system trained on our data beats the large zero-shot model.** A1 learns the spelling directly from our
  transcripts, but it can only write words seen in train, which hurts in agglutinative Uzbek. This is why it stays
  behind C1.
- **WER and CER tell different stories.** C1's CER is 10% but its WER is 34%: one wrong suffix makes the whole word
  wrong, which is why we report both.

## 5. Test results (all 3,836 test clips, one run after the freeze, 21 November)

I ran test once for each system, with the settings chosen on dev and nothing changed. Whisper ran on a Colab T4
(so its RTF is comparable with the team's other GPU systems), and Kaldi on the same Colab CPU type as dev, using
the trained models saved to Drive.

| Type | Row | Model | Dev WER | Test WER | Test CER | Test RTF |
|---|---|---|---:|---:|---:|---:|
| Baseline | A4 | Whisper-small, zero-shot | 136.47% | 119.87% | 53.19% | 0.0379 (T4) |
| Baseline | A5 | Whisper-large-v3, zero-shot | 96.31% | 93.67% | 32.70% | 0.0850 (T4) |
| Baseline | A1 | Kaldi GMM-HMM (tri3b) | 53.61% | 60.12% | 28.77% | 0.217 (CPU, 8 jobs) |
| Ours | C1 | Whisper-small fine-tuned | **33.83%** | **35.56%** | **9.91%** | **0.0201 (T4)** |

The ranking is the same as on dev. I also checked why the systems moved between dev and test, with small
scripts over the prediction files:

- **C1 generalizes:** +1.7 WER points and the same CER, and its WER hardly depends on clip length.
- **A1 drops 6.5 points, and it is not the vocabulary.** 21.6% of dev and of test words are unseen in train, so
  OOV words do not explain it. Test clips are longer (29% are 5-10 s vs 19% on dev) and A1 gets worse on longer
  clips, but it is also about 6 points worse within the same length range. The classic model trained on 24 hours
  is more sensitive to new speakers and recordings than a model pre-trained on far more audio.
- **A4 improves only because it loops less.** 40 test clips (1.0%) vs 78 dev clips (2.1%) fall into the
  repetition loop; without them its WER is the same on both splits (110.0% vs 109.4%).

Two small fixes came out of the test run (details in `baselines/gmm_hmm/README.md`):

- `score.py` crashed in the eval-only run because `data/local/lm_source.txt` is written by the LM stage, which
  eval-only skips. It now falls back to the default LM description. It also logged that run's default grid
  instead of the grid searched on dev; it now reads the grid from `best_params.json`.
- The run logs kept only the hardware of the latest run, so after the T4 test run the dev RTFs looked like T4
  numbers. `whisper_common.write_run_log` now stores the hardware and versions in each split's block
  (`runs.<split>.environment`); I filled in the dev blocks from the earlier logs.

The C1 model card on Hugging Face now has the test results (revision `fdd98ccc`); the weights are the same file
as revision `b38781e`, which made all predictions.

## 6. Problems I solved

- Transformers 5 renamed `torch_dtype` to `dtype`, and an old version stayed loaded in Colab after the install. I
  pinned the versions in `whisper_finetune/requirements-whisper.txt` and restart the runtime after installing.
- `flac` and `kenlm` are not on conda-forge. I used `libflac` for decoding and wrote my own Kneser-Ney LM
  (`lm_arpa.py`).
- The LM weight grid edge (section 3): the grid was widened on dev, without looking at test.
- A Colab disconnect could lose hours of training. C1 checkpoints and Kaldi models are saved to Google Drive, and
  both pipelines can resume (`--resume`, `--stage`).
