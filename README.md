# Uzbek Speech Recognition (COE025 Project 3)

Automatic speech recognition for Uzbek (Latin script). We train and fine-tune
models on 25 hours of the human-verified Uzbek Speech Corpus (USC) and compare
them with every listed baseline that supports Uzbek, on the same test clips and
with one scoring script.

## Data

Source: [Uzbek Speech Corpus (USC), ISSAI](https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus)
([paper](https://arxiv.org/abs/2107.14419)). The data is not covered by our
Apache-2.0 license: the Hugging Face card gives MIT and ISSAI's USC page gives
CC BY 4.0. See [data/README.md](data/README.md) for both licenses and the
citation.

| Split | Source | Selected | After cleaning (kept audio) |
|-------|--------|----------|-----------------------------|
| train | Official USC train speakers that are not in dev or test and have at most 0.25 h of audio, shuffled with seed 42; whole speakers added, skipping any that would push the total past 25.0 h | 25.0 h | 23.92 h |
| dev   | Official USC dev, all clips | 4.0 h | 3.95 h |
| test  | Official USC test, all clips | 4.5 h | 4.40 h |

The "after cleaning" numbers, with speakers and clips per split, come from
`data/stats.py` and are in [data/clean/stats.md](data/clean/stats.md).

USC has no speaker metadata file. Speaker IDs come from clip file names; see
`data/make_splits.py` for the rule. With it, dev and test have exactly the 41
and 38 speakers the USC paper reports.

The USC paper describes the splits as speaker-disjoint, but 10 dev speakers and
16 test speakers also appear in the official train set. We keep dev and test
complete and unchanged and exclude all dev and test speakers from our train
selection, so our splits share no speaker.

Hours per speaker in USC train are very uneven: the median speaker has under a
minute, the largest has 5.2 h. Without a limit, one speaker would make up about
20% of our train set. Only speakers with at most 0.25 h are eligible, so no
single speaker is more than about 1% of train.

Some USC clips share identical audio. In most of these cases the copies have
different transcripts, so at least one transcript is wrong and we cannot tell
which. `data/clean.py` removes all such copies from train; copies with the same
transcript keep the first clip ID. In dev and test, audio shared under
different transcripts is only flagged, never removed.

The other cleaning rules (broken audio, clips over 30 s, Cyrillic look-alike
letters, apostrophe forms, long silence, letters outside the Uzbek Latin
alphabet, text that does not fit the audio length, wrong transcripts and noise
or clipping) are described in `data/clean.py` and `data/rules.py`. Only
clearly broken files are removed from dev and test; everything else there is
flagged, or fixed in the case of apostrophes, look-alike letters and silence
trimming. Every removed, fixed or flagged clip is listed in
`data/clean/removed.csv`, and the counts per rule are in `data/clean/stats.md`.

Some dev and test sentences also appear word for word in train. Their clip IDs
are in `data/clean/test_in_train.csv`, so scores can also be computed without
them.

## Listed models we did not run

Every model on the course list that supports Uzbek was run (A4-A9). These
listed models were not, because they cannot transcribe Uzbek (checked on 18
November 2026):

- **HuBERT + CTC:** we found no HuBERT checkpoint fine-tuned for Uzbek ASR on
  Hugging Face. The published HuBERT ASR checkpoints are English, for example
  [`facebook/hubert-large-ls960-ft`](https://huggingface.co/facebook/hubert-large-ls960-ft)
  (LibriSpeech); running it on Uzbek would be a wrong-language setting. The
  multilingual [`utter-project/mHuBERT-147`](https://huggingface.co/utter-project/mHuBERT-147)
  is only a pre-trained encoder with no ASR head, so it cannot produce a
  transcript.
- **[`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) and
  [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3):**
  25 European languages, no Uzbek ([paper](https://arxiv.org/abs/2509.14128)).
- **[`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and
  [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B):** 30
  languages, no Uzbek ([report](https://arxiv.org/abs/2601.21337)).

Community Uzbek fine-tunes such as
[`Gearnode/qwen3-asr-uzbek-v2`](https://huggingface.co/Gearnode/qwen3-asr-uzbek-v2)
are not on the course list, and its model card says it was also trained on
USC, so it may have seen our test set. We do not use it.

## Dev, test and the freeze

Every setting (learning rate, best checkpoint or epoch, decoding settings) was
chosen on dev only. The settings of A1-A5, A8, A9, C1, C2, C2long and C4 were
fixed and committed with their dev results by Friday 20 November (the freeze).
Each of these systems was then run once on test, on 21 or 22 November; the
scripts refuse the test split unless a flag (`--allow-test` or
`--after-freeze`) is given. C2long and C4 were not in the first plan; they were
added before the freeze because of dev results (see
[nvidia_finetune/README.md](nvidia_finetune/README.md)). A6 and A7 were added
on 22 November; they are ready models with greedy decoding and nothing to
choose, and each was run once on dev and once on test.

## Repository layout

| Path | Contents |
|------|----------|
| `LICENSE` | Apache-2.0, covers the code only |
| `.gitignore` | Keeps Python caches, the virtual environment, model weights and logits out of git |
| `requirements.txt` | Pinned Python packages for the data pipeline, `benchmark/` and the tests. Model folders that need other packages have their own pinned file |
| `pytest.ini` | pytest configuration (repo root on the import path, tests in `tests/`) |
| `REPORT.md` | Written report: data, systems, test results, RTF, confidence intervals and p-values, discussion |
| `data/README.md` | USC source, licenses and citation |
| `data/make_splits.py` | Selects the train speakers and writes `data/clean/train_speakers.txt` |
| `data/get_raw.py` | Copies the clips of our splits from the extracted USC archive into `data/raw/usc/` |
| `data/raw/usc/<split>/<speaker_id>/` | Our clips as lossless 16 kHz mono FLAC (`<clip_id>.flac`) with the unchanged USC transcript (`<clip_id>.txt`) |
| `data/clean.py` | Applies the cleaning rules to `data/raw/` and writes the split CSVs, `removed.csv` and `cleaning_limits.json` |
| `data/rules.py` | The per-clip cleaning rules used by `data/clean.py`, with the limits they compute from the train data |
| `data/stats.py` | Writes `data/clean/stats.md` and `data/clean/test_in_train.csv` from the clean CSVs and the cleaning log |
| `data/normalize.py` | Text normalization used for scoring (and its translit + numbers variant), shared by every system |
| `data/clean/train_speakers.txt` | Speaker IDs of our train split, one per line |
| `data/clean/{train,dev,test}.csv` | Clean splits: `id, path, transcript, speaker_id, duration_s, start_s, end_s`; `path` points into `data/raw/` |
| `data/clean/removed.csv` | Every clip the cleaning removed, fixed or flagged: `id, split, action, reason, detail` |
| `data/clean/cleaning_limits.json` | Limits the rules used (letters per second, SNR, clipping, silence) and the by-ear check results |
| `data/clean/rule8_cer.csv` | CER of the NVIDIA model (A9) on every train clip, input of the wrong-transcript rule |
| `data/clean/manual_check.csv` | Clips sampled for the by-ear checks of the wrong-transcript and noise rules, with the answers |
| `data/clean/stats.md` | Speakers, clips and hours per split, rule counts, limits and by-ear check results |
| `data/clean/test_in_train.csv` | Dev and test clips whose normalized sentence also appears in train |
| `baselines/gmm_hmm/` | A1: Kaldi MFCC + GMM-HMM, trained by us (CPU) |
| `baselines/scratch/` | A2 / A3: Conformer CTC and Conformer RNN-T, trained by us from scratch |
| `baselines/whisper_zero_shot/` | A4 / A5: Whisper small and large-v3, zero-shot; its `transcribe.py` also makes the C1 predictions |
| `baselines/mms/` | A6: Meta MMS-1B (`facebook/mms-1b-all`) with its Uzbek Latin adapter, zero-shot |
| `baselines/xlsr/` | A7: XLS-R 300M fine-tuned on Common Voice 8 Uzbek by others (`lucio/xls-r-uzbek-cv8`), zero-shot; not the same as `xlsr_ctc/` (C2) |
| `baselines/nvidia_fastconformer/` | A8 / A9: NVIDIA Uzbek FastConformer, zero-shot (transducer and CTC heads), and the wrong-transcript scan for cleaning |
| `whisper_finetune/` | C1: Whisper small fine-tuned on our train split |
| `xlsr_ctc/` | C2 / C2long: XLS-R 300M + CTC fine-tuned on our train split |
| `nvidia_finetune/` | C4a / C4b: the NVIDIA FastConformer fine-tuned on our train split (added after C2 stayed far behind A8; the folder README explains why) |
| `common/` | Helpers shared by `baselines/scratch/`, `baselines/nvidia_fastconformer/` and `xlsr_ctc/`: data loading, run logs, a fake dataset for smoke tests, version pinning |
| `notebooks/kaggle_enes_models.ipynb` | Kaggle notebook that ran A2, A3, A8, A9, C2, C2long, C4 and the wrong-transcript scan |
| `benchmark/evaluate.py` | Scores every prediction file against `data/clean/{dev,test}.csv` and writes `benchmark/results/scores.csv` |
| `benchmark/make_tables.py` | Writes the result tables from `scores.csv` |
| `benchmark/significance.py` | 95% confidence intervals of every system's WER and CER and p-values for the paired comparisons (fine-tuned vs start model), by speaker-level bootstrap; writes `benchmark/results/confidence_intervals.csv` and `benchmark/results/significance.csv` |
| `benchmark/timing.py` | Shared real-time factor (RTF) timer |
| `benchmark/predictions/` | One file per system and split, `<system>__<split>.jsonl`, with the model output for every clip |
| `benchmark/results/` | One run log per system, `<system>.json` (model ID and revision, settings, hardware, RTF), plus training logs |
| `contributions/` | One file per team member: what they did, in which files, with which results |
| `tests/` | pytest tests for the data pipeline, cleaning rules, normalizer, evaluator and model code |

Each folder under `baselines/` and each contribution folder has its own README
with settings, commands and, where needed, its own pinned requirements file.

## Reproducing

The USC archive (9.47 GB) is not in this repository; only the clips we use are,
under `data/raw/`. To rebuild them from the original archive:

```bash
pip install -r requirements.txt
curl -L -o ISSAI_USC.tar.gz https://huggingface.co/datasets/issai/Uzbek_Speech_Corpus/resolve/main/ISSAI_USC.tar.gz
shasum -a 256 ISSAI_USC.tar.gz  # 44dc6c70cde073548d975dcdcd3c44ca228ca24ea179905a6cbf5b66ed2fa92d
tar -xzf ISSAI_USC.tar.gz
python data/make_splits.py --usc-dir ISSAI_USC
python data/get_raw.py --usc-dir ISSAI_USC
python data/clean.py --allow-over-1pct  # rule 8 removes 1.6% of train; reviewed by ear (data/clean/stats.md)
python data/stats.py
pytest
```

## Limitations

- **One corpus, read speech only.** Every system is scored on the USC test
  set alone. There is no second, outside test set, so the results may not hold
  for spontaneous speech, other recording conditions or other domains.
- **Small test set in speakers.** Test has 3,836 clips but only 38 speakers.
  The 95% confidence intervals, which resample speakers
  (`benchmark/results/confidence_intervals.csv`), are therefore wide: for
  example 7.96% to 14.96% WER for A8.
- **C4's gain is not significant.** C4a and C4b score lower than A8 and A9,
  but the paired test does not show a significant difference (Holm-adjusted
  p = 0.24 for WER, `benchmark/results/significance.csv`).
- **C4's training data was filtered by its start model.** The wrong-transcript
  rule removed the 448 train clips (1.6%) on which A9, the same NVIDIA model C4
  starts from, has a CER above 0.5. Dev and test were not filtered this way.
- **Sentences shared with train.** 152 dev and 66 test clips have a sentence
  that also appears word for word in train (`data/clean/test_in_train.csv`);
  the scores include them.
- **Training data size.** Our train split is 25 h of selected audio (23.92 h
  after cleaning), chosen for the GPU time we had (free Colab and Kaggle).
- **RTF hardware.** A1 runs on CPU and the other systems on a GPU (test: Tesla
  T4), so A1's RTF is not directly comparable with the others.
- **Normalization and word-final ʻ.** `data/normalize.py` removes the ʻ of oʻ
  and gʻ at the end of a word (`togʻ` and `tog` score as the same word). It
  does so for references and outputs alike; 34 of the 27,042 test reference
  words end this way.
