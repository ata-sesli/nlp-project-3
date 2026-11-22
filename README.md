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

## Metrics

Every system is scored by one script, `benchmark/evaluate.py`, on the same
clips, with `jiwer==4.0.0`. WER = (S + D + I) / N, where S, D and I are the
substituted, deleted and inserted words and N is the number of reference words;
errors and words are summed over all clips before dividing. CER is the same
formula on characters. The choice of metrics follows the
[Hugging Face Audio Course, Unit 5](https://huggingface.co/learn/audio-course/chapter5/evaluation).

| Metric | What it measures | Why we report it |
|---|---|---|
| WER (norm), main metric | Word errors on normalized text | The standard ASR metric. Unit 5 recommends scoring normalized text so that casing and punctuation do not count as errors: some systems write them (the NVIDIA model, for example) and others, such as our CTC models, do not. |
| CER (norm) | Character errors on normalized text | Uzbek is agglutinative: one word carries several suffixes, and one wrong suffix makes the whole word wrong in WER. CER gives credit for the correct part of the word. |
| WER (orthographic) | WER on the text as written, with casing and punctuation | Unit 5 compares orthographic and normalized WER. The gap shows how much of a system's error is only casing and punctuation (A8: 49.25% orthographic, 12.07% normalized). |
| WER (translit+number) | Normalized WER after converting Cyrillic to Latin and digits to Uzbek number words, in references and outputs | Not a main metric. Our normalization keeps digits and Cyrillic as they are, which counts against zero-shot Whisper (A5 wrote Cyrillic in 168 and digits in 105 of the 3,836 test outputs). This column shows how much that changes the score. |
| WER 95% CI | Range of WER over 10,000 resamples of the test speakers | Clips of one speaker are not independent, so the speaker is resampled (`benchmark/significance.py`). Two systems whose intervals overlap may not really differ; `benchmark/results/significance.csv` has the paired tests. |
| RTF and hardware | Processing time / audio duration | Speed next to accuracy. The GPU systems were measured on a Tesla T4; A1 runs on CPU, so its RTF is not comparable with theirs. |
| Parameters | Model size | To compare large and small models fairly. |

WER has no upper limit: a system that inserts many words can pass 100%
(Unit 5), which is what happens with zero-shot Whisper-small (A4).

Normalization (`data/normalize.py`, the same for references and outputs and
for every system): lowercase; every apostrophe form (‘ ’ ` ʻ ʼ and others)
becomes `'`; punctuation is removed, apostrophes inside a word are kept (oʻ,
gʻ and the tutuq belgisi are part of the spelling); spaces are collapsed.
Digits and Cyrillic are not converted; `scores.csv` counts the outputs that
contain them (`hyp_with_digits`, `hyp_with_cyrillic`).

## Results

USC test split, all 3,836 clips. The tables are the output of
`python -m benchmark.make_tables` (from `benchmark/results/scores.csv` and
`benchmark/results/confidence_intervals.csv`); no number is typed by hand. A1,
A2 and A3 are baselines we trained ourselves; the other rows of Table A are
ready models used as they are. Dev scores are in the same `scores.csv`
(`python -m benchmark.make_tables --split dev`).

### Table A: Baselines (test)

| System | WER (norm) | WER 95% CI | CER (norm) | WER (orthographic) | WER (translit+number) | RTF | Hardware | Parameters |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| A1_gmm-hmm | 60.10% | [50.28%, 64.15%] | 28.76% | 60.12% | 60.10% | 0.21749 | CPU: AMD EPYC 7B12, 8 parallel jobs | —* |
| A2_conformer-ctc | 50.17% | [40.80%, 54.24%] | 13.97% | 50.20% | 50.17% | 0.00062 | Tesla T4 | 20120861 |
| A3_conformer-rnnt | 42.19% | [32.85%, 46.42%] | 16.24% | 42.20% | 42.19% | 0.00147 | Tesla T4 | 21146170 |
| A4_whisper-small | 119.88% | [114.38%, 129.61%] | 53.18% | 120.05% | 119.81% | 0.03790 | Tesla T4 | 241734912 |
| A5_whisper-large-v3 | 93.66% | [88.81%, 98.09%] | 32.69% | 97.48% | 93.18% | 0.08499 | Tesla T4 | 1543490560 |
| A6_mms-1b | 29.38% | [22.66%, 35.47%] | 6.60% | 39.59% | 29.37% | 0.01178 | Tesla T4 | 964793329 |
| A7_xls-r-uzbek-cv8 | 40.71% | [33.98%, 44.47%] | 8.78% | 49.66% | 40.71% | 0.00546 | Tesla T4 | 315472545 |
| A8_nvidia-fastconformer-rnnt | 12.07% | [7.96%, 14.96%] | 3.65% | 49.25% | 12.07% | 0.00488 | Tesla T4 | 114621442 |
| A9_nvidia-fastconformer-ctc | 13.08% | [8.86%, 16.68%] | 3.61% | 52.52% | 13.08% | 0.00319 | Tesla T4 | 114621442 |

*\* A1 (GMM-HMM) is a classical acoustic model; instead of neural-network parameter counts, it is characterized by Gaussian mixture components and HMM states.*

### Table B: Our Contribution (test)

| System | WER (norm) | WER 95% CI | CER (norm) | WER (orthographic) | WER (translit+number) | RTF | Hardware | Parameters |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| C1_whisper-small-ft | 35.53% | [27.31%, 39.28%] | 9.89% | 35.57% | 35.53% | 0.02012 | Tesla T4 | 241734912 |
| C2_xlsr-300m-ctc | 35.40% | [27.22%, 39.14%] | 8.45% | 35.43% | 35.40% | 0.00384 | Tesla T4 | 315469470 |
| C2long_xlsr-300m-ctc | 32.96% | [25.17%, 36.61%] | 7.78% | 33.01% | 32.96% | 0.00402 | Tesla T4 | 315469470 |
| C4a_fastconformer-usc-ft-rnnt | 11.38% | [6.71%, 13.44%] | 3.41% | 24.44% | 11.38% | 0.00433 | Tesla T4 | 114621442 |
| C4b_fastconformer-usc-ft-ctc | 12.25% | [7.40%, 14.45%] | 3.31% | 25.17% | 12.25% | 0.00321 | Tesla T4 | 114621442 |

> **Note on C3 (XLS-R + KenLM):** Planned word-level KenLM decoding on C2 logits was not completed due to KenLM build/runtime errors and is omitted from Table B.

Paired comparisons (`benchmark/results/significance.csv`, Holm-adjusted):
fine-tuning Whisper-small (C1 vs A4) and training XLS-R longer (C2long vs C2)
lower WER and CER significantly (p = 0.0008). C4a and C4b score lower than A8
and A9, but the difference is not significant (WER p = 0.24, CER p = 0.16).

## Error Analysis

Error analysis on the best test system (`C4a_fastconformer-usc-ft-rnnt`, 11.38% test WER across 3,836 clips) is in [analysis/error_analysis.md](analysis/error_analysis.md) (generated by `python -m analysis.error_analysis`):

- **Erroneous clips:** 1,783 / 3,836 (46.5%), totaling 3,078 word errors (80.1% substitutions, 14.5% deletions, 5.5% insertions).
- **Test OOV rate (vs. train):** 21.53% token-based (5,823 / 27,042) and 45.15% type-based (4,773 / 10,572).
- **Substitution breakdown (automatic common-prefix rule):** 43.2% suffix errors (1,065), 54.1% root/other substitutions (1,334), and 2.6% apostrophe errors (65).

These categories are an automatic estimate, not a manual classification; the
100 sample sentences in the report were not reviewed by hand. The rule only
compares the first letters of the two words, so 25 "suffix errors" differ from
the reference only by a doubled letter (`elchinning` → `elchining`, 7 times:
the name *Elchin* vs *elchi* "ambassador"), and spelling variants such as
`etarli` → `yetarli` count as root/other errors. The 43.2% suffix share is
therefore an upper estimate. Some references are also wrong: in clip
`1187023182_2_35103_1` the reference has *davlatda aylandi* where the model's
*davlatga aylandi* ("became [the fifth] country") is the grammatical form, and
it still counts as a model error.

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
| `benchmark/make_tables.py` | Writes the result tables in [Results](#results) from `scores.csv` and `confidence_intervals.csv` |
| `benchmark/significance.py` | 95% confidence intervals of every system's WER and CER and p-values for the paired comparisons (fine-tuned vs start model), by speaker-level bootstrap; writes `benchmark/results/confidence_intervals.csv` and `benchmark/results/significance.csv` |
| `benchmark/timing.py` | Shared real-time factor (RTF) timer |
| `benchmark/predictions/` | One file per system and split, `<system>__<split>.jsonl`, with the model output for every clip |
| `benchmark/results/` | One run log per system, `<system>.json` (model ID and revision, settings, hardware, RTF), plus training logs |
| `contributions/` | One file per team member: what they did, in which files, with which results |
| `tests/` | pytest tests for the data pipeline, cleaning rules, normalizer, evaluator and model code |
| `analysis/` | Error analysis script (`error_analysis.py`) and report (`error_analysis.md`) on the best test system |

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
