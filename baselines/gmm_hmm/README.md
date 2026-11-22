# A1: MFCC + GMM-HMM (Kaldi, trained by us, CPU)

Classic pipeline trained on our 25-hour train split: 13 MFCC + per-speaker
CMVN; monophone (5000 shortest utterances), tri1 (deltas, 2000 leaves / 10000
Gaussians), tri2b (LDA+MLLT, 2500 / 15000), tri3b (SAT/fMLLR, 2500 / 15000).
Grapheme lexicon from the train words: each letter is one unit, and o', g',
sh, ch, ng and the tutuq belgisi (') are one unit each. Word 3-gram LM from
train transcripts only. LM weight and word insertion penalty are chosen on dev.

Kaldi is the conda-forge build (`kaldi=5.5.1172`, no compiling); the recipe
scripts (`steps/`, `utils/`) come from the Kaldi GitHub repo, and the run log
records both versions.

| File | Contents |
|------|----------|
| `run.sh` | The recipe, stages 0-9 (see the header); `--eval-only true --splits test --after-freeze true` for the single test run |
| `prepare_data.py` | Kaldi data dirs from `data/clean/*.csv` (silence trim via `segments`), grapheme lexicon, LM text |
| `lm_arpa.py` | Interpolated Kneser-Ney 3-gram ARPA writer; used until Elif's KenLM ARPA is passed with `--arpa` (KenLM is not on conda-forge) |
| `score.py` | LM weight / insertion penalty grid on dev, predictions `benchmark/predictions/A1_gmm-hmm__<split>.jsonl`, run log `benchmark/results/A1_gmm-hmm.json` |
| `colab_gmm_hmm.ipynb` | Colab notebook (CPU runtime): install, run, save models to Drive, test run |

RTF is wall-clock time of MFCC + decoding with the logged number of parallel
CPU jobs divided by audio time; it is not comparable with the GPU RTFs.

## Dev results (all 3,783 dev clips, normalized text)

Trained on Colab CPU (AMD EPYC, 8 parallel jobs): stages 4-7 took about 2.5 h. The final tri3b model has
2,000 states and 15,015 Gaussians; 0.9% of train clips failed to align.

LM weight 7-22 and word insertion penalty 0.0-2.0 were tried on dev (80 pairs). The best pair, LM weight 16
and penalty 1.5, is inside the grid (a first grid of 7-17 x 0.0-1.0 put the best value on its edge); it is
saved in `exp/tri3b/decode_dev/best_params.json` and reused for test.

| Row | Model | WER | CER | RTF (Colab CPU, 8 jobs) |
|---|---|---:|---:|---:|
| A1 | MFCC + GMM-HMM (tri3b, SAT) | 53.61% | 24.66% | 0.178 |

Quick scores with the normalizer copy in `baselines/whisper_zero_shot/whisper_common.py`; the official numbers
come from `benchmark/evaluate.py`.

## Test results (all 3,836 test clips, one run after the freeze)

Run with `--eval-only true --splits test --after-freeze true` on the models restored from Drive, with the dev
choice (LM weight 16, penalty 1.5). Nothing was tuned on test.

| Row | Dev WER | Test WER | Test CER | Test RTF (Colab CPU, 8 jobs) |
|---|---:|---:|---:|---:|
| A1 | 53.61% | 60.12% | 28.77% | 0.217 |

A1 loses 6.5 WER points from dev to test, while the Whisper fine-tune (C1) loses only 1.7. What we checked:

- **Not the vocabulary.** 21.6% of the words in both dev and test never occur in train (out-of-vocabulary for
  our word LM), so test has no more unknown words than dev.
- **Partly clip length.** Test clips are longer (29% are 5-10 s vs 19% in dev), and A1's WER rises with clip
  length (dev: 51.3% under 5 s, 57.8% for 5-10 s), while C1's does not.
- **The rest is the test set itself.** Within the same length range A1 is still about 6 points worse on test
  (under 5 s: 51.3% -> 57.5%). The GMM-HMM, trained on 24 hours only, is more sensitive to new speakers and
  recordings than Whisper, which was pre-trained on far more audio.

RTF (0.217 vs 0.178 on dev) was measured on the same CPU type with the same 8 jobs; Colab CPUs are shared, so
some of this difference may be machine load. We did not measure it further.

`score.py` notes: an eval-only run has no `data/local/lm_source.txt` (written by stage 2), so `score.py` falls
back to the default LM description, and on test it logs the grid that was searched on dev.
