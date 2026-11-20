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
