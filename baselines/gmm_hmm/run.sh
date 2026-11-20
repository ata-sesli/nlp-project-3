#!/usr/bin/env bash
# A1: MFCC + GMM-HMM with Kaldi (CPU), trained by us on our 25-hour USC train split.
#
#   stage 0  data dirs, grapheme lexicon, LM text            (prepare_data.py)
#   stage 1  lang dir                                         (utils/prepare_lang.sh)
#   stage 2  word 3-gram LM -> G.fst                          (--arpa, else lm_arpa.py)
#   stage 3  MFCC (13) + per-speaker CMVN
#   stage 4  mono   (5000 shortest train utterances)
#   stage 5  tri1   deltas, 2000 leaves / 10000 Gaussians
#   stage 6  tri2b  LDA+MLLT, 2500 / 15000
#   stage 7  tri3b  SAT (fMLLR), 2500 / 15000   (risk plan: stop at tri1 if short on time)
#   stage 8  graph + decode of --splits with --model
#   stage 9  LM weight and word insertion penalty on dev; predictions + run log (score.py)
#
# Kaldi binaries come from conda-forge (see README). The working directory
# (features, models, lattices) lives outside the repo; only code is committed.
#
# Usage, inside the Kaldi environment:
#   bash baselines/gmm_hmm/run.sh [--stage N] [--stop-stage N] [--model tri3b] [--nj 12] \
#        [--work ~/gmm_hmm_work] [--arpa lm/<elif>.arpa] [--splits "dev"]
#   bash baselines/gmm_hmm/run.sh --eval-only true --splits test --after-freeze true   # once, after the freeze
set -euo pipefail

stage=0
stop_stage=9
nj=12
model=tri3b
work=$HOME/gmm_hmm_work
arpa=
splits=dev
after_freeze=false
eval_only=false   # true: only stages 0, 3, 8, 9 for --splits (the test run)
kaldi_src=$HOME/kaldi-src
mono_utts=5000
while [ $# -gt 0 ]; do
  key=${1#--}; key=${key//-/_}
  [ $# -ge 2 ] || { echo "missing value for $1" >&2; exit 1; }
  printf -v "$key" '%s' "$2"; shift 2
done

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
here=$repo/baselines/gmm_hmm
if [[ " $splits " == *" test "* && $after_freeze != true ]]; then
  echo "refusing to decode test before the freeze; pass --after-freeze true" >&2; exit 1
fi

mkdir -p "$work/conf"
cd "$work"
ln -sfn "$kaldi_src/egs/wsj/s5/steps" steps
ln -sfn "$kaldi_src/egs/wsj/s5/utils" utils
echo 'export PATH=$PWD/utils:$PATH; export LC_ALL=C' > path.sh
echo 'export train_cmd=run.pl decode_cmd=run.pl' > cmd.sh
printf -- '--use-energy=false\n--sample-frequency=16000\n' > conf/mfcc.conf
. ./path.sh

nj_for() {  # no more jobs than speakers
  local n; n=$(wc -l < "data/$1/spk2utt"); echo $(( n < nj ? n : nj ))
}
run_stage() {
  if $eval_only; then case $1 in 0|3|8|9) ;; *) return 1 ;; esac; fi
  [ "$stage" -le "$1" ] && [ "$stop_stage" -ge "$1" ]
}
timed() {  # timed <file> <command...>: appends wall-clock seconds, for RTF
  local t0; t0=$(date +%s.%N); "${@:2}"; awk -v a="$t0" -v b="$(date +%s.%N)" 'BEGIN{print b - a}' >> "$1"
}

prep_splits="train $splits"
$eval_only && prep_splits=$splits
if run_stage 0; then
  python "$here/prepare_data.py" --work "$work" --splits $prep_splits
  for x in $prep_splits; do
    utils/utt2spk_to_spk2utt.pl "data/$x/utt2spk" > "data/$x/spk2utt"
    utils/fix_data_dir.sh "data/$x"
  done
fi

if run_stage 1; then
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang_tmp data/lang
fi

if run_stage 2; then
  if [ -n "$arpa" ]; then
    cp "$arpa" data/local/lm.arpa
    echo "LM: $arpa" > data/local/lm_source.txt
  else
    python "$here/lm_arpa.py" data/local/lm_train.txt data/local/lm.arpa
    echo "LM: baselines/gmm_hmm/lm_arpa.py (interpolated Kneser-Ney 3-gram, train transcripts)" > data/local/lm_source.txt
  fi
  gzip -f data/local/lm.arpa
  utils/format_lm.sh data/lang data/local/lm.arpa.gz data/local/dict/lexicon.txt data/lang_test
fi

if run_stage 3; then
  for x in $prep_splits; do
    rm -f "exp/timing_$x.txt"
    if [ "$x" = train ]; then t=/dev/null; else t=exp/timing_$x.txt; fi
    mkdir -p exp
    timed "$t" steps/make_mfcc.sh --nj "$(nj_for "$x")" --mfcc-config conf/mfcc.conf "data/$x" "exp/make_mfcc/$x" mfcc
    timed "$t" steps/compute_cmvn_stats.sh "data/$x" "exp/make_mfcc/$x" mfcc
    utils/fix_data_dir.sh "data/$x"
  done
fi

if run_stage 4; then
  utils/subset_data_dir.sh --shortest data/train "$mono_utts" data/train_mono
  steps/train_mono.sh --boost-silence 1.25 --nj "$nj" --cmd run.pl data/train_mono data/lang exp/mono
  steps/align_si.sh --boost-silence 1.25 --nj "$nj" --cmd run.pl data/train data/lang exp/mono exp/mono_ali
fi

if run_stage 5; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd run.pl 2000 10000 data/train data/lang exp/mono_ali exp/tri1
  steps/align_si.sh --nj "$nj" --cmd run.pl data/train data/lang exp/tri1 exp/tri1_ali
fi

if run_stage 6; then
  steps/train_lda_mllt.sh --cmd run.pl --splice-opts "--left-context=3 --right-context=3" \
    2500 15000 data/train data/lang exp/tri1_ali exp/tri2b
  steps/align_si.sh --nj "$nj" --cmd run.pl --use-graphs true data/train data/lang exp/tri2b exp/tri2b_ali
fi

if run_stage 7; then
  steps/train_sat.sh --cmd run.pl 2500 15000 data/train data/lang exp/tri2b_ali exp/tri3b
fi

if run_stage 8; then
  [ -f "exp/$model/graph/HCLG.fst" ] || utils/mkgraph.sh data/lang_test "exp/$model" "exp/$model/graph"
  for x in $splits; do
    decode=steps/decode.sh
    [ "$model" = tri3b ] && decode=steps/decode_fmllr.sh
    timed "exp/timing_$x.txt" $decode --nj "$(nj_for "$x")" --cmd run.pl --skip-scoring true \
      "exp/$model/graph" "data/$x" "exp/$model/decode_$x"
  done
fi

if run_stage 9; then
  for x in $splits; do
    python "$here/score.py" --work "$work" --model "$model" --split "$x" --nj "$nj" \
      $( [ "$x" = test ] && echo --after-freeze )
  done
fi
