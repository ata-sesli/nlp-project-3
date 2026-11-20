"""A1: write the Kaldi data directories, grapheme lexicon and LM text for USC.

Reads data/clean/{train,dev,test}.csv (before Gate 1: the data/raw scan of
whisper_common.load_split) and writes, under <work>/data/:

    {train,dev,test}/wav.scp     <rec> flac -c -d -s <path> |
    {train,dev,test}/segments    <utt> <rec> <start_s> <end_s>   (silence trim from clean.py)
    {train,dev,test}/utt2spk     utterance IDs are "<speaker>-<clip id>", so they sort by speaker
    train/text                   normalized transcripts (shared normalizer)
    local/dict/                  grapheme lexicon of the train words + phone lists
    local/lm_train.txt           train transcripts, one per line, for the n-gram LM
    uttmap_<split>.tsv           Kaldi utterance ID -> clip ID (to write predictions)

Dev and test get no text file: Kaldi never sees their references; scoring uses
the clean CSVs (score.py). Graphemes: every letter is one unit, and the Uzbek
letters o', g', sh, ch, ng and the tutuq belgisi (') are one unit each.

Usage (inside the Kaldi environment, see run.sh):
    python baselines/gmm_hmm/prepare_data.py --work ~/gmm_hmm_work [--splits train dev]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "baselines" / "whisper_zero_shot"))
import whisper_common as wc  # noqa: E402

DIGRAPHS = {"o'": "o_", "g'": "g_", "sh": "sh", "ch": "ch", "ng": "ng"}
LATIN = set("abcdefghijklmnopqrstuvwxyz")


def graphemes(word: str) -> list[str]:
    units, i = [], 0
    while i < len(word):
        pair = word[i: i + 2]
        if pair in DIGRAPHS:
            units.append(DIGRAPHS[pair])
            i += 2
            continue
        ch = word[i]
        units.append(ch if ch in LATIN else "Q" if ch == "'" else f"u{ord(ch):04x}")
        i += 1
    return units


def utt_id(row: dict) -> str:
    return f"{row['speaker_id']}-{row['id']}"


def write_split(split: str, out: Path) -> list[str]:
    rows = sorted(wc.load_split(split), key=utt_id)
    out.mkdir(parents=True, exist_ok=True)
    texts = []
    with open(out / "wav.scp", "w", encoding="utf-8") as wav, \
            open(out / "segments", "w", encoding="utf-8") as seg, \
            open(out / "utt2spk", "w", encoding="utf-8") as u2s, \
            open(out.parent / f"uttmap_{split}.tsv", "w", encoding="utf-8") as umap:
        text_f = open(out / "text", "w", encoding="utf-8") if split == "train" else None
        for r in rows:
            u = utt_id(r)
            if split == "train":
                t = wc.normalize(r["transcript"])
                if not t:
                    continue  # nothing to train on
                text_f.write(f"{u} {t}\n")
                texts.append(t)
            wav.write(f"{u} flac -c -d -s {r['path']} |\n")
            seg.write(f"{u} {u} {r['start_s']:.3f} {r['end_s']:.3f}\n")
            u2s.write(f"{u} {r['speaker_id']}\n")
            umap.write(f"{u}\t{r['id']}\n")
        if text_f:
            text_f.close()
    print(f"{split}: {len(rows)} utterances -> {out}")
    return texts


def write_dict(texts: list[str], dict_dir: Path) -> None:
    dict_dir.mkdir(parents=True, exist_ok=True)
    words = Counter(w for t in texts for w in t.split())
    lex = {w: graphemes(w) for w in sorted(words)}
    phones = sorted({p for units in lex.values() for p in units})
    with open(dict_dir / "lexicon.txt", "w", encoding="utf-8") as f:
        f.write("!SIL SIL\n<unk> SPN\n")
        for w, units in lex.items():
            f.write(f"{w} {' '.join(units)}\n")
    (dict_dir / "nonsilence_phones.txt").write_text("".join(f"{p}\n" for p in phones), encoding="utf-8")
    (dict_dir / "silence_phones.txt").write_text("SIL\nSPN\n", encoding="utf-8")
    (dict_dir / "optional_silence.txt").write_text("SIL\n", encoding="utf-8")
    (dict_dir / "extra_questions.txt").write_text("", encoding="utf-8")
    print(f"lexicon: {len(lex)} words, {len(phones)} grapheme units: {' '.join(phones)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "dev"], choices=["train", "dev", "test"])
    args = p.parse_args()
    data = args.work / "data"
    for split in args.splits:
        texts = write_split(split, data / split)
        if split == "train":
            write_dict(texts, data / "local" / "dict")
            (data / "local" / "lm_train.txt").write_text("".join(f"{t}\n" for t in texts), encoding="utf-8")


if __name__ == "__main__":
    main()
