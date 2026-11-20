"""Make a small FAKE dataset to check that the training and prediction code runs (no real speech).

Each letter becomes a short tone, so a model can learn it in a few minutes. Output folder:
  <out>/clean/{train,dev,test}.csv   same columns as data/clean (absolute paths)
  <out>/raw/usc/train/<speaker>/<clip>.flac+.txt same layout as data/raw (for rule8_flag.py)
Never commit this output and never report numbers from it.

    python common/make_fake_data.py --out /kaggle/working/fake
    python xlsr_ctc/train.py --clean-dir /kaggle/working/fake/clean ...
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

WORDS = ["salom", "o'zbek", "til", "kitob", "maktab", "g'alaba", "bola", "uy", "suv", "non", "shahar", "choy"]
LETTERS = sorted(set("".join(WORDS)))
SR = 16000


def synth(text: str, rng: random.Random) -> np.ndarray:
    parts = [np.zeros(int(0.1 * SR))]
    for ch in text:
        n = int(rng.uniform(0.06, 0.09) * SR)
        if ch == " ":
            parts.append(np.zeros(n))
            continue
        f = 250 + 90 * LETTERS.index(ch)
        t = np.arange(n) / SR
        parts.append(0.3 * np.sin(2 * np.pi * f * t) * np.hanning(n))
    parts.append(np.zeros(int(0.1 * SR)))
    wav = np.concatenate(parts)
    return (wav + 0.003 * np.random.default_rng(rng.randint(0, 1 << 30)).standard_normal(len(wav))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--dev", type=int, default=60)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--max-words", type=int, default=5)
    ap.add_argument("--long", type=int, default=0, help="extra train clips of about 28 s (GPU memory check)")
    args = ap.parse_args()
    rng = random.Random(42)
    out = Path(args.out).resolve()
    sizes = {"train": args.train, "dev": args.dev, "test": args.dev}
    (out / "clean").mkdir(parents=True, exist_ok=True)
    for split, n in sizes.items():
        rows = []
        n_long = args.long if split == "train" else 0
        for k in range(n + n_long):
            spk = f"{split}_spk{k % 7}"
            cid = f"{split}_{k:05d}"
            n_words = rng.randint(args.min_words, args.max_words) if k < n else 66
            words = [rng.choice(WORDS) for _ in range(n_words)]
            text = " ".join(words)
            shown = text.capitalize() + "."  # like a real transcript: capital letter and punctuation
            wav = synth(text, rng)
            sub = "usc/" + split
            path = out / "raw" / sub / spk / f"{cid}.flac"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, wav, SR)
            path.with_suffix(".txt").write_text(shown, encoding="utf-8")
            dur = len(wav) / SR
            rows.append({"id": cid, "path": str(path), "transcript": shown, "speaker_id": spk,
                         "duration_s": f"{dur:.3f}", "start_s": "0.000", "end_s": f"{dur:.3f}"})
        with open(out / "clean" / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"Fake data written to {out}")


if __name__ == "__main__":
    main()
