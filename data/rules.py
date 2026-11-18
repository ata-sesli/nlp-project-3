"""Cleaning rules 2 and 4-10 for data/clean.py (rule 1, duplicate audio, and the
30-second train limit live in clean.py itself).

Every rule is a clean.Rule: a per-clip check that returns a Decision or None.
Rules that need numbers from the whole data (normal letters-per-second range,
noise limits) compute them first in make_rules(). All limits that are used are
returned too, so clean.py can print them and stats.py can report them.

Order matters: fixes run first (text fixes, then silence trimming), so later
checks see the fixed transcript and the trimmed audio.

    rule  name                 splits     action
    2     broken_audio         all        removed (cannot be scored)
    6     cyrillic_lookalike   all        fixed (Latin letter)
    7     apostrophe           all        fixed (ASCII ')
    10    long_silence         all        fixed (start_s / end_s)
    5     non_uzbek_latin      all        removed in train, flagged in dev/test
    4     text_audio_mismatch  all        removed in train, flagged in dev/test
    8     wrong_transcript     train      flagged; removed once the by-ear check confirms the group
    9     noise_clipping       train      flagged; removed once the by-ear check confirms the group

Rules 8 and 9 are confirmed by people: data/clean/manual_check.csv holds a
random sample (seed 42) of the flagged clips with a really_wrong column. When
every sampled row is filled and more than half say "yes", the whole flagged
group is removed from train; otherwise the clips are only flagged. A sampled
clip that a person marked "yes" is always removed from train.
"""
from __future__ import annotations

import csv
import random
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# --------------------------------------------------------------------------- text

# Cyrillic letters that look exactly like Latin ones (rule 6). Only replaced
# inside words that also contain Latin letters, e.g. "mаktab" with a Cyrillic а.
LOOKALIKE = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i", "ј": "j", "ѕ": "s",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "І": "I", "Ј": "J", "Ѕ": "S",
})
# Apostrophe variants used for o‘, g‘ and the tutuq belgisi (rule 7).
APOSTROPHES = "‘’ʻʼ`´′ʹʿ՚＇"
APOSTROPHE_RE = re.compile(f"[{APOSTROPHES}]")
CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"\S+")
# What a clean Uzbek Latin transcript may contain (rule 5): the 26 Latin letters,
# the ASCII apostrophe, spaces and ordinary punctuation. No digits, no Cyrillic,
# no other symbols.
ALLOWED_RE = re.compile(r"""[A-Za-z' .,!?;:\-–—"«»“”„()…]+""")


def fix_lookalikes(text: str) -> str:
    def word(m: re.Match) -> str:
        w = m.group(0)
        if CYRILLIC_RE.search(w) and LATIN_RE.search(w):
            return w.translate(LOOKALIKE)
        return w
    return WORD_RE.sub(word, text)


def fix_apostrophes(text: str) -> str:
    return APOSTROPHE_RE.sub("'", text)


def bad_characters(text: str) -> str:
    """Characters outside the Uzbek Latin alphabet and punctuation, in order of appearance."""
    text = unicodedata.normalize("NFC", text)
    return "".join(dict.fromkeys(ALLOWED_RE.sub("", text)))


def letter_count(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


# --------------------------------------------------------------------------- audio

FRAME_S = 0.025
HOP_S = 0.010
SILENCE_DB_BELOW_PEAK = 35.0   # a frame is silence if it is this far below the loudest frame
MIN_SILENCE_S = 0.5            # rule 10 trims only leading/trailing silence longer than this
KEEP_SILENCE_S = 0.2           # and keeps this much silence next to the speech
CLIP_LEVEL = 32767 * 0.999     # int16 samples at or above this are clipped
SILENT_PEAK = 64               # int16 peak below this (about -54 dBFS): the clip is silent
MAX_CLIPPED = 0.01             # rule 9: more than 1% of samples at full scale


def audio_features(path: str) -> dict:
    """Per-clip numbers for rules 2, 9 and 10. Never raises: unreadable files get ok=False."""
    try:
        x, rate = sf.read(path, dtype="int16")
    except Exception as err:  # noqa: BLE001 - any read error means the file is broken
        return {"ok": False, "error": repr(err)}
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.int16)
    n = len(x)
    out = {"ok": True, "error": "", "samples": n, "rate": rate}
    if n == 0:
        return {**out, "peak": 0, "clipped": 0.0, "snr_db": 0.0, "speech_start": 0.0, "speech_end": 0.0}
    a = np.abs(x.astype(np.int32))
    out["peak"] = int(a.max())
    out["clipped"] = float((a >= CLIP_LEVEL).mean())
    frame, hop = int(FRAME_S * rate), int(HOP_S * rate)
    f = x.astype(np.float64)
    if n < frame:
        energy = np.array([np.mean(f ** 2)])
    else:
        idx = np.arange(0, n - frame + 1, hop)
        sq = np.concatenate([[0.0], np.cumsum(f ** 2)])
        energy = (sq[idx + frame] - sq[idx]) / frame
    db = 10 * np.log10(energy + 1e-9)
    # SNR estimate: loud frames (speech) against quiet frames (background).
    out["snr_db"] = float(np.percentile(db, 95) - np.percentile(db, 10))
    voiced = np.nonzero(db > db.max() - SILENCE_DB_BELOW_PEAK)[0]
    out["speech_start"] = float(voiced[0] * hop / rate)
    out["speech_end"] = float(min(n, voiced[-1] * hop + frame) / rate)
    return out


def all_audio_features(paths: list[str], workers: Optional[int] = None) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(audio_features, paths, chunksize=64))


# --------------------------------------------------------------------------- by-ear checks

CHECK_COLUMNS = ["rule", "id", "path", "score", "transcript", "model_output", "really_wrong", "note"]
CHECK_SAMPLE = {"8": 50, "9": 30}


def read_checks(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


YES_WORDS = ("yes", "y", "evet", "e")
NO_WORDS = ("no", "n", "hayir", "hayır", "h")


def is_yes(row: dict) -> bool:
    return row.get("really_wrong", "").strip().lower() in YES_WORDS


def check_result(rows: list[dict], rule: str) -> tuple[str, int, int, int]:
    """('confirmed' | 'rejected' | 'pending' | 'none', sampled, yes, no) for one rule's by-ear check."""
    mine = [r for r in rows if r.get("rule") == rule]
    if not mine:
        return "none", 0, 0, 0
    yes = sum(is_yes(r) for r in mine)
    no = sum(r.get("really_wrong", "").strip().lower() in NO_WORDS for r in mine)
    if yes + no < len(mine):
        return "pending", len(mine), yes, no
    return ("confirmed" if yes > len(mine) / 2 else "rejected"), len(mine), yes, no


def add_check_sample(path: Path, rule: str, flagged: list[dict]) -> int:
    """Add a random sample (seed 42) of flagged clips for the by-ear check, if the rule has none yet."""
    rows = read_checks(path)
    if any(r.get("rule") == rule for r in rows) or not flagged:
        return 0
    sample = random.Random(42).sample(flagged, min(CHECK_SAMPLE[rule], len(flagged)))
    sample.sort(key=lambda r: r["id"])
    rows += [{c: r.get(c, "") for c in CHECK_COLUMNS} | {"rule": rule, "really_wrong": "", "note": ""} for r in sample]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CHECK_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(sample)


def read_rule8(path: Path) -> dict[str, str]:
    """id -> CER for the clips that baselines/nvidia_fastconformer/rule8_flag.py flagged."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {r["id"]: r["cer"] for r in csv.DictReader(f) if r.get("flagged") == "yes"}
