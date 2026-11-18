"""Turn data/raw/ into the clean split CSVs and the cleaning log.

Reads every clip in data/raw/usc/<split>/<speaker_id>/ and writes:

    data/clean/{train,dev,test}.csv  id, path, transcript, speaker_id,
                                     duration_s, start_s, end_s
    data/clean/removed.csv           id, split, action, reason, detail
    data/clean/cleaning_limits.json  limits used by the rules + by-ear check results
    data/clean/manual_check.csv      rule-9 sample for the by-ear check (added once)

`path` points into data/raw/; audio is never copied or rewritten. `start_s` and
`end_s` are the kept interval (the whole clip unless a rule trims silence).
Every removal, fix and flag is logged in removed.csv; removed clips do not
appear in the split CSVs.

Main principle: only clearly broken files are removed from dev and test;
everything else there is only flagged. Each per-clip rule names the splits it
may remove from (train only by default); a removal anywhere else is recorded as
a flag instead.

Rules, in order:
1. duplicate_audio (all splits): identical decoded audio. Within a split, if
   all copies have the same transcript, the first clip ID is kept. If the
   copies have different transcripts, at least one transcript is wrong and we
   cannot tell which: all copies are removed from train and flagged in dev and
   test. A train copy of dev/test audio is removed from train; the same audio
   in dev and test is flagged in both.
2. too_long (train): clips over 30 s (Whisper does not take longer audio).
3. The per-clip rules of data/rules.py, built by make_rules() from the data:
   broken files, look-alike Cyrillic letters, apostrophes, long silence,
   characters outside the Uzbek Latin alphabet, text that does not fit the
   audio length, wrong transcripts (rule 8, NVIDIA CER scan) and noise or
   clipping (rule 9). Rules 8 and 9 remove clips only after the by-ear check
   in data/clean/manual_check.csv confirms the flagged group.

Before writing, the count for each rule is printed. If a rule removes or flags
more than 1% of a split, nothing is written until the team has reviewed it and
the script is rerun with --allow-over-1pct.

Usage:
    python data/clean.py --dry-run          # print counts only
    python data/clean.py [--allow-over-1pct]
    python data/stats.py                    # then: data/clean/stats.md
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.make_splits import speaker_id  # noqa: E402
from data import rules as R  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "usc"
OUT_DIR = ROOT / "data" / "clean"
SPLITS = ("train", "dev", "test")
CSV_COLUMNS = ["id", "path", "transcript", "speaker_id", "duration_s", "start_s", "end_s"]
REMOVED_COLUMNS = ["id", "split", "action", "reason", "detail"]
MAX_TRAIN_SECONDS = 30.0  # Whisper does not take longer audio
REVIEW_LIMIT = 0.01


@dataclass(frozen=True)
class Clip:
    id: str
    split: str
    speaker_id: str
    path: str
    transcript: str
    duration_s: float
    start_s: float
    end_s: float
    audio_sha256: str


@dataclass(frozen=True)
class Decision:
    action: str  # "removed", "fixed" or "flagged"
    reason: str
    detail: str = ""
    # For "fixed": the corrected values; None leaves the field unchanged.
    transcript: Optional[str] = None
    start_s: Optional[float] = None
    end_s: Optional[float] = None


@dataclass(frozen=True)
class Rule:
    name: str
    check: Callable[[Clip], Optional[Decision]]
    splits: tuple[str, ...] = SPLITS
    removes_from: tuple[str, ...] = ("train",)


def too_long(clip: Clip) -> Optional[Decision]:
    if clip.duration_s > MAX_TRAIN_SECONDS:
        return Decision("removed", "too_long", f"{clip.duration_s:.1f} s > {MAX_TRAIN_SECONDS:.0f} s")
    return None


RULES = [
    Rule("too_long", too_long, splits=("train",)),
]

CHECK_PATH = OUT_DIR / "manual_check.csv"
RULE8_PATH = OUT_DIR / "rule8_cer.csv"
LIMITS_PATH = OUT_DIR / "cleaning_limits.json"


def _tukey(values, k: float, log: bool = False) -> tuple[float, float]:
    """Tukey fences: values outside [Q1 - k*IQR, Q3 + k*IQR] are outliers (k=3: far out).
    With log=True the fences are computed on log(values), for ratios such as letters per second."""
    v = np.log(values) if log else np.asarray(values, dtype=float)
    q1, q3 = np.percentile(v, [25, 75])
    low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return (float(np.exp(low)), float(np.exp(high))) if log else (float(low), float(high))


def _trimmed(clip: Clip, f: dict) -> tuple[float, float]:
    """start_s / end_s after rule 10: cut leading/trailing silence longer than MIN_SILENCE_S."""
    start, end = clip.start_s, clip.end_s
    if f.get("ok") and f.get("samples"):
        if f["speech_start"] - start > R.MIN_SILENCE_S:
            start = max(start, f["speech_start"] - R.KEEP_SILENCE_S)
        if end - f["speech_end"] > R.MIN_SILENCE_S:
            end = min(end, f["speech_end"] + R.KEEP_SILENCE_S)
    return round(start, 3), round(end, 3)


def make_rules(clips: list[Clip], features: list[dict], check_path: Path = CHECK_PATH,
               rule8_path: Path = RULE8_PATH) -> tuple[list[Rule], dict]:
    """All per-clip rules, in order, plus the limits they use (computed on train only)."""
    feats = {(c.split, c.id): f for c, f in zip(clips, features)}
    train = [(c, feats[(c.split, c.id)]) for c in clips if c.split == "train"]
    train_ok = [(c, f) for c, f in train if f.get("ok") and f.get("samples")]

    lps = []
    for c, f in train_ok:
        start, end = _trimmed(c, f)
        n = R.letter_count(R.fix_apostrophes(R.fix_lookalikes(c.transcript)))
        if end > start and n:
            lps.append(n / (end - start))
    lps_low, lps_high = _tukey(lps, k=3.0, log=True)                   # removed without listening: far out only
    snr_low, _ = _tukey([f["snr_db"] for _, f in train_ok], k=1.5)     # only flagged, then checked by ear
    checks = R.read_checks(check_path)
    rule8 = R.read_rule8(rule8_path)
    limits = {
        "letters_per_second": {"low": round(lps_low, 2), "high": round(lps_high, 2),
                               "how": "Tukey far-out fences (3*IQR) on log(letters / trimmed seconds) of train clips"},
        "snr_db": {"low": round(snr_low, 2),
                   "how": "Tukey outlier fence (Q1 - 1.5*IQR) of train clips; SNR = 95th - 10th percentile frame dB"},
        "clipped_share": {"high": R.MAX_CLIPPED, "how": "share of samples at int16 full scale"},
        "silence": {"frame_db_below_peak": R.SILENCE_DB_BELOW_PEAK, "min_trim_s": R.MIN_SILENCE_S,
                    "kept_s": R.KEEP_SILENCE_S},
        "silent_peak_int16": R.SILENT_PEAK,
        "rule8_flagged_clips": len(rule8),
        "by_ear": {rule: dict(zip(("result", "sampled", "yes", "no"), R.check_result(checks, rule)))
                   for rule in ("8", "9")},
    }

    def f_of(clip):
        return feats[(clip.split, clip.id)]

    def broken_file(clip):
        f = f_of(clip)
        if not f.get("ok"):
            return Decision("removed", "broken_file", f"unreadable audio: {f.get('error', '')}")
        if f["samples"] == 0:
            return Decision("removed", "broken_file", "empty audio")
        if f["peak"] < R.SILENT_PEAK:
            return Decision("removed", "broken_file", f"silent audio (peak {f['peak']} of 32767)")
        if R.letter_count(clip.transcript) == 0:
            return Decision("removed", "broken_file", "empty transcript")
        return None

    def cyrillic_lookalike(clip):
        fixed = R.fix_lookalikes(clip.transcript)
        return Decision("fixed", "cyrillic_lookalike", transcript=fixed) if fixed != clip.transcript else None

    def apostrophe(clip):
        fixed = R.fix_apostrophes(clip.transcript)
        return Decision("fixed", "apostrophe", transcript=fixed) if fixed != clip.transcript else None

    def long_silence(clip):
        start, end = _trimmed(clip, f_of(clip))
        if (start, end) == (round(clip.start_s, 3), round(clip.end_s, 3)):
            return None
        return Decision("fixed", "long_silence", f"silence over {R.MIN_SILENCE_S} s trimmed",
                        start_s=start, end_s=end)

    def non_uzbek_latin(clip):
        bad = R.bad_characters(clip.transcript)
        return Decision("removed", "non_uzbek_latin", f"characters {bad!r}") if bad else None

    def text_audio_mismatch(clip):
        seconds = clip.end_s - clip.start_s
        n = R.letter_count(clip.transcript)
        rate = n / seconds if seconds > 0 else float("inf")
        if lps_low <= rate <= lps_high:
            return None
        return Decision("removed", "text_audio_mismatch",
                        f"{rate:.1f} letters/s outside {lps_low:.1f}-{lps_high:.1f}")

    heard_wrong = {(r["rule"], r["id"]) for r in checks if R.is_yes(r)}

    def by_ear(rule, reason, detail, clip):
        result = limits["by_ear"][rule]["result"]
        if result == "confirmed":
            return Decision("removed", reason, f"{detail}; by-ear check confirmed the group")
        if (rule, clip.id) in heard_wrong:
            return Decision("removed", reason, f"{detail}; this clip was checked by ear and is really wrong")
        return Decision("flagged", reason, f"{detail}; by-ear check {result}")

    def wrong_transcript(clip):
        if clip.id not in rule8:
            return None
        return by_ear("8", "wrong_transcript", f"NVIDIA CTC CER {rule8[clip.id]} above the rule-8 threshold", clip)

    def noise_clipping(clip):
        f = f_of(clip)
        problems = []
        if f["snr_db"] < snr_low:
            problems.append(f"SNR {f['snr_db']:.1f} dB < {snr_low:.1f}")
        if f["clipped"] > R.MAX_CLIPPED:
            problems.append(f"clipped {f['clipped']:.2%} > {R.MAX_CLIPPED:.2%}")
        return by_ear("9", "noise_clipping", ", ".join(problems), clip) if problems else None

    rules = [
        Rule("broken_file", broken_file, removes_from=SPLITS),
        Rule("too_long", too_long, splits=("train",)),
        Rule("cyrillic_lookalike", cyrillic_lookalike),
        Rule("apostrophe", apostrophe),
        Rule("long_silence", long_silence),
        Rule("non_uzbek_latin", non_uzbek_latin),
        Rule("text_audio_mismatch", text_audio_mismatch),
        Rule("wrong_transcript", wrong_transcript, splits=("train",)),
        Rule("noise_clipping", noise_clipping, splits=("train",)),
    ]
    return rules, limits


def load_clips(raw_dir: Path = RAW_DIR, root: Path = ROOT, progress: bool = False) -> list[Clip]:
    clips = []
    for split in SPLITS:
        flacs = sorted(Path(raw_dir, split).glob("*/*.flac"), key=lambda p: p.stem)
        for i, flac in enumerate(flacs, 1):
            try:
                data, rate = sf.read(flac, dtype="int16")
            except Exception:  # noqa: BLE001 - an unreadable file is removed by the broken_file rule
                data, rate = np.zeros(0, dtype=np.int16), 16000
            duration = len(data) / rate
            clips.append(Clip(
                id=flac.stem,
                split=split,
                speaker_id=speaker_id(flac.stem),
                path=flac.relative_to(root).as_posix(),
                transcript=flac.with_suffix(".txt").read_text(encoding="utf-8").strip(),
                duration_s=duration,
                start_s=0.0,
                end_s=duration,
                audio_sha256=hashlib.sha256(rate.to_bytes(4, "little") + data.tobytes()).hexdigest(),
            ))
            if progress and (i % 5000 == 0 or i == len(flacs)):
                print(f"loaded {split}: {i}/{len(flacs)}", flush=True)
    return clips


def find_duplicates(clips: list[Clip]) -> list[tuple[Clip, Decision]]:
    groups = defaultdict(list)
    for clip in clips:
        groups[clip.audio_sha256].append(clip)
    log = []
    for group in groups.values():
        if len(group) < 2:
            continue
        first = {}
        for split in SPLITS:
            copies = sorted((c for c in group if c.split == split), key=lambda c: c.id)
            if not copies:
                continue
            if len({c.transcript for c in copies}) > 1:
                # One recording under different transcripts: at least one
                # transcript is wrong and we cannot tell which.
                action = "removed" if split == "train" else "flagged"
                for clip in copies:
                    others = ", ".join(f"{o.id} ({o.transcript!r})" for o in copies if o is not clip)
                    log.append((clip, Decision(
                        action, "duplicate_audio", f"same audio as {others} with a different transcript")))
                if action == "removed":
                    continue
            else:
                for clip in copies[1:]:
                    log.append((clip, Decision(
                        "removed", "duplicate_audio", f"same audio as {split}/{copies[0].id}")))
            first[split] = copies[0]
        held = [first[s] for s in ("dev", "test") if s in first]
        if "train" in first and held:
            log.append((first["train"], Decision(
                "removed", "duplicate_audio", f"same audio as {held[0].split}/{held[0].id}")))
        if len(held) == 2:
            dev, test = held
            log.append((dev, Decision("flagged", "duplicate_audio", f"same audio as test/{test.id}")))
            log.append((test, Decision("flagged", "duplicate_audio", f"same audio as dev/{dev.id}")))
    return log


def _apply_fix(clip: Clip, decision: Decision) -> tuple[Clip, Decision]:
    changes = {f: getattr(decision, f) for f in ("transcript", "start_s", "end_s")
               if getattr(decision, f) is not None}
    detail = "; ".join(f"{f}: {getattr(clip, f)!r} -> {v!r}" for f, v in changes.items())
    if decision.detail:
        detail = f"{decision.detail}; {detail}"
    return dataclasses.replace(clip, **changes), dataclasses.replace(decision, detail=detail)


def clean(clips: list[Clip], rules: list[Rule] = RULES) -> tuple[list[Clip], list[tuple[Clip, Decision]]]:
    log = find_duplicates(clips)
    removed = {(c.split, c.id) for c, d in log if d.action == "removed"}
    kept = []
    for clip in clips:
        if (clip.split, clip.id) in removed:
            continue
        for rule in rules:
            if clip.split not in rule.splits:
                continue
            decision = rule.check(clip)
            if decision is None:
                continue
            if decision.action == "removed" and clip.split not in rule.removes_from:
                decision = dataclasses.replace(
                    decision, action="flagged",
                    detail=f"{decision.detail} (kept: {rule.name} does not remove from {clip.split})")
            if decision.action == "fixed":
                clip, decision = _apply_fix(clip, decision)
            log.append((clip, decision))
            if decision.action == "removed":
                break
        else:
            kept.append(clip)
    return kept, log


def rule_counts(log: list[tuple[Clip, Decision]]) -> Counter:
    return Counter((d.reason, c.split, d.action) for c, d in log)


def rules_over_limit(clips: list[Clip], log: list[tuple[Clip, Decision]], limit: float = REVIEW_LIMIT) -> list[str]:
    sizes = Counter(c.split for c in clips)
    over = []
    for (reason, split, action), n in sorted(rule_counts(log).items()):
        if action != "fixed" and n > limit * sizes[split]:
            over.append(f"{reason}: {n} of {sizes[split]} {split} clips {action} ({n / sizes[split]:.1%})")
    return over


def write_outputs(kept: list[Clip], log: list[tuple[Clip, Decision]], out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with open(out_dir / f"{split}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_COLUMNS)
            for c in kept:
                if c.split == split:
                    w.writerow([c.id, c.path, c.transcript, c.speaker_id,
                                f"{c.duration_s:.3f}", f"{c.start_s:.3f}", f"{c.end_s:.3f}"])
    with open(out_dir / "removed.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(REMOVED_COLUMNS)
        for c, d in sorted(log, key=lambda cd: (SPLITS.index(cd[0].split), cd[0].id, cd[1].reason)):
            w.writerow([c.id, c.split, d.action, d.reason, d.detail])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="print rule counts, write nothing")
    parser.add_argument("--allow-over-1pct", action="store_true",
                        help="write even if a rule removes or flags > 1%% of a split (after team review)")
    args = parser.parse_args()

    clips = load_clips(progress=True)
    print("audio checks (silence, noise, clipping) ...", flush=True)
    features = R.all_audio_features([str(ROOT / c.path) for c in clips])
    rules, limits = make_rules(clips, features)
    kept, log = clean(clips, rules)
    print("limits: " + json.dumps({k: v for k, v in limits.items() if k != "by_ear"}))
    for rule, res in limits["by_ear"].items():
        print(f"by-ear check rule {rule}: {res['result']} "
              f"({res['yes']} yes, {res['no']} no of {res['sampled']} sampled)")

    sizes = Counter(c.split for c in clips)
    print("\nclips per split: " + ", ".join(f"{s} {sizes[s]}" for s in SPLITS))
    print("rule counts (rule, split, action, clips):")
    for (reason, split, action), n in sorted(rule_counts(log).items()):
        print(f"  {reason:20} {split:5} {action:7} {n}")
    if not log:
        print("  (no rule fired)")
    kept_sizes = Counter(c.split for c in kept)
    print("kept: " + ", ".join(f"{s} {kept_sizes[s]}" for s in SPLITS))

    over = rules_over_limit(clips, log)
    for line in over:
        print(f"OVER 1%: {line}")
    if args.dry_run:
        print("dry run: nothing written")
        return
    if over and not args.allow_over_1pct:
        sys.exit("a rule exceeds 1% of a split: review it with the team, then rerun with --allow-over-1pct")
    write_outputs(kept, log)
    LIMITS_PATH.write_text(json.dumps(limits, indent=2) + "\n", encoding="utf-8")
    flagged9 = [{"id": c.id, "path": c.path, "score": d.detail.split(";")[0], "transcript": c.transcript}
                for c, d in log if d.reason == "noise_clipping"]
    added = R.add_check_sample(CHECK_PATH, "9", flagged9)
    print(f"wrote {OUT_DIR}/{{train,dev,test}}.csv, removed.csv and {LIMITS_PATH.name}")
    if added:
        print(f"added {added} rule-9 clips to {CHECK_PATH.name}: listen, fill really_wrong (yes/no), run again")


if __name__ == "__main__":
    main()
