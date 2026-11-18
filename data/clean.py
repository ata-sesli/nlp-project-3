"""Turn data/raw/ into the clean split CSVs and the cleaning log.

Reads every clip in data/raw/usc/<split>/<speaker_id>/ and writes:

    data/clean/{train,dev,test}.csv  id, path, transcript, speaker_id,
                                     duration_s, start_s, end_s
    data/clean/removed.csv           id, split, action, reason, detail

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
2. Per-clip rules in RULES (see the list below for who adds which check).

Before writing, the count for each rule is printed. If a rule removes or flags
more than 1% of a split, nothing is written until the team has reviewed it and
the script is rerun with --allow-over-1pct.

Usage:
    python data/clean.py --dry-run          # print counts only
    python data/clean.py [--allow-over-1pct]
"""

import argparse
import csv
import dataclasses
import hashlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.make_splits import speaker_id  # noqa: E402

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
    # Elif: text checks (Cyrillic look-alike letters, apostrophe variants,
    #   characters outside the Uzbek Latin alphabet, text-audio length mismatch).
    # Serhat: audio checks (empty/unreadable/silent audio, noise and clipping,
    #   long leading/trailing silence via start_s/end_s) and the wrong-transcript
    #   rule from the NVIDIA CTC CER scan.
    # Enes: his checks, once agreed.
]


def load_clips(raw_dir: Path = RAW_DIR, root: Path = ROOT, progress: bool = False) -> list[Clip]:
    clips = []
    for split in SPLITS:
        flacs = sorted(Path(raw_dir, split).glob("*/*.flac"), key=lambda p: p.stem)
        for i, flac in enumerate(flacs, 1):
            data, rate = sf.read(flac, dtype="int16")
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
    kept, log = clean(clips)

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
    print(f"wrote {OUT_DIR}/{{train,dev,test}}.csv and removed.csv")


if __name__ == "__main__":
    main()
