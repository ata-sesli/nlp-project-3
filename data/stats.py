"""Write data/clean/stats.md from the clean split CSVs and the cleaning log.

For each split (after cleaning): speakers, clips and hours. How many dev and
test sentences appear word for word (after normalization) in train; those clip
IDs are listed in data/clean/test_in_train.csv so the scores can also be given
without them. The count of every cleaning rule, the limits the rules used and
the result of the by-ear checks.

Usage (after data/clean.py):
    python data/stats.py
"""
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CLEAN = ROOT / "data" / "clean"
SPLITS = ("train", "dev", "test")


def _fallback_normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[‘’ʻʼ`´′]", "'", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"(?<!\w)'|'(?!\w)", " ", text)
    return " ".join(text.split())


try:  # the team normalizer (data/normalize.py), the same one benchmark/evaluate.py uses
    from data.normalize import normalize
except ImportError:
    normalize = _fallback_normalize


def read(name: str) -> list[dict]:
    path = CLEAN / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_table(rows_by_split: dict) -> list[str]:
    out = ["| Split | Speakers | Clips | Hours (kept audio) | Hours (whole files) |", "|---|---:|---:|---:|---:|"]
    for split in SPLITS:
        rows = rows_by_split[split]
        kept = sum(float(r["end_s"]) - float(r["start_s"]) for r in rows) / 3600
        whole = sum(float(r["duration_s"]) for r in rows) / 3600
        out.append(f"| {split} | {len({r['speaker_id'] for r in rows})} | {len(rows)} | {kept:.2f} | {whole:.2f} |")
    return out


def overlap(rows_by_split: dict) -> tuple[list[str], list[dict]]:
    train = {normalize(r["transcript"]) for r in rows_by_split["train"]}
    lines = ["| Split | Clips | Sentence also in train (word for word, normalized) | Share |", "|---|---:|---:|---:|"]
    found = []
    for split in ("dev", "test"):
        rows = rows_by_split[split]
        hits = [r for r in rows if normalize(r["transcript"]) in train]
        found += [{"id": r["id"], "split": split, "transcript": r["transcript"]} for r in hits]
        share = len(hits) / len(rows) if rows else 0.0
        lines.append(f"| {split} | {len(rows)} | {len(hits)} | {share:.1%} |")
    return lines, found


def rule_table(removed: list[dict]) -> list[str]:
    counts = defaultdict(Counter)
    for r in removed:
        counts[r["reason"]][(r["split"], r["action"])] += 1
    cols = [(s, a) for s in SPLITS for a in ("removed", "fixed", "flagged")]
    out = ["| Rule | " + " | ".join(f"{s} {a}" for s, a in cols) + " |", "|---|" + "---:|" * len(cols)]
    for reason in sorted(counts):
        out.append(f"| {reason} | " + " | ".join(str(counts[reason][c]) for c in cols) + " |")
    return out


def main() -> None:
    rows_by_split = {s: read(f"{s}.csv") for s in SPLITS}
    if not any(rows_by_split.values()):
        sys.exit("No data/clean/*.csv yet: run python data/clean.py first.")
    removed = read("removed.csv")
    limits_path = CLEAN / "cleaning_limits.json"
    limits = json.loads(limits_path.read_text(encoding="utf-8")) if limits_path.exists() else {}
    overlap_lines, found = overlap(rows_by_split)

    with open(CLEAN / "test_in_train.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "split", "transcript"])
        w.writeheader()
        w.writerows(found)

    md = ["# Data statistics (written by data/stats.py, do not edit by hand)", "",
          "## Splits after cleaning", ""] + split_table(rows_by_split)
    md += ["", "Speakers never appear in two splits (checked by tests/test_splits.py).", "",
           "## Dev and test sentences that also appear in train", ""] + overlap_lines
    md += ["", "The clip IDs are in `data/clean/test_in_train.csv`, so scores can also be reported without them.",
           "", "## Cleaning rules (from data/clean/removed.csv)", ""] + rule_table(removed)
    if limits:
        md += ["", "## Limits used by the rules", "", "```json", json.dumps(
            {k: v for k, v in limits.items() if k != "by_ear"}, indent=2), "```", "",
            "## By-ear checks (data/clean/manual_check.csv)", "",
            "| Rule | Result | Sampled | Really wrong | Not wrong |", "|---|---|---:|---:|---:|"]
        names = {"8": "8 wrong transcript", "9": "9 noise / clipping"}
        for rule, r in limits.get("by_ear", {}).items():
            md.append(f"| {names.get(rule, rule)} | {r['result']} | {r['sampled']} | {r['yes']} | {r['no']} |")
    (CLEAN / "stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {CLEAN / 'stats.md'} and {CLEAN / 'test_in_train.csv'}")


if __name__ == "__main__":
    main()
