"""Error analysis on the best test ASR system: OOV rate, automatic suffix/root/apostrophe breakdown, and 100-sentence sample."""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from data.normalize import normalize_text


def load_csv_transcripts(csv_path: Path) -> Dict[str, str]:
    rows: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["id"]] = row["transcript"]
    return rows


def load_predictions(jsonl_path: Path) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            preds[item["id"]] = item.get("hyp", "")
    return preds


def find_best_test_system(scores_csv: Path) -> Tuple[str, float]:
    best_sys, best_wer = None, float("inf")
    with scores_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test" and row["wer_norm"] != "":
                wer = float(row["wer_norm"])
                if wer < best_wer:
                    best_wer = wer
                    best_sys = row["system"]
    if best_sys is None:
        raise ValueError("No test rows found in scores.csv")
    return best_sys, best_wer


def align_words(ref: List[str], hyp: List[str]) -> List[Tuple[str, str, str]]:
    """Standard Levenshtein word alignment returning list of (op, ref_word, hyp_word)."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bp = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bp[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bp[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                bp[i][j] = "E"
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                if best == sub:
                    bp[i][j] = "S"
                elif best == dele:
                    bp[i][j] = "D"
                else:
                    bp[i][j] = "I"

    ops: List[Tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = bp[i][j]
        if op in ("E", "S"):
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            ops.append(("D", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("I", "", hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def common_prefix_len(a: str, b: str) -> int:
    k = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        k += 1
    return k


def classify_substitution(ref_w: str, hyp_w: str) -> str:
    """Classify a word substitution into apostrophe, suffix (common prefix rule), or root/other."""
    if ref_w.replace("'", "") == hyp_w.replace("'", ""):
        return "apostrophe_error"
    lcp = common_prefix_len(ref_w, hyp_w)
    min_len = min(len(ref_w), len(hyp_w))
    if lcp >= 3 and lcp >= 0.5 * min_len:
        return "suffix_error"
    return "root_or_other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR error analysis on the best test model.")
    parser.add_argument("--scores", type=Path, default=Path("benchmark/results/scores.csv"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("benchmark/predictions"))
    parser.add_argument("--clean-dir", type=Path, default=Path("data/clean"))
    parser.add_argument("--system", type=str, default=None, help="Override best system name")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("analysis/error_analysis.md"))
    args = parser.parse_args()

    if args.system:
        system_name = args.system
        best_wer = None
    else:
        system_name, best_wer = find_best_test_system(args.scores)

    train_refs = load_csv_transcripts(args.clean_dir / "train.csv")
    test_refs = load_csv_transcripts(args.clean_dir / "test.csv")
    preds = load_predictions(args.predictions_dir / f"{system_name}__test.jsonl")

    # 1. OOV rate on test relative to train (after normalize_text)
    train_vocab = {w for t in train_refs.values() for w in normalize_text(t).split()}
    test_tokens = [w for t in test_refs.values() for w in normalize_text(t).split()]
    test_types = set(test_tokens)
    oov_tokens = sum(1 for w in test_tokens if w not in train_vocab)
    oov_types = sum(1 for w in test_types if w not in train_vocab)
    oov_token_rate = oov_tokens / len(test_tokens) if test_tokens else 0.0
    oov_type_rate = oov_types / len(test_types) if test_types else 0.0

    # 2. Global error breakdown across all test clips
    op_counts = Counter()
    sub_counts = Counter()
    suffix_examples: Counter = Counter()
    apostrophe_examples: Counter = Counter()
    erroneous_clips = []

    for cid, raw_ref in test_refs.items():
        ref_norm = normalize_text(raw_ref)
        hyp_norm = normalize_text(preds.get(cid, ""))
        ref_words = ref_norm.split()
        hyp_words = hyp_norm.split()

        ops = align_words(ref_words, hyp_words)
        clip_cats = Counter()
        for op, rw, hw in ops:
            if op == "E":
                continue
            op_counts[op] += 1
            if op == "S":
                cat = classify_substitution(rw, hw)
                sub_counts[cat] += 1
                clip_cats[cat] += 1
                if cat == "suffix_error":
                    suffix_examples[(rw, hw)] += 1
                elif cat == "apostrophe_error":
                    apostrophe_examples[(rw, hw)] += 1
            elif op == "D":
                clip_cats["deletion"] += 1
            elif op == "I":
                clip_cats["insertion"] += 1

        if clip_cats:
            erroneous_clips.append({
                "id": cid,
                "ref": ref_norm,
                "hyp": hyp_norm,
                "cats": ", ".join(f"{k}({v})" for k, v in clip_cats.most_common()),
            })

    total_errors = op_counts["S"] + op_counts["D"] + op_counts["I"]
    total_subs = op_counts["S"]

    rng = random.Random(args.seed)
    sample_n = min(args.sample_size, len(erroneous_clips))
    sampled = rng.sample(erroneous_clips, sample_n)

    # 3. Write Markdown report
    lines = [
        f"# Error Analysis — `{system_name}` (USC Test)",
        "",
        f"- **Evaluated model:** `{system_name}`" + (f" (test WER: {best_wer * 100:.2f}%)" if best_wer is not None else ""),
        f"- **Total test clips:** {len(test_refs)} | **Erroneous clips:** {len(erroneous_clips)} ({len(erroneous_clips)/len(test_refs)*100:.1f}%)",
        f"- **Test OOV rate (token-based):** {oov_tokens} / {len(test_tokens)} (**{oov_token_rate * 100:.2f}%**)",
        f"- **Test OOV rate (unique word/type-based):** {oov_types} / {len(test_types)} (**{oov_type_rate * 100:.2f}%**)",
        "",
        "## Overall Error Distribution (Entire Test Set)",
        "",
        "| Error Type | Count | Share of Total Errors |",
        "|---|---:|---:|",
        f"| Substitution | {op_counts['S']} | {op_counts['S']/total_errors*100:.1f}% |",
        f"| Deletion | {op_counts['D']} | {op_counts['D']/total_errors*100:.1f}% |",
        f"| Insertion | {op_counts['I']} | {op_counts['I']/total_errors*100:.1f}% |",
        f"| **Total** | **{total_errors}** | **100.0%** |",
        "",
        "## Substitution Breakdown (Automatic Rule)",
        "",
        "Automatic rule: word pairs that become identical after removing the apostrophe (`'`) are classified as **apostrophe errors**, "
        "pairs with a common prefix of at least 3 characters and at least 50% of the shorter word are classified as **suffix errors**, "
        "and all others are classified as **root/other errors**.",
        "",
        "| Substitution Subtype | Count | Share of Substitutions |",
        "|---|---:|---:|",
        f"| Suffix error (common prefix) | {sub_counts['suffix_error']} | {sub_counts['suffix_error']/total_subs*100:.1f}% |",
        f"| Root/other substitution error | {sub_counts['root_or_other']} | {sub_counts['root_or_other']/total_subs*100:.1f}% |",
        f"| Apostrophe error (`'`) | {sub_counts['apostrophe_error']} | {sub_counts['apostrophe_error']/total_subs*100:.1f}% |",
        "",
        "### Most Frequent Suffix Errors (Top 10)",
        "",
        "| Reference | Prediction | Count |",
        "|---|---|---:|",
    ]
    for (rw, hw), cnt in suffix_examples.most_common(10):
        lines.append(f"| `{rw}` | `{hw}` | {cnt} |")

    lines += [
        "",
        "### Most Frequent Apostrophe Errors (Top 10)",
        "",
        "| Reference | Prediction | Count |",
        "|---|---|---:|",
    ]
    for (rw, hw), cnt in apostrophe_examples.most_common(10):
        lines.append(f"| `{rw}` | `{hw}` | {cnt} |")

    lines += [
        "",
        f"## Random Sample of {sample_n} Erroneous Test Sentences (`seed={args.seed}`)",
        "",
        "| # | Clip ID | Reference (normalized) | Prediction (normalized) | Error Categories |",
        "|---:|---|---|---|---|",
    ]
    for idx, item in enumerate(sampled, 1):
        ref_esc = item["ref"].replace("|", "\\|")
        hyp_esc = item["hyp"].replace("|", "\\|")
        lines.append(f"| {idx} | `{item['id']}` | {ref_esc} | {hyp_esc} | {item['cats']} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote error analysis for {system_name} to {args.out}")


if __name__ == "__main__":
    main()