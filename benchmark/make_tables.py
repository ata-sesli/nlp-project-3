"""Generate Markdown Table A (Baselines) and Table B (Contributions) from scores.csv."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def fmt_pct(val: str) -> str:
    return f"{float(val) * 100:.2f}%" if val != "" else "-"


def build_table(rows: List[Dict[str, str]]) -> str:
    header = "| System | WER (norm) | CER (norm) | WER (orthographic) | WER (translit+numbers) | RTF | Hardware | Parameters |"
    sep = "|---|---:|---:|---:|---:|---:|---|---:|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['system']} | {fmt_pct(r['wer_norm'])} | {fmt_pct(r['cer_norm'])} | "
            f"{fmt_pct(r['wer_ortho'])} | {fmt_pct(r['wer_translit_num'])} | "
            f"{r['rtf'] or '-'} | {r['hardware'] or '-'} | {r['params'] or '-'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build README Markdown tables from scores.csv.")
    parser.add_argument("--scores", type=Path, default=Path("benchmark/results/scores.csv"))
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    with args.scores.open("r", encoding="utf-8") as f:
        all_rows = [r for r in csv.DictReader(f) if r["split"] == args.split]

    table_a = [r for r in all_rows if r["system"].startswith("A")]
    table_b = [r for r in all_rows if r["system"].startswith("C")]

    print(f"### Table A: Baselines ({args.split})\n")
    print(build_table(table_a))
    print(f"\n### Table B: Contributions ({args.split})\n")
    print(build_table(table_b))


if __name__ == "__main__":
    main()