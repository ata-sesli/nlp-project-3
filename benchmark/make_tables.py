"""Generate Markdown Table A (Baselines) and Table B (Contributions) from scores.csv and confidence_intervals.csv."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def fmt_pct(val: str) -> str:
    return f"{float(val) * 100:.2f}%" if val not in ("", None, "None", "null") else "-"


def fmt_ci(ci: Tuple[str, str] | None) -> str:
    if not ci:
        return "-"
    low, high = ci
    return f"[{float(low) * 100:.2f}%, {float(high) * 100:.2f}%]"


def load_cis(ci_path: Path, split: str, metric: str = "wer_norm") -> Dict[str, Tuple[str, str]]:
    """Load system -> (ci_low, ci_high) mapping for the given split and metric."""
    cis: Dict[str, Tuple[str, str]] = {}
    if not ci_path.exists():
        return cis
    with ci_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == split and row["metric"] == metric:
                cis[row["system"]] = (row["ci_low"], row["ci_high"])
    return cis


def fmt_params(system: str, params: str) -> str:
    if params not in ("", None, "None", "null"):
        return params
    if system.startswith("A1"):
        return "—*"
    return "-"


def build_table(rows: List[Dict[str, str]], wer_cis: Dict[str, Tuple[str, str]]) -> str:
    header = (
        "| System | WER (norm) | WER 95% CI | CER (norm) | WER (orthographic) "
        "| WER (translit+number) | RTF | Hardware | Parameters |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---|---:|"
    lines = [header, sep]
    for r in rows:
        sys_name = r["system"]
        lines.append(
            f"| {sys_name} | {fmt_pct(r['wer_norm'])} | {fmt_ci(wer_cis.get(sys_name))} | "
            f"{fmt_pct(r['cer_norm'])} | {fmt_pct(r['wer_ortho'])} | "
            f"{fmt_pct(r['wer_translit_num'])} | {r['rtf'] or '-'} | "
            f"{r['hardware'] or '-'} | {fmt_params(sys_name, r['params'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build README Markdown tables from scores.csv.")
    parser.add_argument("--scores", type=Path, default=Path("benchmark/results/scores.csv"))
    parser.add_argument("--ci", type=Path, default=Path("benchmark/results/confidence_intervals.csv"))
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    with args.scores.open("r", encoding="utf-8") as f:
        all_rows = [r for r in csv.DictReader(f) if r["split"] == args.split]

    wer_cis = load_cis(args.ci, args.split, metric="wer_norm")

    table_a = [r for r in all_rows if r["system"].startswith("A")]
    table_b = [r for r in all_rows if r["system"].startswith("C")]

    print(f"### Table A: Baselines ({args.split})\n")
    print(build_table(table_a, wer_cis))
    if any(r["system"].startswith("A1") for r in table_a):
        print(
            "\n*\\* A1 (GMM-HMM) is a classical acoustic model; instead of neural-network "
            "parameter counts, it is characterized by Gaussian mixture components and HMM states.*"
        )

    print(f"\n### Table B: Our Contribution ({args.split})\n")
    print(build_table(table_b, wer_cis))


if __name__ == "__main__":
    main()