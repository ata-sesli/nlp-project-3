"""Cleaning rule 8: find train clips whose transcript may be wrong.

The NVIDIA model (A9, CTC head) transcribes every train clip. We compute CER between its
output and the human transcript (both normalized). Clips above the threshold are only
"suspects": the human transcript stays the reference, model output is never used as a transcript.

Outputs (Ata's data/clean.py reads them):
  data/clean/rule8_cer.csv      every train clip: id, speaker_id, path, duration_s, cer, flagged, transcript, model_output
  data/clean/manual_check.csv   a random sample of flagged clips (seed 42) for the by-ear check.
                                Rows of other rules (rule 9) in this file are kept.
                                Fill the column really_wrong with yes / no.

Run from the repo root:
  # 1) Model run on all raw train clips (before data/clean/train.csv exists):
  python baselines/nvidia_fastconformer/rule8_flag.py --raw-dir data/raw/usc/train
  # 2) After the by-ear check, if most sampled clips were NOT really wrong, raise the threshold
  #    without running the model again:
  python baselines/nvidia_fastconformer/rule8_flag.py --from-cer --threshold 0.7
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.asr_data import (CLEAN_DIR, REPO_ROOT, RTFTimer, cer, env_info, normalize,  # noqa: E402
                             read_split, set_seed)

CER_FILE = "rule8_cer.csv"
CHECK_FILE = "manual_check.csv"
CER_COLUMNS = ["id", "speaker_id", "path", "duration_s", "cer", "flagged", "transcript", "model_output"]
CHECK_COLUMNS = ["rule", "id", "path", "score", "transcript", "model_output", "really_wrong", "note"]


def rows_from_raw(raw_dir: Path) -> list[dict]:
    """data/raw/usc/train/<speaker_id>/<clip_id>.flac + <clip_id>.txt (plan B2 layout)."""
    rows = []
    for flac in sorted(raw_dir.glob("*/*.flac")):
        txt = flac.with_suffix(".txt")
        if not txt.exists():
            print(f"WARNING: no transcript for {flac}")
            continue
        info = sf.info(flac)
        rows.append({
            "id": flac.stem,
            "speaker_id": flac.parent.name,
            "path": str(flac.relative_to(REPO_ROOT)) if flac.is_relative_to(REPO_ROOT) else str(flac),
            "audio": str(flac),
            "transcript": txt.read_text(encoding="utf-8").strip(),
            "duration_s": info.frames / info.samplerate,
            "start_s": 0.0,
            "end_s": None,
        })
    return rows


def run_model(rows: list[dict], batch_size: int, revision: str | None) -> tuple[list[str], dict]:
    from nemo_utils import MODEL_ID, load_model, set_head, transcribe_rows

    model, sha = load_model(revision)
    set_head(model, "ctc")
    timer = RTFTimer()
    hyps = transcribe_rows(model, rows, batch_size=batch_size, timer=timer)
    return hyps, {"model_id": MODEL_ID, "revision": sha, "head": "ctc", "rtf": timer.rtf, "env": env_info()}


def cer_table(rows: list[dict], hyps: list[str]) -> list[dict]:
    out = []
    for r, h in zip(rows, hyps):
        out.append({
            "id": r["id"], "speaker_id": r.get("speaker_id", ""), "path": r["path"],
            "duration_s": f"{float(r['duration_s']):.3f}",
            "cer": f"{cer(normalize(r['transcript']), normalize(h)):.4f}",
            "flagged": "", "transcript": r["transcript"], "model_output": h,
        })
    out.sort(key=lambda x: -float(x["cer"]))
    return out


def save_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--raw-dir", default=None, help="raw train folder, e.g. data/raw/usc/train")
    src.add_argument("--from-cer", action="store_true", help="reuse rule8_cer.csv, do not run the model")
    ap.add_argument("--clean-dir", default=str(CLEAN_DIR), help="where train.csv is read and outputs are written")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sample-size", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (quick check)")
    args = ap.parse_args()
    set_seed()

    clean_dir = Path(args.clean_dir)
    cer_path = clean_dir / CER_FILE
    if args.from_cer:
        table = load_csv(cer_path)
    else:
        if args.raw_dir:
            rows = rows_from_raw(Path(args.raw_dir).resolve())
        else:
            rows = read_split("train", clean_dir)
        if args.limit:
            rows = rows[:args.limit]
        if not rows:
            raise SystemExit("No train clips found.")
        print(f"Rule 8: transcribing {len(rows)} train clips with the NVIDIA CTC head")
        hyps, meta = run_model(rows, args.batch_size, args.revision)
        print(f"Model {meta['model_id']} @ {meta['revision']}, RTF {meta['rtf']:.4f}, GPU {meta['env'].get('gpu')}")
        table = cer_table(rows, hyps)

    for r in table:
        r["flagged"] = "yes" if float(r["cer"]) > args.threshold else "no"
    save_csv(cer_path, table, CER_COLUMNS)
    flagged = [r for r in table if r["flagged"] == "yes"]
    share = len(flagged) / len(table) if table else 0.0
    print(f"Threshold CER > {args.threshold}: {len(flagged)} of {len(table)} clips flagged ({share:.2%})")
    if share > 0.01:
        print("NOTE: more than 1% of train is flagged. Plan B2: the team checks this before clean.py runs.")

    # Random sample for the by-ear check (seed 42), sorted by id so the list is easy to follow.
    sample = random.Random(42).sample(flagged, min(args.sample_size, len(flagged)))
    sample.sort(key=lambda r: r["id"])
    check_path = clean_dir / CHECK_FILE
    kept = [r for r in load_csv(check_path) if r.get("rule") != "8"] if check_path.exists() else []
    new = [{"rule": "8", "id": r["id"], "path": r["path"], "score": f"cer={r['cer']} threshold={args.threshold}",
            "transcript": r["transcript"], "model_output": r["model_output"], "really_wrong": "", "note": ""}
           for r in sample]
    save_csv(check_path, kept + new, CHECK_COLUMNS)
    print(f"Wrote {cer_path} and {len(new)} rule-8 rows to {check_path} (fill really_wrong with yes/no).")


if __name__ == "__main__":
    main()
