"""Rewrite a requirements.txt with the exact versions installed right now.

The Kaggle notebook runs this at the end, so the committed files list the versions that really ran.
    python common/pin_requirements.py xlsr_ctc/requirements.txt
"""
import re
import sys
from importlib import metadata
from pathlib import Path


def main(path: str) -> None:
    p = Path(path)
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]+\])?==", line)
        if not m:
            out.append(line)
            continue
        try:
            out.append(f"{m.group(1)}{m.group(2) or ''}=={metadata.version(m.group(1))}")
        except metadata.PackageNotFoundError:
            out.append(line)
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        main(arg)
