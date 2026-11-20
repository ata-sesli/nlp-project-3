"""Checks for the C4 helper code (CPU only, no NeMo, no model download).

    pytest tests/test_nvidia_finetune.py
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# loaded under its own name: other folders also have a train.py
_spec = importlib.util.spec_from_file_location("c4_train", ROOT / "nvidia_finetune" / "train.py")
c4_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c4_train)


class FakeTokenizer:
    """Knows the letters a-z, space and only the apostrophe forms given; anything else becomes '?'."""

    def __init__(self, apostrophes):
        self.known = set("abcdefghijklmnopqrstuvwxyz ") | set(apostrophes)

    def text_to_ids(self, text):
        return [ord(c) if c in self.known else 0 for c in text]

    def ids_to_text(self, ids):
        return "".join(chr(i) if i else "?" for i in ids)


def test_pick_apostrophe():
    assert c4_train.pick_apostrophe(FakeTokenizer(["'"])) == "'"
    assert c4_train.pick_apostrophe(FakeTokenizer(["ʻ"])) == "ʻ"
    assert c4_train.pick_apostrophe(FakeTokenizer(["’", "ʼ"])) == "ʼ"  # first known form in APOSTROPHE_FORMS order
    assert c4_train.pick_apostrophe(FakeTokenizer([])) == "'"


def test_apostrophe_rule_like_uzbek_spelling():
    fix, _ = c4_train.apostrophe_rule(FakeTokenizer(["ʻ", "’"]))
    assert fix("o'zbek tilida ma'no bor g'oz") == "oʻzbek tilida ma’no bor gʻoz"
    fix, _ = c4_train.apostrophe_rule(FakeTokenizer(["ʻ"]))
    assert fix("o'zbek ma'no") == "oʻzbek maʻno"
    fix, _ = c4_train.apostrophe_rule(FakeTokenizer(["'"]))
    assert fix("o'zbek ma'no") == "o'zbek ma'no"


def test_unknown_chars():
    assert c4_train.unknown_chars(FakeTokenizer(["'"]), ["o'zbek tili", "x1y"]) == ["1"]


def test_manifest_has_offset_and_duration(tmp_path):
    rows = [{"path": str(tmp_path / "a.flac"), "start_s": "0.25", "end_s": "2.5"}]
    c4_train.write_manifest(tmp_path / "m.jsonl", rows, ["salom dunyo"])
    item = json.loads((tmp_path / "m.jsonl").read_text(encoding="utf-8"))
    assert item["offset"] == 0.25 and item["duration"] == 2.25 and item["text"] == "salom dunyo"
    assert Path(item["audio_filepath"]).is_absolute()


def test_model_card_dry_run(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    info = {"start_revision": "abc123", "train_data": {"clips": 100, "hours": 0.1},
            "training": {"best_epoch": 3, "epochs_done": 5}}
    for name, wer in [("C4a_fastconformer-usc-ft-rnnt", 0.2), ("C4b_fastconformer-usc-ft-ctc", 0.25)]:
        log = {"training": info, "runs": {"dev": {"wer_quick": wer, "cer_quick": 0.05}}}
        (results / f"{name}.json").write_text(json.dumps(log), encoding="utf-8")
    base = {"runs": {"dev": {"wer_quick": 0.3, "cer_quick": 0.1}}}   # A8 as it is (only the transducer head here)
    (results / "A8_nvidia-fastconformer-rnnt.json").write_text(json.dumps(base), encoding="utf-8")
    out = subprocess.run([sys.executable, str(ROOT / "nvidia_finetune" / "push_to_hub.py"), "--nemo", "x.nemo",
                          "--repo", "me/model", "--dry-run"], capture_output=True, text=True,
                         env=dict(os.environ, ASR_OUTPUT_ROOT=str(tmp_path)), check=True).stdout
    assert "| transducer head (C4a) | USC dev | 20.00% | 5.00% |" in out
    assert "| CTC head (C4b) | USC dev | 25.00% | 5.00% |" in out
    assert "| start model, transducer head (A8) | USC dev | 30.00% | 10.00% |" in out
    assert "(A9)" not in out and "start model)" not in out
    assert "best epoch 3 of 5" in out and "abc123" in out
