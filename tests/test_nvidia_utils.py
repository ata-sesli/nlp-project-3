"""Checks for the A8/A9 helper code (CPU only, no model download).

    pytest tests/test_nvidia_utils.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "baselines" / "nvidia_fastconformer")]


def test_nemo_output_to_text():
    from nemo_utils import _texts

    class Hyp:
        def __init__(self, text):
            self.text = text

    assert _texts([Hyp("a"), Hyp("b")]) == ["a", "b"]  # newer NeMo: list of hypotheses
    assert _texts(([Hyp("a")], None)) == ["a"]  # older NeMo: (best, all)
    assert _texts(["x", "y"]) == ["x", "y"]
