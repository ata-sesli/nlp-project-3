import pytest

from benchmark.evaluate import compute_metrics
from benchmark.timing import RTFTimer


def test_known_wer_and_cer():
    # Reference: "salom o'zbekiston bugun" (3 words)
    # Hypothesis: "salom o'zbekiston ertaga" -> 1 substitution -> WER = 1/3
    refs = ["Salom, O‘zbekiston bugun!"]
    hyps = ["salom o'zbekiston ertaga"]
    res = compute_metrics(refs, hyps)

    assert res["n_clips"] == 1
    assert res["wer_norm"] == pytest.approx(1 / 3, abs=1e-5)
    assert res["wer_ortho"] > res["wer_norm"]  # punctuation/case mismatch increases orthographic WER
    assert res["hyp_with_cyrillic"] == 0
    assert res["hyp_with_digits"] == 0


def test_translit_and_number_column():
    refs = ["o'zbekiston ikki ming"]
    hyps = ["Ўзбекистон 2000"]
    res = compute_metrics(refs, hyps)

    assert res["wer_norm"] == pytest.approx(1.0, abs=1e-5)
    assert res["wer_translit_num"] == pytest.approx(0.0, abs=1e-5)
    assert res["hyp_with_cyrillic"] == 1
    assert res["hyp_with_digits"] == 1


def test_rtf_timer():
    timer = RTFTimer()
    timer.elapsed_s = 2.0
    assert timer.rtf(10.0) == pytest.approx(0.2)