"""Checks for the code of A2, A3 and C2 (CPU only, about 30 seconds).

    pytest tests/test_asr_models.py
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # the model tests need torch; skipped where it is not installed

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "baselines" / "scratch"), str(ROOT / "xlsr_ctc")]

from common.asr_data import _fallback_normalize  # noqa: E402
from model import build_model, rnnt_loss_torch  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    ("O‘zbekiston — go‘zal yurt!", "o'zbekiston go'zal yurt"),
    ("Tog' va 'tosh' ma’no", "tog' va tosh ma'no"),
    ("G`ALABA, 2024-yil.", "g'alaba 2024 yil"),
    ("Привет мир", "привет мир"),  # Cyrillic is kept, never transliterated
])
def test_fallback_normalizer(text, expected):
    assert _fallback_normalize(text) == expected


def test_rnnt_loss_matches_torchaudio():
    torchaudio = pytest.importorskip("torchaudio")
    if not hasattr(torchaudio.functional, "rnnt_loss"):
        pytest.skip("this torchaudio has no rnnt_loss")
    torch.manual_seed(0)
    B, T, U, V = 3, 15, 6, 9
    logits = torch.randn(B, T, U + 1, V, requires_grad=True)
    targets = torch.randint(1, V, (B, U), dtype=torch.int32)
    t_len = torch.tensor([15, 11, 8], dtype=torch.int32)
    u_len = torch.tensor([6, 4, 0], dtype=torch.int32)
    ref = torchaudio.functional.rnnt_loss(logits, targets, t_len, u_len, blank=0, reduction="mean")
    (g_ref,) = torch.autograd.grad(ref, logits)
    ours = rnnt_loss_torch(logits, targets, t_len, u_len)
    (g_ours,) = torch.autograd.grad(ours, logits)
    assert torch.allclose(ref, ours, atol=1e-4)
    assert torch.allclose(g_ref, g_ours, atol=1e-5)


@pytest.mark.parametrize("kind", ["ctc", "rnnt"])
def test_scratch_models_learn_one_batch(kind):
    torch.manual_seed(0)
    cfg = dict(d_model=32, layers=1, heads=2, ffn=64, kernel=7, dropout=0.0, pred_dim=32, joint_dim=32, aux_ctc=0.3)
    model = build_model(kind, 6, cfg)
    wav = torch.randn(2, 16000)
    wav_lens = torch.tensor([16000, 12000])
    targets = torch.tensor([[1, 2, 3, 4], [5, 1, 2, 0]])
    target_lens = torch.tensor([4, 3])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(30):
        loss = model.loss(wav, wav_lens, targets, target_lens)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first
    model.eval()
    hyps = model.decode(wav, wav_lens)
    assert len(hyps) == 2 and all(isinstance(t, int) for h in hyps for t in h)


def test_batches_are_reproducible_and_complete():
    from train import seconds_batches
    from xlsr_lib import length_grouped_batches

    lengths = [1 + (i * 7) % 13 for i in range(500)]
    a = length_grouped_batches(lengths, 8, seed=42)
    assert a == length_grouped_batches(lengths, 8, seed=42)
    assert sorted(i for b in a for i in b) == list(range(500))
    s = seconds_batches(lengths, 60, seed=42)
    assert sorted(i for b in s for i in b) == list(range(500))
    assert all(max(lengths[i] for i in b) * len(b) <= 60 or len(b) == 1 for b in s)


def test_xlsr_vocab_and_labels(tmp_path):
    pytest.importorskip("transformers")
    from xlsr_lib import PAD, build_vocab, ctc_labels, make_processor, save_json

    vocab = build_vocab(["salom o'zbek", "g'alaba"])
    save_json(tmp_path / "vocab.json", vocab)
    proc = make_processor(tmp_path / "vocab.json")
    ids = proc.tokenizer("salom o'zbek").input_ids
    assert proc.tokenizer.unk_token_id not in ids
    assert proc.tokenizer.decode(ids) == "salom o'zbek"
    labels = ctc_labels(proc)
    assert labels[vocab[PAD]] == "" and " " in labels
