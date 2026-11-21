from data.normalize import (
    has_cyrillic,
    has_digits,
    int_to_uzbek,
    normalize_text,
    normalize_translit_and_numbers,
    transliterate_cyrillic_to_latin,
)


def test_uzbek_apostrophes_preserved():
    assert normalize_text("O‘zbekiston, gʻisht va maʼno!") == "o'zbekiston g'isht va ma'no"
    assert normalize_text("O`zbek g’isht ma’no") == "o'zbek g'isht ma'no"


def test_punctuation_and_whitespace():
    assert normalize_text("  Salom,   dunyo?!  ") == "salom dunyo"
    assert normalize_text("'salom' 'o'zbek'") == "salom o'zbek"


def test_strict_policy_keeps_digits_and_cyrillic():
    assert normalize_text("2026 yil") == "2026 yil"
    assert normalize_text("Тошкент") == "тошкент"
    assert has_digits("2026 yil") is True
    assert has_cyrillic("Тошкент") is True
    assert has_cyrillic("toshkent") is False


def test_cyrillic_transliteration():
    assert transliterate_cyrillic_to_latin("Ўзбекистон") == "o'zbekiston"
    assert transliterate_cyrillic_to_latin("ғишт") == "g'isht"
    assert transliterate_cyrillic_to_latin("маъно") == "ma'no"
    assert transliterate_cyrillic_to_latin("ер") == "yer"
    assert transliterate_cyrillic_to_latin("мен") == "men"
    assert transliterate_cyrillic_to_latin("цирк") == "sirk"
    assert transliterate_cyrillic_to_latin("акция") == "aksiya"
    assert transliterate_cyrillic_to_latin("лицей") == "litsey"


def test_uzbek_numbers():
    assert int_to_uzbek(0) == "nol"
    assert int_to_uzbek(15) == "o'n besh"
    assert int_to_uzbek(2026) == "ikki ming yigirma olti"


def test_secondary_normalization():
    assert normalize_translit_and_numbers("Ўзбекистон 2026!") == "o'zbekiston ikki ming yigirma olti"