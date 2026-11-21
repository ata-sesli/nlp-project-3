"""Uzbek (Latin) ASR text normalization, Cyrillic transliteration, and number verbalization."""

import re

# All apostrophe / modifier letter variants used in Uzbek Latin texts -> ASCII '
APOSTROPHE_VARIANTS = re.compile(r"[‘’`ʻʼ´ʹ＇]")

# Keep letters (Unicode), digits, spaces, and ASCII apostrophe; drop other punctuation
NON_WORD_EXCEPT_APOSTROPHE = re.compile(r"[^\w\s']", flags=re.UNICODE)
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
DIGIT_RE = re.compile(r"\d")

CYRILLIC_VOWELS = set("аеёиоуыэюяў")

CYRILLIC_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
    "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "x", "ч": "ch", "ш": "sh", "ъ": "'",
    "ь": "", "э": "e", "ю": "yu", "я": "ya", "ў": "o'",
    "қ": "q", "ғ": "g'", "ҳ": "h",
}

ONES = ["", "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz"]
TENS = ["", "o'n", "yigirma", "o'ttiz", "qirq", "ellik", "oltmish", "yetmish", "sakson", "to'qson"]


def has_cyrillic(text: str) -> bool:
    """Return True if text contains any Cyrillic character."""
    return bool(CYRILLIC_RE.search(text or ""))


def has_digits(text: str) -> bool:
    """Return True if text contains any ASCII/Unicode digit."""
    return bool(DIGIT_RE.search(text or ""))


def normalize_text(text: str) -> str:
    """Primary ASR normalization (applied to both reference and hypothesis).

    - Lowercase
    - Standardize apostrophe variants to ASCII '
    - Remove punctuation while preserving word-internal apostrophes (o', g', ma'no)
    - Collapse whitespace
    - Keep numbers and Cyrillic characters unchanged (strict policy)
    """
    if not text:
        return ""

    text = text.lower()
    text = APOSTROPHE_VARIANTS.sub("'", text)
    text = text.replace("_", " ")
    text = NON_WORD_EXCEPT_APOSTROPHE.sub(" ", text)

    # Remove leading/trailing standalone apostrophes, keep word-internal ones (e.g. o'zbek, ma'no)
    tokens = []
    for tok in text.split():
        cleaned = tok.strip("'")
        if cleaned:
            tokens.append(cleaned)

    return " ".join(tokens)


def transliterate_cyrillic_to_latin(text: str) -> str:
    """Transliterate Uzbek Cyrillic text to Uzbek Latin using fixed rules."""
    if not text:
        return ""

    text = text.lower()
    out = []
    n = len(text)

    for i, ch in enumerate(text):
        prev = text[i - 1] if i > 0 else " "
        is_word_start = not prev.isalpha()

        if ch == "е":
            if is_word_start or prev in CYRILLIC_VOWELS:
                out.append("ye")
            else:
                out.append("e")
        elif ch == "ц":
            if prev in CYRILLIC_VOWELS:
                out.append("ts")
            else:
                out.append("s")
        elif ch in CYRILLIC_MAP:
            out.append(CYRILLIC_MAP[ch])
        else:
            out.append(ch)

    return "".join(out)


def _int_under_1000(n: int) -> str:
    parts = []
    hundreds = n // 100
    rem = n % 100
    if hundreds:
        if hundreds == 1:
            parts.append("bir yuz")
        else:
            parts.append(f"{ONES[hundreds]} yuz")
    tens = rem // 10
    ones = rem % 10
    if tens:
        parts.append(TENS[tens])
    if ones:
        parts.append(ONES[ones])
    return " ".join(parts)


def int_to_uzbek(n: int) -> str:
    """Convert a non-negative integer to Uzbek words."""
    if n == 0:
        return "nol"

    scales = [
        (1_000_000_000, "milliard"),
        (1_000_000, "million"),
        (1_000, "ming"),
    ]
    parts = []
    for scale_val, scale_name in scales:
        if n >= scale_val:
            q, n = divmod(n, scale_val)
            parts.append(f"{_int_under_1000(q)} {scale_name}")
    if n > 0:
        parts.append(_int_under_1000(n))
    return " ".join(parts)


def verbalize_numbers(text: str) -> str:
    """Replace digit sequences with Uzbek number words."""
    if not text:
        return ""

    def _replace(match: re.Match) -> str:
        num_str = match.group(0)
        # Preserve leading zeros digit-by-digit (e.g. phone/code fragments)
        if len(num_str) > 1 and num_str.startswith("0"):
            return " ".join(int_to_uzbek(int(d)) for d in num_str)
        val = int(num_str)
        if val >= 1_000_000_000_000:
            return " ".join(int_to_uzbek(int(d)) for d in num_str)
        return int_to_uzbek(val)

    return re.sub(r"\d+", _replace, text)


def normalize_translit_and_numbers(text: str) -> str:
    """Secondary normalization: Cyrillic->Latin transliteration + number verbalization + primary normalization."""
    if not text:
        return ""
    text = transliterate_cyrillic_to_latin(text)
    text = verbalize_numbers(text)
    return normalize_text(text)