"""Word 3-gram LM in ARPA format, interpolated Kneser-Ney, from train transcripts only.

Fallback for A1 until Elif's KenLM ARPA (lm/, same train transcripts) is
ready; run.sh takes that file with --arpa. KenLM is not on conda-forge, so
this is a small pure-Python version: one discount per order,
D = n1 / (n1 + 2 n2), continuation counts for the lower orders, and the
unigram distribution interpolated with a uniform distribution so <unk> has
a probability.

Usage:
    python baselines/gmm_hmm/lm_arpa.py data/local/lm_train.txt data/local/lm.arpa
"""

import argparse
import math
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def discount(counts: Counter) -> float:
    coc = Counter(counts.values())
    n1, n2 = coc.get(1, 0), coc.get(2, 0)
    d = n1 / (n1 + 2 * n2) if n1 + n2 else 0.5
    return min(max(d, 0.1), 0.9)


def build(lines: list[list[str]]):
    c3, c2 = Counter(), Counter()
    for words in lines:
        s = [BOS] + words + [EOS]
        c2.update(zip(s, s[1:]))
        c3.update(zip(s, s[1:], s[2:]))

    # Continuation counts: how many distinct words precede the n-gram.
    # Bigrams starting with <s> have nothing before them: keep raw counts.
    k2 = Counter()
    for (_, v, w) in c3:
        k2[(v, w)] += 1
    for (v, w), n in c2.items():
        if v == BOS:
            k2[(v, w)] = n
    k1 = Counter(w for (_, w) in c2)

    d1, d2, d3 = discount(k1), discount(k2), discount(c3)
    vocab = sorted(set(k1) | {UNK})
    total1 = sum(k1.values())
    uniform = 1.0 / len(vocab)
    p1 = {w: max(k1[w] - d1, 0) / total1 + d1 * len(k1) / total1 * uniform for w in vocab}

    def context_stats(counts):
        tot, types = defaultdict(int), defaultdict(int)
        for ng, n in counts.items():
            tot[ng[:-1]] += n
            types[ng[:-1]] += 1
        return tot, types

    t2, n2 = context_stats(k2)
    bow1 = {h[0]: d2 * n2[h] / t2[h] for h in t2}
    p2 = {(v, w): (k2[(v, w)] - d2) / t2[(v,)] + bow1[v] * p1[w] for (v, w) in k2}

    t3, n3 = context_stats(c3)
    bow2 = {h: d3 * n3[h] / t3[h] for h in t3}
    p3 = {(u, v, w): (n - d3) / t3[(u, v)] + bow2[(u, v)] * p2[(v, w)] for (u, v, w), n in c3.items()}
    return p1, bow1, p2, bow2, p3, (d1, d2, d3)


def log10(x: float) -> str:
    return f"{math.log10(x):.6f}" if x > 0 else "-99"


def write_arpa(path: str, p1, bow1, p2, bow2, p3) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        f.write(f"ngram 1={len(p1) + 1}\nngram 2={len(p2)}\nngram 3={len(p3)}\n\n\\1-grams:\n")
        f.write(f"-99\t{BOS}\t{log10(bow1.get(BOS, 0))}\n")
        for w in sorted(p1):
            bow = f"\t{log10(bow1[w])}" if w in bow1 else ""
            f.write(f"{log10(p1[w])}\t{w}{bow}\n")
        f.write("\n\\2-grams:\n")
        for ng in sorted(p2):
            bow = f"\t{log10(bow2[ng])}" if ng in bow2 else ""
            f.write(f"{log10(p2[ng])}\t{' '.join(ng)}{bow}\n")
        f.write("\n\\3-grams:\n")
        for ng in sorted(p3):
            f.write(f"{log10(p3[ng])}\t{' '.join(ng)}\n")
        f.write("\n\\end\\\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("text")
    p.add_argument("arpa")
    args = p.parse_args()
    with open(args.text, encoding="utf-8") as f:
        lines = [line.split() for line in f if line.strip()]
    p1, bow1, p2, bow2, p3, ds = build(lines)
    write_arpa(args.arpa, p1, bow1, p2, bow2, p3)
    print(f"{len(lines)} sentences -> {args.arpa}: {len(p1)} unigrams, {len(p2)} bigrams, "
          f"{len(p3)} trigrams; discounts {', '.join(f'{d:.3f}' for d in ds)}")


if __name__ == "__main__":
    main()
