"""
Adversarial carry-chain test set, entirely WITHIN the trained digit range
(1-10 digits both operands). Tests whether the model has learned a
length/position-invariant carry-propagation procedure, independent of the
digit-length extrapolation question that test_ood.txt targets.

Construction: one operand is a repunit of 9s (10^d - 1), the other is a
random d-digit number. Since 9 + x + carry_in >= 10 unless x == 0 and
carry_in == 0, this forces carries to propagate almost the entire length of
the number, with the random operand providing example-to-example diversity
(and occasional natural carry breaks) rather than a single fixed example
per digit count.
"""

import os
import random

from generate import digit_range, count_carries

SEED = 2024
MAX_DIGITS = 10
PER_BUCKET = 200

OUT_DIR = os.path.dirname(__file__)


def main():
    rng = random.Random(SEED)

    train_pairs = set()
    train_path = os.path.join(OUT_DIR, 'train.txt')
    if os.path.exists(train_path):
        with open(train_path) as f:
            for line in f:
                lhs, c = line.strip().split('=')
                a_str, b_str = lhs.split('+')
                train_pairs.add((int(a_str), int(b_str)))

    examples = []
    for d in range(1, MAX_DIGITS + 1):
        lo, hi = digit_range(d)
        nines = 10 ** d - 1
        seen = set()
        n = min(PER_BUCKET, hi - lo + 1)
        attempts, max_attempts = 0, PER_BUCKET * 50 + 1000
        while len(seen) < n and attempts < max_attempts:
            attempts += 1
            x = rng.randint(lo, hi)
            if x in seen:
                continue
            # alternate which side holds the 9s so both operand positions are covered
            a, b = (nines, x) if len(seen) % 2 == 0 else (x, nines)
            if (a, b) in train_pairs:
                continue  # avoid contaminating the "held-out" test with memorized pairs
            seen.add(x)
            examples.append((d, a, b))

    with open(os.path.join(OUT_DIR, 'test_carry_heavy.txt'), 'w') as f:
        for _, a, b in examples:
            f.write(f"{a}+{b}={a + b}\n")

    carry_counts = [count_carries(a, b) for _, a, b in examples]
    by_digits = {}
    for (d, a, b), c in zip(examples, carry_counts):
        by_digits.setdefault(d, []).append(c)
    print(f"wrote {len(examples)} examples to test_carry_heavy.txt")
    for d in sorted(by_digits):
        cs = by_digits[d]
        print(f"  digits={d:2d}  mean carries={sum(cs)/len(cs):.2f}  min={min(cs)}  max={max(cs)}  (n={len(cs)})")


if __name__ == '__main__':
    main()
