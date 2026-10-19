"""
Synthetic multi-digit addition task generator.

Produces newline-delimited examples of the form "a+b=c\n". Difficulty is
controlled by digit count: for digit count d, ONE operand (randomly a or b)
is sampled with exactly d digits and the OTHER is sampled with a uniformly
random length from 1..d -- so d = max(len(a), len(b)), matching the
convention eval_addition.py already uses for difficulty bucketing. (An
earlier version of this generator forced BOTH operands to exactly d digits
for every example -- so the model only ever saw same-length operand pairs
and never learned addition as a length-independent procedure in that
respect. Fixed after noticing eval already assumed the general,
mismatched-length case that training data never actually contained.)
(d=1 includes 0-9; d>=2 excludes leading zeros.)

Four length tiers, from a full restart after finding the above bug plus
switching the whole project to RoPE (see project memory):
  - TRAIN range (1..TRAIN_MAX_DIGITS): fully trained, matches every prior
    stage's in-distribution range.
  - GAP range (GAP_MIN..GAP_MAX_DIGITS): ZERO exposure anywhere in
    train/val -- a genuine interpolation-gap test. If the model does well
    here despite never seeing these lengths in any form, that's real
    evidence of a length-general procedure, not just memorized buckets.
  - HINT range (HINT_MIN..HINT_MAX_DIGITS): a SPARSE slice mixed into
    train/val at low density (a "few longer digits" so the model at least
    knows answers this long exist) -- sits beyond the gap, closer to OOD.
  - OOD range (OOD_MIN..OOD_MAX_DIGITS): ZERO exposure anywhere, the
    true "extrapolate past everything it's ever seen" test every prior
    stage was judged on.

Splits:
  - train.txt              : TRAIN range (full, ADVERSARIAL_FRAC carry-heavy) + HINT range (sparse, same mix)
  - val.txt                 : held-out pairs, same digit + adversarial mix as train
  - test_indist.txt         : held-out pairs, TRAIN range, natural distribution (unmixed --
                              kept comparable to every prior stage's eval number)
  - test_gap.txt            : held-out pairs, GAP range, natural distribution (never in train/val)
  - test_hint.txt           : held-out pairs, HINT range, natural distribution (distinct from train's sparse slice)
  - test_ood.txt            : held-out pairs, OOD range, natural distribution (never in train/val)
  - test_carry_heavy_gap.txt: held-out pairs, GAP range, ALL carry-heavy (see below)
  - test_carry_heavy_ood.txt: held-out pairs, OOD range, ALL carry-heavy (see below)

Adversarial carry-heavy examples (sample_carry_heavy): a naive shortcut for
this task is "output digit count = longer operand's digit count" -- true
often enough under uniform random sampling that the model can hit near-zero
training loss without ever really learning when a carry cascades far enough
to push the output a digit longer. sample_carry_heavy constructs the longer
operand with a run of trailing 9s (length chosen uniformly, including the
full-length case, which forces an actual d -> d+1 digit overflow) and a
short operand with a nonzero units digit to trigger the cascade, so a
meaningful fraction of every TRAIN/HINT bucket can't be solved by that
shortcut -- forcing the model to track carry propagation as a real variable,
which (if it generalizes) should transfer to GAP/OOD lengths the same rule
was never directly supervised on. test_carry_heavy_gap/ood isolate exactly
that: same GAP/OOD digit ranges as the natural tests, but every example
carry-heavy, so they directly test whether the rule transferred to unseen
lengths (as opposed to test_gap/test_ood, which mostly exercise the far more
common "output length == input length" case).

Uniqueness is enforced per digit-count bucket via rejection sampling against a
seen-pairs set, so no (a, b) pair appears in more than one split.
"""

import os
import random

SEED = 1337
TRAIN_MAX_DIGITS = 10
GAP_MIN_DIGITS = 11
GAP_MAX_DIGITS = 13
# HINT range extended 15 -> 18 (was the "true OOD, zero exposure" range through
# the whole grokking track this session): user wants to directly test whether
# sparse, correctly-labeled exposure at these lengths fixes what near-miss
# errors there suggested was partial, "confused" competence rather than a
# totally absent capability. OOD pushed out to 19-21 so a genuine
# never-seen-length extrapolation test still exists beyond this.
HINT_MIN_DIGITS = 14
HINT_MAX_DIGITS = 18
OOD_MIN_DIGITS = 19
OOD_MAX_DIGITS = 21
# genuinely untouched holdout: the drill mechanism (train.py's
# drill_on_failures, drill_digit_hi) reaches up to OOD_MAX_DIGITS=21, so
# test_ood is no longer a clean extrapolation test once drilling is in play --
# it's just more exposure, relabeled. FAR_OOD sits past drilling's reach
# entirely (drill_digit_hi must stay <=21 for this to hold), the real test of
# whether patching blind spots at 11-21 induces any generalization beyond
# territory that's ever been trained OR drilled on.
FAR_OOD_MIN_DIGITS = 24
FAR_OOD_MAX_DIGITS = 26

# fraction of every TRAIN/HINT train(+val) bucket built by sample_carry_heavy
# instead of uniform random sampling -- see module docstring
ADVERSARIAL_FRAC = 0.3

TRAIN_PER_BUCKET = 4000
VAL_PER_BUCKET = 200
TEST_INDIST_PER_BUCKET = 200
TEST_GAP_PER_BUCKET = 200
# sparse hint tier: ~2.5% of a normal bucket's train count -- enough to remind
# the model that answers this long exist, not enough to actually solve them
HINT_TRAIN_PER_BUCKET = 100
HINT_VAL_PER_BUCKET = 20
TEST_HINT_PER_BUCKET = 200
TEST_OOD_PER_BUCKET = 200
TEST_CARRY_HEAVY_GAP_PER_BUCKET = 200
TEST_CARRY_HEAVY_OOD_PER_BUCKET = 200

OUT_DIR = os.path.dirname(__file__)


def digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


def bucket_capacity(d):
    # conservative proxy (true capacity is larger for d>=2 since the "other"
    # operand ranges over 1..d) -- only used to soft-cap requests for tiny
    # buckets (d=1) so sampling doesn't spin forever
    lo, hi = digit_range(d)
    n = hi - lo + 1
    return n * n


def count_carries(a, b):
    carries = 0
    carry = 0
    while a > 0 or b > 0:
        da, db = a % 10, b % 10
        s = da + db + carry
        carry = 1 if s >= 10 else 0
        carries += carry
        a //= 10
        b //= 10
    return carries


def sample_bucket(d, n_wanted, seen, rng):
    capacity = bucket_capacity(d)
    n = min(n_wanted, max(0, capacity - len(seen)))
    examples = []
    attempts = 0
    max_attempts = n * 50 + 1000
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        other_len = rng.randint(1, d)
        if rng.random() < 0.5:
            len_a, len_b = d, other_len
        else:
            len_a, len_b = other_len, d
        lo_a, hi_a = digit_range(len_a)
        lo_b, hi_b = digit_range(len_b)
        a = rng.randint(lo_a, hi_a)
        b = rng.randint(lo_b, hi_b)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        examples.append((a, b))
    return examples


def sample_carry_heavy(d, n_wanted, seen, rng):
    """Same (len_a, len_b) convention as sample_bucket (d = max(len_a, len_b),
    other operand's length uniform in 1..d), but the longer operand is built
    with a run of trailing 9s (a "carry chain") and the shorter operand is
    given a nonzero units digit that triggers a carry into that chain --
    guaranteeing the sum can't be computed by a naive same-length-out
    shortcut. chain_len == long_len is the maximal case: the whole longer
    operand is 9s, so the sum overflows into an extra digit (d -> d+1)."""
    capacity = bucket_capacity(d)
    n = min(n_wanted, max(0, capacity - len(seen)))
    examples = []
    attempts = 0
    max_attempts = n * 50 + 1000
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        other_len = rng.randint(1, d)
        if rng.random() < 0.5:
            len_a, len_b = d, other_len
        else:
            len_a, len_b = other_len, d
        long_len = max(len_a, len_b)
        short_len = min(len_a, len_b)

        chain_len = rng.randint(1, long_len)
        if chain_len == long_len:
            long_digits = '9' * long_len
        else:
            head = str(rng.randint(1, 9))
            mid = ''.join(str(rng.randint(0, 9)) for _ in range(long_len - chain_len - 1))
            long_digits = head + mid + '9' * chain_len
        long_val = int(long_digits)

        if short_len == 1:
            short_digits = str(rng.randint(1, 9))  # nonzero: triggers the carry
        else:
            head = str(rng.randint(1, 9))
            mid = ''.join(str(rng.randint(0, 9)) for _ in range(short_len - 2))
            tail = str(rng.randint(1, 9))  # nonzero units digit: triggers the carry
            short_digits = head + mid + tail
        short_val = int(short_digits)

        a, b = (long_val, short_val) if len_a == long_len else (short_val, long_val)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        examples.append((a, b))
    return examples


def sample_mixed(d, n_wanted, seen, rng, adversarial_frac):
    """n_wanted split between sample_carry_heavy and sample_bucket by
    adversarial_frac, sharing one `seen` set so the two never collide."""
    adv_wanted = round(n_wanted * adversarial_frac)
    nat_wanted = n_wanted - adv_wanted
    examples = sample_carry_heavy(d, adv_wanted, seen, rng)
    examples += sample_bucket(d, nat_wanted, seen, rng)
    return examples


def format_line(a, b):
    return f"{a}+{b}={a + b}\n"


def write_lines(path, pairs):
    with open(path, "w") as f:
        for a, b in pairs:
            f.write(format_line(a, b))


def main():
    rng = random.Random(SEED)

    train_pairs, val_pairs, indist_pairs = [], [], []
    gap_pairs, hint_pairs, ood_pairs = [], [], []
    carry_heavy_gap_pairs, carry_heavy_ood_pairs = [], []

    total_wanted = TRAIN_PER_BUCKET + VAL_PER_BUCKET + TEST_INDIST_PER_BUCKET
    for d in range(1, TRAIN_MAX_DIGITS + 1):
        seen = set()
        capacity = bucket_capacity(d)
        # scale each split's request proportionally when a bucket is too small
        # to satisfy all three splits at their default sizes (e.g. d=1 has only
        # 100 possible pairs), preserving the train:val:indist ratio instead of
        # letting an earlier split starve a later one.
        train_wanted = min(TRAIN_PER_BUCKET, round(capacity * TRAIN_PER_BUCKET / total_wanted))
        val_wanted = min(VAL_PER_BUCKET, round(capacity * VAL_PER_BUCKET / total_wanted))
        indist_wanted = min(TEST_INDIST_PER_BUCKET, round(capacity * TEST_INDIST_PER_BUCKET / total_wanted))
        train_pairs += [(d, a, b) for a, b in sample_mixed(d, train_wanted, seen, rng, ADVERSARIAL_FRAC)]
        val_pairs += [(d, a, b) for a, b in sample_mixed(d, val_wanted, seen, rng, ADVERSARIAL_FRAC)]
        # test_indist stays natural distribution -- comparable to every prior stage's eval number
        indist_pairs += [(d, a, b) for a, b in sample_bucket(d, indist_wanted, seen, rng)]

    for d in range(GAP_MIN_DIGITS, GAP_MAX_DIGITS + 1):
        seen = set()
        gap_pairs += [(d, a, b) for a, b in sample_bucket(d, TEST_GAP_PER_BUCKET, seen, rng)]
        carry_heavy_gap_pairs += [(d, a, b) for a, b in sample_carry_heavy(d, TEST_CARRY_HEAVY_GAP_PER_BUCKET, seen, rng)]

    for d in range(HINT_MIN_DIGITS, HINT_MAX_DIGITS + 1):
        seen = set()
        train_pairs += [(d, a, b) for a, b in sample_mixed(d, HINT_TRAIN_PER_BUCKET, seen, rng, ADVERSARIAL_FRAC)]
        val_pairs += [(d, a, b) for a, b in sample_mixed(d, HINT_VAL_PER_BUCKET, seen, rng, ADVERSARIAL_FRAC)]
        hint_pairs += [(d, a, b) for a, b in sample_bucket(d, TEST_HINT_PER_BUCKET, seen, rng)]

    for d in range(OOD_MIN_DIGITS, OOD_MAX_DIGITS + 1):
        seen = set()
        ood_pairs += [(d, a, b) for a, b in sample_bucket(d, TEST_OOD_PER_BUCKET, seen, rng)]
        carry_heavy_ood_pairs += [(d, a, b) for a, b in sample_carry_heavy(d, TEST_CARRY_HEAVY_OOD_PER_BUCKET, seen, rng)]

    far_ood_pairs = []
    for d in range(FAR_OOD_MIN_DIGITS, FAR_OOD_MAX_DIGITS + 1):
        seen = set()
        far_ood_pairs += [(d, a, b) for a, b in sample_bucket(d, TEST_OOD_PER_BUCKET, seen, rng)]

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    write_lines(os.path.join(OUT_DIR, "train.txt"), [(a, b) for _, a, b in train_pairs])
    write_lines(os.path.join(OUT_DIR, "val.txt"), [(a, b) for _, a, b in val_pairs])
    write_lines(os.path.join(OUT_DIR, "test_indist.txt"), [(a, b) for _, a, b in indist_pairs])
    write_lines(os.path.join(OUT_DIR, "test_gap.txt"), [(a, b) for _, a, b in gap_pairs])
    write_lines(os.path.join(OUT_DIR, "test_hint.txt"), [(a, b) for _, a, b in hint_pairs])
    write_lines(os.path.join(OUT_DIR, "test_ood.txt"), [(a, b) for _, a, b in ood_pairs])
    write_lines(os.path.join(OUT_DIR, "test_carry_heavy_gap.txt"), [(a, b) for _, a, b in carry_heavy_gap_pairs])
    write_lines(os.path.join(OUT_DIR, "test_carry_heavy_ood.txt"), [(a, b) for _, a, b in carry_heavy_ood_pairs])
    write_lines(os.path.join(OUT_DIR, "test_far_ood.txt"), [(a, b) for _, a, b in far_ood_pairs])

    def bucket_counts(pairs):
        counts = {}
        for d, _, _ in pairs:
            counts[d] = counts.get(d, 0) + 1
        return counts

    print(f"train:        {len(train_pairs):7d} examples  {bucket_counts(train_pairs)}")
    print(f"val:          {len(val_pairs):7d} examples  {bucket_counts(val_pairs)}")
    print(f"test_indist:  {len(indist_pairs):7d} examples  {bucket_counts(indist_pairs)}")
    print(f"test_gap:     {len(gap_pairs):7d} examples  {bucket_counts(gap_pairs)}")
    print(f"test_hint:    {len(hint_pairs):7d} examples  {bucket_counts(hint_pairs)}")
    print(f"test_ood:     {len(ood_pairs):7d} examples  {bucket_counts(ood_pairs)}")
    print(f"test_carry_heavy_gap: {len(carry_heavy_gap_pairs):7d} examples  {bucket_counts(carry_heavy_gap_pairs)}")
    print(f"test_carry_heavy_ood: {len(carry_heavy_ood_pairs):7d} examples  {bucket_counts(carry_heavy_ood_pairs)}")
    print(f"test_far_ood: {len(far_ood_pairs):7d} examples  {bucket_counts(far_ood_pairs)}")


if __name__ == "__main__":
    main()
