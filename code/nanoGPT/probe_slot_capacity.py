"""
Standalone (no training, no model) test of a DISJOINT-SLOT digit codec at
C=384, matching n_embd, as an alternative to HRR circular-convolution
binding. Instead of superposing every (digit, place) pair into the SAME
384 dims (crosstalk grows with L), each place gets its own exclusive
3-dim slice of the vector -- place p owns dims [3p : 3p+3], nobody else
ever writes there. A digit's value is one of 10 fixed directions in that
3-dim slice, the same 10 directions reused at every place.

Mirrors probe_hrr_capacity.py's methodology exactly (random, UNTRAINED
digit directions -- zero interference is a structural/geometric fact of
disjoint slices, not something that needs learning, so it can be checked
directly) for a fair side-by-side comparison. Key difference in what we
expect: HRR predicts graceful decay starting well before its dim; this
scheme should stay flat near-perfect right up to dim/3 places, then hit
a hard wall (no more slots to allocate) rather than degrading gracefully.
"""
import random

import torch

C = 384
N_DIGITS = 10
DIMS_PER_PLACE = 3
MAX_PLACES = C // DIMS_PER_PLACE  # 128 -- hard capacity, not a soft limit

torch.manual_seed(0)
random.seed(0)

# random UNTRAINED digit direction vectors in the small per-place subspace,
# i.i.d. Gaussian then normalized -- same "no training" spirit as
# probe_hrr_capacity.py's random fillers. Reused identically at every place.
digit_dirs = torch.randn(N_DIGITS, DIMS_PER_PLACE)
digit_dirs = digit_dirs / digit_dirs.norm(dim=-1, keepdim=True)


def encode(digits):
    """digits: list of ints (0-9), digits[p] is the value at place p.
    Returns (C,): disjoint slot concatenation, zero-filled past len(digits)
    or past MAX_PLACES (places beyond MAX_PLACES simply have nowhere to go
    -- unlike HRR there is no fallback superposition, this is the hard
    capacity boundary by construction)."""
    vec = torch.zeros(C)
    for p, d in enumerate(digits):
        if p >= MAX_PLACES:
            break
        vec[DIMS_PER_PLACE * p: DIMS_PER_PLACE * (p + 1)] = digit_dirs[d]
    return vec


def decode_place(vec, p):
    """Nearest-direction classification within place p's own dedicated
    slice -- no unbind, no FFT, literally just indexing. p >= MAX_PLACES
    has no slot at all (was never written in encode()) -- returns None to
    mark it structurally unrepresentable, not just "wrong"."""
    if p >= MAX_PLACES:
        return None
    slot = vec[DIMS_PER_PLACE * p: DIMS_PER_PLACE * (p + 1)]
    sims = digit_dirs @ slot
    return torch.argmax(sims).item()


def run(L, n_trials=200):
    """Scores ALL L places, including any past MAX_PLACES -- those are
    counted as wrong (there is no slot to hold their digit at all, so
    scoring them as correct or simply excluding them would hide the real
    hard-cliff behavior once L exceeds capacity)."""
    per_digit_correct = 0
    per_digit_total = 0
    whole_correct = 0
    for _ in range(n_trials):
        digits = [random.randint(0, 9) for _ in range(L)]
        vec = encode(digits)

        all_right = True
        for p in range(L):
            pred = decode_place(vec, p)
            correct = pred == digits[p]  # None == int is always False -- correctly counts as wrong
            per_digit_correct += int(correct)
            per_digit_total += 1
            all_right = all_right and correct
        whole_correct += int(all_right)
    return per_digit_correct / per_digit_total, whole_correct / n_trials


print(f"C={C} dims, {N_DIGITS} digit classes, {DIMS_PER_PLACE} dims/place "
      f"-> hard capacity = {MAX_PLACES} places, random UNTRAINED directions, no training\n")
print(f"{'L':>4} {'per-digit acc':>14} {'whole-seq acc':>14}  note")
for L in [1, 5, 10, 15, 20, 30, 40, 50, 70, 100, 120, 125, 127, 128, 130, 140, 150, 200]:
    per_digit_acc, whole_acc = run(L, n_trials=200 if L <= 128 else 50)
    note = "" if L <= MAX_PLACES else f"<- L > {MAX_PLACES}, some places have no slot at all (hard cliff expected)"
    print(f"{L:>4} {per_digit_acc:>14.3f} {whole_acc:>14.3f}  {note}")
