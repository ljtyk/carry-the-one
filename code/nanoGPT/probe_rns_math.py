"""
Validates the RNS math is exactly correct BEFORE trusting a full training
run: random large integers -> digit sequence -> encode_operand's residue
formula (replicated here) -> CRT reconstruction -> must recover the exact
original integer, every time, for every modulus set size.
"""
import random
import torch

MODULI_20 = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)  # covers 10^21


def place_weights(moduli, max_places):
    K = len(moduli)
    w = torch.zeros(K, max_places, dtype=torch.long)
    for k, m in enumerate(moduli):
        cur = 1
        for p in range(max_places):
            w[k, p] = cur
            cur = (cur * 10) % m
    return w


def digit_values(v, width):
    return [int(ch) for ch in str(v).zfill(width)]


def residues_from_digits(digit_vals, moduli, weights):
    L = len(digit_vals)
    places = list(range(L - 1, -1, -1))
    d = torch.tensor(digit_vals, dtype=torch.long)
    w = weights[:, places]  # (K, L)
    m = torch.tensor(moduli, dtype=torch.long)
    return (torch.einsum('l,kl->k', d, w) % m).tolist()


def crt_reconstruct(residues, moduli):
    x, M = 0, 1
    for r, m in zip(residues, moduli):
        t = ((r - x) * pow(M, -1, m)) % m
        x = x + M * t
        M *= m
    return x


def main():
    moduli = MODULI_20
    prod = 1
    for m in moduli:
        prod *= m
    print(f"moduli: {moduli}")
    print(f"product: {prod} (~10^{len(str(prod))-1})")

    max_places = 25
    weights = place_weights(moduli, max_places)

    max_valid_width = len(str(prod)) - 1  # largest width guaranteed fully < product
    random.seed(0)
    n_trials = 5000
    failures = []
    for _ in range(n_trials):
        width = random.randint(1, max_valid_width)
        v = random.randint(0, 10**width - 1)
        digits = digit_values(v, width)
        residues = residues_from_digits(digits, moduli, weights)
        # sanity: residues must match direct Python mod on the true value
        expected_residues = [v % m for m in moduli]
        if residues != expected_residues:
            failures.append(('residue_mismatch', v, residues, expected_residues))
            continue
        recovered = crt_reconstruct(residues, moduli)
        if recovered != v:
            failures.append(('crt_mismatch', v, recovered))

    print(f"\n{n_trials} trials, {len(failures)} failures")
    for f in failures[:10]:
        print(' ', f)
    print("ALL PASS" if not failures else "FAILURES FOUND -- do not proceed")

    # also check a value right at the capacity boundary
    v = prod - 1
    digits = digit_values(v, len(str(v)))
    residues = residues_from_digits(digits, moduli, weights)
    recovered = crt_reconstruct(residues, moduli)
    print(f"\nboundary check: v=product-1, recovered == v: {recovered == v}")


if __name__ == '__main__':
    main()
