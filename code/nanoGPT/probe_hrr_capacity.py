"""
Standalone (no training, no model) test of HRR circular-convolution binding
capacity at C=384, matching n_embd. Uses RANDOM digit filler vectors and
constructed-unitary place role vectors -- the graceful-degradation property
is a statistical/mathematical fact about this scheme, not something that
needs to be learned, so this can be checked directly without training
anything.

For increasing L (number of bound (digit,place) pairs superposed into one
vector), measures: (a) per-digit retrieval accuracy at a random target
place, and (b) whole-sequence accuracy (ALL L digits correctly recovered)
-- to directly check the "per-digit accuracy compounds badly across many
positions" concern.
"""
import random

import torch

C = 384
N_DIGITS = 10
N_FREQ = C // 2 + 1

torch.manual_seed(0)
random.seed(0)

# random digit filler vectors, i.i.d. Gaussian
fillers = torch.randn(N_DIGITS, C)

# fixed random base frequency (angle) per rfft bin -- role_vector(p)'s phase
# at bin k is p * base_freq[k], magnitude always 1 (unitary)
base_freq = torch.rand(N_FREQ) * 2 * torch.pi


def role_vector(p):
    phase = base_freq * p
    spectrum = torch.polar(torch.ones(N_FREQ), phase)  # magnitude 1 everywhere
    r = torch.fft.irfft(spectrum, n=C)
    return r / r.norm() * (C ** 0.5)  # normalize to ~sqrt(C), matching a random Gaussian vector's norm


def bind(f, r):
    return torch.fft.irfft(torch.fft.rfft(f) * torch.fft.rfft(r), n=C)


def unbind(vec, r):
    return torch.fft.irfft(torch.fft.rfft(vec) * torch.conj(torch.fft.rfft(r)), n=C)


def run(L, n_trials=200):
    per_digit_correct = 0
    per_digit_total = 0
    whole_correct = 0
    for _ in range(n_trials):
        digits = [random.randint(0, 9) for _ in range(L)]
        roles = [role_vector(p) for p in range(L)]
        vec = sum(bind(fillers[d], r) for d, r in zip(digits, roles))

        all_right = True
        for p in range(L):
            recovered = unbind(vec, roles[p])
            sims = fillers @ recovered
            pred = torch.argmax(sims).item()
            correct = pred == digits[p]
            per_digit_correct += int(correct)
            per_digit_total += 1
            all_right = all_right and correct
        whole_correct += int(all_right)
    return per_digit_correct / per_digit_total, whole_correct / n_trials


print(f"C={C} dims, {N_DIGITS} digit classes, random fillers + unitary role vectors, no training\n")
print(f"{'L':>4} {'per-digit acc':>14} {'whole-seq acc':>14}")
for L in [1, 5, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200]:
    per_digit_acc, whole_acc = run(L, n_trials=100 if L <= 50 else 50)
    print(f"{L:>4} {per_digit_acc:>14.3f} {whole_acc:>14.3f}")
