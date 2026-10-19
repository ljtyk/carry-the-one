"""
Isolates encoder information-loss from decoder generalization failure.

Fits an INDEPENDENT linear probe per output position (distance-from-end k)
on frozen vec_a's from ONLY in-distribution (<=10 digit) operands -- same
data the real decoder (extract_fn) was trained on -- then tests that probe
zero-shot on L=11 (OOD) vec_a's, exactly the same generalization challenge
the real decoder faces. Compares the probe's OOD accuracy against the
model's own decode_from_vector's OOD accuracy at the same position.

If the probe reads L=11 digits meaningfully better than the model's own
decoder: the information IS present in vec_a, and the bottleneck is decoder
generalization (extract_fn failing to generalize to an unfamiliar-shaped
input), not encoder information-loss.

If the probe does about as badly as the model's own decoder: the encoder
itself destroyed the information, and no readout mechanism could recover
it -- a genuine encoder-side failure.
"""
import random
import sys

import torch
import torch.nn.functional as F

from model import GPTConfig, AdditiveComposedGPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load(f'{out_dir}/ckpt.pt', map_location=device)
gptconf = GPTConfig(**ckpt['model_args'])
model = AdditiveComposedGPT(gptconf)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)
print(f"checkpoint: {out_dir}  iter={ckpt.get('iter_num', '?')}\n")

random.seed(7)
torch.manual_seed(7)


def digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


@torch.no_grad()
def sample_vec_and_digit(L, k):
    v = random.randint(*digit_range(L))
    s = str(v).zfill(L)
    digit = int(s[L - 1 - k])
    ids = torch.tensor([[int(ch) for ch in s]], dtype=torch.long, device=device)
    vec_a = model.encode_operand(ids).squeeze(0)
    return vec_a, digit


def gather(k, lengths, n):
    X, Y = [], []
    for _ in range(n):
        L = random.choice(lengths)
        vec, digit = sample_vec_and_digit(L, k)
        X.append(vec)
        Y.append(digit)
    return torch.stack(X), torch.tensor(Y, dtype=torch.long, device=device)


def train_probe(X, Y, n_epochs=400, lr=0.05):
    probe = torch.nn.Linear(X.size(1), 10).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(X), Y)
        loss.backward()
        opt.step()
    return probe


def eval_probe(probe, X, Y):
    with torch.no_grad():
        return (probe(X).argmax(dim=-1) == Y).float().mean().item()


@torch.no_grad()
def model_decode_accuracy(L, k, n=300):
    P = model.config.max_answer_digits
    correct = 0
    for _ in range(n):
        v = random.randint(*digit_range(L))
        s = str(v).zfill(L)
        digit = int(s[L - 1 - k])
        ids = torch.tensor([[int(ch) for ch in s]], dtype=torch.long, device=device)
        vec_a = model.encode_operand(ids)
        logits, _ = model.decode_from_vector(vec_a)
        pred = torch.argmax(logits[0, P - 1 - k], dim=-1).item()
        correct += int(pred == digit)
    return correct / n


print(f"{'k':>3} {'probe in-dist':>14} {'probe OOD(L=11)':>16} {'model OOD(L=11)':>16}")
for k in [0, 3, 6, 8, 9]:
    train_lengths = list(range(k + 1, 11))
    X_train, Y_train = gather(k, train_lengths, n=3000)
    probe = train_probe(X_train, Y_train)

    X_id, Y_id = gather(k, train_lengths, n=500)
    id_acc = eval_probe(probe, X_id, Y_id)

    X_ood, Y_ood = gather(k, [11], n=500)
    ood_acc = eval_probe(probe, X_ood, Y_ood)

    model_ood_acc = model_decode_accuracy(11, k, n=300)

    print(f"{k:>3} {id_acc:>14.3f} {ood_acc:>16.3f} {model_ood_acc:>16.3f}")
