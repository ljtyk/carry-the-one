"""
Two-phase ComposedGPT training, to test whether the encoder/decoder are
secretly doing some of the addition work themselves (contaminated by the
sum-loss gradient during ordinary joint training) rather than addition
being cleanly confined to the combiner:

Phase 1: train ONLY the operand encoder + decoder (+ shared token
embedding), on reconstruction alone (decode(encode(x)) == x for a single
random number x). The combiner and the sum task are never touched here --
this is provably a "pure" representation, shaped by nothing but round-trip
fidelity.

Phase 2: FREEZE everything from phase 1 (requires_grad=False). Train ONLY
the combiner (+ its LayerNorm) to predict a+b from the frozen vec_a/vec_b,
backpropagating the sum loss through the (frozen, but still differentiable)
decoder into the combiner's own weights.

If final sum accuracy holds up close to the jointly-trained baseline
(~62% in-distribution), the encoder/decoder really were neutral and the
combiner alone can learn to add. If it collapses, the joint run's encoder/
decoder had picked up addition-relevant structure beyond pure
reconstruction -- see project discussion.

Usage:
    python train_composed_twophase.py --out_dir=out-addition-composed-twophase
"""
import os
import math
import random

import numpy as np
import torch

from model import GPTConfig, ComposedGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-composed-twophase'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

n_layer = 4
enc_layers = 2
combiner_layers = 1
n_head = 6
n_embd = 384
dropout = 0.0
block_size = 40
batch_size = 64

train_max_digits = 10
adversarial_frac = 0.3

phase1_iters = 20000
phase2_iters = 15000
log_interval = 500
eval_interval = 2000

learning_rate = 1e-3
min_lr = 1e-4
weight_decay = 0.3
beta1, beta2 = 0.9, 0.99
warmup_iters = 100

seed = 1337
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.no_grad().__class__()
from contextlib import nullcontext
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

import pickle
with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']
vocab_size = meta['vocab_size'] if 'vocab_size' in meta else len(stoi)
digit_ids = [stoi[str(i)] for i in range(10)]
nl_id = stoi['\n']
plus_id, eq_id = stoi['+'], stoi['=']


def digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


def sample_pair(len_a, len_b):
    lo_a, hi_a = digit_range(len_a)
    lo_b, hi_b = digit_range(len_b)
    return random.randint(lo_a, hi_a), random.randint(lo_b, hi_b)


def sample_carry_heavy_pair(len_a, len_b):
    long_len, short_len = max(len_a, len_b), min(len_a, len_b)
    chain_len = random.randint(1, long_len)
    if chain_len == long_len:
        long_digits = '9' * long_len
    else:
        head = str(random.randint(1, 9))
        mid = ''.join(str(random.randint(0, 9)) for _ in range(long_len - chain_len - 1))
        long_digits = head + mid + '9' * chain_len
    long_val = int(long_digits)
    if short_len == 1:
        short_digits = str(random.randint(1, 9))
    else:
        head = str(random.randint(1, 9))
        mid = ''.join(str(random.randint(0, 9)) for _ in range(short_len - 2))
        tail = str(random.randint(1, 9))
        short_digits = head + mid + tail
    short_val = int(short_digits)
    return (long_val, short_val) if len_a == long_len else (short_val, long_val)


def to_ids(v, width):
    return [digit_ids[int(ch)] for ch in str(v).zfill(width)]


def get_lr(it, iters):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def make_batch_single():
    """Phase 1: one random number per batch (fixed width), for reconstruction."""
    d = random.randint(1, train_max_digits)
    lo, hi = digit_range(d)
    vals = [random.randint(lo, hi) for _ in range(batch_size)]
    ids = torch.tensor([to_ids(v, d) for v in vals], dtype=torch.long, device=device)
    nl_col = torch.full((batch_size, 1), nl_id, dtype=torch.long, device=device)
    targets = torch.cat([ids, nl_col], dim=1)
    return ids, targets


def make_batch_pair():
    """Phase 2: one (len_a, len_b) pair per batch, for the sum task."""
    d = random.randint(1, train_max_digits)
    other_len = random.randint(1, d)
    len_a, len_b = (d, other_len) if random.random() < 0.5 else (other_len, d)
    adversarial = random.random() < adversarial_frac
    a_vals, b_vals = [], []
    for _ in range(batch_size):
        a, b = sample_carry_heavy_pair(len_a, len_b) if adversarial else sample_pair(len_a, len_b)
        a_vals.append(a)
        b_vals.append(b)
    a_ids = torch.tensor([to_ids(a, len_a) for a in a_vals], dtype=torch.long, device=device)
    b_ids = torch.tensor([to_ids(b, len_b) for b in b_vals], dtype=torch.long, device=device)
    c_strs = [str(a + b) for a, b in zip(a_vals, b_vals)]
    max_L = max(len(s) for s in c_strs)
    dec_x = torch.full((batch_size, max_L), digit_ids[0], dtype=torch.long, device=device)
    dec_y = torch.full((batch_size, max_L + 1), -1, dtype=torch.long, device=device)
    for i, s in enumerate(c_strs):
        ds = torch.tensor([digit_ids[int(ch)] for ch in s], dtype=torch.long, device=device)
        L = ds.size(0)
        dec_x[i, :L] = ds
        dec_y[i, :L] = ds
        dec_y[i, L] = nl_id
    return a_ids, b_ids, dec_x, dec_y


gptconf = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head,
                     n_embd=n_embd, dropout=dropout, bias=True, pos_encoding='rope',
                     enc_layers=enc_layers, combiner_layers=combiner_layers)
model = ComposedGPT(gptconf).to(device)

# ============================== PHASE 1 ==============================
print("=" * 60)
print("PHASE 1: reconstruction-only pretraining (encoder + decoder)")
print("=" * 60)
phase1_params = [p for n, p in model.named_parameters() if not n.startswith('combiner.') and not n.startswith('comb_ln.')]
decay = [p for p in phase1_params if p.dim() >= 2]
nodecay = [p for p in phase1_params if p.dim() < 2]
opt1 = torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay},
                           {'params': nodecay, 'weight_decay': 0.0}],
                          lr=learning_rate, betas=(beta1, beta2))

for it in range(phase1_iters + 1):
    lr = get_lr(it, phase1_iters)
    for g in opt1.param_groups:
        g['lr'] = lr
    ids, targets = make_batch_single()
    with ctx:
        vec = model.encode_operand(ids)
        _, loss = model.decode_from_vector(vec, ids, targets)
    opt1.zero_grad(set_to_none=True)
    loss.backward()
    opt1.step()
    if it % log_interval == 0:
        print(f"phase1 iter {it}: recon_loss {loss.item():.4f}")

# ============================== PHASE 2 ==============================
print("=" * 60)
print("PHASE 2: combiner-only training (encoder + decoder FROZEN)")
print("=" * 60)
for p in model.parameters():
    p.requires_grad = False
for p in model.combiner.parameters():
    p.requires_grad = True
for p in model.comb_ln.parameters():
    p.requires_grad = True

combiner_params = list(model.combiner.parameters()) + list(model.comb_ln.parameters())
decay2 = [p for p in combiner_params if p.dim() >= 2]
nodecay2 = [p for p in combiner_params if p.dim() < 2]
opt2 = torch.optim.AdamW([{'params': decay2, 'weight_decay': weight_decay},
                           {'params': nodecay2, 'weight_decay': 0.0}],
                          lr=learning_rate, betas=(beta1, beta2))

for it in range(phase2_iters + 1):
    lr = get_lr(it, phase2_iters)
    for g in opt2.param_groups:
        g['lr'] = lr
    a_ids, b_ids, dec_x, dec_y = make_batch_pair()
    with ctx:
        with torch.no_grad():
            vec_a = model.encode_operand(a_ids)
            vec_b = model.encode_operand(b_ids)
        vec_c = model.combine(vec_a, vec_b, plus_id, eq_id)
        _, loss = model.decode_from_vector(vec_c, dec_x, dec_y)
    opt2.zero_grad(set_to_none=True)
    loss.backward()
    opt2.step()
    if it % log_interval == 0:
        print(f"phase2 iter {it}: sum_loss {loss.item():.4f}")

# unfreeze before saving (state_dict doesn't care, but keeps the checkpoint's
# model consistent with a normally-trained one if reloaded for further training)
for p in model.parameters():
    p.requires_grad = True

checkpoint = {
    'model': model.state_dict(),
    'model_args': dict(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head,
                        n_embd=n_embd, dropout=dropout, bias=True, pos_encoding='rope',
                        enc_layers=enc_layers, combiner_layers=combiner_layers, composed_recon_weight=1.0),
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
print(f"saved two-phase checkpoint to {os.path.join(out_dir, 'ckpt.pt')}")
