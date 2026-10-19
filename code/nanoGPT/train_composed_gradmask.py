"""
Concurrent gradient-masked ComposedGPT training -- a cleaner version of
train_composed_twophase.py's contamination test that removes the "combiner
has to learn from cold against an already-frozen, unfamiliar
representation" confound.

Instead of two SEQUENTIAL phases (train encoder/decoder to completion,
freeze, then train the combiner from scratch against the now-static
result), this interleaves both updates on EVERY iteration, using the same
paired (a, b) addition-problem batch throughout:

  1. Encoder + decoder (+ shared token embedding) are updated ONLY by the
     reconstruction loss (decode(encode(a))==a, decode(encode(b))==b) --
     they are trained "on the problem" (real operands from real addition
     examples) but never see the answer; no gradient from the sum loss
     ever reaches them.
  2. The combiner is updated ONLY by the sum loss, using vec_a/vec_b
     computed FRESH from the current (just-updated) encoder every step,
     but DETACHED before combine() so no gradient flows back into the
     encoder. It never sees raw digit tokens, only the encoded vectors.

Because both updates happen concurrently, step by step, the combiner is
always chasing the encoder's CURRENT representation rather than meeting an
already-fixed, alien one cold -- isolating "does the encoder need
addition-relevant contamination to support a good combiner" from the
cold-start optimization confound in the sequential two-phase version.

Usage:
    python train_composed_gradmask.py --out_dir=out-addition-composed-gradmask
"""
import os
import math
import random
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, ComposedGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-composed-gradmask'
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

iters = 30000
log_interval = 500

learning_rate = 1e-3
min_lr = 1e-4
weight_decay = 0.3
beta1, beta2 = 0.9, 0.99
warmup_iters = 100

seed = 1337
warm_start_ckpt = ''  # optional: path to a ComposedGPT checkpoint (.pt) to initialize weights from before training -- e.g. out-addition-composed-twophase/ckpt.pt, whose encoder/decoder are ALREADY a pure phase-1-only (reconstruction-only) result (phase 2 never touched them, since it freezes everything but the combiner), so warm-starting from it skips re-deriving that from scratch and only needs to correct the combiner's cold-start training
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def make_batch_pair():
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
    nl_col = torch.full((batch_size, 1), nl_id, dtype=torch.long, device=device)
    a_targets = torch.cat([a_ids, nl_col], dim=1)
    b_targets = torch.cat([b_ids, nl_col], dim=1)
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
    return a_ids, b_ids, a_targets, b_targets, dec_x, dec_y


gptconf = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head,
                     n_embd=n_embd, dropout=dropout, bias=True, pos_encoding='rope',
                     enc_layers=enc_layers, combiner_layers=combiner_layers)
model = ComposedGPT(gptconf).to(device)
if warm_start_ckpt:
    print(f"warm-starting from {warm_start_ckpt}")
    src = torch.load(warm_start_ckpt, map_location=device)
    src_sd = src['model']
    for k in list(src_sd.keys()):
        if k.startswith('_orig_mod.'):
            src_sd[k[len('_orig_mod.'):]] = src_sd.pop(k)
    missing, unexpected = model.load_state_dict(src_sd, strict=False)
    print(f"warm start: {len(missing)} missing, {len(unexpected)} unexpected: {missing} {unexpected}")

encdec_params = [p for n, p in model.named_parameters() if not n.startswith('combiner.') and not n.startswith('comb_ln.')]
decay1 = [p for p in encdec_params if p.dim() >= 2]
nodecay1 = [p for p in encdec_params if p.dim() < 2]
opt_encdec = torch.optim.AdamW([{'params': decay1, 'weight_decay': weight_decay},
                                  {'params': nodecay1, 'weight_decay': 0.0}],
                                 lr=learning_rate, betas=(beta1, beta2))

combiner_params = list(model.combiner.parameters()) + list(model.comb_ln.parameters())
decay2 = [p for p in combiner_params if p.dim() >= 2]
nodecay2 = [p for p in combiner_params if p.dim() < 2]
opt_combiner = torch.optim.AdamW([{'params': decay2, 'weight_decay': weight_decay},
                                    {'params': nodecay2, 'weight_decay': 0.0}],
                                   lr=learning_rate, betas=(beta1, beta2))

print("=" * 60)
print("Concurrent gradient-masked training: encoder/decoder <- recon only, combiner <- sum only")
print("=" * 60)

for it in range(iters + 1):
    lr = get_lr(it)
    for g in opt_encdec.param_groups:
        g['lr'] = lr
    for g in opt_combiner.param_groups:
        g['lr'] = lr

    a_ids, b_ids, a_targets, b_targets, dec_x, dec_y = make_batch_pair()

    # --- step 1: encoder + decoder, reconstruction loss only ---
    with ctx:
        vec_a = model.encode_operand(a_ids)
        vec_b = model.encode_operand(b_ids)
        _, loss_a = model.decode_from_vector(vec_a, a_ids, a_targets)
        _, loss_b = model.decode_from_vector(vec_b, b_ids, b_targets)
        recon_loss = loss_a + loss_b
    model.zero_grad(set_to_none=True)
    recon_loss.backward()
    opt_encdec.step()
    model.zero_grad(set_to_none=True)

    # --- step 2: combiner, sum loss only, encoder re-run fresh but DETACHED ---
    with ctx:
        with torch.no_grad():
            vec_a2 = model.encode_operand(a_ids)
            vec_b2 = model.encode_operand(b_ids)
        vec_c = model.combine(vec_a2, vec_b2, plus_id, eq_id)
        _, sum_loss = model.decode_from_vector(vec_c, dec_x, dec_y)
    model.zero_grad(set_to_none=True)
    sum_loss.backward()
    opt_combiner.step()
    model.zero_grad(set_to_none=True)

    if it % log_interval == 0:
        print(f"iter {it}: recon_loss {recon_loss.item():.4f}  sum_loss {sum_loss.item():.4f}")

checkpoint = {
    'model': model.state_dict(),
    'model_args': dict(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head,
                        n_embd=n_embd, dropout=dropout, bias=True, pos_encoding='rope',
                        enc_layers=enc_layers, combiner_layers=combiner_layers, composed_recon_weight=1.0),
}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
print(f"saved gradient-masked checkpoint to {os.path.join(out_dir, 'ckpt.pt')}")
