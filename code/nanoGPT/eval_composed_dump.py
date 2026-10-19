"""
Dump concrete wrong (and a few right, for contrast) sum predictions from
ComposedGPT, alongside the reconstruction predictions for the same operands
-- lets us see qualitatively HOW it's wrong (off-by-a-digit? right digits,
wrong order? unrelated to the operands?) rather than just the aggregate
accuracy number.

Usage:
    python eval_composed_dump.py --out_dir=out-addition-composed --digits=8 --n=15
"""
import os
import pickle
import random
from contextlib import nullcontext

import torch

from model import GPTConfig, ComposedGPT

out_dir = 'out-addition-composed'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
encoder_position_scheme = ''  # override for checkpoints predating this flag (saved model_args lack it) -- 'sequential' or 'place_value'
digits = 8       # len_a == len_b == digits, for a clean same-length sample
n = 15           # how many examples to sample
seed = 0
exec(open('configurator.py').read())

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
if encoder_position_scheme:
    model_args['encoder_position_scheme'] = encoder_position_scheme
gptconf = GPTConfig(**model_args)
model = ComposedGPT(gptconf)
state_dict = checkpoint['model']
for k, v in list(state_dict.items()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
digit_ids = [stoi[str(i)] for i in range(10)]
nl_id = stoi['\n']
plus_id, eq_id = stoi['+'], stoi['=']


def to_ids(v, width):
    return [digit_ids[int(ch)] for ch in str(v).zfill(width)]


@torch.no_grad()
def greedy_decode_from_vector(memory_vec, max_new_tokens):
    B = memory_vec.size(0)
    dec_x = torch.zeros((B, 0), dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        with ctx:
            logits, _ = model.decode_from_vector(memory_vec, dec_x)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        dec_x = torch.cat([dec_x, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(1) == nl_id)
        if finished.all():
            break
    return dec_x


def decode_str(row):
    out = []
    for t in row:
        if t == nl_id:
            break
        out.append(itos[t])
    return ''.join(out)


rng = random.Random(seed)
lo, hi = 10 ** (digits - 1), 10 ** digits - 1
a_vals = [rng.randint(lo, hi) for _ in range(n)]
b_vals = [rng.randint(lo, hi) for _ in range(n)]

a_ids = torch.tensor([to_ids(a, digits) for a in a_vals], dtype=torch.long, device=device)
b_ids = torch.tensor([to_ids(b, digits) for b in b_vals], dtype=torch.long, device=device)
with ctx:
    vec_a = model.encode_operand(a_ids)
    vec_b = model.encode_operand(b_ids)
    vec_c = model.combine(vec_a, vec_b, plus_id, eq_id)

gen_a = greedy_decode_from_vector(vec_a, digits + 3)
gen_b = greedy_decode_from_vector(vec_b, digits + 3)
gen_c = greedy_decode_from_vector(vec_c, digits + 4)

n_wrong = 0
for i in range(n):
    a, b = a_vals[i], b_vals[i]
    true_c = a + b
    pred_a = decode_str(gen_a[i].tolist())
    pred_b = decode_str(gen_b[i].tolist())
    pred_c = decode_str(gen_c[i].tolist())
    correct = pred_c == str(true_c)
    n_wrong += not correct
    tag = "OK  " if correct else "WRONG"
    print(f"[{tag}] a={a}  b={b}")
    print(f"        true sum = {true_c}")
    print(f"        pred sum = {pred_c}")
    if pred_a != str(a) or pred_b != str(b):
        print(f"        (recon_a={pred_a!r} recon_b={pred_b!r} -- operand itself mis-encoded)")
    print()

print(f"{n_wrong}/{n} wrong at {digits} digits")
