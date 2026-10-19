"""
Grounding diagnostic for the segment-embeddings ablation (mirrors the test
run earlier on RefineGPT v1 and PositionCoupledGPT): fix operand B, vary
operand A across several distinct values (all same digit length, in the OOD
range), and check whether the model's greedy output actually tracks the
value of A -- or whether it collapses to the same (wrong) output regardless
of A, which would indicate an operand-grounding failure rather than merely
an arithmetic-precision failure.

Usage:
    python eval_segments_grounding.py --out_dir=out-addition-segments --digits=20
"""
import os
import pickle
import random
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT

out_dir = 'out-addition-segments'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
digits = 20          # OOD range is 19-21
n_a_values = 6        # how many distinct operand-A values to try
seed = 0
exec(open('configurator.py').read())

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
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


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return ''.join(itos[i] for i in ids)


@torch.no_grad()
def greedy_decode_batch(prompts, max_new_tokens):
    idx = torch.tensor(prompts, dtype=torch.long, device=device)
    b_sz, plen = idx.size()
    plus_id, eq_id = stoi['+'], stoi['=']
    seg = torch.zeros((b_sz, plen), dtype=torch.long, device=device)
    for i in range(b_sz):
        row = prompts[i]
        plus_pos = row.index(plus_id)
        eq_pos = row.index(eq_id)
        seg[i, :plus_pos] = 1
        seg[i, plus_pos:eq_pos] = 2
        seg[i, eq_pos:] = 3
    for _ in range(max_new_tokens):
        with ctx:
            logits, _ = model(idx, segment_ids=seg)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        seg = torch.cat((seg, torch.full((b_sz, 1), 3, dtype=torch.long, device=device)), dim=1)
    return idx[:, plen:].tolist()


rng = random.Random(seed)
lo, hi = 10 ** (digits - 1), 10 ** digits - 1
b_fixed = rng.randint(lo, hi)
a_values = [rng.randint(lo, hi) for _ in range(n_a_values)]

prompts = [encode(f"\n{a}+{b_fixed}=") for a in a_values]
assert len(set(len(p) for p in prompts)) == 1
max_new_tokens = digits + 3
gen = greedy_decode_batch(prompts, max_new_tokens)

print(f"digits={digits}  b_fixed={b_fixed}\n")
preds = []
for a, g in zip(a_values, gen):
    text = decode(g)
    pred_str = text.split('\n')[0]
    true_c = a + b_fixed
    preds.append(pred_str)
    print(f"a={a}")
    print(f"  true = {true_c}")
    print(f"  pred = {pred_str}")
    print(f"  correct: {str(true_c) == pred_str}")
    print()

n_unique = len(set(preds))
print(f"unique predictions across {len(preds)} distinct operand-A values: {n_unique}")
if n_unique == 1:
    print("=> UNGROUNDED: prediction is identical regardless of operand A")
else:
    print("=> GROUNDED (at least partially): prediction changes with operand A")
