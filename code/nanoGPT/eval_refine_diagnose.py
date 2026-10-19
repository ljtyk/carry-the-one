"""
Diagnose which (digit count, window position, carry-heaviness) combinations
still produce high loss on the current RefineGPT checkpoint. Replicates
get_batch_refine's construction (train.py) but computes PER-EXAMPLE loss
(not batch-mean) and tags each example with its digit count, window start t,
and carry count, so we can see exactly what's still hard.

Usage:
    python eval_refine_diagnose.py --out_dir=out-addition-refine --ckpt_name=ckpt.pt
"""
import os
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from model import GPTConfig, RefineGPT

out_dir = 'out-addition-refine'
dataset = 'addition_refine'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
refine_window = 3
n_samples_per_bucket = 40  # how many (digit, t) combos to probe per prompt-length bucket
exec(open('configurator.py').read())

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = RefineGPT(gptconf)
state_dict = checkpoint['model']
for k, v in list(state_dict.items()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
pad_id, mask_id, max_len = meta['pad_id'], meta['mask_id'], meta['max_len']
equals_id = meta['stoi']['=']

split = 'val'
bin_path = os.path.join(data_dir, f'{split}.bin')
data = np.memmap(bin_path, dtype=np.uint16, mode='r').reshape(-1, max_len)
lengths_path = os.path.join(data_dir, f'{split}_lengths.bin')
lengths_data = np.memmap(lengths_path, dtype=np.int32, mode='r')
prompt_lens = (data == equals_id).argmax(axis=1) + 1


def count_carries(a, b):
    carries, carry = 0, 0
    while a > 0 or b > 0:
        da, db = a % 10, b % 10
        s = da + db + carry
        carry = 1 if s >= 10 else 0
        carries += carry
        a //= 10
        b //= 10
    return carries


def row_to_ab(row_ids, plen):
    itos = meta['itos']
    text = ''.join(itos[i] for i in row_ids[:plen].tolist())
    a_str, b_str = text.strip('\n')[:-1].split('+')
    return int(a_str), int(b_str)


unique_plens = sorted(set(prompt_lens.tolist()))
records = []

with torch.no_grad():
    for plen in unique_plens:
        row_idx = np.nonzero(prompt_lens == plen)[0]
        if len(row_idx) == 0:
            continue
        sample_idx = np.random.choice(row_idx, size=min(n_samples_per_bucket, len(row_idx)), replace=False)
        rows = torch.from_numpy(data[sample_idx].astype(np.int64))
        real_lens = lengths_data[sample_idx]
        answer_span = max_len - plen
        max_t = max(0, int(real_lens.min()) - plen - 1)
        digits = max(len(str(row_to_ab(rows[i], plen)[0])) for i in range(len(rows)))

        for t in range(0, max_t + 1, max(1, (max_t + 1) // 6)):  # sample ~6 spread-out window positions
            win_start, win_end = plen + t, min(plen + t + refine_window, max_len)
            x = rows.clone()
            y = torch.full((len(rows), max_len), -1, dtype=torch.int64)
            y[:, win_start:win_end] = rows[:, win_start:win_end]
            y[y == pad_id] = -1
            x[:, win_start:] = mask_id
            x, y = x.to(device), y.to(device)
            with ctx:
                logits, _ = model(x)
            per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1, reduction='none')
            per_tok = per_tok.view(len(rows), max_len)
            valid = y != -1
            for i in range(len(rows)):
                if valid[i].any():
                    ex_loss = per_tok[i][valid[i]].mean().item()
                    a, b = row_to_ab(rows[i], plen)
                    carries = count_carries(a, b)
                    records.append({'digits': digits, 't': t, 'max_t': max_t, 'loss': ex_loss, 'carries': carries})

print(f"{len(records)} (example, window) probes collected\n")

# bucket by digit count
by_digits = {}
for r in records:
    by_digits.setdefault(r['digits'], []).append(r['loss'])
print("mean loss by digit count:")
for d, losses in sorted(by_digits.items()):
    print(f"  {d:2d} digits: {np.mean(losses):.3f} (n={len(losses)})")

# bucket by relative window position (early/mid/late)
def frac_bucket(r):
    if r['max_t'] == 0:
        return 'only'
    f = r['t'] / r['max_t']
    return 'early' if f < 0.33 else ('mid' if f < 0.67 else 'late')

by_pos = {}
for r in records:
    by_pos.setdefault(frac_bucket(r), []).append(r['loss'])
print("\nmean loss by window position (fraction through the answer):")
for pos in ['only', 'early', 'mid', 'late']:
    if pos in by_pos:
        print(f"  {pos:5s}: {np.mean(by_pos[pos]):.3f} (n={len(by_pos[pos])})")

# bucket by carry count
by_carry = {}
for r in records:
    by_carry.setdefault(r['carries'], []).append(r['loss'])
print("\nmean loss by carry count:")
for c, losses in sorted(by_carry.items()):
    print(f"  {c:2d} carries: {np.mean(losses):.3f} (n={len(losses)})")

# the single worst combination
worst = sorted(records, key=lambda r: -r['loss'])[:15]
print("\nworst individual (example, window) probes:")
for r in worst:
    print(f"  digits={r['digits']:2d} t={r['t']:2d}/{r['max_t']:2d} carries={r['carries']} loss={r['loss']:.3f}")
