"""Quick standalone: eval just test_gap (or another single split), dumping
both correct and incorrect examples. Faster than eval_addition.py's full
sweep across all splits when you only care about one.

Usage:
    python eval_gap_only.py --out_dir=out-addition-latent-grok --split=test_gap --device=cuda
"""
import os
import re
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, AdaptiveLatentLoopedGPT, LatentLoopedGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-latent-grok'
dataset_dir = os.path.join('data', 'addition')
split = 'test_gap'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
n_show = 15
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
model_type = checkpoint.get('model_type')
model_cls = {'adaptive_latent_looped': AdaptiveLatentLoopedGPT, 'latent_looped': LatentLoopedGPT}[model_type]
gptconf = GPTConfig(**checkpoint['model_args'])
model = model_cls(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

train_dataset = checkpoint.get('config', {}).get('dataset', 'addition_padded')
meta_dir = os.path.join('data', train_dataset)
with open(os.path.join(meta_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

LINE_RE = re.compile(r'^(\d+)\+(\d+)=(\d+)$')


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


examples = []
with open(os.path.join(dataset_dir, f'{split}.txt')) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d = max(len(str(a)), len(str(b)))
        examples.append({'a': a, 'b': b, 'c': c, 'digits': d, 'carries': count_carries(a, b)})

by_plen = {}
for ex in examples:
    plen = len(f"{ex['a']}+{ex['b']}=")
    by_plen.setdefault(plen, []).append(ex)

results = []
with torch.no_grad():
    for plen, exs in sorted(by_plen.items()):
        max_new = max(ex['digits'] for ex in exs) + 3
        for i in range(0, len(exs), 200):
            batch = exs[i:i + 200]
            prompts = [encode(f"\n{ex['a']}+{ex['b']}=") for ex in batch]
            idx = torch.tensor(prompts, dtype=torch.long, device=device)
            prompt_len = idx.size(1)
            for _ in range(max_new):
                with ctx:
                    logits, _ = model(idx, prompt_len=prompt_len)
                idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
            gen = idx[:, prompt_len:].tolist()
            for ex, g in zip(batch, gen):
                text = decode(g)
                pred_str = text.split('\n')[0]
                try:
                    pred_val = int(pred_str) if pred_str != '' else None
                except ValueError:
                    pred_val = None
                correct = pred_val == ex['c']
                results.append({**ex, 'pred': pred_str, 'correct': correct})

right = [r for r in results if r['correct']]
wrong = [r for r in results if not r['correct']]
acc = len(right) / len(results)
print(f"{split}: {len(right)}/{len(results)} correct ({acc*100:.2f}%)")

print(f"\n--- {min(n_show, len(right))} CORRECT examples ---")
for r in right[:n_show]:
    print(f"  {r['a']}+{r['b']}={r['c']}  digits={r['digits']} carries={r['carries']} len_a={len(str(r['a']))} len_b={len(str(r['b']))}")

print(f"\n--- {min(n_show, len(wrong))} INCORRECT examples ---")
for r in wrong[:n_show]:
    print(f"  {r['a']}+{r['b']}={r['c']}  (pred={r['pred']!r})  digits={r['digits']} carries={r['carries']} len_a={len(str(r['a']))} len_b={len(str(r['b']))}")

a_longer = [r for r in results if len(str(r['a'])) > len(str(r['b']))]
b_longer = [r for r in results if len(str(r['b'])) > len(str(r['a']))]
print(f"\n--- breakdown by which operand is longer ---")
if a_longer:
    print(f"  len(a) > len(b): {sum(r['correct'] for r in a_longer)}/{len(a_longer)} correct ({100*sum(r['correct'] for r in a_longer)/len(a_longer):.2f}%)")
if b_longer:
    print(f"  len(b) > len(a): {sum(r['correct'] for r in b_longer)}/{len(b_longer)} correct ({100*sum(r['correct'] for r in b_longer)/len(b_longer):.2f}%)")
