"""
How wrong is wrong? For every incorrect prediction, computes the numeric gap
between predicted and true answer (not just exact-match accuracy), to
distinguish "close/confused" errors (small numeric difference, right digit
count, plausible carry slip) from "garbage" errors (unparseable, wildly
different magnitude, wrong digit count entirely).

Usage:
    python eval_error_magnitude.py --out_dir=out-addition-latent-drill --ckpt_name=ckpt_5000.pt --splits=test_gap,test_hint,test_ood
"""
import os
import re
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT, LoopedGPT, ACTLoopedGPT, PerExampleLoopedGPT, TwoBlockGPT, LatentLoopedGPT, AdaptiveLatentLoopedGPT

LATENT_MODEL_TYPES = ('latent_looped', 'adaptive_latent_looped')

out_dir = 'out-addition-latent-drill'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
splits = 'test_gap,test_hint,test_ood'
n_show = 15  # example rows to print per split, spread across the error-magnitude spectrum
exec(open('configurator.py').read())

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
model_type = checkpoint.get('model_type', 'gpt')
model_cls = {'gpt': GPT, 'looped': LoopedGPT, 'act_looped': ACTLoopedGPT, 'per_example_looped': PerExampleLoopedGPT,
             'two_block': TwoBlockGPT, 'latent_looped': LatentLoopedGPT, 'adaptive_latent_looped': AdaptiveLatentLoopedGPT}[model_type]
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

meta_path = os.path.join(dataset_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return ''.join(itos[i] for i in ids)


LINE_RE = re.compile(r'^(\d+)\+(\d+)=(\d+)$')


def load_examples(path):
    examples = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            d = max(len(str(a)), len(str(b)))
            examples.append({'a': a, 'b': b, 'c': c, 'digits': d})
    return examples


@torch.no_grad()
def greedy_decode_batch(prompts, max_new_tokens, latent_prompt_len=None):
    idx = torch.tensor(prompts, dtype=torch.long, device=device)
    prompt_len = idx.size(1)
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        with ctx:
            if latent_prompt_len is not None:
                logits, _ = model(idx_cond, prompt_len=latent_prompt_len)
            else:
                logits, _ = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[:, prompt_len:].tolist()


def eval_split(name):
    examples = load_examples(os.path.join(dataset_dir, f'{name}.txt'))
    by_prompt_len = {}
    for ex in examples:
        plen = len(f"{ex['a']}+{ex['b']}=")
        by_prompt_len.setdefault(plen, []).append(ex)

    results = []
    for plen, exs in sorted(by_prompt_len.items()):
        max_d = max(ex['digits'] for ex in exs)
        max_new_tokens = max_d + 3
        for i in range(0, len(exs), 200):
            batch = exs[i:i + 200]
            prompts = [encode(f"\n{ex['a']}+{ex['b']}=") for ex in batch]
            latent_prompt_len = plen + 1 if model_type in LATENT_MODEL_TYPES else None
            gen = greedy_decode_batch(prompts, max_new_tokens, latent_prompt_len=latent_prompt_len)
            for ex, g in zip(batch, gen):
                text = decode(g)
                pred_str = text.split('\n')[0]
                try:
                    pred_val = int(pred_str) if pred_str not in ('', '-') else None
                except ValueError:
                    pred_val = None
                results.append({**ex, 'pred_str': pred_str, 'pred_val': pred_val})
    return results


def summarize(name, results):
    n = len(results)
    unparseable = sum(1 for r in results if r['pred_val'] is None)
    correct = sum(1 for r in results if r['pred_val'] == r['c'])
    wrong = [r for r in results if r['pred_val'] != r['c']]

    print(f"\n=== {name} (n={n}) ===")
    print(f"correct: {correct} ({100*correct/n:.1f}%)   unparseable output: {unparseable} ({100*unparseable/n:.1f}%)")

    parseable_wrong = [r for r in wrong if r['pred_val'] is not None]
    if parseable_wrong:
        abs_errs = [abs(r['pred_val'] - r['c']) for r in parseable_wrong]
        rel_errs = [abs(r['pred_val'] - r['c']) / max(r['c'], 1) for r in parseable_wrong]
        digit_diffs = [len(str(r['pred_val'])) - len(str(r['c'])) for r in parseable_wrong]

        same_digit_count = sum(1 for d in digit_diffs if d == 0)
        rel_err_lt_1pct = sum(1 for r in rel_errs if r < 0.01)
        rel_err_lt_10pct = sum(1 for r in rel_errs if r < 0.10)

        print(f"of {len(parseable_wrong)} wrong-but-parseable predictions:")
        print(f"  same digit count as true answer: {same_digit_count} ({100*same_digit_count/len(parseable_wrong):.1f}%)")
        print(f"  relative error < 1%:  {rel_err_lt_1pct} ({100*rel_err_lt_1pct/len(parseable_wrong):.1f}%)")
        print(f"  relative error < 10%: {rel_err_lt_10pct} ({100*rel_err_lt_10pct/len(parseable_wrong):.1f}%)")
        print(f"  median abs error: {sorted(abs_errs)[len(abs_errs)//2]:.3e}   median rel error: {sorted(rel_errs)[len(rel_errs)//2]:.4f}")

    # show a spread: closest few, median few, farthest few (by relative error; unparseable sorts last)
    def sort_key(r):
        if r['pred_val'] is None:
            return float('inf')
        return abs(r['pred_val'] - r['c']) / max(r['c'], 1)
    wrong_sorted = sorted(wrong, key=sort_key)
    picks = []
    if wrong_sorted:
        picks += wrong_sorted[:5]
        mid = len(wrong_sorted) // 2
        picks += wrong_sorted[max(0, mid-2):mid+3]
        picks += wrong_sorted[-5:]
    print(f"sample wrong predictions (closest -> median -> farthest):")
    for r in picks:
        print(f"  {r['a']}+{r['b']}={r['c']}  pred={r['pred_str']!r}  digits={r['digits']}")


for split in splits.split(','):
    results = eval_split(split)
    summarize(split, results)
