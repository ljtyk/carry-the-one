"""
Evaluate the index-hints ablation (plain GPT, ordinary sequential RoPE
positions, normal digit order, each digit tagged with a place-value letter).
Standard greedy decoding; the generated answer is untagged (strip the
letters, keep the digits in order) before comparing to the true sum.

Usage:
    python eval_hints.py --out_dir=out-addition-hints --device=cuda
"""
import os
import re
import json
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT

INDEX_CHARS = set([chr(ord('a') + i) for i in range(26)] + ['!'])

# -----------------------------------------------------------------------------
out_dir = 'out-addition-hints'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
dump_errors_split = ''
dump_errors_n = 30
results_path = None
exec(open('configurator.py').read())
if results_path is None:
    results_path = os.path.join(out_dir, 'eval_hints_results.json')
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

with open(os.path.join('data', 'addition_hints', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return ''.join(itos[i] for i in ids)


def tag(number_str):
    L = len(number_str)
    idx_chars = [chr(ord('a') + i) for i in range(26)] + ['!']
    return ''.join(f"{idx_chars[L - 1 - i]}{ch}" for i, ch in enumerate(number_str))


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
def greedy_decode_batch(prompts, max_new_tokens):
    idx = torch.tensor(prompts, dtype=torch.long, device=device)
    prompt_len = idx.size(1)
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        with ctx:
            logits, _ = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[:, prompt_len:].tolist()


def eval_examples(examples, batch_size=200):
    by_prompt_len = {}
    for ex in examples:
        tagged_a, tagged_b = tag(str(ex['a'])), tag(str(ex['b']))
        plen = len(f"{tagged_a}+{tagged_b}=")
        by_prompt_len.setdefault(plen, []).append((ex, tagged_a, tagged_b))

    per_example_results = []
    for plen, group in sorted(by_prompt_len.items()):
        max_d = max(ex['digits'] for ex, _, _ in group)
        max_new_tokens = 2 * (max_d + 2) + 1  # (digit+tag) pairs for up to max_d+1 digits, plus newline, plus slack
        for i in range(0, len(group), batch_size):
            batch = group[i:i + batch_size]
            prompts = [encode(f"\n{ta}+{tb}=") for ex, ta, tb in batch]
            assert len(set(len(p) for p in prompts)) == 1, "prompt lengths must match within a batch"
            gen = greedy_decode_batch(prompts, max_new_tokens)
            for (ex, ta, tb), g in zip(batch, gen):
                text = decode(g)
                tagged_answer = text.split('\n')[0]
                digits_only = ''.join(ch for ch in tagged_answer if ch.isdigit())
                try:
                    pred_val = int(digits_only) if digits_only != '' else None
                except ValueError:
                    pred_val = None
                correct = pred_val == ex['c']
                per_example_results.append({**ex, 'pred': digits_only, 'correct': correct})
    return per_example_results


def summarize(results, key):
    buckets = {}
    for r in results:
        buckets.setdefault(r[key], []).append(r['correct'])
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}


def main():
    all_results = {}
    splits = ['test_indist', 'test_ood']
    for extra_split in ['test_hint', 'test_gap']:
        if os.path.exists(os.path.join(dataset_dir, f'{extra_split}.txt')):
            splits.insert(1, extra_split)
    for extra_split in ['test_far_ood']:
        if os.path.exists(os.path.join(dataset_dir, f'{extra_split}.txt')):
            splits.append(extra_split)
    for split in splits:
        path = os.path.join(dataset_dir, f'{split}.txt')
        examples = load_examples(path)
        results = eval_examples(examples)
        overall_acc = sum(r['correct'] for r in results) / len(results)
        acc_by_digits = summarize(results, 'digits')
        print(f"\n=== {split} ===")
        print(f"overall accuracy: {overall_acc:.4f} (n={len(results)})")
        print(f"accuracy by digit count: {acc_by_digits}")
        if split == dump_errors_split:
            wrong = [r for r in results if not r['correct']]
            print(f"\n--- {len(wrong)} incorrect examples in {split} (showing up to {dump_errors_n}) ---")
            for r in wrong[:dump_errors_n]:
                print(f"  {r['a']}+{r['b']}={r['c']}  (pred={r['pred']!r})  digits={r['digits']}")
        all_results[split] = {'overall_accuracy': overall_acc, 'accuracy_by_digits': acc_by_digits, 'n': len(results)}

    os.makedirs(out_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved results to {results_path}")


if __name__ == '__main__':
    main()
