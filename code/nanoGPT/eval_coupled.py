"""
Evaluate a PositionCoupledGPT checkpoint: exact-match accuracy as a function
of digit count, using standard greedy decoding but with the position-
coupling id scheme (same as get_batch_coupled/model.py's PositionCoupledGPT
docstring) instead of sequential positions, and un-reversing/un-padding the
generated answer before comparing to the true sum.

Since position ids depend only on the (known-in-advance) operand length L,
not on what's actually generated, the full position_ids array can be
precomputed before decoding starts -- greedy decoding just slices it to
however much has been generated so far each step.

Usage:
    python eval_coupled.py --out_dir=out-addition-coupled --device=cuda
"""
import os
import re
import json
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, PositionCoupledGPT

BASE = 10  # must match train.py's _COUPLED_BASE

# -----------------------------------------------------------------------------
out_dir = 'out-addition-coupled'
dataset_dir = os.path.join('data', 'addition')  # source (a, b, c) triples, same as every other eval script
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
dump_errors_split = ''
dump_errors_n = 30
results_path = None
exec(open('configurator.py').read())
if results_path is None:
    results_path = os.path.join(out_dir, 'eval_coupled_results.json')
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
assert checkpoint.get('model_type') == 'coupled', f"expected a PositionCoupledGPT checkpoint, got {checkpoint.get('model_type')}"
gptconf = GPTConfig(**checkpoint['model_args'])
model = PositionCoupledGPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

with open(os.path.join('data', 'addition_coupled', 'meta.pkl'), 'rb') as f:
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


def coupled_prompt_and_positions(a, b):
    """Builds the (token ids, position ids) for '\\n{padded_a}+{padded_b}='
    plus the position ids for the L+1 answer digits and trailing newline
    that greedy decoding will fill in -- same scheme as get_batch_coupled."""
    a_str, b_str = str(a), str(b)
    L = max(len(a_str), len(b_str))
    padded_a, padded_b = a_str.zfill(L), b_str.zfill(L)
    prompt_str = f"\n{padded_a}+{padded_b}="
    prompt_ids = encode(prompt_str)

    pos = [0] * len(prompt_ids)
    pos[0] = BASE - 2  # leading '\n'
    for i in range(L):
        pos[1 + i] = BASE + i  # operand a, MSD->LSD
    plus_pos = 1 + L
    pos[plus_pos] = BASE + L
    for i in range(L):
        pos[plus_pos + 1 + i] = BASE + i  # operand b, coupled to a
    eq_pos = plus_pos + 1 + L
    pos[eq_pos] = BASE + L

    # position ids for the L+1 answer digits (descending) + trailing '\n'
    answer_pos = list(range(BASE + L - 1, BASE - 2, -1)) + [BASE - 2]
    return prompt_ids, pos, answer_pos, L


@torch.no_grad()
def coupled_decode_batch(examples):
    """No KV cache anywhere in this codebase, so each example is decoded
    individually (position-id construction and answer length both vary per
    example) rather than batched -- fine for eval-time throughput here."""
    results = []
    for ex in examples:
        prompt_ids, prompt_pos, answer_pos, L = coupled_prompt_and_positions(ex['a'], ex['b'])
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        full_pos = torch.tensor([prompt_pos + answer_pos], dtype=torch.long, device=device)
        n_answer_tokens = L + 2  # L+1 digits + trailing newline
        for step in range(n_answer_tokens):
            cur_len = idx.size(1)
            with ctx:
                logits, _ = model(idx, position_ids=full_pos[:, :cur_len])
            idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, idx_next], dim=1)
        gen_ids = idx[0, len(prompt_ids):].tolist()
        text = decode(gen_ids)
        reversed_part = text.split('\n')[0]  # the L+1 reversed sum digits (before the newline, if predicted correctly)
        try:
            # un-reverse, then int() strips any leading zeros from the padding
            pred_val = int(reversed_part[::-1]) if reversed_part != '' else None
        except ValueError:
            pred_val = None
        results.append({**ex, 'pred': reversed_part[::-1], 'correct': pred_val == ex['c']})
    return results


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
        results = coupled_decode_batch(examples)
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
