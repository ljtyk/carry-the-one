"""
Evaluate the segment-embeddings ablation: plain GPT, ordinary sequential
RoPE positions, natural (non-padded) digit order -- the ONLY addition is a
deterministic segment id per token (0=other, 1=operand a, 2=operand b,
3=answer) fed through a small learned embedding added at the input. Standard
greedy decoding, extended each step with the new token's segment id (always
3, since it's part of the answer).

Usage:
    python eval_segments.py --out_dir=out-addition-segments --device=cuda
"""
import os
import re
import json
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-segments'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
dump_errors_split = ''
dump_errors_n = 30
results_path = None
exec(open('configurator.py').read())
if results_path is None:
    results_path = os.path.join(out_dir, 'eval_segments_results.json')
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

with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
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
def greedy_decode_batch(prompts, max_new_tokens):
    """prompts: list of equal-length token-id lists, e.g. encode(f"\\n{a}+{b}=").
    Segment ids for the prompt are computed the same way as
    get_batch_segments; each newly generated token gets segment=3 (answer)."""
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
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        seg_cond = seg if seg.size(1) <= model.config.block_size else seg[:, -model.config.block_size:]
        with ctx:
            logits, _ = model(idx_cond, segment_ids=seg_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
        seg = torch.cat((seg, torch.full((b_sz, 1), 3, dtype=torch.long, device=device)), dim=1)
    return idx[:, plen:].tolist()


def eval_examples(examples, batch_size=200):
    by_prompt_len = {}
    for ex in examples:
        plen = len(f"{ex['a']}+{ex['b']}=")
        by_prompt_len.setdefault(plen, []).append(ex)

    per_example_results = []
    for plen, exs in sorted(by_prompt_len.items()):
        max_d = max(ex['digits'] for ex in exs)
        max_new_tokens = max_d + 3
        for i in range(0, len(exs), batch_size):
            batch = exs[i:i + batch_size]
            prompts = [encode(f"\n{ex['a']}+{ex['b']}=") for ex in batch]
            assert len(set(len(p) for p in prompts)) == 1, "prompt lengths must match within a batch"
            gen = greedy_decode_batch(prompts, max_new_tokens)
            for ex, g in zip(batch, gen):
                text = decode(g)
                pred_str = text.split('\n')[0]
                try:
                    pred_val = int(pred_str) if pred_str != '' else None
                except ValueError:
                    pred_val = None
                correct = pred_val == ex['c']
                per_example_results.append({**ex, 'pred': pred_str, 'correct': correct})
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
