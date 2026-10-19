"""
Evaluate RefineGPT: exact-match accuracy via the sliding-window refinement
procedure (see model.py's RefineGPT docstring), not greedy autoregressive
decoding -- eval_addition.py's greedy_decode_batch doesn't apply here
(RefineGPT is bidirectional and predicts a window, not one causal next
token).

Inference: L = max_new_tokens passes. At pass t, the input canvas is
(prompt + locked positions [0, t) + MASK for [t, L)) -- exactly the same
construction get_batch_refine uses during training (the whole window plus
everything after it is MASK every pass, no carried-forward within-window
draft; see get_batch_refine's docstring for why this simplification was
chosen). The model's prediction for position t (the window's leftmost slot)
is written into the canvas and locked; positions t+1, t+2 are also predicted
this pass but get overwritten (re-predicted with more context) on
subsequent passes, so only the pass-t prediction for position t is kept.

Usage:
    python eval_refine.py --out_dir=out-addition-refine --dataset_dir=data/addition --device=cuda
"""
import os
import re
import json
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, RefineGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-refine'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
dump_errors_split = ''
dump_errors_n = 30
results_path = None
exec(open('configurator.py').read())
if results_path is None:
    results_path = os.path.join(out_dir, 'eval_refine_results.json')
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
assert checkpoint.get('model_type') == 'refine', f"expected a RefineGPT checkpoint, got {checkpoint.get('model_type')}"
gptconf = GPTConfig(**checkpoint['model_args'])
model = RefineGPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# dataset's own meta.pkl (data/addition_refine) has the mask_id RefineGPT was
# trained with -- NOT dataset_dir's (data/addition has no meta.pkl at all,
# it's raw txt; the padded variants' vocabs don't all agree on token ids)
train_meta_path = os.path.join('data', 'addition_refine', 'meta.pkl')
with open(train_meta_path, 'rb') as f:
    train_meta = pickle.load(f)
stoi, itos, mask_id = train_meta['stoi'], train_meta['itos'], train_meta['mask_id']


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
def refine_decode_batch(prompts, max_new_tokens):
    """prompts: list of equal-length token-id lists (already includes the
    leading '\\n'). Returns list of generated (post-prompt) id lists, one
    per example, built via the sliding-window refinement procedure."""
    b = len(prompts)
    prompt_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    plen = prompt_ids.size(1)
    canvas = torch.full((b, max_new_tokens), mask_id, dtype=torch.long, device=device)
    locked = torch.zeros((b, max_new_tokens), dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        window_input = canvas.clone()
        window_input[:, t:] = mask_id  # everything from t onward is "not yet visited", matching training
        idx = torch.cat([prompt_ids, window_input], dim=1)
        with ctx:
            logits, _ = model(idx)
        pred_t = torch.argmax(logits[:, plen + t, :], dim=-1)
        canvas[:, t] = pred_t
        locked[:, t] = True

    return canvas.tolist()


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
            gen = refine_decode_batch(prompts, max_new_tokens)
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
