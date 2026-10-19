"""
Plot the per-example adaptive-loop model's chosen loop count (n) against
problem digit count, across all length tiers (in-dist, gap, hint, ood).

Only meaningful for model_type='per_example_looped' (PerExampleLoopedGPT),
since it's the only model with a genuine per-example loop-count decision.
n is read straight off the prompt (before any answer digits are generated),
matching what the model's difficulty probe actually sees when it commits to
a loop count -- this is model._last_n_per_example, set during forward().

Usage:
    python eval_loop_vs_digits.py --out_dir=out-addition-perexample-rope-masked --device=cuda
"""
import os
import re
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, PerExampleLoopedGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-perexample-rope-masked'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
plot_path = None
exec(open('configurator.py').read())
if plot_path is None:
    plot_path = os.path.join(out_dir, 'loop_n_vs_digits.png')
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
model_type = checkpoint.get('model_type', 'gpt')
assert model_type == 'per_example_looped', f"loop count vs digits only applies to per_example_looped, got {model_type!r}"
gptconf = GPTConfig(**checkpoint['model_args'])
model = PerExampleLoopedGPT(gptconf)
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
stoi = meta['stoi']
encode = lambda s: [stoi[c] for c in s]

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
            examples.append({'a': a, 'b': b, 'digits': d})
    return examples


@torch.no_grad()
def loop_n_for_examples(examples, batch_size=200):
    """Returns list of (digits, n) -- n read from the prompt-only forward pass."""
    by_prompt_len = {}
    for ex in examples:
        plen = len(f"{ex['a']}+{ex['b']}=")
        by_prompt_len.setdefault(plen, []).append(ex)

    out = []
    for plen, exs in sorted(by_prompt_len.items()):
        for i in range(0, len(exs), batch_size):
            batch = exs[i:i + batch_size]
            prompts = [encode(f"\n{ex['a']}+{ex['b']}=") for ex in batch]
            idx = torch.tensor(prompts, dtype=torch.long, device=device)
            with ctx:
                model(idx)
            n = model._last_n_per_example.float().cpu().tolist()
            for ex, ni in zip(batch, n):
                out.append((ex['digits'], ni))
    return out


def main():
    splits = ['test_indist', 'test_gap', 'test_hint', 'test_ood']
    all_pairs = []
    for split in splits:
        path = os.path.join(dataset_dir, f'{split}.txt')
        if not os.path.exists(path):
            continue
        examples = load_examples(path)
        pairs = loop_n_for_examples(examples)
        all_pairs.extend(pairs)

    by_digits = {}
    for d, n in all_pairs:
        by_digits.setdefault(d, []).append(n)

    print(f"\n{'digits':>7} {'n_examples':>11} {'mean_n':>8} {'std_n':>8} {'min_n':>7} {'max_n':>7}")
    digits_sorted = sorted(by_digits.keys())
    means, stds = [], []
    for d in digits_sorted:
        vals = torch.tensor(by_digits[d])
        mean_n = vals.mean().item()
        std_n = vals.std().item() if len(vals) > 1 else 0.0
        means.append(mean_n)
        stds.append(std_n)
        print(f"{d:>7} {len(vals):>11} {mean_n:>8.3f} {std_n:>8.3f} {vals.min().item():>7.3f} {vals.max().item():>7.3f}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(digits_sorted, means, yerr=stds, marker='o', capsize=3)
        ax.axvline(10.5, color='gray', linestyle='--', linewidth=1, label='train/gap boundary')
        ax.axvline(13.5, color='lightgray', linestyle=':', linewidth=1, label='gap/hint boundary')
        ax.axvline(15.5, color='lightgray', linestyle='-.', linewidth=1, label='hint/ood boundary')
        max_loop_iters = gptconf.max_loop_iters
        ax.axhline(max_loop_iters, color='red', linestyle='--', linewidth=1, label=f'max_loop_iters={max_loop_iters}')
        ax.set_xlabel('digit count (d = max(len(a), len(b)))')
        ax.set_ylabel('model-chosen loop count n (mean +/- std)')
        ax.set_title('Per-example loop count vs. problem digit count')
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"\nsaved plot to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == '__main__':
    main()
