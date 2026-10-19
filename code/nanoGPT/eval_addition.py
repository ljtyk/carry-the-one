"""
Evaluate a trained addition model: exact-match accuracy as a function of
digit count and carry count, on both in-distribution and out-of-distribution
(longer digit count) held-out test sets.

Unlike sample.py this does greedy (argmax) batched decoding -- addition has a
single correct answer, so we want the model's best guess, not a stochastic
sample -- and stops each sequence at its first generated newline.

Usage:
    python eval_addition.py --out_dir=out-addition --device=cuda
"""
import os
import re
import json
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT, LoopedGPT, ACTLoopedGPT, PerExampleLoopedGPT, TwoBlockGPT, LatentLoopedGPT, AdaptiveLatentLoopedGPT, LocalLoopedGPT

LATENT_MODEL_TYPES = ('latent_looped', 'adaptive_latent_looped')
# LocalLoopedGPT also needs prompt_len passed into every forward call (same
# requirement as the latent classes, different reason -- see model.py)
PROMPT_GROUPED_MODEL_TYPES = LATENT_MODEL_TYPES + ('local_looped',)

# -----------------------------------------------------------------------------
out_dir = 'out-addition'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
ckpt_name = 'ckpt.pt'  # override to a numbered snapshot (e.g. 'ckpt_15000.pt') to eval mid-run without touching the live training checkpoint
dump_errors_split = ''  # set to a split name (e.g. 'test_indist') to print its incorrect examples
dump_errors_n = 30  # max incorrect examples to print
plot_path = None  # default: <out_dir>/accuracy_vs_difficulty.png, set below after out_dir is finalized
results_path = None  # default: <out_dir>/eval_results.json
exec(open('configurator.py').read())
if plot_path is None:
    plot_path = os.path.join(out_dir, 'accuracy_vs_difficulty.png')
if results_path is None:
    results_path = os.path.join(out_dir, 'eval_results.json')
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
model_type = checkpoint.get('model_type', 'gpt')
model_cls = {'gpt': GPT, 'looped': LoopedGPT, 'act_looped': ACTLoopedGPT, 'per_example_looped': PerExampleLoopedGPT, 'two_block': TwoBlockGPT, 'latent_looped': LatentLoopedGPT, 'adaptive_latent_looped': AdaptiveLatentLoopedGPT, 'local_looped': LocalLoopedGPT}[model_type]
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
if compile:
    model = torch.compile(model)

# vocab must match what the checkpoint was actually trained with -- e.g.
# addition_padded's vocab has an extra PAD char, shifting every other
# char's id, so this can't just always be dataset_dir's meta.pkl (that's
# only guaranteed to hold the eval test files, not necessarily match training)
train_dataset = checkpoint.get('config', {}).get('dataset', 'addition')
meta_dir = os.path.join('data', train_dataset)
with open(os.path.join(meta_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

LINE_RE = re.compile(r'^(\d+)\+(\d+)=(\d+)$')


def count_carries(a, b):
    carries = 0
    carry = 0
    while a > 0 or b > 0:
        da, db = a % 10, b % 10
        s = da + db + carry
        carry = 1 if s >= 10 else 0
        carries += carry
        a //= 10
        b //= 10
    return carries


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
            examples.append({'a': a, 'b': b, 'c': c, 'digits': d, 'carries': count_carries(a, b)})
    return examples


@torch.no_grad()
def greedy_decode_batch(prompts, max_new_tokens, latent_prompt_len=None):
    """prompts: list of equal-length token-id lists. Returns list of generated (post-prompt) id lists.
    latent_prompt_len: LatentLoopedGPT only -- the real prompt length (INCLUDING
    the leading '\\n', matching training's convention) within idx, needed on
    every call since the model recomputes its K latent-thought positions
    fresh each step (no KV cache anywhere in this codebase) -- see
    LatentLoopedGPT.forward. None for every other model type."""
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


def eval_examples(examples, batch_size=200):
    # group by actual prompt length, not just the digit-count label `d` --
    # operands can now have mismatched lengths within the same d-bucket
    # (d = max(len(a), len(b))), so two examples with the same `d` can have
    # different "a+b=" prompt lengths. greedy_decode_batch needs a
    # rectangular tensor, so every prompt in a batch must be the same length.
    by_prompt_len = {}
    for ex in examples:
        plen = len(f"{ex['a']}+{ex['b']}=")
        by_prompt_len.setdefault(plen, []).append(ex)

    per_example_results = []
    for plen, exs in sorted(by_prompt_len.items()):
        max_d = max(ex['digits'] for ex in exs)
        max_new_tokens = max_d + 3  # up to max_d+1 digit sum, plus newline, plus slack
        for i in range(0, len(exs), batch_size):
            batch = exs[i:i + batch_size]
            # leading '\n' matters: every training example is preceded by the
            # previous example's newline, so a bare "a+b=" prefix is off-distribution
            prompts = [encode(f"\n{ex['a']}+{ex['b']}=") for ex in batch]
            assert len(set(len(p) for p in prompts)) == 1, "prompt lengths must match within a batch"
            # plen (the loop variable) is len(f"{a}+{b}=") -- doesn't include the
            # leading '\n' that IS part of the encoded prompt tensor above, so the
            # real prompt_len LatentLoopedGPT needs is one more than that
            latent_prompt_len = plen + 1 if model_type in PROMPT_GROUPED_MODEL_TYPES else None
            gen = greedy_decode_batch(prompts, max_new_tokens, latent_prompt_len=latent_prompt_len)
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
    for extra_split in ['test_hint', 'test_gap']:  # length-tiered datasets only, inserted right after test_indist
        if os.path.exists(os.path.join(dataset_dir, f'{extra_split}.txt')):
            splits.insert(1, extra_split)
    for extra_split in ['test_carry_heavy_ood', 'test_carry_heavy_gap']:  # carry-cascade transfer tests, if present
        if os.path.exists(os.path.join(dataset_dir, f'{extra_split}.txt')):
            splits.append(extra_split)
    for split in splits:
        path = os.path.join(dataset_dir, f'{split}.txt')
        examples = load_examples(path)
        results = eval_examples(examples)
        overall_acc = sum(r['correct'] for r in results) / len(results)
        acc_by_digits = summarize(results, 'digits')
        acc_by_carries = summarize(results, 'carries')
        print(f"\n=== {split} ===")
        print(f"overall accuracy: {overall_acc:.4f} (n={len(results)})")
        print(f"accuracy by digit count: {acc_by_digits}")
        print(f"accuracy by carry count: {acc_by_carries}")
        if split == dump_errors_split:
            wrong = [r for r in results if not r['correct']]
            print(f"\n--- {len(wrong)} incorrect examples in {split} (showing up to {dump_errors_n}) ---")
            for r in wrong[:dump_errors_n]:
                len_a, len_b = len(str(r['a'])), len(str(r['b']))
                print(f"  {r['a']}+{r['b']}={r['c']}  (pred={r['pred']!r})  "
                      f"digits={r['digits']} carries={r['carries']} len_a={len_a} len_b={len_b}")
        all_results[split] = {
            'overall_accuracy': overall_acc,
            'accuracy_by_digits': acc_by_digits,
            'accuracy_by_carries': acc_by_carries,
            'n': len(results),
        }

    os.makedirs(out_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved results to {results_path}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ax = axes[0]
        for split, marker in [('test_indist', 'o'), ('test_ood', 's')]:
            d = all_results[split]['accuracy_by_digits']
            ax.plot(list(d.keys()), list(d.values()), marker=marker, label=split)
        ax.axvline(10.5, color='gray', linestyle='--', linewidth=1, label='train/OOD boundary')
        ax.set_xlabel('digit count')
        ax.set_ylabel('exact-match accuracy')
        ax.set_title('Accuracy vs. digit count')
        ax.set_ylim(-0.02, 1.02)
        ax.legend()

        ax = axes[1]
        for split, marker in [('test_indist', 'o'), ('test_ood', 's')]:
            d = all_results[split]['accuracy_by_carries']
            ax.plot(list(d.keys()), list(d.values()), marker=marker, label=split)
        ax.set_xlabel('carry count')
        ax.set_ylabel('exact-match accuracy')
        ax.set_title('Accuracy vs. carry count')
        ax.set_ylim(-0.02, 1.02)
        ax.legend()

        fig.suptitle('Stage 1 fixed-depth baseline')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"saved plot to {plot_path}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == '__main__':
    main()
