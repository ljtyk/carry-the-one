"""
Diagnostic: does the model even attempt an answer of the correct LENGTH on
OOD (11-15 digit) problems, or does it truncate to something close to the
trained max (~10-11 digit sums) regardless of what the problem actually
requires? Tests the "output-length calibration" hypothesis raised after
RoPE's null result on OOD accuracy: RoPE fixes attention's ability to
represent unseen relative positions, but doesn't necessarily fix a learned
habit about how many digits an answer "should" have.

Reuses eval_addition.py's loading/decoding logic; reports generated answer
length vs. true answer length instead of exact-match accuracy.

Usage:
    python eval_length_calibration.py --out_dir=out-addition-rope --device=cuda
"""
import os
import re
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT, LoopedGPT, ACTLoopedGPT, PerExampleLoopedGPT, TwoBlockGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-rope'
dataset_dir = os.path.join('data', 'addition')
ckpt_name = 'ckpt.pt'  # override to a numbered snapshot (e.g. 'ckpt_15000.pt') to eval mid-run without touching the live training checkpoint
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
model_type = checkpoint.get('model_type', 'gpt')
model_cls = {'gpt': GPT, 'looped': LoopedGPT, 'act_looped': ACTLoopedGPT, 'per_example_looped': PerExampleLoopedGPT, 'two_block': TwoBlockGPT}[model_type]
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

train_dataset = checkpoint.get('config', {}).get('dataset', 'addition')
meta_dir = os.path.join('data', train_dataset)
with open(os.path.join(meta_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

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
            examples.append({'a': a, 'b': b, 'c': c, 'digits': d, 'true_len': len(str(c))})
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


def run_split(split, extra_slack=8):
    path = os.path.join(dataset_dir, f'{split}.txt')
    examples = load_examples(path)
    # group by actual prompt length, not just the digit-count label `d` --
    # operands can have mismatched lengths within the same d-bucket, so two
    # examples with the same `d` can have different "a+b=" prompt lengths.
    # greedy_decode_batch needs a rectangular tensor, so every prompt in a
    # batch must be the same length.
    by_prompt_len = {}
    for ex in examples:
        plen = len(f"{ex['a']}+{ex['b']}=")
        by_prompt_len.setdefault(plen, []).append(ex)

    results = []
    for plen, exs in sorted(by_prompt_len.items()):
        max_d = max(ex['digits'] for ex in exs)
        max_new_tokens = max_d + 1 + extra_slack  # generous margin beyond any plausible true answer length
        for i in range(0, len(exs), 200):
            batch = exs[i:i + 200]
            prompts = [encode(f"\n{ex['a']}+{ex['b']}=") for ex in batch]
            assert len(set(len(p) for p in prompts)) == 1
            gen = greedy_decode_batch(prompts, max_new_tokens)
            for ex, g in zip(batch, gen):
                text = decode(g)
                pred_str = text.split('\n')[0]
                gen_len = len(pred_str)  # how many chars before the model chose to emit '\n' (or ran out of budget)
                hit_budget = '\n' not in text  # never emitted a stop at all within the generous budget
                results.append({**ex, 'pred': pred_str, 'gen_len': gen_len, 'hit_budget': hit_budget})
    return results


def summarize(results, label):
    print(f"\n=== {label} ===")
    by_digits = {}
    for r in results:
        by_digits.setdefault(r['digits'], []).append(r)
    for d, rs in sorted(by_digits.items()):
        mean_true = sum(r['true_len'] for r in rs) / len(rs)
        mean_gen = sum(r['gen_len'] for r in rs) / len(rs)
        frac_short = sum(r['gen_len'] < r['true_len'] for r in rs) / len(rs)
        frac_exact = sum(r['gen_len'] == r['true_len'] for r in rs) / len(rs)
        frac_long = sum(r['gen_len'] > r['true_len'] for r in rs) / len(rs)
        frac_hit_budget = sum(r['hit_budget'] for r in rs) / len(rs)
        print(f"  digits={d:2d}: true_len={mean_true:.2f}  gen_len={mean_gen:.2f}  "
              f"short={frac_short:.0%} exact={frac_exact:.0%} long={frac_long:.0%}  never_stopped={frac_hit_budget:.0%}")


def main():
    splits = ['test_indist', 'test_ood']
    for extra_split in ['test_hint', 'test_gap']:  # length-tiered datasets only, inserted right after test_indist
        if os.path.exists(os.path.join(dataset_dir, f'{extra_split}.txt')):
            splits.insert(1, extra_split)
    for split in splits:
        results = run_split(split)
        summarize(results, f'{split} (length calibration)')


if __name__ == '__main__':
    main()
