"""
Compare two checkpoints on a more sensitive metric than exact-match: did the
model at least predict the correct output DIGIT COUNT? Exact match requires
every digit right, which can mask whether adversarial (carry-heavy) training
improved length-tracking specifically, even if it didn't yet produce fully
correct answers.

Usage:
    python eval_length_accuracy.py out-addition-rope-masked out-addition-rope-masked-adversarial
"""
import os
import re
import sys
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT

LINE_RE = re.compile(r'^(\d+)\+(\d+)=(\d+)$')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)


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


def load_model(out_dir):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
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

    train_dataset = checkpoint.get('config', {}).get('dataset', 'addition')
    meta_dir = os.path.join('data', train_dataset)
    with open(os.path.join(meta_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return model, meta['stoi'], meta['itos']


@torch.no_grad()
def greedy_decode_batch(model, encode, prompts, max_new_tokens):
    idx = torch.tensor(prompts, dtype=torch.long, device=device)
    prompt_len = idx.size(1)
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        with ctx:
            logits, _ = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[:, prompt_len:].tolist()


def eval_length_acc(model, stoi, itos, examples, batch_size=200):
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    by_plen = {}
    for ex in examples:
        plen = len(f"{ex['a']}+{ex['b']}=")
        by_plen.setdefault(plen, []).append(ex)

    results = []
    for plen, exs in sorted(by_plen.items()):
        max_d = max(ex['digits'] for ex in exs)
        max_new_tokens = max_d + 3
        for i in range(0, len(exs), batch_size):
            batch = exs[i:i + batch_size]
            prompts = [encode(f"\n{ex['a']}+{ex['b']}=") for ex in batch]
            gen = greedy_decode_batch(model, encode, prompts, max_new_tokens)
            for ex, g in zip(batch, gen):
                text = decode(g)
                pred_str = text.split('\n')[0]
                pred_len = len(pred_str) if pred_str.isdigit() else -1
                true_len = len(str(ex['c']))
                exact = pred_str.isdigit() and int(pred_str) == ex['c']
                results.append({
                    'digits': ex['digits'],
                    'len_correct': pred_len == true_len,
                    'len_off_by': (pred_len - true_len) if pred_len >= 0 else None,
                    'exact': exact,
                })
    return results


def summarize(results):
    by_digits = {}
    for r in results:
        by_digits.setdefault(r['digits'], []).append(r)
    out = {}
    for d, rs in sorted(by_digits.items()):
        n = len(rs)
        len_acc = sum(r['len_correct'] for r in rs) / n
        exact_acc = sum(r['exact'] for r in rs) / n
        offs = [r['len_off_by'] for r in rs if r['len_off_by'] is not None]
        mean_off = sum(offs) / len(offs) if offs else float('nan')
        out[d] = (n, len_acc, exact_acc, mean_off)
    return out


def main():
    out_dirs = sys.argv[1:]
    assert len(out_dirs) >= 1
    splits = ['test_gap', 'test_hint', 'test_ood', 'test_carry_heavy_gap', 'test_carry_heavy_ood']
    dataset_dir = os.path.join('data', 'addition')

    for out_dir in out_dirs:
        print(f"\n########## {out_dir} ##########")
        model, stoi, itos = load_model(out_dir)
        for split in splits:
            path = os.path.join(dataset_dir, f'{split}.txt')
            if not os.path.exists(path):
                continue
            examples = load_examples(path)
            results = eval_length_acc(model, stoi, itos, examples)
            summary = summarize(results)
            print(f"\n=== {split} ===")
            print(f"{'digits':>7} {'n':>6} {'len_acc':>9} {'exact_acc':>10} {'mean_len_off':>13}")
            for d, (n, len_acc, exact_acc, mean_off) in summary.items():
                print(f"{d:>7} {n:>6} {len_acc:>9.3f} {exact_acc:>10.3f} {mean_off:>13.3f}")
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
