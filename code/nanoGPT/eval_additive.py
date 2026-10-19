"""
Evaluate AdditiveComposedGPT: same three numbers as eval_composed.py
(reconstruct-A, reconstruct-B, sum accuracy), adapted for this
architecture's differences -- raw digit VALUES (0-9) instead of vocab
token ids, and parallel fixed-width (max_answer_digits) classification
with an 11th "blank" class instead of autoregressive generation.
Reconstruct accuracy is bucketed by each operand's OWN actual length (not
the pair-max 'digits' convention), for the same reason as eval_composed.py.

Usage:
    python eval_additive.py --out_dir=out-addition-additive --device=cuda
"""
import os
import re
import json
from contextlib import nullcontext

import torch

from model import GPTConfig, AdditiveComposedGPT

out_dir = 'out-addition-additive'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
results_path = None
exec(open('configurator.py').read())
if results_path is None:
    results_path = os.path.join(out_dir, 'eval_additive_results.json')

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = AdditiveComposedGPT(gptconf)
state_dict = checkpoint['model']
for k, v in list(state_dict.items()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

import pickle
with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
plus_id, eq_id = meta['stoi']['+'], meta['stoi']['=']


def digit_values(v, width):
    return [int(ch) for ch in str(v).zfill(width)]


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
            examples.append({'a': a, 'b': b, 'c': c, 'digits': max(len(str(a)), len(str(b)))})
    return examples


def logits_to_str(logits_row):
    """logits_row: (max_answer_digits, n_classes). Returns the predicted
    digit string. codec_zero_pad: n_classes=10, no blank to filter --
    every place is a genuine digit 0-9, leading zeros included, parsed as
    an int downstream the same way int("00042") naturally drops them.
    Otherwise (blank-padded, n_classes=11): skips blank (class 10)
    predictions wherever they occur."""
    preds = torch.argmax(logits_row, dim=-1).tolist()
    if getattr(gptconf, 'codec_zero_pad', False):
        return ''.join(str(p) for p in preds)
    return ''.join(str(p) for p in preds if p != 10)


def _crt_reconstruct(residues, moduli):
    """residues/moduli: same-length int sequences, moduli pairwise
    coprime. Returns the unique integer in [0, product(moduli)) with
    these residues -- deterministic CRT reconstruction, the RNS decode
    counterpart to logits_to_str's digit-string assembly. See model.py's
    GPTConfig.codec_rns_moduli."""
    x, M = 0, 1
    for r, m in zip(residues, moduli):
        t = ((r - x) * pow(M, -1, m)) % m
        x = x + M * t
        M *= m
    return x


def logits_to_val(logits_row):
    """logits_row: (K, max_modulus). codec_type == 'rns' counterpart to
    logits_to_str -- argmax each channel (padding classes beyond that
    channel's own modulus are already masked to -inf by decode_from_
    vector), then CRT-reconstruct the integer value directly (no string
    parsing -- RNS has no digit-string intermediate at all)."""
    residues = torch.argmax(logits_row, dim=-1).tolist()
    return _crt_reconstruct(residues, gptconf.codec_rns_moduli)


@torch.no_grad()
def eval_examples(examples, batch_size=200):
    by_lens = {}
    for ex in examples:
        by_lens.setdefault((len(str(ex['a'])), len(str(ex['b']))), []).append(ex)

    per_example_results = []
    for (len_a, len_b), exs in sorted(by_lens.items()):
        for i in range(0, len(exs), batch_size):
            batch = exs[i:i + batch_size]
            a_ids = torch.tensor([digit_values(ex['a'], len_a) for ex in batch], dtype=torch.long, device=device)
            b_ids = torch.tensor([digit_values(ex['b'], len_b) for ex in batch], dtype=torch.long, device=device)
            with ctx:
                vec_a = model.encode_operand(a_ids)
                vec_b = model.encode_operand(b_ids)
                vec_c, _ = model.combine(vec_a, vec_b, plus_id, eq_id)
                logits_a, _ = model.decode_from_vector(vec_a)
                logits_b, _ = model.decode_from_vector(vec_b)
                logits_c, _ = model.decode_from_vector(vec_c)

            is_rns = getattr(gptconf, 'codec_type', None) == 'rns'
            for j, ex in enumerate(batch):
                if is_rns:
                    pred_a_val = logits_to_val(logits_a[j])
                    pred_b_val = logits_to_val(logits_b[j])
                    pred_c_val = logits_to_val(logits_c[j])
                else:
                    pred_a_str = logits_to_str(logits_a[j])
                    pred_b_str = logits_to_str(logits_b[j])
                    pred_c_str = logits_to_str(logits_c[j])
                    try:
                        pred_a_val = int(pred_a_str) if pred_a_str != '' else None
                    except ValueError:
                        pred_a_val = None
                    try:
                        pred_b_val = int(pred_b_str) if pred_b_str != '' else None
                    except ValueError:
                        pred_b_val = None
                    try:
                        pred_c_val = int(pred_c_str) if pred_c_str != '' else None
                    except ValueError:
                        pred_c_val = None
                per_example_results.append({
                    **ex,
                    'len_a': len_a, 'len_b': len_b,
                    'recon_a_correct': pred_a_val == ex['a'],
                    'recon_b_correct': pred_b_val == ex['b'],
                    'sum_correct': pred_c_val == ex['c'],
                })
    return per_example_results


def summarize(results, key, field):
    buckets = {}
    for r in results:
        buckets.setdefault(r[key], []).append(r[field])
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
        n = len(results)
        recon_a_acc = sum(r['recon_a_correct'] for r in results) / n
        recon_b_acc = sum(r['recon_b_correct'] for r in results) / n
        sum_acc = sum(r['sum_correct'] for r in results) / n
        print(f"\n=== {split} ===")
        print(f"reconstruct-A accuracy: {recon_a_acc:.4f}  by operand-A's own length: {summarize(results, 'len_a', 'recon_a_correct')}")
        print(f"reconstruct-B accuracy: {recon_b_acc:.4f}  by operand-B's own length: {summarize(results, 'len_b', 'recon_b_correct')}")
        print(f"sum accuracy:           {sum_acc:.4f}  by digit count: {summarize(results, 'digits', 'sum_correct')}")
        all_results[split] = {
            'n': n,
            'recon_a_accuracy': recon_a_acc, 'recon_a_by_len_a': summarize(results, 'len_a', 'recon_a_correct'),
            'recon_b_accuracy': recon_b_acc, 'recon_b_by_len_b': summarize(results, 'len_b', 'recon_b_correct'),
            'sum_accuracy': sum_acc, 'sum_by_digits': summarize(results, 'digits', 'sum_correct'),
        }

    os.makedirs(out_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved results to {results_path}")


if __name__ == '__main__':
    main()
