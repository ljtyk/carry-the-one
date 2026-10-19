"""
Evaluate ComposedGPT: reports THREE separate numbers per split --
reconstruct-A accuracy (decode(encode(a)) == a, bypassing the combiner
entirely), reconstruct-B accuracy, and sum accuracy (decode(combine(encode(a),
encode(b))) == a+b). Comparing reconstruction vs. sum accuracy isolates
whether a failure is an encoder/decoder fidelity problem (reconstruction
itself breaks) or a combiner/addition-function problem (reconstruction holds
but the sum doesn't) -- see model.py's ComposedGPT docstring.

Usage:
    python eval_composed.py --out_dir=out-addition-composed --device=cuda
"""
import os
import re
import json
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, ComposedGPT

out_dir = 'out-addition-composed'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
encoder_position_scheme = ''  # override for checkpoints predating this flag (saved model_args lack it) -- 'sequential' or 'place_value'
results_path = None
exec(open('configurator.py').read())
if results_path is None:
    results_path = os.path.join(out_dir, 'eval_composed_results.json')

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
if encoder_position_scheme:
    model_args['encoder_position_scheme'] = encoder_position_scheme
gptconf = GPTConfig(**model_args)
model = ComposedGPT(gptconf)
state_dict = checkpoint['model']
for k, v in list(state_dict.items()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
digit_ids = [stoi[str(i)] for i in range(10)]
nl_id = stoi['\n']
plus_id, eq_id = stoi['+'], stoi['=']


def to_ids(v, width):
    return [digit_ids[int(ch)] for ch in str(v).zfill(width)]


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


@torch.no_grad()
def greedy_decode_from_vector(memory_vec, max_new_tokens):
    B = memory_vec.size(0)
    dec_x = torch.zeros((B, 0), dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_new_tokens):
        with ctx:
            logits, _ = model.decode_from_vector(memory_vec, dec_x)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        dec_x = torch.cat([dec_x, next_tok], dim=1)
        finished = finished | (next_tok.squeeze(1) == nl_id)
        if finished.all():
            break
    return dec_x


def decode_str(row):
    """row: list of token ids. Cuts at the first '\\n' (exclusive)."""
    out = []
    for t in row:
        if t == nl_id:
            break
        out.append(itos[t])
    return ''.join(out)


@torch.no_grad()
def eval_examples(examples, batch_size=200):
    by_lens = {}
    for ex in examples:
        by_lens.setdefault((len(str(ex['a'])), len(str(ex['b']))), []).append(ex)

    per_example_results = []
    for (len_a, len_b), exs in sorted(by_lens.items()):
        for i in range(0, len(exs), batch_size):
            batch = exs[i:i + batch_size]
            a_ids = torch.tensor([to_ids(ex['a'], len_a) for ex in batch], dtype=torch.long, device=device)
            b_ids = torch.tensor([to_ids(ex['b'], len_b) for ex in batch], dtype=torch.long, device=device)
            with ctx:
                vec_a = model.encode_operand(a_ids)
                vec_b = model.encode_operand(b_ids)
                vec_c = model.combine(vec_a, vec_b, plus_id, eq_id)

            max_c_new = max(ex['digits'] for ex in batch) + 3
            gen_a = greedy_decode_from_vector(vec_a, len_a + 3)
            gen_b = greedy_decode_from_vector(vec_b, len_b + 3)
            gen_c = greedy_decode_from_vector(vec_c, max_c_new)

            for j, ex in enumerate(batch):
                pred_a_str = decode_str(gen_a[j].tolist())
                pred_b_str = decode_str(gen_b[j].tolist())
                pred_c_str = decode_str(gen_c[j].tolist())
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
        # recon accuracy is bucketed by each operand's OWN actual length (len_a/len_b),
        # NOT by 'digits' (= max(len_a,len_b), the pair-difficulty convention) -- a
        # 'digits'-bucketed average blends together examples where the reconstructed
        # operand is actually within the trained 1-10 digit range (trivially ~100%)
        # with ones where it's genuinely out-of-range, which inflates the apparent
        # OOD reconstruction accuracy and (worse) makes the composite number mostly a
        # property of the test file's length-mixture rather than of the model. See
        # project discussion -- this masked a real difference between two checkpoints
        # that turned out to have identical composite numbers for exactly this reason.
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
