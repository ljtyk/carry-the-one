"""
Push v33 (or any AdditiveComposedGPT slot-codec model) past the test_far_ood
ceiling: on-the-fly problems at digit counts far beyond any test file, bucketed,
natural vs. worst-case-carry (run-of-9s), and optionally with extra combiner
rounds. Finds where length generalization actually breaks.

Hard limits to expect:
  - slot capacity = n_embd // codec_dims_per_place places (384//3 = 128).
  - carry wavefront ~1.85 rounds/place (probe F4), so combiner_repeat_max=80
    covers ~43-digit worst-case carry chains; longer full chains need more
    rounds (v33 holds fixed points, so extra rounds are safe to add).

Usage:
    python eval_additive_limits.py --out_dir=out-addition-additive-v33 \
        --ckpt_name=ckpt_15000.pt --device=cpu
"""
import os
import sys
import json

import numpy as np
import torch

from model import GPTConfig, AdditiveComposedGPT

out_dir = 'out-addition-additive-v33'
ckpt_name = 'ckpt_15000.pt'
device = 'cpu'
n_per_bucket = 200
extra_rounds = 0          # >0: run combiner_repeat_max + this many rounds
exec(open('configurator.py').read())

rng = np.random.RandomState(0)
ckpt = torch.load(os.path.join(out_dir, ckpt_name), map_location=device, weights_only=False)
cfg = GPTConfig(**ckpt['model_args'])
model = AdditiveComposedGPT(cfg)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)

assert cfg.codec_type == 'slot', "this script assumes the slot codec"
CAP = cfg.n_embd // cfg.codec_dims_per_place
BASE_ROUNDS = cfg.combiner_repeat_max
N_ROUNDS = BASE_ROUNDS + extra_rounds
print(f"{out_dir}/{ckpt_name} iter={ckpt.get('iter_num')}  slot cap={CAP} places  "
      f"max_answer_digits={cfg.max_answer_digits}  rounds={N_ROUNDS} (base {BASE_ROUNDS})")

import pickle
with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
PLUS_ID, EQ_ID = meta['stoi']['+'], meta['stoi']['=']


def rand_digits(L):
    if L <= 0:
        return '0'
    return str(rng.randint(1, 10)) + ''.join(str(rng.randint(0, 10)) for _ in range(L - 1))


def natural_pair(d):
    a = int(rand_digits(d))
    b = int(rand_digits(int(rng.randint(1, d + 1))))
    return a, b


def carry_heavy_pair(d):
    """Longer operand = head + run of trailing 9s (carry chain up to full length);
    shorter operand has a nonzero units digit to trigger the carry. Mirrors
    train.py's _composed_sample_carry_heavy_pair."""
    chain = int(rng.randint(max(1, d // 2), d + 1))     # bias toward long chains
    if chain >= d:
        long_s = '9' * d
    else:
        long_s = rand_digits(d - chain) + '9' * chain
    short_len = int(rng.randint(1, d + 1))
    if short_len == 1:
        short_s = str(rng.randint(1, 10))
    else:
        short_s = str(rng.randint(1, 10)) + \
            ''.join(str(rng.randint(0, 10)) for _ in range(short_len - 2)) + str(rng.randint(1, 10))
    return int(long_s), int(short_s)


def digit_values(v, width):
    return [int(c) for c in str(v).zfill(width)]


@torch.no_grad()
def solve(a_vals, b_vals):
    la = max(len(str(v)) for v in a_vals)
    lb = max(len(str(v)) for v in b_vals)
    a_ids = torch.tensor([digit_values(v, la) for v in a_vals], dtype=torch.long, device=device)
    b_ids = torch.tensor([digit_values(v, lb) for v in b_vals], dtype=torch.long, device=device)
    vec_a = model.encode_operand(a_ids)
    vec_b = model.encode_operand(b_ids)
    vec_c, _ = model.combine(vec_a, vec_b, PLUS_ID, EQ_ID, n_rounds=N_ROUNDS)
    logits, _ = model.decode_from_vector(vec_c)          # (B, max_answer_digits, 10)
    preds = logits.argmax(-1)
    return [int(''.join(str(x) for x in row.tolist())) for row in preds]


buckets = [24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 96, 112, 124]
buckets = [d for d in buckets if d <= min(CAP, cfg.max_answer_digits)]

results = {'ckpt': f'{out_dir}/{ckpt_name}', 'iter': ckpt.get('iter_num'),
           'slot_cap': CAP, 'rounds': N_ROUNDS, 'base_rounds': BASE_ROUNDS,
           'n_per_bucket': n_per_bucket, 'natural': {}, 'carry_heavy': {}}

print(f"\n{'digits':>7} | {'natural':>9} | {'carry-heavy':>11}   (n={n_per_bucket} each, exact-match)")
print("-" * 40)
for d in buckets:
    nat = [natural_pair(d) for _ in range(n_per_bucket)]
    chy = [carry_heavy_pair(d) for _ in range(n_per_bucket)]
    for tag, pairs, store in [('natural', nat, results['natural']),
                              ('carry_heavy', chy, results['carry_heavy'])]:
        # group by (len_a, len_b) for batched solve
        groups = {}
        for i, (a, b) in enumerate(pairs):
            groups.setdefault((len(str(a)), len(str(b))), []).append(i)
        correct = 0
        for _, idxs in groups.items():
            av = [pairs[i][0] for i in idxs]
            bv = [pairs[i][1] for i in idxs]
            pred = solve(av, bv)
            correct += sum(int(pred[j] == av[j] + bv[j]) for j in range(len(idxs)))
        store[d] = correct / len(pairs)
    print(f"{d:>7} | {results['natural'][d]:>9.3f} | {results['carry_heavy'][d]:>11.3f}")

# where does each drop below 0.99 / 0.5?
def first_below(store, thr):
    return next((d for d in buckets if store[d] < thr), None)
for tag in ('natural', 'carry_heavy'):
    s = results[tag]
    print(f"{tag}: first <0.99 at {first_below(s, 0.99)} digits, first <0.5 at {first_below(s, 0.5)} digits")
    results[f'{tag}_first_below_0.99'] = first_below(s, 0.99)
    results[f'{tag}_first_below_0.5'] = first_below(s, 0.5)

out_path = os.path.join(out_dir, f'eval_limits_r{N_ROUNDS}.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nsaved -> {out_path}")
