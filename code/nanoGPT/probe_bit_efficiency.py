"""
Bit-efficiency check for vec_a/vec_b (encoder output) and, newly, vec_c
(combiner output): for a given place, how many of the 384 dimensions
actually carry that place's digit-value information? Measured as
between-group variance -- group samples by the true digit value at that
place (0-9), compute the per-dimension variance ACROSS the 10 group means,
and count how many dims (sorted descending) are needed to capture 95% of
the cumulative variance. Small count = compact/modular code; large count =
distributed/holographic code (spread across most of the 384 dims).

For vec_a: places are isolated directly via combine_fn(digit,place) terms
(no other digits present, so grouping is exact). For vec_c: samples full
(a,b) pairs at a fixed length, group by the TRUE SUM's digit at a given
place -- vec_c isn't decomposable into isolated per-digit terms the way
vec_a is, so this ANOVA-style grouping is the natural analogous measure.
"""
import random
import sys

import numpy as np
import torch

from model import GPTConfig, AdditiveComposedGPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ckpt = torch.load(f'{out_dir}/ckpt.pt', map_location=device)
gptconf = GPTConfig(**ckpt['model_args'])
model = AdditiveComposedGPT(gptconf)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)
print(f"checkpoint: {out_dir}  iter={ckpt.get('iter_num', '?')}\n")

import pickle
with open('data/addition_padded/meta.pkl', 'rb') as f:
    m = pickle.load(f)
plus_id, eq_id = m['stoi']['+'], m['stoi']['=']

random.seed(4)
torch.manual_seed(4)


def live_dim_count(group_means, threshold=0.95):
    """group_means: (10, C). Returns dims needed (sorted desc by per-dim
    between-group variance) to reach `threshold` of cumulative variance."""
    var = group_means.var(axis=0)  # (C,)
    order = np.argsort(-var)
    cum = np.cumsum(var[order])
    cum = cum / cum[-1]
    return int(np.searchsorted(cum, threshold) + 1), var


def term(digit, place):
    d = torch.tensor([[digit]], device=device)
    p = torch.tensor([place], device=device)
    d_emb = model.digit_embed(d)
    place_sig = model.place_signal[p].unsqueeze(0)
    combined = torch.cat([d_emb, place_sig], dim=-1)
    return model.combine_fn(combined).squeeze(0).squeeze(0)


print("=== vec_a (encoder): bit efficiency per place ===")
with torch.no_grad():
    for place in [0, 4, 9]:
        terms = torch.stack([term(d, place) for d in range(10)]).cpu().numpy()  # (10, C)
        n_live, var = live_dim_count(terms)
        print(f"place={place}: {n_live}/384 dims needed for 95% of variance  (top dim var={var.max():.4f})")

print("\n=== vec_c (combiner): bit efficiency per sum-digit place ===")


def digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


LEN_A = LEN_B = 6
N_PER_DIGIT = 300

with torch.no_grad():
    for k_from_end in [0, 3, 5]:  # ones, mid, near-MSD of a ~6-7 digit sum
        vecs_by_digit = {d: [] for d in range(10)}
        attempts = 0
        while min(len(v) for v in vecs_by_digit.values()) < N_PER_DIGIT and attempts < 200000:
            attempts += 1
            batch = 200
            a_vals = [random.randint(*digit_range(LEN_A)) for _ in range(batch)]
            b_vals = [random.randint(*digit_range(LEN_B)) for _ in range(batch)]
            c_vals = [a + b for a, b in zip(a_vals, b_vals)]
            a_ids = torch.tensor([[int(ch) for ch in str(v).zfill(LEN_A)] for v in a_vals], dtype=torch.long, device=device)
            b_ids = torch.tensor([[int(ch) for ch in str(v).zfill(LEN_B)] for v in b_vals], dtype=torch.long, device=device)
            vec_a = model.encode_operand(a_ids)
            vec_b = model.encode_operand(b_ids)
            vec_c, _ = model.combine(vec_a, vec_b, plus_id, eq_id)
            vec_c = vec_c.cpu().numpy()
            for i, c in enumerate(c_vals):
                s = str(c)
                if k_from_end >= len(s):
                    continue
                dig = int(s[len(s) - 1 - k_from_end])
                if len(vecs_by_digit[dig]) < N_PER_DIGIT:
                    vecs_by_digit[dig].append(vec_c[i])

        counts = [len(v) for v in vecs_by_digit.values()]
        group_means = np.stack([np.mean(vecs_by_digit[d], axis=0) for d in range(10)])
        n_live, var = live_dim_count(group_means)
        print(f"sum place k={k_from_end} (from end): {n_live}/384 dims needed for 95% of variance  "
              f"(top dim var={var.max():.4f}, samples/digit min={min(counts)})")
