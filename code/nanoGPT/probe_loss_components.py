"""
Diagnostic: the v2 additive run's total loss exploded into the thousands
while "val loss (robust)" (which only recomputes sum-decode CE) stayed flat
around 0.73. This loads the live checkpoint and recomputes the THREE loss
components (sum CE, recon CE, consistency MSE) separately on a fresh batch,
matching AdditiveComposedGPT.forward()'s exact composition, to confirm which
term is actually exploding.
"""
import random
import sys

import torch
import torch.nn.functional as F

from model import GPTConfig, AdditiveComposedGPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v2'
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
print(f"loaded checkpoint from iter {ckpt.get('iter_num', '?')}")

random.seed(1)
torch.manual_seed(1)


def _composed_digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


def sample_pair(len_a, len_b):
    lo_a, hi_a = _composed_digit_range(len_a)
    lo_b, hi_b = _composed_digit_range(len_b)
    return random.randint(lo_a, hi_a), random.randint(lo_b, hi_b)


def sample_skewed_length(max_digits, power=2.0):
    u = random.random()
    return max_digits - int((max_digits - 1) * (u ** power))


batch_size = 64
max_answer_digits = gptconf.max_answer_digits
len_a = sample_skewed_length(10)
len_b = sample_skewed_length(10)
a_vals, b_vals = [], []
for _ in range(batch_size):
    a, b = sample_pair(len_a, len_b)
    a_vals.append(a)
    b_vals.append(b)
c_vals = [a + b for a, b in zip(a_vals, b_vals)]


def digit_values(v, width):
    return [int(ch) for ch in str(v).zfill(width)]


def right_justified_targets(vals, width):
    t = torch.full((batch_size, width), 10, dtype=torch.long)
    for i, v in enumerate(vals):
        s = str(v)
        L = len(s)
        t[i, width - L:] = torch.tensor([int(ch) for ch in s], dtype=torch.long)
    return t


a_ids = torch.tensor([digit_values(a, len_a) for a in a_vals], dtype=torch.long, device=device)
b_ids = torch.tensor([digit_values(b, len_b) for b in b_vals], dtype=torch.long, device=device)
a_targets = right_justified_targets(a_vals, max_answer_digits).to(device)
b_targets = right_justified_targets(b_vals, max_answer_digits).to(device)
dec_y = right_justified_targets(c_vals, max_answer_digits).to(device)

with open('data/addition_padded/meta.pkl', 'rb') as f:
    import pickle
    m = pickle.load(f)
plus_id, eq_id = m['stoi']['+'], m['stoi']['=']

with torch.no_grad():
    vec_a = model.encode_operand(a_ids)
    vec_b = model.encode_operand(b_ids)
    vec_c = model.combine(vec_a, vec_b, plus_id, eq_id)

    logits_c, sum_ce = model.decode_from_vector(vec_c, dec_y)
    _, recon_a = model.decode_from_vector(vec_a, a_targets)
    _, recon_b = model.decode_from_vector(vec_b, b_targets)

    by_len = {}
    for i, v in enumerate(c_vals):
        by_len.setdefault(len(str(v)), []).append(i)
    vec_sum_target = torch.zeros_like(vec_c)
    for L, idxs in by_len.items():
        digit_tensor = torch.tensor([[int(ch) for ch in str(c_vals[i])] for i in idxs], dtype=torch.long, device=device)
        vec_sum_target[idxs] = model.encode_operand(digit_tensor)
    consistency_mse = F.mse_loss(vec_c, vec_sum_target)

    print(f"len_a={len_a} len_b={len_b}")
    print(f"||vec_a|| mean={vec_a.norm(dim=-1).mean().item():.3f}  ||vec_b|| mean={vec_b.norm(dim=-1).mean().item():.3f}")
    print(f"||vec_c|| mean={vec_c.norm(dim=-1).mean().item():.3f}  ||vec_sum_target|| mean={vec_sum_target.norm(dim=-1).mean().item():.3f}")
    print(f"||vec_c - vec_sum_target|| mean={(vec_c - vec_sum_target).norm(dim=-1).mean().item():.3f}")
    print()
    print(f"sum_ce         = {sum_ce.item():.4f}")
    print(f"recon_a        = {recon_a.item():.4f}")
    print(f"recon_b        = {recon_b.item():.4f}")
    print(f"consistency_mse= {consistency_mse.item():.4f}")
    print()
    recon_weight = gptconf.composed_recon_weight
    cons_weight = gptconf.composed_consistency_weight
    total = sum_ce + recon_weight * (recon_a + recon_b) + cons_weight * consistency_mse
    print(f"weights: recon={recon_weight} consistency={cons_weight}")
    print(f"reconstructed total loss = {total.item():.4f}")
    print(f"  sum_ce contributes            {sum_ce.item():.2f}")
    print(f"  recon contributes             {(recon_weight*(recon_a+recon_b)).item():.2f}")
    print(f"  consistency contributes       {(cons_weight*consistency_mse).item():.2f}")
