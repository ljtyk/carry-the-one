"""
Re-run of the ComposedGPT-era "is the combiner close to simple addition
(W_a, W_b near identity)" check, for the CURRENT AdditiveComposedGPT combiner
-- a fresh architecture, never checked before. Learned lessons applied from
the original (ComposedGPT) investigation:
  1. n >> parameters (avoid the underdetermined-regression artifact that
     produced a spurious "exactly linear, R^2=1.0" claim with n=500).
  2. Restrict to high-variance ("live") dimensions before fitting/reporting
     weight magnitudes (avoid the collinearity artifact where near-constant
     input dims produce wildly unstable, uninterpretable coefficients).
  3. Use a held-out test split for R^2, not train-set R^2.
"""
import random
import sys

import numpy as np
import torch

from model import GPTConfig, AdditiveComposedGPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
K_LIVE = 24  # number of high-variance dims to keep per space

ckpt = torch.load(f'{out_dir}/ckpt.pt', map_location=device)
gptconf = GPTConfig(**ckpt['model_args'])
model = AdditiveComposedGPT(gptconf)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)
print(f"checkpoint: {out_dir}  iter={ckpt.get('iter_num', '?')}")

import pickle
with open('data/addition_padded/meta.pkl', 'rb') as f:
    m = pickle.load(f)
plus_id, eq_id = m['stoi']['+'], m['stoi']['=']

random.seed(3)
torch.manual_seed(3)


def digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


N_PER_LEN = 90
all_vec_a, all_vec_b, all_vec_c = [], [], []

with torch.no_grad():
    for len_a in range(1, 11):
        for len_b in range(1, 11):
            a_vals = [random.randint(*digit_range(len_a)) for _ in range(N_PER_LEN)]
            b_vals = [random.randint(*digit_range(len_b)) for _ in range(N_PER_LEN)]
            a_ids = torch.tensor([[int(ch) for ch in str(v).zfill(len_a)] for v in a_vals], dtype=torch.long, device=device)
            b_ids = torch.tensor([[int(ch) for ch in str(v).zfill(len_b)] for v in b_vals], dtype=torch.long, device=device)
            vec_a = model.encode_operand(a_ids)
            vec_b = model.encode_operand(b_ids)
            vec_c = model.combine(vec_a, vec_b, plus_id, eq_id)
            all_vec_a.append(vec_a.cpu())
            all_vec_b.append(vec_b.cpu())
            all_vec_c.append(vec_c.cpu())

vec_a = torch.cat(all_vec_a).numpy()
vec_b = torch.cat(all_vec_b).numpy()
vec_c = torch.cat(all_vec_c).numpy()
n = vec_a.shape[0]
print(f"n={n} examples (len_a,len_b each 1..10, {N_PER_LEN} per combo)")

# live dims: highest-variance dims of vec_a (shared encode_operand space --
# check vec_b agrees) and, separately, of vec_c (different space, post-combiner+LN)
var_a = vec_a.var(axis=0)
var_b = vec_b.var(axis=0)
var_c = vec_c.var(axis=0)
live_in = np.argsort(-var_a)[:K_LIVE]
live_in_b_check = np.argsort(-var_b)[:K_LIVE]
live_out = np.argsort(-var_c)[:K_LIVE]
overlap_ab = len(set(live_in.tolist()) & set(live_in_b_check.tolist()))
overlap_in_out = len(set(live_in.tolist()) & set(live_out.tolist()))
print(f"live-dim overlap: vec_a-top{K_LIVE} vs vec_b-top{K_LIVE} = {overlap_ab}/{K_LIVE}")
print(f"live-dim overlap: vec_a-top{K_LIVE} vs vec_c-top{K_LIVE} = {overlap_in_out}/{K_LIVE}")

Xa = vec_a[:, live_in]
Xb = vec_b[:, live_in]
Y = vec_c[:, live_out]
X = np.concatenate([Xa, Xb, np.ones((n, 1))], axis=1)  # bias column

# held-out split
idx = np.random.RandomState(0).permutation(n)
n_train = int(n * 0.8)
train_idx, test_idx = idx[:n_train], idx[n_train:]

W, residuals, rank, sv = np.linalg.lstsq(X[train_idx], Y[train_idx], rcond=None)
pred_test = X[test_idx] @ W
ss_res = ((Y[test_idx] - pred_test) ** 2).sum()
ss_tot = ((Y[test_idx] - Y[test_idx].mean(axis=0)) ** 2).sum()
r2 = 1 - ss_res / ss_tot

W_a = W[:K_LIVE, :]
W_b = W[K_LIVE:2 * K_LIVE, :]

print(f"\nheld-out R^2 (vec_c_live ~ W_a@vec_a_live + W_b@vec_b_live + bias) = {r2:.4f}")
print(f"train samples={len(train_idx)}  test samples={len(test_idx)}  params={W.size}  (n >> params: {len(train_idx) > 5*W.size})")
print(f"mean|W_a| = {np.abs(W_a).mean():.4f}   mean|W_b| = {np.abs(W_b).mean():.4f}")
print(f"||W_a||_F = {np.linalg.norm(W_a):.4f}   ||W_b||_F = {np.linalg.norm(W_b):.4f}")

if overlap_in_out >= K_LIVE // 2:
    # only meaningful if input/output live dims substantially correspond
    common = sorted(set(live_in.tolist()) & set(live_out.tolist()))
    ia = [np.where(live_in == d)[0][0] for d in common]
    io = [np.where(live_out == d)[0][0] for d in common]
    W_a_sq = W_a[np.ix_(ia, io)]
    W_b_sq = W_b[np.ix_(ia, io)]
    I = np.eye(len(common))
    print(f"\non the {len(common)} dims common to both in/out live sets:")
    print(f"||W_a - I||_F = {np.linalg.norm(W_a_sq - I):.4f}   ||W_b - I||_F = {np.linalg.norm(W_b_sq - I):.4f}   (vs ||I||_F={np.linalg.norm(I):.4f})")
else:
    print(f"\ninput/output live-dim sets barely overlap ({overlap_in_out}/{K_LIVE}) -- vec_c does not appear to live in the same coordinate basis as vec_a/vec_b, so an identity-distance comparison isn't meaningful here")
