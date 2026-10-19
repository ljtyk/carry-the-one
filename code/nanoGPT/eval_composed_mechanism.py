"""
Probe HOW the (now near-perfect in-distribution) combiner solves addition.

1. Linear probe: is a's numeric value linearly readable from vec_a? (per
   digit-position, and as a scalar). Tests whether the encoder settled on
   a "clean", roughly linear/place-value-like representation, or something
   more entangled that still happens to be exactly invertible via the
   nonlinear decoder.
2. Combiner linearity: is vec_c well-approximated by an affine function of
   vec_a and vec_b, i.e. vec_c ~= W_a @ vec_a + W_b @ vec_b + bias? If so,
   the combiner (despite being a full attention+MLP block) has essentially
   learned to implement addition as a LINEAR operation in vector space.
3. Cross-composition: mix vec_a from example i with vec_b from a
   DIFFERENT, unrelated example j (never jointly seen this way), and check
   whether decode(combine(vec_a_i, vec_b_j)) == a_i + b_j. Tests genuine
   compositionality (a general two-argument function) vs. some kind of
   pair-specific shortcut.

Usage:
    python eval_composed_mechanism.py --out_dir=out-addition-composed-300k --digits=6 --n=500
"""
import os
import pickle
import random
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, ComposedGPT

out_dir = 'out-addition-composed-300k'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
encoder_position_scheme = ''  # override for checkpoints predating this flag (saved model_args lack it) -- 'sequential' or 'place_value'
digits = 6
n = 500
seed = 0
exec(open('configurator.py').read())

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
    out = []
    for t in row:
        if t == nl_id:
            break
        out.append(itos[t])
    return ''.join(out)


rng = random.Random(seed)
lo, hi = 10 ** (digits - 1), 10 ** digits - 1
a_vals = [rng.randint(lo, hi) for _ in range(n)]
b_vals = [rng.randint(lo, hi) for _ in range(n)]

a_ids = torch.tensor([to_ids(a, digits) for a in a_vals], dtype=torch.long, device=device)
b_ids = torch.tensor([to_ids(b, digits) for b in b_vals], dtype=torch.long, device=device)
with torch.no_grad(), ctx:
    vec_a = model.encode_operand(a_ids).float()
    vec_b = model.encode_operand(b_ids).float()
    vec_c = model.combine(vec_a.to(ptdtype), vec_b.to(ptdtype), plus_id, eq_id).float()

vec_a_np = vec_a.cpu().numpy()
vec_b_np = vec_b.cpu().numpy()
vec_c_np = vec_c.cpu().numpy()


def r2_score(Y, Y_pred):
    ss_res = np.sum((Y - Y_pred) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2, axis=0)
    return 1 - ss_res / np.maximum(ss_tot, 1e-8)


print("=" * 60)
print("1. LINEAR PROBE: is a's value linearly readable from vec_a?")
print("=" * 60)
digit_matrix = np.array([[int(ch) for ch in str(a).zfill(digits)] for a in a_vals], dtype=np.float64)
X_aug = np.hstack([vec_a_np, np.ones((n, 1))])
W, _, _, _ = np.linalg.lstsq(X_aug, digit_matrix, rcond=None)
pred = X_aug @ W
r2_per_digit = r2_score(digit_matrix, pred)
print(f"per-digit-position R² (position 0 = most significant): {np.round(r2_per_digit, 4).tolist()}")

log_a = np.log10(np.array(a_vals, dtype=np.float64)).reshape(-1, 1)
W2, _, _, _ = np.linalg.lstsq(X_aug, log_a, rcond=None)
r2_log = r2_score(log_a, X_aug @ W2)[0]
print(f"R² for log10(a) as a single linear readout: {r2_log:.4f}")

print()
print("=" * 60)
print("2. COMBINER LINEARITY: vec_c ~= W_a @ vec_a + W_b @ vec_b + bias ?")
print("=" * 60)
Xc = np.hstack([vec_a_np, vec_b_np, np.ones((n, 1))])
Wc, _, _, _ = np.linalg.lstsq(Xc, vec_c_np, rcond=None)
vec_c_pred = Xc @ Wc
r2_combiner = r2_score(vec_c_np, vec_c_pred)
print(f"mean R² across the {vec_c_np.shape[1]} output dims: {r2_combiner.mean():.4f}")
print(f"min/max per-dim R²: {r2_combiner.min():.4f} / {r2_combiner.max():.4f}")

# also check the naive hypothesis vec_c ~= vec_a + vec_b directly (no learned weights)
naive_pred = vec_a_np + vec_b_np
r2_naive = r2_score(vec_c_np, naive_pred)
print(f"naive vec_c ~= vec_a + vec_b (no fitting): mean R² = {r2_naive.mean():.4f}")

n_embd_dim = vec_a_np.shape[1]
W_a = Wc[:n_embd_dim, :]
W_b = Wc[n_embd_dim:2 * n_embd_dim, :]
bias_c = Wc[-1, :]
I_mat = np.eye(n_embd_dim)


def identity_report(name, W):
    diag = np.diag(W)
    off_diag_rms = np.sqrt((np.sum(W ** 2) - np.sum(diag ** 2)) / (W.size - len(diag)))
    dist_from_I = np.linalg.norm(W - I_mat, 'fro') / np.linalg.norm(I_mat, 'fro')
    print(f"{name}: diagonal mean={diag.mean():.3f} (std={diag.std():.3f}), "
          f"off-diagonal RMS={off_diag_rms:.4f}, ||{name}-I||/||I|| = {dist_from_I:.3f}")


print()
print("is the combiner's linear map close to the identity (i.e. literally vec_a + vec_b)?")
identity_report("W_a", W_a)
identity_report("W_b", W_b)
print(f"bias term: ||bias|| = {np.linalg.norm(bias_c):.3f}  (compare to ||vec_a|| mean = {np.linalg.norm(vec_a_np, axis=1).mean():.3f})")

print()
print("=" * 60)
print("2b. CODEC CONSISTENCY: is vec_c actually close to encode(true_sum)?")
print("=" * 60)
c_vals = [a + b for a, b in zip(a_vals, b_vals)]
sum_width = max(len(str(c)) for c in c_vals)
sum_ids = torch.tensor([to_ids(c, sum_width) for c in c_vals], dtype=torch.long, device=device)
with torch.no_grad(), ctx:
    vec_sum_true = model.encode_operand(sum_ids).float()
vec_sum_true_np = vec_sum_true.cpu().numpy()
mse = np.mean((vec_c_np - vec_sum_true_np) ** 2)
cos_sim = np.sum(vec_c_np * vec_sum_true_np, axis=1) / (
    np.linalg.norm(vec_c_np, axis=1) * np.linalg.norm(vec_sum_true_np, axis=1) + 1e-8)
r2_consistency = r2_score(vec_sum_true_np, vec_c_np)
print(f"MSE(vec_c, encode(true_sum)) = {mse:.4f}   (compare to typical ||vec||^2/384 = "
      f"{(np.linalg.norm(vec_a_np, axis=1).mean() ** 2 / n_embd_dim):.4f})")
print(f"mean cosine similarity = {cos_sim.mean():.4f} (std {cos_sim.std():.4f})")
print(f"R²(encode(true_sum) ~ vec_c, no fitting -- are they literally the same point?) mean = {r2_consistency.mean():.4f}")

print()
print("=" * 60)
print("3. CROSS-COMPOSITION: mix vec_a[i] with vec_b[j], i != j (never jointly trained on)")
print("=" * 60)
perm = list(range(n))
rng.shuffle(perm)
# ensure no fixed points (i == perm[i])
for i in range(n):
    if perm[i] == i:
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
vec_b_shuffled = vec_b[perm]
b_vals_shuffled = [b_vals[p] for p in perm]

with torch.no_grad(), ctx:
    vec_c_cross = model.combine(vec_a.to(ptdtype), vec_b_shuffled.to(ptdtype), plus_id, eq_id)
gen_cross = greedy_decode_from_vector(vec_c_cross, digits + 4)

n_correct = 0
examples_shown = 0
for i in range(n):
    pred_str = decode_str(gen_cross[i].tolist())
    true_c = a_vals[i] + b_vals_shuffled[i]
    correct = pred_str == str(true_c)
    n_correct += correct
    if examples_shown < 5:
        print(f"  a={a_vals[i]} (orig paired with b={b_vals[i]}), b_swapped_in={b_vals_shuffled[i]}"
              f"  true={true_c}  pred={pred_str}  {'OK' if correct else 'WRONG'}")
        examples_shown += 1
print(f"cross-composition accuracy: {n_correct}/{n} = {n_correct/n:.4f}")

print()
print("=" * 60)
print("4. PACKING EFFICIENCY: how many of the 384 dims is vec_a actually using?")
print("=" * 60)


def effective_rank_report(name, X):
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    var = s ** 2
    total = var.sum()
    cum = np.cumsum(var) / total
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    n999 = int(np.searchsorted(cum, 0.999) + 1)
    participation_ratio = (var.sum() ** 2) / (var ** 2).sum()
    print(f"{name}: dims for 90%/99%/99.9% variance = {n90}/{n99}/{n999}  "
          f"(participation-ratio effective dim = {participation_ratio:.1f}, of {X.shape[1]} total)")
    return s


s_a = effective_rank_report("vec_a", vec_a_np)
s_b = effective_rank_report("vec_b", vec_b_np)
s_c = effective_rank_report("vec_c", vec_c_np)

print()
print(f"(for reference: {digits} digits at ~3.32 bits/digit needs only ~{digits*3.32:.1f} bits of "
      f"content, i.e. in principle a handful of well-separated directions would suffice)")

print()
print("per-digit linear-readout weight concentration (how many of the 384 dims carry")
print("90% of that digit's readout weight L2 norm -- low = sparse/dedicated code, high = distributed):")
W_no_bias = W[:-1, :]  # drop the bias row, shape (384, digits)
for d in range(digits):
    w = np.abs(W_no_bias[:, d])
    order = np.argsort(-w)
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted ** 2) / np.sum(w_sorted ** 2)
    n90_w = int(np.searchsorted(cum_w, 0.90) + 1)
    print(f"  digit position {d}: {n90_w} dims carry 90% of readout weight")
