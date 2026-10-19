"""
Probe D for the v22 AdditiveComposedGPT combiner: attention structure in the
shared (24x weight-tied) block, round by round.

A+E+C established a carry-lookahead / parallel-prefix algorithm: each place
computes local sum + carry-out in round ~1, then a carry wavefront resolves
carry-in over rounds ~2-10, chain endpoints last. D asks HOW information moves:
is the wavefront nearest-neighbour (place p <- place p-1, where its carry-in
comes from), longer-range lookahead, or global? Do the 4 thinking tokens act as
broadcast hubs?

Needs CausalSelfAttention._attn_store (hook in model.py): set it to a list and
the module takes the manual attention path and appends post-softmax weights
(B, n_head, T, T) every call. The combiner block calls attn once per round, so
with n_rounds=24 the list holds 24 maps.

Sequence layout (T=34): indices 0..29 = place tokens (index p = decimal place p,
units=0), indices 30..33 = thinking tokens.

Usage: python probe_combiner_attn.py [ckpt_dir] [ckpt_name]
"""
import os
import sys
import json
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import GPTConfig, AdditiveComposedGPT

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v22'
ckpt_name = sys.argv[2] if len(sys.argv) > 2 else 'ckpt.pt'
device = 'cpu'
rng = np.random.RandomState(0)

ckpt = torch.load(os.path.join(ckpt_dir, ckpt_name), map_location=device, weights_only=False)
cfg = GPTConfig(**ckpt['model_args'])
model = AdditiveComposedGPT(cfg)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)

R = cfg.combiner_repeat_max
P = cfg.max_answer_digits          # 30
NT = cfg.combiner_n_thinking       # 4
T = P + NT                         # 34
NH = cfg.n_head
print(f"{ckpt_dir}/{ckpt_name} iter={ckpt.get('iter_num')}  R={R} P={P} n_think={NT} T={T} heads={NH}")

with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
PLUS_ID, EQ_ID = meta['stoi']['+'], meta['stoi']['=']
outdir = ckpt_dir


def rand_int(L):
    if L <= 0:
        return 0
    return int(str(rng.randint(1, 10)) + ''.join(str(rng.randint(0, 10)) for _ in range(L - 1)))


def make_bucket(kind, L, n):
    a_vals, b_vals = [], []
    for _ in range(n):
        if kind == 'nines':
            clen = int(rng.randint(1, L + 1))
            head = '' if clen == L else str(rng.randint(1, 10)) + \
                ''.join(str(rng.randint(0, 10)) for _ in range(L - clen - 1))
            a_vals.append(int(head + '9' * clen))
            b_vals.append(max(rand_int(int(rng.randint(1, L + 1))), 1))
        else:  # random same length
            a_vals.append(rand_int(L))
            b_vals.append(rand_int(L))
    return a_vals, b_vals


@torch.no_grad()
def capture(a_vals, b_vals):
    la = max(len(str(v)) for v in a_vals)
    lb = max(len(str(v)) for v in b_vals)
    a_ids = torch.tensor([[int(c) for c in str(v).zfill(la)] for v in a_vals], dtype=torch.long)
    b_ids = torch.tensor([[int(c) for c in str(v).zfill(lb)] for v in b_vals], dtype=torch.long)
    vec_a = model.encode_operand(a_ids)
    vec_b = model.encode_operand(b_ids)
    model.combiner_block.attn._attn_store = []
    model.combine(vec_a, vec_b, PLUS_ID, EQ_ID, n_rounds=R)
    maps = torch.stack(model.combiner_block.attn._attn_store, dim=0)  # (R, B, NH, T, T)
    model.combiner_block.attn._attn_store = None
    return maps


buckets = {}
for kind in ['rand', 'nines']:
    for L in [6, 10, 16, 20]:
        buckets[f'{kind}{L}'] = (make_bucket(kind, L, 24), L)

results = {'ckpt': f'{ckpt_dir}/{ckpt_name}', 'iter': ckpt.get('iter_num'), 'buckets': {}}

# aggregate offset profile + thinking usage per bucket
off_bins = np.arange(-(P - 1), P)      # key_place - query_place
agg = {}
for name, ((a_vals, b_vals), L) in buckets.items():
    maps = capture(a_vals, b_vals)                       # (R,B,NH,T,T)
    m = maps.mean(dim=1)                                 # (R,NH,T,T) mean over batch
    m_h = m.mean(dim=1)                                  # (R,T,T) mean over heads too

    # offset profile among ACTIVE place tokens (query,key in 0..L)
    off_profile = np.zeros((R, len(off_bins)))
    for r in range(R):
        A = m_h[r, :L + 1, :L + 1].numpy()               # (L+1,L+1)
        for qi in range(L + 1):
            for ki in range(L + 1):
                off_profile[r, (ki - qi) + (P - 1)] += A[qi, ki]
        off_profile[r] /= (L + 1)                         # per-query-token mass
    # thinking-token traffic
    place_to_think = m_h[:, :L + 1, P:].sum(dim=-1).mean(dim=-1).numpy()   # (R,)
    think_to_place = m_h[:, P:, :L + 1].sum(dim=-1).mean(dim=-1).numpy()   # (R,)
    place_to_self = np.array([np.diag(m_h[r, :L + 1, :L + 1].numpy()).mean() for r in range(R)])
    # net directional flow among place tokens: mass toward MORE significant
    # (key>query, offset>0) minus toward LESS significant (key<query)
    up = off_profile[:, off_bins > 0].sum(axis=1)
    down = off_profile[:, off_bins < 0].sum(axis=1)

    agg[name] = dict(off_profile=off_profile, place_to_think=place_to_think,
                     think_to_place=think_to_place, place_to_self=place_to_self,
                     up=up, down=down, L=L)
    results['buckets'][name] = {
        'L': L,
        'place->think_by_round': place_to_think.tolist(),
        'think->place_by_round': think_to_place.tolist(),
        'self_attn_by_round': place_to_self.tolist(),
        'flow_up_by_round': up.tolist(), 'flow_down_by_round': down.tolist(),
        'offset_profile_round6': off_profile[6].tolist(),
        'offset_profile_round16': off_profile[16].tolist(),
        'offset_bins': off_bins.tolist(),
    }
    print(f"  {name:8s} L={L:2d}  r6: down(-)={down[6]:.3f} up(+)={up[6]:.3f} "
          f"self={place_to_self[6]:.3f} ->think={place_to_think[6]:.3f}   "
          f"peak-offset r6={off_bins[off_profile[6].argmax()]:+d}")

# ---- plots -------------------------------------------------------------
# 1. attention heatmaps, rand16, selected rounds
name = 'rand16'
maps = capture(*buckets[name][0])
mh = maps.mean(dim=1).mean(dim=1)   # (R,T,T)
sel = [1, 2, 4, 8, 16, 23]
fig, axes = plt.subplots(1, len(sel), figsize=(3 * len(sel), 3.2))
for ax, r in zip(axes, sel):
    ax.imshow(mh[r].numpy(), cmap='magma', vmin=0, vmax=float(mh[sel].max()) * 0.6)
    ax.set_title(f'round {r}'); ax.set_xlabel('key idx');
    ax.axhline(P - .5, color='c', lw=.5); ax.axvline(P - .5, color='c', lw=.5)
axes[0].set_ylabel('query idx')
fig.suptitle(f'v22 combiner attention (mean over heads+batch) — {name} '
             f'(cyan line = place|thinking boundary at {P})')
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'dyn_D_attn_maps.png'), dpi=110)
plt.close(fig)

# 2. offset profile vs round (rand16 + nines16), zoomed to |offset|<=8
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, bn in zip(axes, ['rand16', 'nines16']):
    op = agg[bn]['off_profile']
    lo, hi = (P - 1) - 8, (P - 1) + 9
    im = ax.imshow(op[:, lo:hi], aspect='auto', cmap='viridis',
                   extent=[-8, 8, R, 0])
    ax.set_title(f'{bn}: attention offset (key_place - query_place) vs round')
    ax.set_xlabel('offset  (<0 = toward less-significant / carry source)')
    ax.set_ylabel('round'); ax.axvline(0, color='w', lw=.5)
    fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'dyn_D_offset_vs_round.png'), dpi=110)
plt.close(fig)

# 3. thinking-token traffic + directional flow vs round
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for bn in ['rand6', 'rand10', 'rand16', 'rand20', 'nines16', 'nines20']:
    a = agg[bn]
    axes[0].plot(range(R), a['place_to_think'], label=bn)
    axes[1].plot(range(R), a['think_to_place'], label=bn)
    axes[2].plot(range(R), a['down'], label=bn + ' down(-)')
axes[0].set_title('place -> thinking (total mass)'); axes[0].set_xlabel('round')
axes[1].set_title('thinking -> place (mean per thinking token)'); axes[1].set_xlabel('round')
axes[2].set_title('place->place mass toward less-significant (carry source)'); axes[2].set_xlabel('round')
for ax in axes:
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'dyn_D_thinking_traffic.png'), dpi=110)
plt.close(fig)

# 4. per-head offset signature at round 6, rand16
maps = capture(*buckets['rand16'][0])
L = 16
perhead = maps.mean(dim=1)  # (R,NH,T,T)
fig, axes = plt.subplots(1, NH, figsize=(3 * NH, 3), sharey=True)
xb = np.arange(-8, 9)
for h in range(NH):
    A = perhead[6, h, :L + 1, :L + 1].numpy()
    prof = np.zeros(len(xb))
    for qi in range(L + 1):
        for ki in range(L + 1):
            o = ki - qi
            if -8 <= o <= 8:
                prof[o + 8] += A[qi, ki]
    prof /= (L + 1)
    axes[h].bar(xb, prof); axes[h].set_title(f'head {h}'); axes[h].set_xlabel('offset')
axes[0].set_ylabel('attn mass / query')
fig.suptitle('v22 combiner: per-head place-place attention offset, round 6, rand16')
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'dyn_D_per_head.png'), dpi=110)
plt.close(fig)

with open(os.path.join(outdir, 'probe_combiner_attn.json'), 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nsaved -> {os.path.join(outdir, 'probe_combiner_attn.json')}")
print(f"plots -> {outdir}/dyn_D_attn_maps.png, dyn_D_offset_vs_round.png, "
      f"dyn_D_thinking_traffic.png, dyn_D_per_head.png")
