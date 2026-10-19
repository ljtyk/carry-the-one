"""
Probe C for the v22 AdditiveComposedGPT combiner: linear probes on per-place
token states across combiner rounds.

A+E (probe_combiner_dynamics.py) showed the combiner is a length-proportional
parallel relaxation, not ripple carry, with the chain ENDPOINTS resolving
slowest and a "carry hump" around rounds 15-22 for long inputs. This asks WHAT
each place token represents and WHEN:

  - sum_no_carry  = (da + db) % 10          -- local, no cross-place info
  - carry_in[p]   = carry_out[p-1]  (0/1)
  - carry_out[p]                     (0/1)
  - final_digit[p] = (da + db + carry_in) % 10
  - da[p], db[p]  (the raw operands -- sanity: should be trivially decodable)

For each round t and target, fit a ridge linear probe (standardize -> closed
form -> argmax) on 70% of examples, report test accuracy on the held-out 30%,
broken down by place TYPE:
  chain_origin  (carry_in=0, carry_out=1)   chain_interior (1,1)
  chain_terminus(carry_in=1, carry_out=0)   no_carry       (0,0)
  leading_zero  (place beyond both operands' length)

Also probes the 4 thinking tokens (mean-pooled) for a global signal:
  total carry-out count, and per-round "is the decode already correct".

Relies on model._combiner_trace (hook in AdditiveComposedGPT.combine()).

Usage: python probe_combiner_carry.py [ckpt_dir] [ckpt_name]
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
N_EXAMPLES = 500
RIDGE_LAMBDA = 10.0
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

R = cfg.combiner_repeat_max          # 24
P = cfg.max_answer_digits           # 30
NTHINK = cfg.combiner_n_thinking    # 4
print(f"{ckpt_dir}/{ckpt_name}  iter={ckpt.get('iter_num')}  R={R} P={P}")

with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
PLUS_ID, EQ_ID = meta['stoi']['+'], meta['stoi']['=']
outdir = ckpt_dir


# ---------------------------------------------------------------- example set
def rand_int(L):
    """Random L-digit integer (leading digit 1-9), arbitrary precision."""
    if L <= 0:
        return 0
    s = str(rng.randint(1, 10)) + ''.join(str(rng.randint(0, 10)) for _ in range(L - 1))
    return int(s)


def sample_pair():
    kind = rng.choice(['match', 'mismatch', 'carry9', 'carry9'])
    if kind == 'match':
        L = int(rng.randint(4, 19))
        a, b = rand_int(L), rand_int(L)
    elif kind == 'mismatch':
        a, b = rand_int(int(rng.randint(2, 19))), rand_int(int(rng.randint(2, 19)))
    else:  # run of 9s of random length in the longer operand
        L = int(rng.randint(6, 19))
        clen = int(rng.randint(1, L + 1))
        head = '' if clen == L else str(rng.randint(1, 10)) + \
            ''.join(str(rng.randint(0, 10)) for _ in range(L - clen - 1))
        a = int(head + '9' * clen)
        b = max(rand_int(int(rng.randint(1, L + 1))), 1)
    return int(a), int(b)


examples = [sample_pair() for _ in range(N_EXAMPLES)]


def digit_at(v, p):
    return (v // 10 ** p) % 10


def carry_seq(a, b, width):
    outs, c = [], 0
    for p in range(width):
        s = digit_at(a, p) + digit_at(b, p) + c
        c = s // 10
        outs.append(c)
    return outs


def longest_run(bits):
    best = cur = 0
    for x in bits:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


# ---------------------------------------------------------------- collect states
@torch.no_grad()
def collect():
    groups = {}
    for ei, (a, b) in enumerate(examples):
        groups.setdefault((len(str(a)), len(str(b))), []).append(ei)

    place_feats = np.zeros((R + 1, N_EXAMPLES * P, cfg.n_embd), dtype=np.float32)
    think_feats = np.zeros((R + 1, N_EXAMPLES, cfg.n_embd), dtype=np.float32)
    tgt = {k: np.zeros(N_EXAMPLES * P, dtype=np.int64)
           for k in ['da', 'db', 'carry_in', 'carry_out', 'sum_nc', 'final']}
    place_ex = np.zeros(N_EXAMPLES * P, dtype=np.int64)
    place_idx = np.zeros(N_EXAMPLES * P, dtype=np.int64)
    place_len = np.zeros(N_EXAMPLES * P, dtype=np.int64)     # max operand len
    ex_total_carry = np.zeros(N_EXAMPLES, dtype=np.int64)
    ex_maxchain = np.zeros(N_EXAMPLES, dtype=np.int64)
    ex_solved = np.zeros((R + 1, N_EXAMPLES), dtype=np.int64)

    row = 0
    for (la, lb), eis in groups.items():
        a_vals = [examples[ei][0] for ei in eis]
        b_vals = [examples[ei][1] for ei in eis]
        a_ids = torch.tensor([[int(c) for c in str(v).zfill(la)] for v in a_vals], dtype=torch.long)
        b_ids = torch.tensor([[int(c) for c in str(v).zfill(lb)] for v in b_vals], dtype=torch.long)
        vec_a = model.encode_operand(a_ids)
        vec_b = model.encode_operand(b_ids)
        model._combiner_trace = []
        model.combine(vec_a, vec_b, PLUS_ID, EQ_ID, n_rounds=R)
        states = torch.stack(model._combiner_trace, dim=0)     # (R+1, B, 34, C)
        model._combiner_trace = None
        # per-round decode for the "solved" flag
        T, B = states.shape[:2]
        x = model.comb_ln(states.reshape(T * B, states.shape[2], -1))
        oc = model.combiner_out_slot(x[:, :P, :]).float()
        vc = oc.new_zeros(T * B, cfg.n_embd)
        vc[:, :P * cfg.codec_dims_per_place] = oc.reshape(T * B, P * cfg.codec_dims_per_place)
        dlog, _ = model.decode_from_vector(vc)
        dpred = dlog.argmax(-1).reshape(T, B, P)               # MSD-first

        for j, ei in enumerate(eis):
            a, b = a_vals[j], b_vals[j]
            c = a + b
            cs = carry_seq(a, b, P)
            mlen = max(la, lb)
            ex_total_carry[ei] = sum(cs)
            ex_maxchain[ei] = longest_run(cs)
            for t in range(R + 1):
                pred_int = int(''.join(str(d) for d in dpred[t, j].tolist()))
                ex_solved[t, ei] = int(pred_int == c)
            for p in range(P):
                da, db = digit_at(a, p), digit_at(b, p)
                cin = cs[p - 1] if p > 0 else 0
                cout = cs[p]
                tgt['da'][row] = da
                tgt['db'][row] = db
                tgt['carry_in'][row] = cin
                tgt['carry_out'][row] = cout
                tgt['sum_nc'][row] = (da + db) % 10
                tgt['final'][row] = (da + db + cin) % 10
                place_ex[row] = ei
                place_idx[row] = p
                place_len[row] = mlen
                # place-token state for place p is index p in the sequence
                place_feats[:, row, :] = states[:, j, p, :].numpy()
                row += 1
            think_feats[:, ei, :] = states[:, j, P:, :].mean(axis=1).numpy()
    assert row == N_EXAMPLES * P
    return dict(place_feats=place_feats, think_feats=think_feats, tgt=tgt,
               place_ex=place_ex, place_idx=place_idx, place_len=place_len,
               ex_total_carry=ex_total_carry, ex_maxchain=ex_maxchain,
               ex_solved=ex_solved)


print("collecting combiner states ...")
D = collect()

# place types
cin, cout = D['tgt']['carry_in'], D['tgt']['carry_out']
beyond = D['place_idx'] >= D['place_len']
ptype = np.full(len(cin), 'active', dtype=object)
ptype[(cin == 0) & (cout == 1)] = 'chain_origin'
ptype[(cin == 1) & (cout == 1)] = 'chain_interior'
ptype[(cin == 1) & (cout == 0)] = 'chain_terminus'
ptype[(cin == 0) & (cout == 0)] = 'no_carry'
ptype[beyond] = 'leading_zero'

ex_ids = np.unique(D['place_ex'])
rng.shuffle(ex_ids)
n_tr = int(len(ex_ids) * 0.7)
train_ex = set(ex_ids[:n_tr].tolist())
tr_mask = np.array([e in train_ex for e in D['place_ex']])
te_mask = ~tr_mask


def ridge_probe(Xtr, ytr, Xte, n_classes):
    mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd_
    Xte = (Xte - mu) / sd_
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float32)])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1), np.float32)])
    Y = np.eye(n_classes, dtype=np.float32)[ytr]
    A = Xtr.T @ Xtr + RIDGE_LAMBDA * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Y)
    return (Xte @ W).argmax(1)


TARGETS = {'da': 10, 'db': 10, 'sum_nc': 10, 'carry_in': 2, 'carry_out': 2, 'final': 10}
rounds = list(range(R + 1))
results = {'ckpt': f'{ckpt_dir}/{ckpt_name}', 'iter': ckpt.get('iter_num'),
           'n_examples': N_EXAMPLES, 'place_probe': {}, 'place_probe_by_type': {},
           'baseline': {}}

print("\n[C] place-token linear probes  (test accuracy by round)")
hdr = "round " + ' '.join(f'{t:>6s}' for t in ['da', 'sum_nc', 'cin', 'cout', 'final'])
for name, nc in TARGETS.items():
    y = D['tgt'][name]
    vals, cnts = np.unique(y[te_mask], return_counts=True)
    results['baseline'][name] = float(cnts.max() / cnts.sum())
    accs = []
    by_type = {pt: [] for pt in ['chain_origin', 'chain_interior', 'chain_terminus',
                                 'no_carry', 'leading_zero']}
    for t in rounds:
        Xtr = D['place_feats'][t][tr_mask]
        Xte = D['place_feats'][t][te_mask]
        pred = ridge_probe(Xtr, y[tr_mask], Xte, nc)
        yte = y[te_mask]
        accs.append(float((pred == yte).mean()))
        pt_te = ptype[te_mask]
        for pt in by_type:
            m = pt_te == pt
            by_type[pt].append(float((pred[m] == yte[m]).mean()) if m.any() else None)
    results['place_probe'][name] = accs
    results['place_probe_by_type'][name] = by_type
    print(f"  {name:9s} baseline={results['baseline'][name]:.3f}  "
          f"r0={accs[0]:.3f} r1={accs[1]:.3f} r2={accs[2]:.3f} r4={accs[4]:.3f} "
          f"r8={accs[8]:.3f} r12={accs[12]:.3f} r{R}={accs[R]:.3f}")

# thinking-token probes
print("\n[C-think] thinking-token (mean-pooled) probes")
tr_e = ex_ids[:n_tr]; te_e = ex_ids[n_tr:]
tc = D['ex_total_carry']
# bucket total carry count into classes 0..12+ for a classification probe
tc_cls = np.clip(tc, 0, 12)
think_curve = []
solved_curve = []
for t in rounds:
    Xtr = D['think_feats'][t][tr_e]; Xte = D['think_feats'][t][te_e]
    pred = ridge_probe(Xtr, tc_cls[tr_e], Xte, 13)
    within1 = float((np.abs(pred - tc_cls[te_e]) <= 1).mean())
    exact = float((pred == tc_cls[te_e]).mean())
    think_curve.append({'exact': exact, 'within1': within1})
    # "is the decode already correct" from thinking tokens
    sy = D['ex_solved'][t]
    if len(np.unique(sy[tr_e])) == 2 and len(np.unique(sy[te_e])) >= 1:
        sp = ridge_probe(D['think_feats'][t][tr_e], sy[tr_e], D['think_feats'][t][te_e], 2)
        solved_curve.append(float((sp == sy[te_e]).mean()))
    else:
        solved_curve.append(None)
    print(f"  r{t:2d}: total-carry-count exact={think_curve[-1]['exact']:.2f} "
          f"within1={think_curve[-1]['within1']:.2f}   solved-flag acc={solved_curve[-1]}")
results['think_total_carry'] = think_curve
results['think_solved_flag'] = solved_curve
results['frac_solved_by_round'] = [float(D['ex_solved'][t].mean()) for t in rounds]

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(figsize=(9, 6))
for name in ['da', 'sum_nc', 'carry_in', 'carry_out', 'final']:
    ax.plot(rounds, results['place_probe'][name], marker='.', label=name)
    ax.axhline(results['baseline'][name], color='gray', lw=.5, ls=':')
ax.set_xlabel('combiner round'); ax.set_ylabel('linear-probe test accuracy')
ax.set_title('v22 combiner: what is linearly decodable from a place token, by round')
ax.legend(); ax.set_ylim(0, 1.02); fig.tight_layout()
fig.savefig(os.path.join(outdir, 'dyn_C_place_probes.png'), dpi=110)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, name in zip(axes, ['final', 'carry_in', 'carry_out']):
    for pt, series in results['place_probe_by_type'][name].items():
        s = [np.nan if v is None else v for v in series]
        ax.plot(rounds, s, marker='.', label=pt)
    ax.set_title(f'probe: {name}'); ax.set_xlabel('round'); ax.set_ylim(0, 1.02)
axes[0].set_ylabel('test accuracy'); axes[0].legend(fontsize=8)
fig.suptitle('v22 combiner: probe accuracy by place type (chain endpoints resolve last)')
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'dyn_C_by_placetype.png'), dpi=110)
plt.close(fig)

with open(os.path.join(outdir, 'probe_combiner_carry.json'), 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nsaved -> {os.path.join(outdir, 'probe_combiner_carry.json')}")
print(f"plots -> {outdir}/dyn_C_place_probes.png, dyn_C_by_placetype.png")
