"""
Probe A + E for the v22 AdditiveComposedGPT combiner (the project's best model).

The combiner is a single shared transformer block iterated combiner_repeat_max
(=24) times over 30 per-place tokens + 4 thinking tokens. Everything else in the
model (slot codec encode/decode) is a near-fixed numeral code -- the combiner is
where the arithmetic actually happens. This script asks two things:

  A. Per-round decode / carry-front tracking. Run the combiner for n_rounds =
     1..24, decode after each round. Ripple carry predicts the answer solidifies
     from the least-significant place upward, ~one place per round, and on a
     run-of-9s example you should literally watch the carry front sweep left.

  E. Fixed point vs tuned trajectory. Run PAST the trained budget (64 rounds).
     Measure the per-round state update norm ||x_t - x_{t-1}|| -- does it decay
     to ~0 (true converged fixed point) or keep moving (limit cycle / drift)?
     Does decode accuracy hold past round 24 or degrade (trajectory tuned to
     exactly 24 rounds)?

Relies on the model._combiner_trace hook added to AdditiveComposedGPT.combine().

Usage:
    python probe_combiner_dynamics.py [ckpt_dir] [ckpt_name]
    (defaults: out-addition-additive-v22  ckpt.pt)
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
EXTENDED_ROUNDS = None        # set below to combiner_repeat_max + 40 (run past the trained budget)
dataset_dir = os.path.join('data', 'addition')
device = 'cpu'                # 1.79M params -- fp32 on CPU, no autocast, cleanest numbers

# ----------------------------------------------------------------------------
# load
ckpt = torch.load(os.path.join(ckpt_dir, ckpt_name), map_location=device, weights_only=False)
cfg = GPTConfig(**ckpt['model_args'])
model = AdditiveComposedGPT(cfg)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval().to(device)

REPEAT_MAX = cfg.combiner_repeat_max        # 24
EXTENDED_ROUNDS = REPEAT_MAX + 40           # run past the trained budget
P = cfg.max_answer_digits                   # 30
DPP = cfg.codec_dims_per_place              # 3
NTHINK = cfg.combiner_n_thinking            # 4
SEQ = P + NTHINK                            # 34
print(f"{ckpt_dir}/{ckpt_name}  iter={ckpt.get('iter_num')}  "
      f"repeat_max={REPEAT_MAX}  max_answer_digits={P}  dpp={DPP}  n_think={NTHINK}")

with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
PLUS_ID, EQ_ID = meta['stoi']['+'], meta['stoi']['=']

outdir = ckpt_dir


# ----------------------------------------------------------------------------
# helpers

def digit_values(v, width):
    return [int(ch) for ch in str(v).zfill(width)]


@torch.no_grad()
def trace_combine(a_vals, b_vals, n_rounds):
    """Encode a batch of (a,b) integer pairs (grouped so every a shares one
    width and every b shares one width), run the combiner for n_rounds with
    the trace hook on. Returns states: (n_rounds+1, B, SEQ, C) tensor --
    states[0] is the pre-round-1 input, states[t] the state after round t."""
    len_a = max(len(str(v)) for v in a_vals)
    len_b = max(len(str(v)) for v in b_vals)
    a_ids = torch.tensor([digit_values(v, len_a) for v in a_vals], dtype=torch.long, device=device)
    b_ids = torch.tensor([digit_values(v, len_b) for v in b_vals], dtype=torch.long, device=device)
    vec_a = model.encode_operand(a_ids)
    vec_b = model.encode_operand(b_ids)
    model._combiner_trace = []
    model.combine(vec_a, vec_b, PLUS_ID, EQ_ID, n_rounds=n_rounds)
    states = torch.stack(model._combiner_trace, dim=0)   # (n_rounds+1, B, SEQ, C)
    model._combiner_trace = None
    return states


@torch.no_grad()
def decode_states(states):
    """states: (T, B, SEQ, C). Apply the exact combine() readout (comb_ln ->
    combiner_out_slot on the P place tokens -> scatter into vec_c slots) at
    every T, then decode_from_vector. Returns preds: (T, B, P) argmax digit
    per output slot, MSD-first (slot i has place value P-1-i)."""
    T, B = states.shape[:2]
    flat = states.reshape(T * B, SEQ, -1)
    preds_flat = torch.empty(T * B, P, dtype=torch.long)
    CHUNK = max(1, 4000 // max(B, 1)) * max(B, 1)   # ~4k rows/chunk, keeps big models off the memory wall
    for s in range(0, T * B, CHUNK):
        xb = model.comb_ln(flat[s:s + CHUNK])
        oc = model.combiner_out_slot(xb[:, :P, :]).float()
        vc = oc.new_zeros(oc.shape[0], cfg.n_embd)
        vc[:, :P * DPP] = oc.reshape(oc.shape[0], P * DPP)
        lg, _ = model.decode_from_vector(vc)
        preds_flat[s:s + CHUNK] = lg.argmax(-1)
    return preds_flat.reshape(T, B, P)


def preds_to_ints(preds_tb):
    """preds_tb: (B, P) MSD-first digit slots -> list of B python ints."""
    out = []
    for row in preds_tb.tolist():
        out.append(int(''.join(str(d) for d in row)))
    return out


def digit_at(v, p):
    return (v // (10 ** p)) % 10


def carryout_per_place(a, b, width):
    """carry-OUT of each place 0..width-1 for a+b. Longest consecutive run of
    1s = the worst carry-ripple depth this example demands."""
    carries, carry = [], 0
    for p in range(width):
        s = digit_at(a, p) + digit_at(b, p) + carry
        carry = s // 10
        carries.append(carry)
    return carries


def longest_run(bits):
    best = cur = 0
    for x in bits:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def load_tier(name, k):
    path = os.path.join(dataset_dir, f'{name}.txt')
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lhs, c = line.split('=')
            a, b = lhs.split('+')
            rows.append((int(a), int(b), int(c)))
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(rows))[:k]
    return [rows[i] for i in idx]


results = {'ckpt': f'{ckpt_dir}/{ckpt_name}', 'iter': ckpt.get('iter_num'),
           'repeat_max': REPEAT_MAX, 'extended_rounds': EXTENDED_ROUNDS}


# ============================================================================
# A. per-round decode / carry-front
# ============================================================================
print("\n" + "=" * 70 + "\nA. per-round decode / carry-front tracking\n" + "=" * 70)

# --- A1: run-of-9s ripple (a = 10^L - 1, b = 1 -> sum = 10^L) -------------
print("\n[A1] run-of-9s full-length carry ripple  (rows = round, showing sum)")
nines_L = [4, 6, 8, 10, 13, 16, 20, 24]
a1 = {}
for L in nines_L:
    a_val, b_val, c_val = 10 ** L - 1, 1, 10 ** L
    states = trace_combine([a_val], [b_val], REPEAT_MAX)          # (25,1,SEQ,C)
    preds = decode_states(states)[:, 0, :]                        # (25, P)
    per_round_int = [int(''.join(map(str, r))) for r in preds.tolist()]
    # correctness of each answer place p across rounds
    ans_w = L + 1
    grid = np.zeros((REPEAT_MAX + 1, ans_w), dtype=int)
    for t in range(REPEAT_MAX + 1):
        for p in range(ans_w):
            grid[t, p] = int(digit_at(per_round_int[t], p) == digit_at(c_val, p))
    a1[L] = {'true_sum': c_val,
             'per_round_sum': per_round_int,
             'first_all_correct_round': next((t for t in range(REPEAT_MAX + 1)
                                              if grid[t].all()), None),
             'place_correct_grid': grid.tolist()}
    conv = a1[L]['first_all_correct_round']
    print(f"  L={L:2d} nines: correct by round {conv}   "
          f"(true sum has {ans_w} digits)")
    # compact text carry-front (every place, . = wrong, # = right), rounds 0..min
    show_to = min(REPEAT_MAX, (conv or REPEAT_MAX) + 2)
    for t in range(show_to + 1):
        line = ''.join('#' if grid[t, p] else '.' for p in range(ans_w - 1, -1, -1))
        print(f"    r{t:2d} |{line}|  {a1[L]['per_round_sum'][t]}")

results['A1_nines_ripple'] = a1

# heatmap: for the L=20 case, place-correct vs round
L = 20
grid = np.array(a1[L]['place_correct_grid'])
fig, ax = plt.subplots(figsize=(9, 6))
ax.imshow(grid[:, ::-1], aspect='auto', cmap='Greens', vmin=0, vmax=1,
          extent=[0, grid.shape[1], REPEAT_MAX, 0])
ax.set_xlabel('answer place value (units = 0, right)')
ax.set_ylabel('combiner round')
ax.set_title(f'v22 combiner: place-correct vs round\n{10**L-1} + 1  (L={L} run-of-9s ripple)')
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'dyn_A1_carry_front.png'), dpi=110)
plt.close(fig)

# --- A2: settle round vs place, across crafted carry chains ---------------
print("\n[A2] first-correct round vs answer place  (crafted carry chains)")
# chain_j_m: carry starts at place j and ripples m places.
#   a: '9' at places j..j+m-1, random 1-8 above, 0 below j
#   b: '1' at place j, 0 elsewhere  -> forces the carry in at j
chain_specs = []
for j in [0, 4, 9]:
    for m in [1, 3, 6, 10]:
        top = j + m
        if top + 2 > 26:
            continue
        chain_specs.append((j, m))
pts_place, pts_first = [], []
a2 = []
for (j, m) in chain_specs:
    hi = j + m
    a_digits = [0] * 28
    for p in range(j, hi):
        a_digits[p] = 9
    a_digits[hi] = np.random.RandomState(j * 10 + m).randint(1, 9)
    a_val = sum(d * 10 ** p for p, d in enumerate(a_digits))
    b_val = 10 ** j
    c_val = a_val + b_val
    ans_w = len(str(c_val))
    states = trace_combine([a_val], [b_val], REPEAT_MAX)
    preds = decode_states(states)[:, 0, :]
    per_round_int = [int(''.join(map(str, r))) for r in preds.tolist()]
    firsts = {}
    for p in range(ans_w):
        fc = None
        for t in range(REPEAT_MAX + 1):
            if all(digit_at(per_round_int[tt], p) == digit_at(c_val, p)
                   for tt in range(t, REPEAT_MAX + 1)):
                fc = t
                break
        firsts[p] = fc
        if p >= j and fc is not None:          # only the places the carry touches
            pts_place.append(p)
            pts_first.append(fc)
    a2.append({'j': j, 'm': m, 'a': a_val, 'b': b_val, 'sum': c_val,
               'first_correct_by_place': firsts})
    print(f"  j={j:2d} m={m:2d}: carry {j}->{j+m}   first-correct by place "
          + ' '.join(f'{p}:{firsts[p]}' for p in range(j, min(ans_w, j + m + 2))))

# linear fit first_correct ~ place, over carry-touched places
if len(pts_place) > 2:
    A = np.polyfit(pts_place, pts_first, 1)
    results['A2_settle_slope'] = {'slope_rounds_per_place': float(A[0]),
                                  'intercept': float(A[1]),
                                  'n_points': len(pts_place)}
    print(f"\n  linear fit: first-correct-round ~= {A[0]:.2f} * place + {A[1]:.2f}"
          f"   (slope ~1.0 => one place per round = ripple carry)")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pts_place, pts_first, alpha=0.6)
    xs = np.array(sorted(set(pts_place)))
    ax.plot(xs, A[0] * xs + A[1], 'r-', label=f'{A[0]:.2f}·place + {A[1]:.2f}')
    ax.set_xlabel('answer place value'); ax.set_ylabel('first-correct round')
    ax.set_title('v22 combiner: when each place locks in vs its position')
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'dyn_A2_settle_slope.png'), dpi=110)
    plt.close(fig)
results['A2_chains'] = a2

# --- A3 / A4: natural examples, accuracy & convergence vs rounds ----------
print("\n[A3] full-sum accuracy vs combiner rounds, per tier")
tiers = ['test_indist', 'test_gap', 'test_hint', 'test_ood', 'test_far_ood']
K_NAT = 150 if P <= 40 else 60   # trace tensor scales with P*rounds; cap for the big (128-wide) models
acc_curves = {}
conv_scatter = {'tier': [], 'conv_round': [], 'chain_len': [], 'digits': []}
for tier in tiers:
    rows = load_tier(tier, K_NAT)
    # group by (len_a, len_b) for batched encode
    groups = {}
    for (a, b, c) in rows:
        groups.setdefault((len(str(a)), len(str(b))), []).append((a, b, c))
    per_round_correct = np.zeros(EXTENDED_ROUNDS + 1)
    n = 0
    conv_rounds, chain_lens, digit_counts = [], [], []
    for (la, lb), g in groups.items():
        a_vals = [a for (a, b, c) in g]
        b_vals = [b for (a, b, c) in g]
        c_vals = [c for (a, b, c) in g]
        states = trace_combine(a_vals, b_vals, EXTENDED_ROUNDS)     # (65,B,SEQ,C)
        preds = decode_states(states)                              # (65,B,P)
        ints = np.array([preds_to_ints(preds[t]) for t in range(EXTENDED_ROUNDS + 1)])  # (65,B)
        truth = np.array(c_vals)
        correct = (ints == truth[None, :])                         # (65,B)
        per_round_correct += correct.sum(axis=1)
        n += len(g)
        final = ints[REPEAT_MAX]
        for bi in range(len(g)):
            # convergence: first round t (<=REPEAT_MAX) after which the decode
            # never changes through REPEAT_MAX
            cr = REPEAT_MAX
            for t in range(REPEAT_MAX + 1):
                if all(ints[tt, bi] == ints[REPEAT_MAX, bi] for tt in range(t, REPEAT_MAX + 1)):
                    cr = t
                    break
            w = max(len(str(a_vals[bi])), len(str(b_vals[bi]))) + 1
            cl = longest_run(carryout_per_place(a_vals[bi], b_vals[bi], w))
            conv_rounds.append(cr); chain_lens.append(cl)
            digit_counts.append(max(len(str(a_vals[bi])), len(str(b_vals[bi]))))
            conv_scatter['tier'].append(tier)
            conv_scatter['conv_round'].append(cr)
            conv_scatter['chain_len'].append(cl)
            conv_scatter['digits'].append(digit_counts[-1])
    curve = (per_round_correct / n).tolist()
    acc_curves[tier] = curve
    at24 = curve[REPEAT_MAX]
    at64 = curve[EXTENDED_ROUNDS]
    reach99 = next((t for t in range(REPEAT_MAX + 1) if curve[t] >= 0.99), None)
    print(f"  {tier:14s} n={n:4d}  acc@r24={at24:.3f}  acc@r64={at64:.3f}  "
          f"first r>=99% : {reach99}")
    results.setdefault('A3_acc_by_tier', {})[tier] = {
        'curve': curve, 'acc_at_24': at24, 'acc_at_64': at64, 'first_round_99': reach99}

fig, ax = plt.subplots(figsize=(8, 5))
for tier in tiers:
    ax.plot(range(EXTENDED_ROUNDS + 1), acc_curves[tier], label=tier)
ax.axvline(REPEAT_MAX, color='k', ls=':', lw=1, label='trained budget (24)')
ax.set_xlabel('combiner rounds'); ax.set_ylabel('full-sum accuracy')
ax.set_title('v22: accuracy vs combiner rounds, by length tier')
ax.legend(); ax.set_ylim(-0.02, 1.02); fig.tight_layout()
fig.savefig(os.path.join(outdir, 'dyn_A3_acc_vs_rounds.png'), dpi=110)
plt.close(fig)

# A4 scatter: convergence round vs carry-chain length
cr = np.array(conv_scatter['conv_round']); cl = np.array(conv_scatter['chain_len'])
dg = np.array(conv_scatter['digits'])
if cr.std() > 0 and cl.std() > 0:
    r_chain = float(np.corrcoef(cr, cl)[0, 1])
    r_dig = float(np.corrcoef(cr, dg)[0, 1])
    results['A4_convergence'] = {
        'corr_convround_vs_chainlen': r_chain,
        'corr_convround_vs_digits': r_dig,
        'mean_conv_round': float(cr.mean()),
        'mean_slack_vs_chainlen': float((cr - cl).mean())}
    print(f"\n[A4] convergence round: corr with carry-chain length = {r_chain:.3f}, "
          f"with digit count = {r_dig:.3f}")
    print(f"     mean convergence round = {cr.mean():.1f}  "
          f"(mean carry-chain length = {cl.mean():.1f})")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for tier in tiers:
        mask = np.array(conv_scatter['tier']) == tier
        axes[0].scatter(cl[mask] + np.random.uniform(-.15, .15, mask.sum()),
                        cr[mask] + np.random.uniform(-.15, .15, mask.sum()),
                        s=12, alpha=.5, label=tier)
        axes[1].scatter(dg[mask] + np.random.uniform(-.15, .15, mask.sum()),
                        cr[mask] + np.random.uniform(-.15, .15, mask.sum()),
                        s=12, alpha=.5, label=tier)
    axes[0].plot([0, cl.max()], [0, cl.max()], 'k--', lw=1, label='y = x')
    axes[0].set_xlabel('carry-chain length (longest run)'); axes[0].set_ylabel('convergence round')
    axes[1].set_xlabel('digit count'); axes[1].set_ylabel('convergence round')
    axes[0].legend(fontsize=8); axes[0].set_title('convergence vs carry chain')
    axes[1].set_title('convergence vs length')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'dyn_A4_convergence.png'), dpi=110)
    plt.close(fig)


# ============================================================================
# E. fixed point vs tuned trajectory  (run to EXTENDED_ROUNDS)
# ============================================================================
print("\n" + "=" * 70 + "\nE. fixed-point / stability analysis\n" + "=" * 70)

def make_family(kind, rng):
    if kind == 'nocarry_10':
        a = int(''.join(str(rng.randint(0, 5)) for _ in range(10)))
        b = int(''.join(str(rng.randint(0, 5)) for _ in range(10)))
    elif kind == 'single_8':
        a = 5 * 10 ** 8 + rng.randint(0, 10 ** 8)
        b = 5 * 10 ** 8 + rng.randint(0, 10 ** 8)
    elif kind == 'nines_10':
        a, b = 10 ** 10 - 1, 1
    elif kind == 'nines_24':
        a, b = 10 ** 24 - 1, 1
    else:
        raise ValueError(kind)
    return a, b

families = ['nocarry_10', 'single_8', 'nines_10', 'nines_24']
BATCH_E = 24
e_update = {}
for fam in families:
    rng = np.random.RandomState(hash(fam) % 2 ** 31)
    if fam.startswith('nines'):
        a_vals, b_vals = zip(*[make_family(fam, rng)])
        a_vals, b_vals = list(a_vals), list(b_vals)
    else:
        pairs = [make_family(fam, rng) for _ in range(BATCH_E)]
        a_vals = [p[0] for p in pairs]; b_vals = [p[1] for p in pairs]
    states = trace_combine(a_vals, b_vals, EXTENDED_ROUNDS)         # (65,B,SEQ,C)
    d = (states[1:] - states[:-1])                                  # (64,B,SEQ,C)
    denom = states[:-1].norm(dim=-1).clamp(min=1e-6)                # (64,B,SEQ)
    rel = (d.norm(dim=-1) / denom)                                  # (64,B,SEQ)
    rel_place = rel[:, :, :P].mean(dim=(1, 2)).tolist()
    rel_think = rel[:, :, P:].mean(dim=(1, 2)).tolist()
    # limit-cycle check on tokens: ||x_t - x_{t-2}|| relative, late rounds
    d2 = (states[2:] - states[:-2]).norm(dim=-1) / states[:-2].norm(dim=-1).clamp(min=1e-6)
    late_1 = float(rel[30:, :, :P].mean())
    late_2 = float(d2[30:, :, :P].mean())
    # direction stability of consecutive updates
    du = d[:, :, :P, :]
    cos = torch.nn.functional.cosine_similarity(du[1:], du[:-1], dim=-1)  # (63,B,P)
    late_cos = float(cos[30:].mean())
    e_update[fam] = {'rel_update_place': rel_place, 'rel_update_think': rel_think,
                     'late_rel_step1': late_1, 'late_rel_step2': late_2,
                     'late_update_cos': late_cos}
    print(f"  {fam:12s}  rel-update place: r1={rel_place[0]:.3f} "
          f"r12={rel_place[11]:.4f} r24={rel_place[23]:.4f} r48={rel_place[47]:.4f}   "
          f"late |Δ1|={late_1:.4f} |Δ2|={late_2:.4f} cos(Δt,Δt-1)={late_cos:+.2f}")

results['E_update_norms'] = e_update

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for fam in families:
    axes[0].plot(range(1, EXTENDED_ROUNDS + 1), e_update[fam]['rel_update_place'], label=fam)
    axes[1].plot(range(1, EXTENDED_ROUNDS + 1), e_update[fam]['rel_update_think'], label=fam)
for ax, ttl in zip(axes, ['place tokens', 'thinking tokens']):
    ax.axvline(REPEAT_MAX, color='k', ls=':', lw=1)
    ax.set_yscale('log'); ax.set_xlabel('round'); ax.set_ylabel('mean ||Δx|| / ||x||')
    ax.set_title(f'v22 combiner state update — {ttl}'); ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'dyn_E_update_norms.png'), dpi=110)
plt.close(fig)

# E2: decode accuracy past round 24 (already have curves from A3)
print("\n[E2] decode accuracy past the trained budget (from A3 curves)")
for tier in tiers:
    c = results['A3_acc_by_tier'][tier]['curve']
    peak_t = int(np.argmax(c))
    print(f"  {tier:14s} acc  r24={c[REPEAT_MAX]:.3f}  r32={c[32]:.3f}  "
          f"r48={c[48]:.3f}  r64={c[64]:.3f}   peak at round {peak_t} ({c[peak_t]:.3f})")
    results['A3_acc_by_tier'][tier]['peak_round'] = peak_t
    results['A3_acc_by_tier'][tier]['acc_at_32'] = c[32]
    results['A3_acc_by_tier'][tier]['acc_at_48'] = c[48]

with open(os.path.join(outdir, 'probe_combiner_dynamics.json'), 'w') as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nsaved -> {os.path.join(outdir, 'probe_combiner_dynamics.json')}")
print(f"plots -> {outdir}/dyn_A1_carry_front.png, dyn_A2_settle_slope.png, "
      f"dyn_A3_acc_vs_rounds.png, dyn_A4_convergence.png, dyn_E_update_norms.png")
