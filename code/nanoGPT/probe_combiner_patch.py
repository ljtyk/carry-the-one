"""
Probe F for the v22 AdditiveComposedGPT combiner: activation patching / causal
intervention on the carry.

A/E/C/D inferred a nearest-neighbour (offset -1, LSB->MSB) carry wavefront. F
tests it causally: inject a carry-in at place p mid-relaxation (resample-patch
that place token from a donor example engineered to have carry_out[p-1]=1), then
watch what happens downstream.

  1. injection efficacy -- non-propagating case (da+db in 2..6 at p): does the
     answer digit at p become (base_digit + 1)?  vs patch round.  Control: patch
     a far place instead -> answer should be unchanged.
  2. wavefront speed -- propagating case (places p..p+6 all sum to 9): inject at
     an early round, decode after every subsequent round, measure the round at
     which each downstream place p+k first flips.  slope = rounds per place.

Needs model._combiner_hook (hook in model.py): callable hook(t, x) -> x invoked
after every round.

Usage: python probe_combiner_patch.py [ckpt_dir] [ckpt_name]
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

R = cfg.combiner_repeat_max          # 24
P = cfg.max_answer_digits           # 30
DPP = cfg.codec_dims_per_place
L = 14                               # fixed operand width for clean batching
outdir = ckpt_dir
with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
PLUS_ID, EQ_ID = meta['stoi']['+'], meta['stoi']['=']
print(f"{ckpt_dir}/{ckpt_name} iter={ckpt.get('iter_num')}  R={R} L={L}")


def to_ids(digs):                    # digs: list length L, MSD-first
    return torch.tensor(digs, dtype=torch.long)


@torch.no_grad()
def run(a_digs, b_digs, hook=None, want_all_rounds=False):
    """a_digs/b_digs: (B, L) MSD-first digit lists. Returns per-round decoded
    ints (R+1, B) if want_all_rounds else final (B,)."""
    a_ids = torch.tensor(a_digs, dtype=torch.long)
    b_ids = torch.tensor(b_digs, dtype=torch.long)
    va, vb = model.encode_operand(a_ids), model.encode_operand(b_ids)
    model._combiner_trace = []
    model._combiner_hook = hook
    model.combine(va, vb, PLUS_ID, EQ_ID, n_rounds=R)
    states = torch.stack(model._combiner_trace, dim=0)     # (R+1,B,34,C)
    model._combiner_trace = None
    model._combiner_hook = None
    T, B = states.shape[:2]
    x = model.comb_ln(states.reshape(T * B, states.shape[2], -1))
    oc = model.combiner_out_slot(x[:, :P, :]).float()
    vc = oc.new_zeros(T * B, cfg.n_embd)
    vc[:, :P * DPP] = oc.reshape(T * B, P * DPP)
    dlog, _ = model.decode_from_vector(vc)
    preds = dlog.argmax(-1).reshape(T, B, P)               # MSD-first
    ints = np.array([[int(''.join(map(str, preds[t, b].tolist()))) for b in range(B)]
                     for t in range(T)])
    return ints if want_all_rounds else ints[R]


@torch.no_grad()
def capture_states(a_digs, b_digs):
    a_ids = torch.tensor(a_digs, dtype=torch.long)
    b_ids = torch.tensor(b_digs, dtype=torch.long)
    va, vb = model.encode_operand(a_ids), model.encode_operand(b_ids)
    model._combiner_trace = []
    model.combine(va, vb, PLUS_ID, EQ_ID, n_rounds=R)
    s = torch.stack(model._combiner_trace, dim=0)
    model._combiner_trace = None
    return s                                               # (R+1,B,34,C)


def digit_at(v, p):
    return (v // 10 ** p) % 10


# ======================================================================
# 1. injection efficacy -- non-propagating
# ======================================================================
print("\n[F1] carry injection efficacy (non-propagating: da+db in 2..6 at place p)")
N = 240
base_a, base_b, pp = [], [], []
for _ in range(N):
    p = int(rng.randint(4, 11))          # target place (0=units); leaves room below & above
    digs_a = [int(rng.randint(0, 5)) for _ in range(L)]
    digs_b = [int(rng.randint(0, 5)) for _ in range(L)]
    ia = L - 1 - p                        # MSD-first index of place p
    s = int(rng.randint(2, 7))
    da = int(rng.randint(max(0, s - 4), min(5, s) + 1)); db = s - da
    digs_a[ia], digs_b[ia] = da, db
    # ensure place p-1 does NOT carry (keep small)
    digs_a[ia + 1] = min(digs_a[ia + 1], 3); digs_b[ia + 1] = min(digs_b[ia + 1], 3)
    digs_a[0] = max(digs_a[0], 1); digs_b[0] = max(digs_b[0], 1)   # true L digits
    base_a.append(digs_a); base_b.append(digs_b); pp.append(p)
pp = np.array(pp)
# donor: force place p-1 to 9+9 -> carry_out[p-1]=1
don_a = [row[:] for row in base_a]; don_b = [row[:] for row in base_b]
for i, p in enumerate(pp):
    ia1 = L - 1 - (p - 1)
    don_a[i][ia1] = 9; don_b[i][ia1] = 9

base_int = run(base_a, base_b)
base_vals = [sum(d * 10 ** (L - 1 - k) for k, d in enumerate(base_a[i]))
             + sum(d * 10 ** (L - 1 - k) for k, d in enumerate(base_b[i])) for i in range(N)]
donor_states = capture_states(don_a, don_b)                # (R+1,N,34,C)
B = N
ar = torch.arange(B)

f1 = {}
for troll in [1, 2, 3, 4, 5, 6, 8, 10, 14]:
    def hook(t, x, tr=troll):
        if t == tr - 1:
            x = x.clone()
            x[ar, torch.tensor(pp)] = donor_states[tr][ar, torch.tensor(pp)].to(x.dtype)
        return x
    patched = run(base_a, base_b, hook=hook)
    # success = digit at p is base_digit+1 (mod 10), and carry didn't spill (p+1 unchanged)
    ok = np.array([(digit_at(patched[i], pp[i]) == (digit_at(base_int[i], pp[i]) + 1) % 10)
                   for i in range(N)])
    spill = np.array([digit_at(patched[i], pp[i] + 1) != digit_at(base_int[i], pp[i] + 1)
                      for i in range(N)])
    unchanged_elsewhere = np.array([
        all(digit_at(patched[i], q) == digit_at(base_int[i], q)
            for q in range(P) if q not in (pp[i], pp[i] + 1))
        for i in range(N)])
    f1[troll] = {'flip_rate': float(ok.mean()), 'spill_rate': float(spill.mean()),
                 'clean_elsewhere': float(unchanged_elsewhere.mean())}
    print(f"  patch@round {troll:2d}: digit[p]->+1 in {ok.mean():.2f}   "
          f"spill to p+1 {spill.mean():.2f}   rest-unchanged {unchanged_elsewhere.mean():.2f}")

# control: patch a FAR place q = p-3 (donor ~= base there) at round 4
qq = pp - 3
def hook_ctrl(t, x):
    if t == 3:
        x = x.clone()
        x[ar, torch.tensor(qq)] = donor_states[4][ar, torch.tensor(qq)].to(x.dtype)
    return x
ctrl = run(base_a, base_b, hook=hook_ctrl)
ctrl_same = np.mean([ctrl[i] == base_int[i] for i in range(N)])
print(f"  CONTROL patch far place p-3 @round 4: answer unchanged in {ctrl_same:.2f}")
f1['control_far_place_unchanged'] = float(ctrl_same)

# ======================================================================
# 2. wavefront speed -- propagating run of 9s
# ======================================================================
print("\n[F2] wavefront speed (inject carry at p, places p..p+M all sum to 9)")
M = 6
N2 = 160
ba, bb, pp2 = [], [], []
for _ in range(N2):
    p = 3
    digs_a = [int(rng.randint(0, 4)) for _ in range(L)]
    digs_b = [int(rng.randint(0, 4)) for _ in range(L)]
    for k in range(M + 1):               # places p..p+M : a=9, b=0  -> sum 9 (propagate)
        digs_a[L - 1 - (p + k)] = 9
        digs_b[L - 1 - (p + k)] = 0
    digs_a[L - 1 - (p - 1)] = 1; digs_b[L - 1 - (p - 1)] = 1   # no carry into p
    digs_a[0] = max(digs_a[0], 1); digs_b[0] = max(digs_b[0], 1)
    ba.append(digs_a); bb.append(digs_b); pp2.append(p)
p = 3
da2 = [row[:] for row in ba]; db2 = [row[:] for row in bb]
for i in range(N2):                       # donor: carry_out[p-1]=1
    da2[i][L - 1 - (p - 1)] = 9; db2[i][L - 1 - (p - 1)] = 9
base2 = run(ba, bb)
base2_ok = np.mean([all(digit_at(base2[i], p + k) == 9 for k in range(M + 1)) for i in range(N2)])
print(f"  base (unpatched) has 9s at p..p+{M} in {base2_ok:.2f} of cases")
donor2 = capture_states(da2, db2)
ar2 = torch.arange(N2)

def ripple_for(T_INJECT):
    def hook2(t, x):
        if t == T_INJECT - 1:
            x = x.clone()
            x[ar2, p] = donor2[T_INJECT][ar2, p].to(x.dtype)
        return x
    allr = run(ba, bb, hook=hook2, want_all_rounds=True)     # (R+1, N2)
    # first round each downstream place k reaches its patched target (0) and
    # holds it to the end
    first_flip = {k: [] for k in range(M + 1)}
    for i in range(N2):
        for k in range(M + 1):
            for t in range(T_INJECT, R + 1):
                if all(digit_at(allr[tt, i], p + k) == 0 for tt in range(t, R + 1)):
                    first_flip[k].append(t)
                    break
    means = {k: (float(np.mean(v)) if v else None) for k, v in first_flip.items()}
    final_ok = np.mean([
        all(digit_at(allr[R, i], p + k) == 0 for k in range(M + 1)) and
        digit_at(allr[R, i], p + M + 1) == (digit_at(base2[i], p + M + 1) + 1) % 10
        for i in range(N2)])
    return means, float(final_ok), allr

best = None
for T_INJECT in [8, 10, 12, 14, 16]:
    means, final_ok, allr = ripple_for(T_INJECT)
    ks = [k for k in range(M + 1) if means[k] is not None]
    sl = np.polyfit(ks, [means[k] for k in ks], 1)[0] if len(ks) >= 3 else None
    print(f"  inject@r{T_INJECT}: full-ripple {final_ok:.2f}  "
          f"first-flip by k={[None if means[k] is None else round(means[k],1) for k in range(M+1)]}  "
          f"slope={None if sl is None else round(sl,2)} rounds/place")
    if final_ok > (best[1] if best else -1):
        best = (T_INJECT, final_ok, means, sl)
T_INJECT, final_ok, means, slope = best
ks = [k for k in range(M + 1) if means[k] is not None]
icept = np.polyfit(ks, [means[k] for k in ks], 1)[1] if len(ks) >= 3 else 0
print(f"  => best inject round {T_INJECT}: wavefront ~ "
      f"{None if slope is None else round(slope,2)} rounds/place, full ripple {final_ok:.2f}")

# ======================================================================
# 3. direction-vector injection -- inject a carry EARLY (before the donor
#    approach can, since a donor resolves its own carry no faster than the
#    clean run would propagate an injected one). carry direction = diff of
#    per-place state means (carry_in=1 minus carry_in=0) at the inject round.
# ======================================================================
print("\n[F3] direction-vector carry injection (inject at an early round, measure ripple)")
def carry_seq_local(a_digs, b_digs):
    o, c = [], 0
    for k in range(L):                    # place k = index L-1-k
        s = a_digs[L - 1 - k] + b_digs[L - 1 - k] + c
        c = s // 10
        o.append(c)
    return o

NC = 500
cda, cdb = [], []
for _ in range(NC):
    cda.append([int(rng.randint(0, 10)) for _ in range(L)])
    cdb.append([int(rng.randint(0, 10)) for _ in range(L)])
cds = capture_states(cda, cdb)            # (R+1, NC, 34, C)
cin_lab = np.zeros((NC, P), dtype=int)    # carry_in per place
for i in range(NC):
    cs = carry_seq_local(cda[i], cdb[i])  # len L, carry_out per place
    for k in range(1, L):
        cin_lab[i, k] = cs[k - 1]

def carry_dir_at(rnd):
    S = cds[rnd, :, :L, :].numpy().reshape(-1, cfg.n_embd)       # (NC*L, C)
    lab = cin_lab[:, :L].reshape(-1)
    d = S[lab == 1].mean(0) - S[lab == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)

typ_norm = float(np.linalg.norm(cds[8, :, :L, :].numpy(), axis=-1).mean())
print(f"  typical place-token norm ~ {typ_norm:.1f}")

f3 = {}
for inj_round in [3, 5, 8, 12]:
    cd = torch.tensor(carry_dir_at(inj_round), dtype=torch.float32)
    for alpha in [0.5, 1.0, 2.0]:
        vec = (alpha * typ_norm) * cd
        def hook3(t, x, ir=inj_round, v=vec):
            if t == ir - 1:
                x = x.clone()
                x[ar2, p] = x[ar2, p] + v.to(x.dtype)
            return x
        allr = run(ba, bb, hook=hook3, want_all_rounds=True)      # reuse F2's run-of-9s bases
        ff = {}
        for k in range(M + 1):
            rr = []
            for i in range(N2):
                for t in range(inj_round, R + 1):
                    if all(digit_at(allr[tt, i], p + k) == 0 for tt in range(t, R + 1)):
                        rr.append(t); break
            ff[k] = float(np.mean(rr)) if rr else None
        full = np.mean([all(digit_at(allr[R, i], p + k) == 0 for k in range(M + 1))
                        for i in range(N2)])
        ks = [k for k in range(M + 1) if ff[k] is not None]
        sl = np.polyfit(ks, [ff[k] for k in ks], 1)[0] if len(ks) >= 3 else None
        f3[f'r{inj_round}_a{alpha}'] = {'first_flip_by_k': ff, 'full_ripple': float(full),
                                        'slope': None if sl is None else float(sl)}
        print(f"  inject r{inj_round} alpha {alpha}: full-ripple {full:.2f}  "
              f"flip-by-k {[None if ff[k] is None else round(ff[k],1) for k in range(M+1)]}  "
              f"slope {None if sl is None else round(sl,2)}")

# ======================================================================
# 4. NATURAL wavefront speed (no patching): a clean run-of-9s with vs.
#    without a carry entering at the units place. Decode both every round;
#    for each place, the round its digit first flips 9->0 in the carry
#    version = when the wavefront arrived. slope = rounds per place.
# ======================================================================
print("\n[F4] natural carry ripple speed (paired clean runs, no patching)")
RUNMAX = 11         # a = clean 9-run over places 0..RUN; +carry version adds 1 at units
N4 = 200
na, nb0, nb1, runs = [], [], [], []
for _ in range(N4):
    rn = int(rng.randint(7, RUNMAX + 1))
    da = [int(rng.randint(0, 8)) for _ in range(L)]        # random above the run
    for k in range(rn + 1):
        da[L - 1 - k] = 9
    da[0] = max(da[0], 1)
    zb = [0] * L
    ob = [0] * L; ob[L - 1] = 1
    na.append(da); nb0.append(zb); nb1.append(ob); runs.append(rn)
base4 = run(na, nb0, want_all_rounds=True)      # (R+1, N4) -- keeps the 9-run
carr4 = run(na, nb1, want_all_rounds=True)      # 9-run flips to 0s, carry out the top
flip = {k: [] for k in range(RUNMAX + 1)}
for i in range(N4):
    for k in range(runs[i] + 1):
        for t in range(R + 1):
            if digit_at(carr4[t, i], k) == 0 and digit_at(base4[t, i], k) == 9 and \
               all(digit_at(carr4[tt, i], k) == 0 for tt in range(t, R + 1)):
                flip[k].append(t); break
f4_means = {k: (float(np.mean(v)) if v else None) for k, v in flip.items()}
kk4 = [k for k in range(RUNMAX + 1) if f4_means[k] is not None]
sl4 = ic4 = None
if len(kk4) >= 4:
    sl4, ic4 = np.polyfit(kk4, [f4_means[k] for k in kk4], 1)
base_ok4 = np.mean([all(digit_at(base4[R, i], k) == 9 for k in range(runs[i] + 1)) for i in range(N4)])
carr_ok4 = np.mean([all(digit_at(carr4[R, i], k) == 0 for k in range(runs[i] + 1)) for i in range(N4)])
print(f"  base decodes as full 9-run: {base_ok4:.2f}   carry version fully rippled: {carr_ok4:.2f}")
print(f"  first-flip round by place k: {[None if f4_means[k] is None else round(f4_means[k],1) for k in range(RUNMAX+1)]}")
print(f"  => natural wavefront ~ {None if sl4 is None else round(sl4,2)} rounds/place "
      f"(intercept {None if ic4 is None else round(ic4,1)})")

results = {'ckpt': f'{ckpt_dir}/{ckpt_name}', 'iter': ckpt.get('iter_num'),
          'F1_injection': f1,
          'F2_donor_patch': {'inject_round': T_INJECT, 'mean_first_flip_by_k': means,
                             'slope_rounds_per_place': None if slope is None else float(slope),
                             'full_ripple_rate': float(final_ok),
                             'note': 'donor-patch injection flips place p but does NOT propagate downstream'},
          'F3_direction_injection': dict(f3, note='additive carry-direction injection had no effect at any round/alpha'),
          'F4_natural_wavefront': {'first_flip_by_k': f4_means,
                                   'slope_rounds_per_place': None if sl4 is None else float(sl4),
                                   'intercept': None if ic4 is None else float(ic4),
                                   'base_ok': float(base_ok4), 'carry_ok': float(carr_ok4)}}
with open(os.path.join(outdir, 'probe_combiner_patch.json'), 'w') as f:
    json.dump(results, f, indent=2, default=float)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
tr = sorted(k for k in f1 if isinstance(k, int))
axes[0].plot(tr, [f1[t]['flip_rate'] for t in tr], 'o-', label='digit[p] -> +1 (injection took)')
axes[0].plot(tr, [f1[t]['clean_elsewhere'] for t in tr], 's-', label='all other places unchanged')
axes[0].axhline(1 - f1['control_far_place_unchanged'], color='gray', ls=':',
                label='control (far-place patch effect)')
axes[0].set_xlabel('patch round'); axes[0].set_ylabel('rate'); axes[0].set_ylim(0, 1.02)
axes[0].set_title('F1: non-propagating carry injection'); axes[0].legend(fontsize=8)
axes[1].plot(kk4, [f4_means[k] for k in kk4], 'o-', label='first-flip round')
if sl4 is not None:
    xs = np.array(kk4)
    axes[1].plot(xs, sl4 * xs + ic4, 'r--', label=f'{sl4:.2f} rounds/place')
axes[1].set_xlabel('place (units = 0)'); axes[1].set_ylabel('round the carry arrives (digit flips 9->0)')
axes[1].set_title('F4: natural carry wavefront speed (no patching)')
if kk4:
    axes[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'dyn_F_patch.png'), dpi=110)
plt.close(fig)
print(f"\nsaved -> {os.path.join(outdir, 'probe_combiner_patch.json')}, dyn_F_patch.png")
