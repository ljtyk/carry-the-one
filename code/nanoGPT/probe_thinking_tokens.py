"""
Probe B for the v22 AdditiveComposedGPT combiner: what do the 4 thinking tokens do?

D showed them acting as a global load/broadcast bus (heavy traffic rounds 1-2,
re-engaging rounds 15-22 for hard cases) and v24 showed n_thinking=0 hurts. This
pins down (a) how much each individual token matters and where, (b) whether each
specializes (reader vs broadcaster; which global quantity it carries).

  1. ablation sweep: zero / mean-of-others ablate each token (and all), eval sum
     accuracy per length tier + run-of-9s.
  2. per-token attention role (via _attn_store): attention received from place
     tokens, attention given to place tokens vs other thinking tokens, per round.
  3. per-token linear probe: token j state at round r -> total carry count,
     digit count, max carry-chain length, "already solved".

Usage: python probe_thinking_tokens.py [ckpt_dir] [ckpt_name]
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
dataset_dir = os.path.join('data', 'addition')
rng = np.random.RandomState(0)
RIDGE = 10.0

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
P = cfg.max_answer_digits
NT = cfg.combiner_n_thinking          # 4
DPP = cfg.codec_dims_per_place
NH = cfg.n_head
outdir = ckpt_dir
with open(os.path.join('data', 'addition_padded', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
PLUS_ID, EQ_ID = meta['stoi']['+'], meta['stoi']['=']
ORIG_THINK = model.thinking_tokens.data.clone()
print(f"{ckpt_dir}/{ckpt_name} iter={ckpt.get('iter_num')}  R={R} n_think={NT}")


def digit_at(v, p):
    return (v // 10 ** p) % 10


def carry_seq(a, b, w):
    o, c = [], 0
    for p in range(w):
        s = digit_at(a, p) + digit_at(b, p) + c
        c = s // 10
        o.append(c)
    return o


def longest_run(bits):
    best = cur = 0
    for x in bits:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


@torch.no_grad()
def decode_final(a_vals, b_vals):
    """batched by shared (len_a,len_b); returns list of predicted ints (round R)."""
    out = [None] * len(a_vals)
    groups = {}
    for i, (a, b) in enumerate(zip(a_vals, b_vals)):
        groups.setdefault((len(str(a)), len(str(b))), []).append(i)
    for (la, lb), ids in groups.items():
        ai = torch.tensor([[int(c) for c in str(a_vals[i]).zfill(la)] for i in ids], dtype=torch.long)
        bi = torch.tensor([[int(c) for c in str(b_vals[i]).zfill(lb)] for i in ids], dtype=torch.long)
        vc, _ = model.combine(model.encode_operand(ai), model.encode_operand(bi), PLUS_ID, EQ_ID)
        dl, _ = model.decode_from_vector(vc)
        pr = dl.argmax(-1)
        for k, i in enumerate(ids):
            out[i] = int(''.join(map(str, pr[k].tolist())))
    return out


def load_tier(name, k):
    rows = []
    with open(os.path.join(dataset_dir, f'{name}.txt')) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lhs, c = line.split('=')
            a, b = lhs.split('+')
            rows.append((int(a), int(b), int(c)))
    idx = np.random.RandomState(0).permutation(len(rows))[:k]
    return [rows[i] for i in idx]


def nines_set(Ls, per=40):
    r = np.random.RandomState(1)
    rows = []
    for Lv in Ls:
        for _ in range(per):
            clen = int(r.randint(1, Lv + 1))
            head = '' if clen == Lv else str(r.randint(1, 10)) + \
                ''.join(str(r.randint(0, 10)) for _ in range(Lv - clen - 1))
            a = int(head + '9' * clen)
            bl = int(r.randint(1, Lv + 1))
            b = int(str(r.randint(1, 10)) + ''.join(str(r.randint(0, 10)) for _ in range(bl - 1)))
            rows.append((a, b, a + b))
    return rows


EVAL = {t: load_tier(t, 200) for t in ['test_indist', 'test_hint', 'test_ood', 'test_far_ood']}
EVAL['nines_8_16_22'] = nines_set([8, 16, 22])


def eval_all():
    res = {}
    for name, rows in EVAL.items():
        a = [r[0] for r in rows]; b = [r[1] for r in rows]; c = [r[2] for r in rows]
        pred = decode_final(a, b)
        res[name] = float(np.mean([pred[i] == c[i] for i in range(len(rows))]))
    return res


# ---------------------------------------------------------------- 1. ablation
print("\n[B1] ablation sweep (sum accuracy)")
base = eval_all()
print("  baseline      " + '  '.join(f'{k}={v:.3f}' for k, v in base.items()))
rows_ab = {'baseline': base}
for method in ['zero', 'mean']:
    for j in list(range(NT)) + ['all']:
        model.thinking_tokens.data.copy_(ORIG_THINK)
        if method == 'zero':
            if j == 'all':
                model.thinking_tokens.data.zero_()
            else:
                model.thinking_tokens.data[j] = 0
        else:  # mean of the OTHER tokens
            if j == 'all':
                model.thinking_tokens.data[:] = ORIG_THINK.mean(0, keepdim=True)
            else:
                others = [k for k in range(NT) if k != j]
                model.thinking_tokens.data[j] = ORIG_THINK[others].mean(0)
        r = eval_all()
        rows_ab[f'{method}_{j}'] = r
        d = {k: r[k] - base[k] for k in base}
        print(f"  {method:4s} tok {str(j):3s}  " +
              '  '.join(f'{k.replace("test_","")}={r[k]:.3f}({d[k]:+.2f})' for k in base))
model.thinking_tokens.data.copy_(ORIG_THINK)

# ---------------------------------------------------------------- 2. attention role
print("\n[B2] per-thinking-token attention role (rand L=16 + nines L=20)")
def _ri(r, Lv):
    return int(str(r.randint(1, 10)) + ''.join(str(r.randint(0, 10)) for _ in range(Lv - 1)))


def mk(kind, Lv, n=32):
    r = np.random.RandomState(2)
    a, b = [], []
    for _ in range(n):
        if kind == 'nines':
            clen = int(r.randint(1, Lv + 1))
            head = '' if clen == Lv else str(r.randint(1, 10)) + ''.join(str(r.randint(0, 10)) for _ in range(Lv - clen - 1))
            a.append(int(head + '9' * clen)); b.append(_ri(r, int(r.randint(1, Lv + 1))))
        else:
            a.append(_ri(r, Lv)); b.append(_ri(r, Lv))
    return a, b


@torch.no_grad()
def attn_capture(a_vals, b_vals):
    la = max(len(str(v)) for v in a_vals); lb = max(len(str(v)) for v in b_vals)
    ai = torch.tensor([[int(c) for c in str(v).zfill(la)] for v in a_vals], dtype=torch.long)
    bi = torch.tensor([[int(c) for c in str(v).zfill(lb)] for v in b_vals], dtype=torch.long)
    model.combiner_block.attn._attn_store = []
    model.combine(model.encode_operand(ai), model.encode_operand(bi), PLUS_ID, EQ_ID, n_rounds=R)
    maps = torch.stack(model.combiner_block.attn._attn_store, 0).mean(1).mean(1)  # (R,T,T) mean B,heads
    model.combiner_block.attn._attn_store = None
    return maps

b2 = {}
for tag, (kind, Lv) in {'rand16': ('rand', 16), 'nines20': ('nines', 20)}.items():
    a_vals, b_vals = mk(kind, Lv)
    m = attn_capture(a_vals, b_vals)              # (R, 34, 34)
    Lm = Lv + 1
    perj = {}
    for j in range(NT):
        col = P + j
        recv = m[:, :Lm, col].mean(-1).numpy()               # place -> this token
        give_place = m[:, col, :Lm].sum(-1).numpy()          # this token -> place tokens
        give_think = m[:, col, P:].sum(-1).numpy()           # this token -> thinking tokens
        self_a = m[:, col, col].numpy()
        perj[j] = {'received_from_place': recv.tolist(), 'to_place': give_place.tolist(),
                   'to_think': give_think.tolist(), 'self': self_a.tolist()}
        print(f"  {tag} tok{j}: recv(r1,r8,r20)=({recv[1]:.3f},{recv[8]:.3f},{recv[20]:.3f})  "
              f"->place(r1,r8,r20)=({give_place[1]:.2f},{give_place[8]:.2f},{give_place[20]:.2f})  "
              f"->think r1={give_think[1]:.2f}")
    b2[tag] = perj

# ---------------------------------------------------------------- 3. per-token probe
print("\n[B3] per-thinking-token linear probe of global quantities")
Np = 500
pa, pb = [], []
for _ in range(Np):
    Lv = int(rng.randint(3, 20))
    if rng.random() < 0.4:
        cl = int(rng.randint(1, Lv + 1))
        head = '' if cl == Lv else str(rng.randint(1, 10)) + ''.join(str(rng.randint(0, 10)) for _ in range(Lv - cl - 1))
        a = int(head + '9' * cl)
        b = _ri(rng, int(rng.randint(1, Lv + 1)))
    else:
        a = _ri(rng, Lv)
        b = _ri(rng, int(rng.randint(1, Lv + 1)))
    pa.append(a); pb.append(b)

groups = {}
for i, (a, b) in enumerate(zip(pa, pb)):
    groups.setdefault((len(str(a)), len(str(b))), []).append(i)
think_states = np.zeros((R + 1, Np, NT, cfg.n_embd), dtype=np.float32)
ycarry = np.zeros(Np, np.int64); ydig = np.zeros(Np, np.int64); ychain = np.zeros(Np, np.int64)
ysolved = np.zeros((R + 1, Np), np.int64)
with torch.no_grad():
    for (la, lb), ids in groups.items():
        ai = torch.tensor([[int(c) for c in str(pa[i]).zfill(la)] for i in ids], dtype=torch.long)
        bi = torch.tensor([[int(c) for c in str(pb[i]).zfill(lb)] for i in ids], dtype=torch.long)
        model._combiner_trace = []
        model.combine(model.encode_operand(ai), model.encode_operand(bi), PLUS_ID, EQ_ID, n_rounds=R)
        s = torch.stack(model._combiner_trace, 0); model._combiner_trace = None
        T, Bg = s.shape[:2]
        xr = model.comb_ln(s.reshape(T * Bg, s.shape[2], -1))
        oc = model.combiner_out_slot(xr[:, :P, :]).float()
        vc = oc.new_zeros(T * Bg, cfg.n_embd); vc[:, :P * DPP] = oc.reshape(T * Bg, P * DPP)
        dl, _ = model.decode_from_vector(vc)
        pr = dl.argmax(-1).reshape(T, Bg, P)
        for k, i in enumerate(ids):
            cs = carry_seq(pa[i], pb[i], P)
            ycarry[i] = min(sum(cs), 12); ychain[i] = longest_run(cs)
            ydig[i] = min(max(la, lb), 19)
            for t in range(T):
                ysolved[t, i] = int(int(''.join(map(str, pr[t, k].tolist()))) == pa[i] + pb[i])
            think_states[:, i] = s[:, k, P:, :].numpy()

ex = np.arange(Np); rng.shuffle(ex)
tr, te = ex[:350], ex[350:]

def ridge(Xtr, ytr, Xte, nc):
    mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = np.hstack([(Xtr - mu) / sd_, np.ones((len(Xtr), 1), np.float32)])
    Xte = np.hstack([(Xte - mu) / sd_, np.ones((len(Xte), 1), np.float32)])
    Y = np.eye(nc, dtype=np.float32)[ytr]
    W = np.linalg.solve(Xtr.T @ Xtr + RIDGE * np.eye(Xtr.shape[1], dtype=np.float32), Xtr.T @ Y)
    return (Xte @ W).argmax(1)

probe_rounds = [1, 2, 4, 8, 16, 24]
b3 = {}
for tgt, (y, nc) in {'total_carry': (ycarry, 13), 'digit_count': (ydig, 20),
                     'max_chain': (ychain, int(ychain.max()) + 1)}.items():
    b3[tgt] = {}
    for j in range(NT):
        accs = []
        for t in probe_rounds:
            pred = ridge(think_states[t][tr, j], y[tr], think_states[t][te, j], nc)
            accs.append(float((pred == y[te]).mean()))
        b3[tgt][j] = dict(zip(probe_rounds, accs))
    # all 4 concatenated
    accs = []
    for t in probe_rounds:
        pred = ridge(think_states[t][tr].reshape(len(tr), -1), y[tr],
                     think_states[t][te].reshape(len(te), -1), nc)
        accs.append(float((pred == y[te]).mean()))
    b3[tgt]['concat'] = dict(zip(probe_rounds, accs))
    base_acc = np.bincount(y[te]).max() / len(te)
    print(f"  {tgt:12s} baseline={base_acc:.2f}  concat@r8={b3[tgt]['concat'][8]:.2f} "
          f"@r24={b3[tgt]['concat'][24]:.2f}   per-tok@r8=" +
          ','.join(f'{b3[tgt][j][8]:.2f}' for j in range(NT)))

results = {'ckpt': f'{ckpt_dir}/{ckpt_name}', 'iter': ckpt.get('iter_num'),
          'B1_ablation': rows_ab, 'B2_attn_role': b2, 'B3_probe': b3,
          'frac_solved_by_round': [float(ysolved[t].mean()) for t in range(R + 1)]}
with open(os.path.join(outdir, 'probe_thinking_tokens.json'), 'w') as f:
    json.dump(results, f, indent=2, default=float)

# plot: ablation delta heatmap + attn role
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
tiers = list(base.keys())
labels = [k for k in rows_ab if k != 'baseline']
Z = np.array([[rows_ab[l][t] - base[t] for t in tiers] for l in labels])
im = axes[0].imshow(Z, cmap='RdBu', vmin=-.5, vmax=.5, aspect='auto')
axes[0].set_xticks(range(len(tiers))); axes[0].set_xticklabels([t.replace('test_', '') for t in tiers], rotation=30)
axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels)
axes[0].set_title('B1: accuracy delta vs baseline'); fig.colorbar(im, ax=axes[0])
for tag in b2:
    for j in range(NT):
        axes[1].plot(range(R), b2[tag][j]['received_from_place'], label=f'{tag} tok{j}')
axes[1].set_title('B2: attention received from place tokens'); axes[1].set_xlabel('round')
axes[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(outdir, 'dyn_B_thinking.png'), dpi=110)
plt.close(fig)
print(f"\nsaved -> {os.path.join(outdir, 'probe_thinking_tokens.json')}, dyn_B_thinking.png")
