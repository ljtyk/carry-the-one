"""
How does v37 actually add? v22's mech-interp found a nearest-neighbour
(offset -1) carry wavefront among its 30 per-place INPUT tokens. v37 has
no per-place input tokens at all -- just 2 content tokens (vec_a, vec_b)
and 64 RoPE-addressed thinking tokens that self-organized into a working
algorithm (83%->100% in-range accuracy). Does the SAME offset-(-1)
wavefront shape show up among the thinking tokens, relocated onto
self-organized workspace instead of encoder-provided slots? Or did it
find something else?

Two things measured, restricted to WITHIN the trained range (places
0-19, where we already know a real mechanism exists, per probe_v37_margin.py):
  A. attention-mass-by-relative-offset among thinking tokens, at several
     rounds (direct analogue of v22's probe D).
  B. per-round decode accuracy (direct analogue of v22's probe A) --
     does convergence look length-proportional/onset-then-propagate like
     v22, or something else.
"""
import torch
import torch.nn.functional as F
import random
import model

def digit_values(v, width):
    return [int(ch) for ch in str(v).zfill(width)]

ckpt = torch.load('out-addition-additive-v37/ckpt.pt', map_location='cuda', weights_only=False)
gptconf = model.GPTConfig(**ckpt['model_args'])
m = model.AdditiveComposedGPT(gptconf)
m.load_state_dict(ckpt['model'])
m.eval().to('cuda')

n_think = gptconf.combiner_n_thinking  # 64
CONTENT_START = n_think

random.seed(11)
N = 40  # examples, well within trained range so we're probing the WORKING mechanism
LENS = list(range(8, 20))

# ---------------- Part A: attention offset structure among thinking tokens ----------------
print("=== Part A: attention mass by relative offset (think-token to think-token), by round ===")
attn_store = []
m.combiner_block.attn._attn_store = attn_store

offset_mass = {}  # round -> {offset: [mass values across examples/heads]}
ROUNDS_TO_CHECK = [0, 1, 2, 4, 8, 16, 23]

for _ in range(N):
    L = random.choice(LENS)
    a = random.randint(10**(L-1) if L > 1 else 0, 10**L - 1)
    b = random.randint(10**(L-1) if L > 1 else 0, 10**L - 1)
    a_ids = torch.tensor([digit_values(a, L)], device='cuda')
    b_ids = torch.tensor([digit_values(b, L)], device='cuda')
    attn_store.clear()
    with torch.no_grad():
        vec_a = m.encode_operand(a_ids)
        vec_b = m.encode_operand(b_ids)
        m.combine(vec_a, vec_b, None, None, n_rounds=24)
    for r in ROUNDS_TO_CHECK:
        if r >= len(attn_store):
            continue
        att = attn_store[r][0].mean(dim=0)  # (T,T), avg over heads
        think_att = att[:n_think, :n_think]  # restrict to think-token x think-token block
        for qi in range(n_think):
            for offset in range(-6, 7):
                ki = qi + offset
                if 0 <= ki < n_think and ki != qi:
                    offset_mass.setdefault(r, {}).setdefault(offset, []).append(think_att[qi, ki].item())

print(f"{'round':>6}", *[f"off{o:+d}" for o in range(-6, 7)])
for r in ROUNDS_TO_CHECK:
    if r not in offset_mass:
        continue
    row = [sum(offset_mass[r].get(o, [0])) / max(1, len(offset_mass[r].get(o, [1]))) for o in range(-6, 7)]
    print(f"{r:>6}", *[f"{v:.3f}" for v in row])

m.combiner_block.attn._attn_store = None

# ---------------- Part B: per-round decode accuracy ----------------
print("\n=== Part B: per-round decode accuracy (within trained range) ===")
trace_holder = []
m._combiner_trace = trace_holder

def per_round_decode(L, n_rounds=24):
    a = random.randint(10**(L-1) if L > 1 else 0, 10**L - 1)
    b = random.randint(10**(L-1) if L > 1 else 0, 10**L - 1)
    true_sum = a + b
    a_ids = torch.tensor([digit_values(a, L)], device='cuda')
    b_ids = torch.tensor([digit_values(b, L)], device='cuda')
    trace_holder.clear()
    with torch.no_grad():
        vec_a = m.encode_operand(a_ids)
        vec_b = m.encode_operand(b_ids)
        m.combine(vec_a, vec_b, None, None, n_rounds=n_rounds)
    true_str = str(true_sum).zfill(30)
    accs = []
    for state in trace_holder:
        x = m.comb_ln(state)
        vec_c = x[:, -1, :]  # combiner_thinking_rope=True layout: vec_b is last
        logits, _ = m.decode_from_vector(vec_c)
        preds = torch.argmax(logits[0], dim=-1).tolist()
        pred_str = ''.join(str(p) for p in preds)
        W = len(str(true_sum))
        accs.append(sum(int(x_==y_) for x_, y_ in zip(true_str[-W:], pred_str[-W:])) / W)
    return accs

for L in [5, 10, 15, 19]:
    all_accs = [per_round_decode(L) for _ in range(15)]
    n_rounds_seen = len(all_accs[0])
    mean_by_round = [sum(a[r] for a in all_accs) / len(all_accs) for r in range(n_rounds_seen)]
    first99 = next((r for r, v in enumerate(mean_by_round) if v >= 0.99), None)
    print(f"L={L:2d}: first round >=99% digit-acc: {first99}   trace: {[round(v,2) for v in mean_by_round]}")

m._combiner_trace = None
