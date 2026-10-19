"""
Why does v37 have ZERO margin past composed_train_max_digits=20 (hard
cliff at L=21) instead of v22/v33-style graceful decay?

Already known: reconstruct-A/B is 100% even at L=26, so the ENCODER
correctly represents high-place digits -- the failure is specifically in
the COMBINER's extraction/use of that information. This probes two
candidate mechanisms:
  (1) high-index thinking tokens (nominal "place" >= 20, never exercised
      with real non-padding content in training) never learned ANY
      reliable attention pattern into vec_a/vec_b -- i.e. they're
      effectively unwired, defaulting to whatever their generic
      initialization implies (a bias toward the padding digit, 0).
  (2) they DID learn to attend, but imprecisely/inconsistently (a RoPE
      long-relative-offset addressing problem), producing wrong-but-
      non-degenerate guesses.
Distinguishing evidence: whether wrong predictions at high places are
statistically biased toward 0 (supports 1) vs. roughly uniform over
1-9 (supports 2); and whether high-index thinking tokens' attention
mass into vec_a/vec_b is sharp+consistent (like low-index ones) or
diffuse/inconsistent across examples.
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

TRAIN_MAX = 20
N = 200
random.seed(7)

# ---- Part 1: is the wrong-digit distribution at high places biased toward 0? ----
print("=== Part 1: prediction bias at trained vs untrained place indices ===")
low_place_wrong_digits = []   # errors at place index < TRAIN_MAX
high_place_wrong_digits = []  # errors at place index >= TRAIN_MAX
low_place_total = high_place_total = 0

with torch.no_grad():
    for _ in range(N):
        L = random.randint(21, 26)
        a = random.randint(10**(L-1), 10**L - 1)
        b = random.randint(10**(L-1), 10**L - 1)
        true_sum = a + b
        a_ids = torch.tensor([digit_values(a, L)], device='cuda')
        b_ids = torch.tensor([digit_values(b, L)], device='cuda')
        vec_a = m.encode_operand(a_ids)
        vec_b = m.encode_operand(b_ids)
        vec_c, _ = m.combine(vec_a, vec_b, None, None)
        logits_c, _ = m.decode_from_vector(vec_c)
        preds = torch.argmax(logits_c[0], dim=-1).tolist()  # (30,) MSD-first, slot 0 = highest place value = place 29
        true_str = str(true_sum).zfill(30)
        for slot in range(30):
            place_idx = 29 - slot  # slot 0 is place 29 (highest), slot 29 is place 0 (units)
            true_d = int(true_str[slot])
            pred_d = preds[slot]
            if pred_d == true_d:
                continue
            if place_idx < TRAIN_MAX:
                low_place_wrong_digits.append(pred_d)
                low_place_total += 1
            else:
                high_place_wrong_digits.append(pred_d)
                high_place_total += 1

def dist(xs):
    from collections import Counter
    c = Counter(xs)
    n = len(xs)
    return {d: round(c.get(d, 0) / n, 3) for d in range(10)}

print(f"low-place  (<{TRAIN_MAX}) errors: n={low_place_total}, pred-digit distribution: {dist(low_place_wrong_digits)}")
print(f"high-place (>={TRAIN_MAX}) errors: n={high_place_total}, pred-digit distribution: {dist(high_place_wrong_digits)}")
print(f"fraction of high-place errors that predict 0: {sum(1 for d in high_place_wrong_digits if d==0)/max(1,high_place_total):.3f}")
print(f"fraction of low-place errors that predict 0:  {sum(1 for d in low_place_wrong_digits if d==0)/max(1,low_place_total):.3f}")

# ---- Part 2: attention mass from thinking tokens into vec_a/vec_b, low vs high nominal place ----
print("\n=== Part 2: thinking-token attention into content tokens, low vs high place index ===")
n_think = gptconf.combiner_n_thinking  # 64
# sequence layout with combiner_thinking_rope=True: [think_0..think_63, vec_a, vec_b]
CONTENT_START = n_think  # index of vec_a
attn_store = []
m.combiner_block.attn._attn_store = attn_store

def get_attn_round1(L):
    attn_store.clear()
    a = random.randint(10**(L-1), 10**L - 1)
    b = random.randint(10**(L-1), 10**L - 1)
    a_ids = torch.tensor([digit_values(a, L)], device='cuda')
    b_ids = torch.tensor([digit_values(b, L)], device='cuda')
    with torch.no_grad():
        vec_a = m.encode_operand(a_ids)
        vec_b = m.encode_operand(b_ids)
        m.combine(vec_a, vec_b, None, None, n_rounds=3)  # just need the first few rounds
    return attn_store[0][0]  # round 0, batch 0: (n_head, T, T)

low_idxs = [3, 8, 15]     # well within train range
high_idxs = [22, 24, 28]  # beyond train range (only ever padding during training)

for label, idxs in [("LOW (trained)", low_idxs), ("HIGH (untrained)", high_idxs)]:
    print(f"-- {label} thinking-token indices {idxs} --")
    for _ in range(3):
        L = random.randint(21, 26)
        att = get_attn_round1(L)  # (n_head, T, T)
        att_mean_heads = att.mean(dim=0)  # (T, T)
        for tok in idxs:
            mass_to_content = att_mean_heads[tok, CONTENT_START:CONTENT_START+2].sum().item()
            mass_to_self = att_mean_heads[tok, tok].item()
            top_target = att_mean_heads[tok].argmax().item()
            print(f"   L={L} think_tok={tok}: mass->content(vecA,vecB)={mass_to_content:.3f}  mass->self={mass_to_self:.3f}  argmax_target={top_target}")

m.combiner_block.attn._attn_store = None
