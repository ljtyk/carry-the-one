"""
Does vec_c (combiner output) behave like a genuine encode() output, the way
vec_a/vec_b do? Samples many (a,b) pairs across a range of lengths, computes
vec_a, vec_b, vec_c=combine(vec_a,vec_b), and vec_sum_target=encode(true_sum)
(the exact quantity the consistency loss targets), then compares norms and
vec_c's cosine similarity / decodability against the genuine target.
"""
import random
import sys

import torch
import torch.nn.functional as F

from model import GPTConfig, AdditiveComposedGPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v3'
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
print(f"checkpoint: {out_dir}  iter={ckpt.get('iter_num', '?')}")

import pickle
with open('data/addition_padded/meta.pkl', 'rb') as f:
    m = pickle.load(f)
plus_id, eq_id = m['stoi']['+'], m['stoi']['=']

random.seed(2)
torch.manual_seed(2)


def digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


N_PER_LEN = 100

all_vec_a, all_vec_b, all_vec_c, all_vec_target = [], [], [], []
correct_c, correct_target_decode = 0, 0
total = 0

with torch.no_grad():
    for len_a in range(1, 11):
        for len_b in range(1, 11):
            a_vals = [random.randint(*digit_range(len_a)) for _ in range(N_PER_LEN)]
            b_vals = [random.randint(*digit_range(len_b)) for _ in range(N_PER_LEN)]
            c_vals = [a + b for a, b in zip(a_vals, b_vals)]

            a_ids = torch.tensor([[int(ch) for ch in str(v).zfill(len_a)] for v in a_vals], dtype=torch.long, device=device)
            b_ids = torch.tensor([[int(ch) for ch in str(v).zfill(len_b)] for v in b_vals], dtype=torch.long, device=device)

            vec_a = model.encode_operand(a_ids)
            vec_b = model.encode_operand(b_ids)
            vec_c, _ = model.combine(vec_a, vec_b, plus_id, eq_id)

            by_len = {}
            for i, v in enumerate(c_vals):
                by_len.setdefault(len(str(v)), []).append(i)
            vec_target = torch.zeros_like(vec_c)
            for L, idxs in by_len.items():
                dt = torch.tensor([[int(ch) for ch in str(c_vals[i])] for i in idxs], dtype=torch.long, device=device)
                vec_target[idxs] = model.encode_operand(dt)

            all_vec_a.append(vec_a)
            all_vec_b.append(vec_b)
            all_vec_c.append(vec_c)
            all_vec_target.append(vec_target)

    vec_a = torch.cat(all_vec_a)
    vec_b = torch.cat(all_vec_b)
    vec_c = torch.cat(all_vec_c)
    vec_target = torch.cat(all_vec_target)

    print(f"\nn={vec_a.size(0)} examples across len_a,len_b in 1..10\n")
    print(f"||vec_a||       mean={vec_a.norm(dim=-1).mean():.3f}  std={vec_a.norm(dim=-1).std():.3f}")
    print(f"||vec_b||       mean={vec_b.norm(dim=-1).mean():.3f}  std={vec_b.norm(dim=-1).std():.3f}")
    print(f"||vec_c||       mean={vec_c.norm(dim=-1).mean():.3f}  std={vec_c.norm(dim=-1).std():.3f}")
    print(f"||vec_target||  mean={vec_target.norm(dim=-1).mean():.3f}  std={vec_target.norm(dim=-1).std():.3f}  (genuine encode(true_sum))")

    cos = F.cosine_similarity(vec_c, vec_target, dim=-1)
    print(f"\ncos(vec_c, vec_target) mean={cos.mean():.4f}  std={cos.std():.4f}  min={cos.min():.4f}")

    mse = F.mse_loss(vec_c, vec_target, reduction='none').mean(dim=-1)
    print(f"per-example MSE(vec_c, vec_target) mean={mse.mean():.4f}  std={mse.std():.4f}")

    # decodability: does decoding vec_c vs decoding vec_target (the genuine
    # encode of the same true sum) give the same answer at the same rate?
    logits_c, _ = model.decode_from_vector(vec_c)
    logits_target, _ = model.decode_from_vector(vec_target)
    preds_c = torch.argmax(logits_c, dim=-1)
    preds_target = torch.argmax(logits_target, dim=-1)
    agree = (preds_c == preds_target).all(dim=-1).float().mean()
    print(f"\nfraction where decode(vec_c) == decode(vec_target) exactly: {agree:.4f}")
