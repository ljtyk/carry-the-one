"""
Same check as probe_vec_family.py (does vec_c track a genuine
encode(true_sum)?), but specifically at OOD operand lengths, to see WHY
the combiner fails there even though reconstruction (encode+decode alone)
now generalizes gracefully past L=10.
"""
import random
import sys

import torch
import torch.nn.functional as F

from model import GPTConfig, AdditiveComposedGPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v12'
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

random.seed(9)
torch.manual_seed(9)


def digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


N_PER_LEN = 150

with torch.no_grad():
    for len_a, len_b in [(10, 10), (11, 10), (11, 11), (12, 11), (12, 12)]:
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

        cos = F.cosine_similarity(vec_c, vec_target, dim=-1)
        mse = F.mse_loss(vec_c, vec_target, reduction='none').mean(dim=-1)

        logits_c, _ = model.decode_from_vector(vec_c)
        logits_target, _ = model.decode_from_vector(vec_target)
        preds_c = torch.argmax(logits_c, dim=-1)
        preds_target = torch.argmax(logits_target, dim=-1)
        agree = (preds_c == preds_target).all(dim=-1).float().mean()

        # does vec_c decode to the actual CORRECT sum, and -- the check that
        # was missing -- does vec_target (a genuine encode(true_sum), the
        # thing vec_c is supposed to match) ALSO decode correctly at this
        # length? If vec_target itself doesn't decode correctly, the
        # decoder -- not just the combiner -- is part of the bottleneck.
        P = model.config.max_answer_digits
        correct_c = 0
        correct_target = 0
        for i in range(N_PER_LEN):
            s = str(c_vals[i])
            pred_c = torch.argmax(logits_c[i, P - len(s):], dim=-1).tolist()
            pred_t = torch.argmax(logits_target[i, P - len(s):], dim=-1).tolist()
            correct_c += int(''.join(str(d) for d in pred_c) == s)
            correct_target += int(''.join(str(d) for d in pred_t) == s)

        print(f"\nlen_a={len_a} len_b={len_b} (sum length ~{len_a if len_a>=len_b else len_b}-{max(len_a,len_b)+1})")
        print(f"  ||vec_a||={vec_a.norm(dim=-1).mean():.2f}  ||vec_b||={vec_b.norm(dim=-1).mean():.2f}  ||vec_c||={vec_c.norm(dim=-1).mean():.2f}  ||vec_target||={vec_target.norm(dim=-1).mean():.2f}")
        print(f"  cos(vec_c, vec_target)={cos.mean():.4f} (std={cos.std():.4f})   MSE={mse.mean():.4f}")
        print(f"  decode(vec_c)==decode(vec_target) exactly: {agree:.4f}")
        print(f"  vec_c decodes to the ACTUAL correct sum:      {correct_c/N_PER_LEN:.4f}")
        print(f"  vec_target decodes to the ACTUAL correct sum: {correct_target/N_PER_LEN:.4f}  <- decoder alone, no combiner involved")
