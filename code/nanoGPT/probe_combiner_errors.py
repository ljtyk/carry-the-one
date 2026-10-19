"""Show concrete wrong examples from the repeating combiner (v13) at OOD
lengths, digit-by-digit, to see the actual failure SHAPE rather than just
aggregate accuracy."""
import random
import sys

import torch

from model import GPTConfig, AdditiveComposedGPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v13'
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
print(f"checkpoint: {out_dir}  iter={ckpt.get('iter_num', '?')}\n")

random.seed(11)
torch.manual_seed(11)


def digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


def mark(pred_str, true_str):
    """Right-align both, mark mismatches with ^ underneath."""
    w = max(len(pred_str), len(true_str))
    p, t = pred_str.rjust(w), true_str.rjust(w)
    marks = ''.join(' ' if a == b else '^' for a, b in zip(p, t))
    return p, t, marks


@torch.no_grad()
def show(len_a, len_b, n=8):
    P = model.config.max_answer_digits
    print(f"=== len_a={len_a} len_b={len_b} ===")
    shown = 0
    attempts = 0
    while shown < n and attempts < 200:
        attempts += 1
        a = random.randint(*digit_range(len_a))
        b = random.randint(*digit_range(len_b))
        true_sum = a + b
        a_ids = torch.tensor([[int(c) for c in str(a).zfill(len_a)]], dtype=torch.long, device=device)
        b_ids = torch.tensor([[int(c) for c in str(b).zfill(len_b)]], dtype=torch.long, device=device)
        vec_a = model.encode_operand(a_ids)
        vec_b = model.encode_operand(b_ids)
        vec_c, _ = model.combine(vec_a, vec_b, 0, 0)
        logits, _ = model.decode_from_vector(vec_c)
        s = str(true_sum)
        pred_digits = torch.argmax(logits[0, P - len(s):], dim=-1).tolist()
        pred_str = ''.join(str(d) if d != 10 else '_' for d in pred_digits)
        if pred_str == s and shown >= n // 2:
            continue  # after showing a couple correct ones, prioritize wrong ones
        shown += 1
        p, t, marks = mark(pred_str, s)
        status = "OK" if pred_str == s else "WRONG"
        print(f"  {a} + {b} = {t}  [{status}]")
        if pred_str != s:
            print(f"    pred: {p}")
            print(f"          {marks}")
    print()


show(2, 2)
show(3, 3)
show(5, 5)
show(8, 8)
show(10, 10)
show(12, 12)
show(15, 15)
show(18, 18)
show(20, 20)




