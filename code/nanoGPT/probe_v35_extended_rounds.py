"""
Fast, BATCHED check: does running v35's trained combiner for more rounds
than it was trained with (at inference only, no retraining) find any
working addition algorithm, or is it flat regardless of depth?
"""
import sys
import random
import torch
import model

def digit_values(v, width):
    return [int(ch) for ch in str(v).zfill(width)]

ckpt = torch.load('out-addition-additive-v35/ckpt.pt', map_location='cuda', weights_only=False)
gptconf = model.GPTConfig(**ckpt['model_args'])
m = model.AdditiveComposedGPT(gptconf)
m.load_state_dict(ckpt['model'])
m.eval().to('cuda')
print(f'trained combiner_repeat_max: {gptconf.combiner_repeat_max}', flush=True)

N = 200
random.seed(2)
examples = []
for _ in range(N):
    L = random.randint(1, 10)
    a = random.randint(10**(L-1) if L > 1 else 0, 10**L - 1)
    b = random.randint(10**(L-1) if L > 1 else 0, 10**L - 1)
    examples.append((a, b, L))

# batch by length so digit_values width is consistent within a batch
by_len = {}
for a, b, L in examples:
    by_len.setdefault(L, []).append((a, b))

def meaningful_acc(n_rounds):
    total_correct = 0
    total_digits = 0
    with torch.no_grad():
        for L, pairs in by_len.items():
            a_ids = torch.tensor([digit_values(a, L) for a, b in pairs], device='cuda')
            b_ids = torch.tensor([digit_values(b, L) for a, b in pairs], device='cuda')
            vec_a = m.encode_operand(a_ids)
            vec_b = m.encode_operand(b_ids)
            vec_c, _ = m.combine(vec_a, vec_b, None, None, n_rounds=n_rounds)
            logits_c, _ = m.decode_from_vector(vec_c)
            preds = torch.argmax(logits_c, dim=-1)  # (B, 30)
            for i, (a, b) in enumerate(pairs):
                true_sum = a + b
                true_str = str(true_sum).zfill(30)
                pred_str = ''.join(str(p) for p in preds[i].tolist())
                W = len(str(true_sum))
                total_correct += sum(int(x == y) for x, y in zip(true_str[-W:], pred_str[-W:]))
                total_digits += W
    return total_correct / total_digits

for nr in [24, 48, 96, 200, 500]:
    acc = meaningful_acc(nr)
    print(f'n_rounds={nr:4d}: meaningful-digit accuracy = {acc:.3f}', flush=True)
