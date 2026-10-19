"""
Assumption-free interference probe: uses ONLY complete, valid, in-distribution-
style numbers (every place 0..L-1 populated, correct leading digit 1-9, no
partial sums, no invented operations). For each length L, measures decode
accuracy AT EACH position, indexed by distance-from-end k (k=0 is the ones
digit, k=L-1 is the most-significant digit) -- to see whether growing L
degrades accuracy uniformly across all positions (diffuse interference) or
concentrates at specific positions (a localized mechanism, like the old
attention encoder's demotion bug).
"""
import sys
import random

import torch
import torch.nn.functional as F

from model import GPTConfig, AdditiveComposedGPT

out_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-150k'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES = 300
L_RANGE = list(range(1, 16))

ckpt = torch.load(f'{out_dir}/ckpt.pt', map_location=device)
gptconf = GPTConfig(**ckpt['model_args'])
model = AdditiveComposedGPT(gptconf)
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)

old_extract2_w = old_extract2_b = None
if sd['extract_fn.2.weight'].shape[0] != model.config.n_embd:
    old_extract2_w = sd.pop('extract_fn.2.weight').to(device)
    old_extract2_b = sd.pop('extract_fn.2.bias').to(device)
model.load_state_dict(sd, strict=False)
model.eval().to(device)


def get_logits(memory_vec):
    B = memory_vec.size(0)
    P = model.config.max_answer_digits
    places = torch.arange(P - 1, -1, -1, device=device)
    place_sig = model.place_signal[places].unsqueeze(0).expand(B, -1, -1)
    mem_expanded = memory_vec.unsqueeze(1).expand(-1, P, -1)
    combined = torch.cat([mem_expanded, place_sig], dim=-1)
    if old_extract2_w is not None:
        h = F.gelu(model.extract_fn[0](combined))
        return F.linear(h, old_extract2_w, old_extract2_b)
    query = model.extract_fn(combined)
    class_vectors = torch.cat([model.digit_embed.weight, model.blank_embed.unsqueeze(0)], dim=0)
    return query @ class_vectors.t()


random.seed(0)
torch.manual_seed(0)
P = model.config.max_answer_digits

print(f"checkpoint: {out_dir}\n")
print(f"{'L':>3} {'whole':>7}  per-position accuracy by distance-from-end k (k=0=ones ... k=L-1=MSD)")

with torch.no_grad():
    for L in L_RANGE:
        digits = torch.zeros(N_SAMPLES, L, dtype=torch.long)
        digits[:, 0] = torch.randint(1, 10, (N_SAMPLES,))
        if L > 1:
            digits[:, 1:] = torch.randint(0, 10, (N_SAMPLES, L - 1))
        digits = digits.to(device)

        vec = model.encode_operand(digits)
        logits = get_logits(vec)
        preds = torch.argmax(logits, dim=-1)

        true_slots = digits
        pred_slots = preds[:, P - L:P]
        correct = (pred_slots == true_slots)  # (N, L) col 0=MSD .. L-1=ones

        per_k = [correct[:, L - 1 - k].float().mean().item() for k in range(L)]
        whole_acc = correct.all(dim=1).float().mean().item()
        row = "  ".join(f"k{k}:{a:.2f}" for k, a in enumerate(per_k))
        print(f"{L:>3} {whole_acc:>7.3f}  {row}")
