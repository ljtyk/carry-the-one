"""Check the raw magnitude of contrib.abs().mean() (the new sparsity loss's
un-weighted value) against the current checkpoint, across a range of
lengths, so the weight can be picked from an actual measurement instead of
a guess -- learned the hard way from the consistency-loss blowup."""
import random
import sys

import torch

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

random.seed(5)
torch.manual_seed(5)


def digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


def hoyer_ratio(x):
    return (x.abs().sum(dim=-1) / x.norm(dim=-1).clamp(min=1e-8)).mean()


with torch.no_grad():
    for L in [1, 5, 10]:
        vals = [random.randint(*digit_range(L)) for _ in range(200)]
        ids = torch.tensor([[int(ch) for ch in str(v).zfill(L)] for v in vals], dtype=torch.long, device=device)
        vec, contrib = model.encode_operand(ids, return_contributions=True)
        print(f"L={L}: contrib.abs().mean()={contrib.abs().mean().item():.5f}  "
              f"hoyer_ratio(contrib)={hoyer_ratio(contrib).item():.4f}  (max dense=sqrt(384)={384**0.5:.2f}, max sparse=1.0)  "
              f"contrib per-digit norm mean={contrib.norm(dim=-1).mean().item():.3f}  "
              f"vec norm mean={vec.norm(dim=-1).mean().item():.3f}")
