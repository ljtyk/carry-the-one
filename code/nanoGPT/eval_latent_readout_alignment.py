"""
Does direct supervision (the last latent, trained as a real next-token
predictor) leave a different signature than the unsupervised latents (0
through K-2, gradient only via downstream effect)? Logit-lens confidence
alone conflates two things: how much of the vector's norm sits in the
subspace lm_head actually reads from (the span of the digit-token
directions, since wte/lm_head are weight-tied), vs how sharp the
distribution looks *within* that subspace. This measures both, plus raw
norm, averaged over many examples per latent step.

Usage:
    python eval_latent_readout_alignment.py --out_dir=out-addition-latent-rope --device=cuda --n_examples=100
"""
import os
import pickle
import random
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from model import GPTConfig, LatentLoopedGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-latent-rope'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
n_examples = 100
seed = 0
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = LatentLoopedGPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

train_dataset = checkpoint.get('config', {}).get('dataset', 'addition_padded')
meta_dir = os.path.join('data', train_dataset)
with open(os.path.join(meta_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]

# subspace the readout actually cares about: the rows of lm_head (== wte,
# weight-tied) for the 10 digit tokens -- an orthonormal basis for that span
digit_ids = [stoi[str(d)] for d in range(10)]
with torch.no_grad():
    digit_dirs = model.lm_head.weight[digit_ids].float()  # (10, C)
    basis, _ = torch.linalg.qr(digit_dirs.T)  # (C, 10), orthonormal columns spanning the same subspace


def readout_fraction(vec):
    """vec: (C,) -> fraction of ||vec|| that lies in the digit-readout subspace."""
    vec = vec.float()
    proj = basis.T @ vec  # (10,)
    proj_norm = proj.norm()
    total_norm = vec.norm()
    return (proj_norm / total_norm.clamp(min=1e-8)).item(), total_norm.item()


random.seed(seed)
examples = []
for _ in range(n_examples):
    d = random.randint(1, 10)
    other = random.randint(1, d)
    a_len, b_len = (d, other) if random.random() < 0.5 else (other, d)
    a = random.randint(10 ** (a_len - 1) if a_len > 1 else 0, 10 ** a_len - 1)
    b = random.randint(10 ** (b_len - 1) if b_len > 1 else 0, 10 ** b_len - 1)
    examples.append((a, b))

K = gptconf.n_latent_steps
sum_conf = [0.0] * K
sum_readout_frac = [0.0] * K
sum_norm = [0.0] * K
n_seen = 0

with torch.no_grad():
    for a, b in examples:
        prompt = f"\n{a}+{b}="
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        with ctx:
            model(idx, prompt_len=idx.size(1))
        latents = model._last_latents[0]  # (K, C)
        with ctx:
            h = model.transformer.ln_f(latents)
            logits = model.lm_head(h).float()
        probs = F.softmax(logits, dim=-1)
        for step in range(K):
            sum_conf[step] += probs[step].max().item()
            frac, norm = readout_fraction(latents[step])
            sum_readout_frac[step] += frac
            sum_norm[step] += norm
        n_seen += 1

print(f"\naveraged over {n_seen} examples, K={K} latent steps (step {K-1} is the directly-supervised one)")
print(f"{'step':>5} {'top1_conf':>10} {'readout_frac':>13} {'raw_norm':>9}")
for step in range(K):
    print(f"{step:>5} {sum_conf[step]/n_seen:>10.3f} {sum_readout_frac[step]/n_seen:>13.3f} {sum_norm[step]/n_seen:>9.2f}")
