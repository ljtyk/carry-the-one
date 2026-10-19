"""
"Logit lens" probe for LatentLoopedGPT: project each of the K latent-thought
vectors through the model's own final layernorm + lm_head (the same
projection normally only applied to the final hidden state) to see which
vocabulary token each latent "looks like" from the model's own point of
view. Also reports cosine similarity between consecutive latent steps, to
see how much each step actually changes the state vs. barely moving.

Not a claim that the latents literally encode tokens -- they're continuous
vectors optimized only through their downstream effect on the real answer,
never directly supervised to look like anything -- but it's a cheap way to
see whether they drift toward interpretable directions in the model's own
output space or stay opaque throughout.

Usage:
    python eval_latent_probe.py --out_dir=out-addition-latent-rope --device=cuda
"""
import os
import pickle
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
top_k = 3
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
assert checkpoint.get('model_type') == 'latent_looped'
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


@torch.no_grad()
def probe(a, b, label):
    prompt = f"\n{a}+{b}="
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    prompt_len = idx.size(1)
    with ctx:
        model(idx, prompt_len=prompt_len)  # populates model._last_latents
    latents = model._last_latents[0]  # (K, C)
    K = latents.size(0)

    print(f"\n=== {label}: {a}+{b}={a + b}  (prompt_len={prompt_len}) ===")
    with ctx:
        h = model.transformer.ln_f(latents)
        logits = model.lm_head(h)  # (K, vocab)
    probs = F.softmax(logits.float(), dim=-1)
    for step in range(K):
        topv, topi = probs[step].topk(top_k)
        decoded = ', '.join(f"{itos[i.item()]!r}:{v.item():.2f}" for v, i in zip(topv, topi))
        print(f"  latent {step}: top-{top_k} by logit-lens -> {decoded}")

    if K > 1:
        sims = []
        for step in range(K - 1):
            sim = F.cosine_similarity(latents[step].float(), latents[step + 1].float(), dim=0).item()
            sims.append(sim)
        sims_str = ', '.join(f"{s:.3f}" for s in sims)
        print(f"  consecutive cosine similarities (latent[i] vs latent[i+1]): {sims_str}")


examples = [
    ('indist WRONG: 62754+37759 (overflow missed)', 62754, 37759),
    ('indist WRONG: 8252+395882 (1st digit off)', 8252, 395882),
    ('gap WRONG: 19169327529+2', 19169327529, 2),
    ('gap WRONG: 7+44243402027', 7, 44243402027),
    ('ood WRONG: 5+4821377179253895', 5, 4821377179253895),
    ('ood WRONG: 7980826103072298+4', 7980826103072298, 4),
]

for label, a, b in examples:
    probe(a, b, label)
