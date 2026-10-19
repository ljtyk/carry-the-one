"""
Extends eval_latent_bits.py across ALL K latent steps, not just the last
(directly-supervised) one: for each step i, fit a b-bit k-means codebook on
step i's own vectors (norm/scale differs a lot across steps, so codebooks
aren't shared), quantize ONLY step i during a forward pass (steps before
and after are left untouched), and measure accuracy. Finds, per step,
roughly where accuracy stops improving with more bits -- the effective
information content the "thinking" phase is building up at that point.

Usage:
    python eval_latent_bits_per_step.py --out_dir=out-addition-latent-rope --device=cuda --n_examples=150
"""
import os
import re
import random
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, LatentLoopedGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-latent-rope'
dataset_dir = os.path.join('data', 'addition')
split = 'test_indist'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
n_examples = 150
n_codebook_samples = 1000
bit_budgets = [1, 3, 5, 7, 9]
kmeans_iters = 20
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
decode = lambda l: ''.join([itos[i] for i in l])

K = gptconf.n_latent_steps
random.seed(seed)


def random_examples(n):
    out = []
    for _ in range(n):
        d = random.randint(1, 10)
        other = random.randint(1, d)
        a_len, b_len = (d, other) if random.random() < 0.5 else (other, d)
        a = random.randint(10 ** (a_len - 1) if a_len > 1 else 0, 10 ** a_len - 1)
        b = random.randint(10 ** (b_len - 1) if b_len > 1 else 0, 10 ** b_len - 1)
        out.append((a, b))
    return out


@torch.no_grad()
def collect_all_step_latents(pairs, batch_size=100):
    """Returns (N, K, C) tensor: every latent step for every pair."""
    by_plen = {}
    for a, b in pairs:
        plen = len(f"{a}+{b}=")
        by_plen.setdefault(plen, []).append((a, b))
    out = []
    for plen, group in by_plen.items():
        for i in range(0, len(group), batch_size):
            batch = group[i:i + batch_size]
            prompts = [encode(f"\n{a}+{b}=") for a, b in batch]
            idx = torch.tensor(prompts, dtype=torch.long, device=device)
            with ctx:
                model(idx, prompt_len=idx.size(1))
            out.append(model._last_latents.float().cpu())  # (batch, K, C)
    return torch.cat(out, dim=0)


def kmeans(vecs, n_clusters, iters):
    N = vecs.size(0)
    perm = torch.randperm(N)[:n_clusters]
    centroids = vecs[perm].clone()
    for _ in range(iters):
        d = torch.cdist(vecs, centroids)
        assign = d.argmin(dim=1)
        new_centroids = centroids.clone()
        for c in range(n_clusters):
            mask = assign == c
            if mask.any():
                new_centroids[c] = vecs[mask].mean(dim=0)
        centroids = new_centroids
    return centroids


@torch.no_grad()
def forward_quantized(idx, prompt_len, target_step, codebook):
    prompt_idx = idx[:, :prompt_len]
    emb = model._embed_tokens(prompt_idx, pos_start=0)
    for step in range(K):
        with ctx:
            h = model._run_stack(emb)
        raw_latent = h[:, -1, :]
        if step == target_step and codebook is not None:
            vec = raw_latent.float()
            d = torch.cdist(vec, codebook.to(device))
            nearest = d.argmin(dim=1)
            raw_latent = codebook.to(device)[nearest].to(raw_latent.dtype)
        last = model._embed_latent(raw_latent.unsqueeze(1), pos=prompt_len + step)
        emb = torch.cat([emb, last], dim=1)
    answer_idx = idx[:, prompt_len:]
    answer_emb = model._embed_tokens(answer_idx, pos_start=prompt_len + K)
    emb = torch.cat([emb, answer_emb], dim=1)
    with ctx:
        h = model._run_stack(emb)
        h = model.transformer.ln_f(h)
        logits = model.lm_head(h[:, [-1], :])
    return logits


@torch.no_grad()
def greedy_decode_quantized(prompts, max_new_tokens, target_step, codebook):
    idx = torch.tensor(prompts, dtype=torch.long, device=device)
    prompt_len = idx.size(1)
    for _ in range(max_new_tokens):
        logits = forward_quantized(idx, prompt_len, target_step, codebook)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[:, prompt_len:].tolist()


LINE_RE = re.compile(r'^(\d+)\+(\d+)=(\d+)$')
examples = []
with open(os.path.join(dataset_dir, f'{split}.txt')) as f:
    lines = [l.strip() for l in f if l.strip()]
random.shuffle(lines)
for line in lines[:n_examples]:
    m = LINE_RE.match(line)
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    d = max(len(str(a)), len(str(b)))
    examples.append({'a': a, 'b': b, 'c': c, 'digits': d})

by_plen = {}
for ex in examples:
    plen = len(f"{ex['a']}+{ex['b']}=")
    by_plen.setdefault(plen, []).append(ex)


def run_condition(target_step, codebook):
    correct, total = 0, 0
    for plen, exs in sorted(by_plen.items()):
        max_new = max(e['digits'] for e in exs) + 3
        for i in range(0, len(exs), 100):
            batch = exs[i:i + 100]
            prompts = [encode(f"\n{e['a']}+{e['b']}=") for e in batch]
            gen = greedy_decode_quantized(prompts, max_new, target_step, codebook)
            for e, g in zip(batch, gen):
                text = decode(g)
                pred_str = text.split('\n')[0]
                try:
                    pred_val = int(pred_str) if pred_str != '' else None
                except ValueError:
                    pred_val = None
                correct += (pred_val == e['c'])
                total += 1
    return correct / total


print(f"\ncollecting {n_codebook_samples} examples' worth of all-step latents...")
codebook_pairs = random_examples(n_codebook_samples)
all_step_latents = collect_all_step_latents(codebook_pairs)  # (N, K, C)
print(f"collected {all_step_latents.size(0)} examples x {K} steps")

baseline_acc = run_condition(None, None)
print(f"\n{split}, n={len(examples)} examples, K={K}")
print(f"continuous baseline (no quantization): {baseline_acc*100:.2f}%\n")

results = {}
for step in range(K):
    print(f"--- step {step} (norm~{all_step_latents[:, step, :].norm(dim=1).mean():.1f}) ---")
    step_vecs = all_step_latents[:, step, :]
    row = {}
    for b in bit_budgets:
        n_clusters = 2 ** b
        codebook = kmeans(step_vecs, n_clusters, kmeans_iters)
        acc = run_condition(step, codebook)
        row[b] = acc
        print(f"  {b}-bit ({n_clusters:>4}): {acc*100:.2f}%")
    results[step] = row

print(f"\n{'step':>5}", *[f"{b}b" for b in bit_budgets], "cont", sep="  ")
for step in range(K):
    vals = [f"{results[step][b]*100:5.1f}" for b in bit_budgets]
    print(f"{step:>5}", *vals, f"{baseline_acc*100:5.1f}", sep="  ")
