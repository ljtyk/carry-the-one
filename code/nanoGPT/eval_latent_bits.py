"""
How many bits does the final latent actually carry, operationally? A
discrete token is hard-capped at log2(vocab_size) ~= 3.8 bits. The
continuous vector has no such cap in principle, but "how much of that
capacity is actually load-bearing" is an empirical question -- answered
here via rate-distortion: quantize the latent to the nearest centroid of a
size-2^b codebook (built via k-means on real latent vectors, so it's close
to the best possible b-bit encoding, not an arbitrary one), sweep b, and
see where accuracy transitions from the continuous ceiling down toward the
full-discretization floor from eval_latent_ablation.py.

Usage:
    python eval_latent_bits.py --out_dir=out-addition-latent-rope --device=cuda --n_examples=300
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
n_examples = 300
n_codebook_samples = 1500  # latent vectors collected to fit each k-means codebook
bit_budgets = [1, 2, 3, 4, 5, 6, 8, 10, 12]
kmeans_iters = 25
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
def collect_final_latents(pairs, batch_size=100):
    """Returns (N, C) tensor of the final (step K-1) latent for each pair."""
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
            out.append(model._last_latents[:, -1, :].float().cpu())
    return torch.cat(out, dim=0)


def kmeans(vecs, n_clusters, iters):
    """vecs: (N, C) cpu float tensor. Returns (n_clusters, C) centroids."""
    N = vecs.size(0)
    perm = torch.randperm(N)[:n_clusters]
    centroids = vecs[perm].clone()
    for _ in range(iters):
        d = torch.cdist(vecs, centroids)  # (N, n_clusters)
        assign = d.argmin(dim=1)
        new_centroids = centroids.clone()
        for c in range(n_clusters):
            mask = assign == c
            if mask.any():
                new_centroids[c] = vecs[mask].mean(dim=0)
        centroids = new_centroids
    return centroids


@torch.no_grad()
def forward_quantized(idx, prompt_len, codebook):
    prompt_idx = idx[:, :prompt_len]
    emb = model._embed_tokens(prompt_idx, pos_start=0)
    for step in range(K):
        with ctx:
            h = model._run_stack(emb)
        raw_latent = h[:, -1, :]
        if step == K - 1 and codebook is not None:
            vec = raw_latent.float()
            d = torch.cdist(vec, codebook.to(device))  # (B, n_clusters)
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
def greedy_decode_quantized(prompts, max_new_tokens, codebook):
    idx = torch.tensor(prompts, dtype=torch.long, device=device)
    prompt_len = idx.size(1)
    for _ in range(max_new_tokens):
        logits = forward_quantized(idx, prompt_len, codebook)
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


def run_condition(codebook, label):
    correct, total = 0, 0
    for plen, exs in sorted(by_plen.items()):
        max_new = max(e['digits'] for e in exs) + 3
        for i in range(0, len(exs), 100):
            batch = exs[i:i + 100]
            prompts = [encode(f"\n{e['a']}+{e['b']}=") for e in batch]
            gen = greedy_decode_quantized(prompts, max_new, codebook)
            for e, g in zip(batch, gen):
                text = decode(g)
                pred_str = text.split('\n')[0]
                try:
                    pred_val = int(pred_str) if pred_str != '' else None
                except ValueError:
                    pred_val = None
                correct += (pred_val == e['c'])
                total += 1
    acc = correct / total
    print(f"{label:>28}: {correct}/{total} correct ({acc*100:.2f}%)")
    return acc


print(f"\ncollecting {n_codebook_samples} final-latent vectors to fit codebooks...")
codebook_pairs = random_examples(n_codebook_samples)
codebook_latents = collect_final_latents(codebook_pairs)
print(f"collected {codebook_latents.size(0)} vectors, dim={codebook_latents.size(1)}")

print(f"\n{split}, n={len(examples)} examples, K={K}\n")
run_condition(None, "continuous (no quantization)")
for b in bit_budgets:
    n_clusters = 2 ** b
    print(f"fitting k-means codebook, 2^{b} = {n_clusters} clusters...")
    codebook = kmeans(codebook_latents, n_clusters, kmeans_iters)
    run_condition(codebook, f"{b}-bit codebook ({n_clusters})")

print(f"\nfor reference (from eval_latent_ablation.py): true token discretization (~3.8 bits, vocab=14) collapses to ~15%")
