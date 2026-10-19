"""
Causal test for the hypothesis that the readout-orthogonal component of the
final latent (the ~48-91% of its norm that logit-lens can't see, per
eval_latent_readout_alignment.py) carries real computation the model
depends on, vs. being unused capacity / residual-stream norm growth.

Three conditions, same examples, greedy-decoded exact-match accuracy:
  - baseline: no intervention
  - keep_aligned_only: zero out everything except the readout-subspace
    projection (what logit-lens can see) -- if accuracy holds up, the
    orthogonal part wasn't load-bearing
  - keep_orthogonal_only: the complement -- zero out the readout-aligned
    part, keep only what logit-lens is blind to -- if this alone still
    solves problems, real computation really does live off the readout
    basis

The ablation is reapplied at every autoregressive decode step (this
codebase has no KV cache, so the K latent steps get recomputed fresh each
time anyway).

Usage:
    python eval_latent_ablation.py --out_dir=out-addition-latent-rope --device=cuda --n_examples=300
"""
import os
import re
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

digit_ids = [stoi[str(d)] for d in range(10)]
with torch.no_grad():
    digit_dirs = model.lm_head.weight[digit_ids].float()
    basis, _ = torch.linalg.qr(digit_dirs.T)  # (C, 10)


def ablate(raw_latent, mode):
    """raw_latent: (B, C). mode: None / 'keep_aligned_only' / 'keep_orthogonal_only' /
    'discretize' (the true token-based-CoT simulation: argmax the FULL vocab
    distribution -- not just the digit subspace -- then re-embed that single
    token via wte, exactly as if it had been decoded and fed back in)."""
    if mode is None:
        return raw_latent
    orig_dtype = raw_latent.dtype
    if mode == 'discretize':
        with torch.no_grad():
            full_logits = model.lm_head(model.transformer.ln_f(raw_latent))
            top_id = full_logits.argmax(dim=-1)
            out = model.transformer.wte(top_id)
        return out.to(orig_dtype)
    vec = raw_latent.float()
    coords = vec @ basis            # (B, 10)
    aligned = coords @ basis.T      # (B, C)
    if mode == 'keep_aligned_only':
        out = aligned
    elif mode == 'keep_orthogonal_only':
        out = vec - aligned
    else:
        raise ValueError(mode)
    return out.to(orig_dtype)


@torch.no_grad()
def forward_ablated(idx, prompt_len, mode):
    """Same computation as LatentLoopedGPT.forward(idx, prompt_len=...),
    reusing the model's own helper methods, with the LAST latent step
    ablated per `mode`."""
    K = model.config.n_latent_steps
    prompt_idx = idx[:, :prompt_len]
    emb = model._embed_tokens(prompt_idx, pos_start=0)
    for step in range(K):
        with ctx:
            h = model._run_stack(emb)
        raw_latent = h[:, -1, :]
        if step == K - 1:
            raw_latent = ablate(raw_latent, mode)
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
def greedy_decode_ablated(prompts, max_new_tokens, mode):
    idx = torch.tensor(prompts, dtype=torch.long, device=device)
    prompt_len = idx.size(1)
    for _ in range(max_new_tokens):
        logits = forward_ablated(idx, prompt_len, mode)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[:, prompt_len:].tolist()


LINE_RE = re.compile(r'^(\d+)\+(\d+)=(\d+)$')
import random
random.seed(seed)
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


def run_condition(mode, label):
    correct = 0
    total = 0
    for plen, exs in sorted(by_plen.items()):
        max_new = max(e['digits'] for e in exs) + 3
        for i in range(0, len(exs), 100):
            batch = exs[i:i + 100]
            prompts = [encode(f"\n{e['a']}+{e['b']}=") for e in batch]
            gen = greedy_decode_ablated(prompts, max_new, mode)
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
    print(f"{label:>22}: {correct}/{total} correct ({acc*100:.2f}%)")
    return acc


print(f"\n{split}, n={len(examples)} examples, K={gptconf.n_latent_steps}, ablating step {gptconf.n_latent_steps-1} only\n")
run_condition(None, "baseline (no ablation)")
run_condition('keep_aligned_only', "keep_aligned_only")
run_condition('keep_orthogonal_only', "keep_orthogonal_only")
run_condition('discretize', "discretize (true CoT sim)")
