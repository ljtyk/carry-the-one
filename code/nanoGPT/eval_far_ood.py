"""
Evaluate on test_far_ood.txt (24-26 digits) -- a range never touched by
training OR by the drill mechanism (drill_digit_hi stays <=21), the real test
of whether patching blind spots at 11-21 digits induces any generalization
beyond territory the model has ever been trained or drilled on.

Needs a larger block_size than the checkpoint was trained with (26-digit
operands push the total sequence past the original 104), so the model is
reconstructed with an extended block_size before loading weights -- valid
for RoPE (deterministic, non-persistent tables recomputed at construction,
no learned position parameters), per model.py's CausalSelfAttention comment
on why RoPE should extend beyond its trained block_size.

Usage:
    python eval_far_ood.py --out_dir=out-addition-latent-drill --ckpt_name=ckpt_20000.pt
"""
import os
import re
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT, LoopedGPT, ACTLoopedGPT, PerExampleLoopedGPT, TwoBlockGPT, LatentLoopedGPT, AdaptiveLatentLoopedGPT, LocalLoopedGPT

LATENT_MODEL_TYPES = ('latent_looped', 'adaptive_latent_looped', 'local_looped')

out_dir = 'out-addition-latent-drill'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
eval_block_size = 160  # comfortably covers 26-digit operands: prompt(~55) + cap(24) + answer(~29)
exec(open('configurator.py').read())

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
model_type = checkpoint.get('model_type', 'gpt')
model_cls = {'gpt': GPT, 'looped': LoopedGPT, 'act_looped': ACTLoopedGPT, 'per_example_looped': PerExampleLoopedGPT,
             'two_block': TwoBlockGPT, 'latent_looped': LatentLoopedGPT, 'adaptive_latent_looped': AdaptiveLatentLoopedGPT,
             'local_looped': LocalLoopedGPT}[model_type]
model_args = dict(checkpoint['model_args'])
orig_block_size = model_args['block_size']
model_args['block_size'] = max(orig_block_size, eval_block_size)
print(f"extending block_size {orig_block_size} -> {model_args['block_size']} for far-OOD eval (RoPE, no retraining needed)")
gptconf = GPTConfig(**model_args)
model = model_cls(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
assert not unexpected, f"unexpected keys not loaded: {unexpected}"
assert all('rope' in k or 'window_mask' in k for k in missing), f"unexpected keys missing (real problem): {missing}"
model.eval()
model.to(device)

meta_path = os.path.join(dataset_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return ''.join(itos[i] for i in ids)


LINE_RE = re.compile(r'^(\d+)\+(\d+)=(\d+)$')


def load_examples(path):
    examples = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            d = max(len(str(a)), len(str(b)))
            examples.append({'a': a, 'b': b, 'c': c, 'digits': d})
    return examples


@torch.no_grad()
def greedy_decode_batch(prompts, max_new_tokens, latent_prompt_len=None):
    idx = torch.tensor(prompts, dtype=torch.long, device=device)
    prompt_len = idx.size(1)
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        with ctx:
            if latent_prompt_len is not None:
                logits, _ = model(idx_cond, prompt_len=latent_prompt_len)
            else:
                logits, _ = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[:, prompt_len:].tolist()


examples = load_examples(os.path.join(dataset_dir, 'test_far_ood.txt'))
by_prompt_len = {}
for ex in examples:
    plen = len(f"{ex['a']}+{ex['b']}=")
    by_prompt_len.setdefault(plen, []).append(ex)

results = []
for plen, exs in sorted(by_prompt_len.items()):
    max_d = max(ex['digits'] for ex in exs)
    max_new_tokens = max_d + 3
    for i in range(0, len(exs), 200):
        batch = exs[i:i + 200]
        prompts = [encode(f"\n{ex['a']}+{ex['b']}=") for ex in batch]
        latent_prompt_len = plen + 1 if model_type in LATENT_MODEL_TYPES else None
        gen = greedy_decode_batch(prompts, max_new_tokens, latent_prompt_len=latent_prompt_len)
        for ex, g in zip(batch, gen):
            text = decode(g)
            pred_str = text.split('\n')[0]
            try:
                pred_val = int(pred_str) if pred_str not in ('', '-') else None
            except ValueError:
                pred_val = None
            results.append({**ex, 'pred_str': pred_str, 'correct': pred_val == ex['c']})

overall_acc = sum(r['correct'] for r in results) / len(results)
by_digits = {}
for r in results:
    by_digits.setdefault(r['digits'], []).append(r['correct'])
by_digits = {k: sum(v) / len(v) for k, v in sorted(by_digits.items())}

print(f"\n=== test_far_ood (n={len(results)}) ===")
print(f"overall accuracy: {overall_acc:.4f}")
print(f"accuracy by digit count: {by_digits}")
n_correct = sum(r['correct'] for r in results)
print(f"\n{n_correct}/{len(results)} correct")
if n_correct > 0:
    print("sample correct predictions:")
    for r in [r for r in results if r['correct']][:10]:
        print(f"  {r['a']}+{r['b']}={r['c']}  pred={r['pred_str']!r}")
