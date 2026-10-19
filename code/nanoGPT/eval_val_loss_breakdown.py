"""
Diagnostic: break down val loss by digit-range bucket instead of trusting
the single blended average train.py reports. val.txt now mixes the
in-distribution (1-10 digit) range with a sparse slice of the hint (14-15
digit) range (see data/addition/generate.py) -- if the model is overfitting
its ~100 memorized hint-training pairs while failing to generalize to the
*different* held-out hint pairs in val, that alone could drag the blended
average up a lot even while the much larger in-dist portion stays flat.
This computes teacher-forced next-token cross-entropy directly per bucket,
as a sanity check on train.py's reported (packed-random-window) val loss.

Usage:
    python eval_val_loss_breakdown.py --out_dir=out-addition-rope --ckpt_name=ckpt_15000.pt --device=cuda
"""
import os
import re
import pickle
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from model import GPTConfig, GPT, LoopedGPT, ACTLoopedGPT, PerExampleLoopedGPT, TwoBlockGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-rope'
dataset_dir = os.path.join('data', 'addition')
ckpt_name = 'ckpt.pt'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
exec(open('configurator.py').read())
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
model_type = checkpoint.get('model_type', 'gpt')
model_cls = {'gpt': GPT, 'looped': LoopedGPT, 'act_looped': ACTLoopedGPT, 'per_example_looped': PerExampleLoopedGPT, 'two_block': TwoBlockGPT}[model_type]
gptconf = GPTConfig(**checkpoint['model_args'])
model = model_cls(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)
print(f"loaded {ckpt_name} from {out_dir}, iter {checkpoint.get('iter_num')}")

train_dataset = checkpoint.get('config', {}).get('dataset', 'addition')
meta_dir = os.path.join('data', train_dataset)
with open(os.path.join(meta_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']
encode = lambda s: [stoi[c] for c in s]

LINE_RE = re.compile(r'^(\d+)\+(\d+)=(\d+)$')


def bucket_of(line):
    m = LINE_RE.match(line)
    a, b = int(m.group(1)), int(m.group(2))
    return max(len(str(a)), len(str(b)))


@torch.no_grad()
def mean_loss_over_lines(lines, block_size):
    # concatenate lines back into one blob (each line already ends in '\n',
    # matching how train.py packs val.bin), encode, then teacher-forced
    # next-token cross-entropy over sequential block_size chunks -- a direct
    # from-scratch recomputation, independent of train.py's random-window
    # sampling in get_batch(), as a cross-check on its reported val loss
    text = ''.join(lines)
    ids = torch.tensor(encode(text), dtype=torch.long, device=device)
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(ids) - 1, block_size):
        chunk = ids[i:i + block_size + 1]
        if len(chunk) < 2:
            continue
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)
        with ctx:
            _, loss = model(x, y)
        n = y.numel()
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / max(total_tokens, 1), total_tokens


def main():
    val_path = os.path.join(dataset_dir, 'val.txt')
    with open(val_path, 'r') as f:
        lines = [l for l in f.readlines() if l.strip()]

    by_bucket = {}
    for line in lines:
        d = bucket_of(line.strip())
        by_bucket.setdefault(d, []).append(line)

    block_size = model.config.block_size
    print(f"\n{'digits':>6}  {'n_examples':>10}  {'n_tokens':>9}  {'mean_loss':>10}")
    indist_loss_num, indist_tokens = 0.0, 0
    hint_loss_num, hint_tokens = 0.0, 0
    for d in sorted(by_bucket):
        lns = by_bucket[d]
        loss, ntok = mean_loss_over_lines(lns, block_size)
        print(f"{d:6d}  {len(lns):10d}  {ntok:9d}  {loss:10.4f}")
        if d <= 10:
            indist_loss_num += loss * ntok
            indist_tokens += ntok
        else:
            hint_loss_num += loss * ntok
            hint_tokens += ntok

    print(f"\nin-dist (1-10) blended: {indist_loss_num / max(indist_tokens,1):.4f}  ({indist_tokens} tokens)")
    print(f"hint (14-15) blended:   {hint_loss_num / max(hint_tokens,1):.4f}  ({hint_tokens} tokens)")
    overall = (indist_loss_num + hint_loss_num) / max(indist_tokens + hint_tokens, 1)
    print(f"overall blended:        {overall:.4f}  ({indist_tokens + hint_tokens} tokens)")


if __name__ == '__main__':
    main()
