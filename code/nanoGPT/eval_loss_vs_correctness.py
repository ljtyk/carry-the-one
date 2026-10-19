"""
Diagnostic: split per-example ANSWER loss (teacher-forced cross-entropy on
just the "c\n" portion of "a+b=c\n", excluding the random/incompressible
operand digits) by whether the example is greedy-decoded correctly or not.
Answers the question: is rising val loss a broad confidence softening
across still-correct examples, or a worsening tail of already-wrong
examples getting more confidently wrong (which -log(p) being unbounded as
p->0 would let dominate the mean without moving accuracy at all)?

Usage:
    python eval_loss_vs_correctness.py --out_dir=out-addition-rope --ckpt_name=ckpt_15000.pt --device=cuda
"""
import os
import re
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT, LoopedGPT, ACTLoopedGPT, PerExampleLoopedGPT, TwoBlockGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-rope'
dataset_dir = os.path.join('data', 'addition')
split = 'test_indist'
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
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

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
            examples.append({'a': a, 'b': b, 'c': c})
    return examples


@torch.no_grad()
def greedy_decode(prompt_ids, max_new_tokens):
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with ctx:
            logits, _ = model(idx)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[0, len(prompt_ids):].tolist()


@torch.no_grad()
def answer_loss(prompt_ids, answer_ids):
    # teacher-forced loss on JUST the answer+newline portion, excluding the
    # random/incompressible operand digits in the prompt
    full = prompt_ids + answer_ids
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full[1:]], dtype=torch.long, device=device)
    with ctx:
        logits, _ = model(x, targets=y)
    # logits: (1, T, vocab). answer portion targets start at index len(prompt_ids)-1 in y
    ans_start = len(prompt_ids) - 1
    ans_logits = logits[0, ans_start:, :]
    ans_targets = y[0, ans_start:]
    losses = torch.nn.functional.cross_entropy(ans_logits, ans_targets, reduction='none')
    return losses.mean().item(), losses.max().item()


def main():
    examples = load_examples(os.path.join(dataset_dir, f'{split}.txt'))

    correct_losses, incorrect_losses = [], []
    correct_max, incorrect_max = [], []
    n_correct = 0
    for ex in examples:
        prompt = f"\n{ex['a']}+{ex['b']}="
        prompt_ids = encode(prompt)
        answer_str = f"{ex['c']}\n"
        answer_ids = encode(answer_str)

        gen = greedy_decode(prompt_ids, len(answer_ids) + 3)
        pred_text = decode(gen).split('\n')[0]
        try:
            pred_val = int(pred_text) if pred_text != '' else None
        except ValueError:
            pred_val = None
        correct = pred_val == ex['c']

        mean_l, max_l = answer_loss(prompt_ids, answer_ids)
        if correct:
            n_correct += 1
            correct_losses.append(mean_l)
            correct_max.append(max_l)
        else:
            incorrect_losses.append(mean_l)
            incorrect_max.append(max_l)

    def stats(name, vals):
        if not vals:
            print(f"{name}: (none)")
            return
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        mean = sum(vals_sorted) / n
        median = vals_sorted[n // 2]
        p90 = vals_sorted[int(n * 0.9)]
        print(f"{name}: n={n}  mean={mean:.4f}  median={median:.4f}  p90={p90:.4f}  max={vals_sorted[-1]:.4f}")

    print(f"\naccuracy: {n_correct}/{len(examples)} = {n_correct/len(examples):.4f}\n")
    print("--- mean per-token answer loss ---")
    stats("correct examples  ", correct_losses)
    stats("incorrect examples", incorrect_losses)
    print("\n--- max single-token answer loss (worst digit in the answer) ---")
    stats("correct examples  ", correct_max)
    stats("incorrect examples", incorrect_max)


if __name__ == '__main__':
    main()
