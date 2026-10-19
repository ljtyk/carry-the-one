"""
Evaluate a RefineGPT checkpoint trained with refine_v2=True (get_batch_refine2,
random-fraction corruption / self-correction training) using CONFIDENCE-BASED
iterative unmasking instead of v1's fixed left-to-right sliding window --
the other half of the redesign discussed alongside self-correction training:
commit whichever positions the model is currently most sure about first,
regardless of position, so easy/certain digits can scaffold harder ones,
rather than being forced to decide the hardest (earliest, least-context)
position first every time.

Procedure: canvas starts all MASK. Over num_rounds rounds, following a
linear unmasking schedule (round r commits enough additional positions so
cumulative committed count == round(total * r / num_rounds)), pick the
currently-highest-confidence not-yet-committed positions across the WHOLE
answer region and lock them in with the model's current prediction. Unlike
v1, committed positions stay visible as normal tokens (not forced back to
MASK) for the rest of generation -- they're locked, not means re-masked.

Usage:
    python eval_refine2.py --out_dir=out-addition-refine2 --dataset_dir=data/addition --num_rounds=10
"""
import os
import re
import json
import pickle
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from model import GPTConfig, RefineGPT

# -----------------------------------------------------------------------------
out_dir = 'out-addition-refine2'
dataset_dir = os.path.join('data', 'addition')
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ckpt_name = 'ckpt.pt'
num_rounds = 10  # unmasking schedule steps; independent of answer length (unlike v1's L passes)
dump_errors_split = ''
dump_errors_n = 30
results_path = None
exec(open('configurator.py').read())
if results_path is None:
    results_path = os.path.join(out_dir, 'eval_refine2_results.json')
# -----------------------------------------------------------------------------

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

ckpt_path = os.path.join(out_dir, ckpt_name)
checkpoint = torch.load(ckpt_path, map_location=device)
assert checkpoint.get('model_type') == 'refine', f"expected a RefineGPT checkpoint, got {checkpoint.get('model_type')}"
gptconf = GPTConfig(**checkpoint['model_args'])
model = RefineGPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

train_meta_path = os.path.join('data', 'addition_refine', 'meta.pkl')
with open(train_meta_path, 'rb') as f:
    train_meta = pickle.load(f)
stoi, itos, mask_id = train_meta['stoi'], train_meta['itos'], train_meta['mask_id']


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
def refine_decode_batch(prompts, max_new_tokens):
    """Genuine revision, not lock-on-commit: every round, ALL currently-
    revealed positions get REFRESHED to the model's latest prediction (given
    the fuller context now available), not frozen at whatever they were
    guessed as when first revealed. `revealed` only tracks which positions
    are shown as a real digit vs MASK (grows monotonically per the
    confidence schedule) -- it does NOT freeze their VALUE. Fixes a real gap
    found empirically: locking values on first commit meant a wrong early
    high-confidence guess (observed directly: 3+4=7 predicted as '1' at 73%
    confidence from an all-MASK canvas) could never be corrected, which
    defeated the entire point of self-correction training -- the model was
    never actually given the chance to use it."""
    b = len(prompts)
    prompt_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    plen = prompt_ids.size(1)
    canvas = torch.full((b, max_new_tokens), mask_id, dtype=torch.long, device=device)
    revealed = torch.zeros((b, max_new_tokens), dtype=torch.bool, device=device)

    for r in range(1, num_rounds + 1):
        idx = torch.cat([prompt_ids, canvas], dim=1)
        with ctx:
            logits, _ = model(idx)
        answer_logits = logits[:, plen:, :]  # (b, max_new_tokens, vocab)
        probs = F.softmax(answer_logits, dim=-1)
        conf, pred = probs.max(dim=-1)  # (b, max_new_tokens) each

        # refresh every already-revealed position to its latest prediction
        canvas[revealed] = pred[revealed]

        target_revealed = round(max_new_tokens * r / num_rounds)
        for i in range(b):
            n_more = target_revealed - int(revealed[i].sum().item())
            if n_more <= 0:
                continue
            unrevealed_idx = (~revealed[i]).nonzero(as_tuple=True)[0]
            if len(unrevealed_idx) == 0:
                continue
            unrevealed_conf = conf[i, unrevealed_idx]
            k = min(n_more, len(unrevealed_idx))
            top_k = torch.topk(unrevealed_conf, k).indices
            chosen = unrevealed_idx[top_k]
            canvas[i, chosen] = pred[i, chosen]
            revealed[i, chosen] = True

    return canvas.tolist()


def eval_examples(examples, batch_size=200):
    by_prompt_len = {}
    for ex in examples:
        plen = len(f"{ex['a']}+{ex['b']}=")
        by_prompt_len.setdefault(plen, []).append(ex)

    per_example_results = []
    for plen, exs in sorted(by_prompt_len.items()):
        max_d = max(ex['digits'] for ex in exs)
        max_new_tokens = max_d + 3
        for i in range(0, len(exs), batch_size):
            batch = exs[i:i + batch_size]
            prompts = [encode(f"\n{ex['a']}+{ex['b']}=") for ex in batch]
            assert len(set(len(p) for p in prompts)) == 1, "prompt lengths must match within a batch"
            gen = refine_decode_batch(prompts, max_new_tokens)
            for ex, g in zip(batch, gen):
                text = decode(g)
                pred_str = text.split('\n')[0]
                try:
                    pred_val = int(pred_str) if pred_str != '' else None
                except ValueError:
                    pred_val = None
                correct = pred_val == ex['c']
                per_example_results.append({**ex, 'pred': pred_str, 'correct': correct})
    return per_example_results


def summarize(results, key):
    buckets = {}
    for r in results:
        buckets.setdefault(r[key], []).append(r['correct'])
    return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}


def main():
    all_results = {}
    splits = ['test_indist', 'test_ood']
    for extra_split in ['test_hint', 'test_gap']:
        if os.path.exists(os.path.join(dataset_dir, f'{extra_split}.txt')):
            splits.insert(1, extra_split)
    for split in splits:
        path = os.path.join(dataset_dir, f'{split}.txt')
        examples = load_examples(path)
        results = eval_examples(examples)
        overall_acc = sum(r['correct'] for r in results) / len(results)
        acc_by_digits = summarize(results, 'digits')
        print(f"\n=== {split} ===")
        print(f"overall accuracy: {overall_acc:.4f} (n={len(results)})")
        print(f"accuracy by digit count: {acc_by_digits}")
        if split == dump_errors_split:
            wrong = [r for r in results if not r['correct']]
            print(f"\n--- {len(wrong)} incorrect examples in {split} (showing up to {dump_errors_n}) ---")
            for r in wrong[:dump_errors_n]:
                print(f"  {r['a']}+{r['b']}={r['c']}  (pred={r['pred']!r})  digits={r['digits']}")
        all_results[split] = {'overall_accuracy': overall_acc, 'accuracy_by_digits': acc_by_digits, 'n': len(results)}

    os.makedirs(out_dir, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved results to {results_path}")


if __name__ == '__main__':
    main()
