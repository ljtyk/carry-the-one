"""
Prepare the addition dataset as one-problem-per-sequence, fixed-length,
right-padded examples -- instead of data/addition/prepare.py's concatenated-
stream format.

Needed for the global-per-problem adaptive-loop model: a single loop-count
decision per training example only makes sense if one forward pass covers
exactly one problem, not several concatenated ones (see project discussion).

Right-padding + the model's existing causal masking means no attention
changes are needed: a real token at position i can only attend to positions
<= i, which are never padding (padding always comes after the real content
in a right-padded sequence), so padding cannot contaminate real predictions.
Padding target positions get ignore_index=-1 so they contribute no loss.

Reads train.txt / val.txt from the sibling data/addition/ dir (run
data/addition/generate.py first if those don't exist yet; test_indist.txt /
test_ood.txt are handled dynamically at eval time, not pre-padded here).

Produces train.bin / val.bin (uint16, shape (N, MAX_LEN), row i = one
right-padded example), train_lengths.bin / val_lengths.bin (int32, shape
(N,), real length of example i before padding), and meta.pkl (vocab_size,
stoi, itos, pad_id, max_len) -- named to match train.py's standard
`dataset = 'addition_padded'` convention.

Also writes train_mask.bin / val_mask.bin (uint8, same shape as the ids
arrays): 0 at operand-digit positions (digits of a or b, before the '='),
1 everywhere else -- same convention and rationale as data/addition/prepare.py
(operand digits are random and genuinely unpredictable from context, so loss
on them is pure irreducible-entropy noise). get_batch (train.py) applies
this on top of the existing pad-position masking.
"""
import os
import pickle
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'addition')
OUT_DIR = os.path.dirname(__file__)
MAX_LEN = 64  # covers train's longest example (hint range now extends to 18 digits, ~59 chars worst case)

PAD_CHAR = '#'
chars = sorted(list("0123456789+=\n") + [PAD_CHAR])
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
pad_id = stoi[PAD_CHAR]

print("vocab:", repr(''.join(chars)), "pad_id:", pad_id)


def encode(s):
    return [stoi[c] for c in s]


def compute_keep_mask(text):
    """1 = valid loss target (answer digit, final newline), 0 = operand
    digit, '+', or '='. Predicting '+'/'=' requires knowing an operand just
    ended, but operand length is randomly sampled -- same root-cause noise
    as the digit values themselves, not a property of the addition. The
    final '\n', by contrast, reflects whether the sum carried into an extra
    digit -- a genuine arithmetic fact -- so it stays a valid target. Same
    state machine as data/addition/prepare.py, applied to one row's full
    "\na+b=c\n" content instead of a concatenated multi-example stream."""
    keep = np.ones(len(text), dtype=np.uint8)
    in_operand = True
    for j, ch in enumerate(text):
        if in_operand and ch.isdigit():
            keep[j] = 0
        if ch in ('+', '='):
            keep[j] = 0
        if ch == '=':
            in_operand = False
        elif ch == '\n':
            in_operand = True
    return keep


def prepare_split(name):
    path = os.path.join(SRC_DIR, f'{name}.txt')
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l]

    arr = np.full((len(lines), MAX_LEN), pad_id, dtype=np.uint16)
    # pad positions default to 1 (kept) -- harmless, they're separately zeroed
    # out via the pad_id check in get_batch regardless of this mask's value there
    mask = np.ones((len(lines), MAX_LEN), dtype=np.uint8)
    lengths = np.zeros(len(lines), dtype=np.int32)
    for i, line in enumerate(lines):
        full = f"\n{line}\n"
        assert len(full) <= MAX_LEN, f"example too long for MAX_LEN={MAX_LEN}: {full!r}"
        ids = encode(full)
        arr[i, :len(ids)] = ids
        mask[i, :len(ids)] = compute_keep_mask(full)
        lengths[i] = len(ids)

    arr.tofile(os.path.join(OUT_DIR, f'{name}.bin'))
    mask.tofile(os.path.join(OUT_DIR, f'{name}_mask.bin'))
    lengths.tofile(os.path.join(OUT_DIR, f'{name}_lengths.bin'))
    print(f"{name}: {len(lines)} examples, max real length {lengths.max()}, padded to {MAX_LEN}, "
          f"{mask.mean()*100:.1f}% of all positions (incl. padding) kept as loss targets")


prepare_split('train')
prepare_split('val')

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'pad_id': pad_id,
    'max_len': MAX_LEN,
}
with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
print("saved meta.pkl")
