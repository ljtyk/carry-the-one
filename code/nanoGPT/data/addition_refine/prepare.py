"""
Same one-problem-per-sequence, right-padded format as data/addition_padded/
prepare.py, but with a MASK_CHAR added to the vocab -- needed for
RefineGPT's iterative window-refinement training (see model.py), which
constructs its own masked/partially-revealed training views on the fly in
get_batch_refine (train.py) from these same raw padded rows. Kept as a
SEPARATE dataset directory (not added to addition_padded's shared vocab)
specifically so existing checkpoints trained against that dataset's
vocab_size/token-id mapping are never put at risk of silently shifting.

Reads train.txt / val.txt from the sibling data/addition/ dir (same source
data every other dataset variant here uses).

Produces train.bin / val.bin (uint16, shape (N, MAX_LEN)), train_lengths.bin
/ val_lengths.bin (int32, shape (N,)), train_mask.bin / val_mask.bin (uint8,
same convention as addition_padded -- 0 at operand-digit/+/= positions, 1
elsewhere), and meta.pkl (vocab_size, stoi, itos, pad_id, mask_id, max_len).
"""
import os
import pickle
import numpy as np

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'addition')
OUT_DIR = os.path.dirname(__file__)
MAX_LEN = 64

PAD_CHAR = '#'
MASK_CHAR = '?'
chars = sorted(list("0123456789+=\n") + [PAD_CHAR, MASK_CHAR])
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
pad_id = stoi[PAD_CHAR]
mask_id = stoi[MASK_CHAR]

print("vocab:", repr(''.join(chars)), "pad_id:", pad_id, "mask_id:", mask_id)


def encode(s):
    return [stoi[c] for c in s]


def compute_keep_mask(text):
    """Same convention as data/addition_padded/prepare.py: 1 = valid loss
    target (answer digit, final newline), 0 = operand digit / '+' / '='."""
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
    'mask_id': mask_id,
    'max_len': MAX_LEN,
}
with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
print("saved meta.pkl")
