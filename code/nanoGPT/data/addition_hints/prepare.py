"""
Same one-problem-per-sequence, right-padded format as data/addition_padded/
prepare.py, applied to the index-hints reformatted data (generate.py in this
dir). Vocab extends the usual digit/+/=/\\n/pad set with 27 index-tag
characters (a-z plus '!', covering place values 0-26 -- see generate.py).
"""
import os
import pickle
import numpy as np

OUT_DIR = os.path.dirname(__file__)
MAX_LEN = 165  # covers far_ood's reformatted lines (up to 160 chars) + leading/trailing \n

PAD_CHAR = '#'
INDEX_CHARS = [chr(ord('a') + i) for i in range(26)] + ['!']
chars = sorted(list("0123456789+=\n") + [PAD_CHAR] + INDEX_CHARS)
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
pad_id = stoi[PAD_CHAR]

print("vocab:", repr(''.join(chars)), "pad_id:", pad_id, "vocab_size:", vocab_size)


def encode(s):
    return [stoi[c] for c in s]


def compute_keep_mask(text):
    """1 = valid loss target (answer digit/index-tag, final newline), 0 =
    operand digit/index-tag / '+' / '='. Same convention as
    data/addition_padded/prepare.py -- operand content (now including its
    index tags) is still fully determined by random sampling, not by the
    arithmetic, so still excluded from the loss."""
    keep = np.ones(len(text), dtype=np.uint8)
    in_operand = True
    for j, ch in enumerate(text):
        if in_operand and (ch.isdigit() or ch.isalpha() or ch == '!'):
            keep[j] = 0
        if ch in ('+', '='):
            keep[j] = 0
        if ch == '=':
            in_operand = False
        elif ch == '\n':
            in_operand = True
    return keep


def prepare_split(name):
    path = os.path.join(OUT_DIR, f'{name}.txt')
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
    'max_len': MAX_LEN,
}
with open(os.path.join(OUT_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
print("saved meta.pkl")
