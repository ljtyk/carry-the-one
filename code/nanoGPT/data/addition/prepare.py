"""
Prepare the synthetic addition dataset for character-level language modeling,
mirroring data/shakespeare_char/prepare.py so it plugs into train.py unmodified.

Run generate.py first to produce train.txt / val.txt / test_indist.txt / test_ood.txt.
This script encodes train.txt and val.txt into train.bin / val.bin (uint16 ids)
plus meta.pkl (encoder/decoder). The held-out test files are left as plain text
for the eval harness, which needs per-example generation, not a loss over a
concatenated byte stream.

Vocab is fixed (not derived from the data) so that test files -- which are
encoded separately by the eval harness using this same meta.pkl -- can never
contain a character missing from train's alphabet.

Also writes train_mask.bin / val_mask.bin (uint8, same shape as the ids
arrays): 0 at operand-digit positions and at '+'/'=' (all governed by the
randomly-sampled operand lengths, not by the addition itself -- see
compute_keep_mask), 1 at answer digits and the final '\n' (which reflects a
genuine arithmetic fact: whether the sum carried into an extra digit).
Predicting the masked positions is pure irreducible noise that dilutes the
gradient signal actually reflecting whether the model is learning the
addition procedure. get_batch (train.py) uses this mask to set those target
positions to ignore_index=-1 when the file is present.
"""
import os
import pickle
import numpy as np

DATA_DIR = os.path.dirname(__file__)

chars = sorted(list("0123456789+=\n"))
vocab_size = len(chars)
print("all the unique characters:", repr(''.join(chars)))
print(f"vocab size: {vocab_size:,}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return ''.join([itos[i] for i in l])


def compute_keep_mask(text):
    """1 = valid loss target (answer digit, final newline), 0 = operand
    digit, '+', or '='. Predicting '+'/'=' requires knowing an operand just
    ended, but operand length is randomly sampled -- same root-cause noise
    as the digit values themselves, not a property of the addition. The
    final '\n', by contrast, reflects whether the sum carried into an extra
    digit -- a genuine arithmetic fact -- so it stays a valid target."""
    keep = np.ones(len(text), dtype=np.uint8)
    in_operand = True  # start of first line: reading digits of operand a
    for j, ch in enumerate(text):
        if in_operand and ch.isdigit():
            keep[j] = 0
        if ch in ('+', '='):
            keep[j] = 0
        if ch == '=':
            in_operand = False
        elif ch == '\n':
            in_operand = True  # next line starts a new operand a
    return keep


with open(os.path.join(DATA_DIR, 'train.txt'), 'r') as f:
    train_data = f.read()
with open(os.path.join(DATA_DIR, 'val.txt'), 'r') as f:
    val_data = f.read()

print(f"train has {len(train_data):,} characters")
print(f"val has {len(val_data):,} characters")

train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids.tofile(os.path.join(DATA_DIR, 'train.bin'))
val_ids.tofile(os.path.join(DATA_DIR, 'val.bin'))

train_mask = compute_keep_mask(train_data)
val_mask = compute_keep_mask(val_data)
print(f"train: {train_mask.mean()*100:.1f}% of positions kept as loss targets (rest are masked operand digits)")
print(f"val:   {val_mask.mean()*100:.1f}% of positions kept as loss targets")
train_mask.tofile(os.path.join(DATA_DIR, 'train_mask.bin'))
val_mask.tofile(os.path.join(DATA_DIR, 'val_mask.bin'))

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(DATA_DIR, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
