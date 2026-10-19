"""
Reformats the existing (a, b, a+b) triples from data/addition/*.txt into the
INDEX HINTS format: each digit is prefixed with a letter tagging its place
value (units=index 0, tens=index 1, ...), the SAME letter used for that
place value across both operands AND the answer -- letting the model
exploit ordinary attention/induction-head copying (find the matching index
letter, read the digit after it) instead of needing hidden position
machinery (position coupling) or a reordered output (reversal). Digit order
stays NORMAL (most-significant-digit-first), unlike position coupling.

Example: 653 + 49 = 702, place values 0(units)/1(tens)/2(hundreds):
  operand a=653 (padded to match b's length isn't needed here since each
  digit carries its own place-value tag): c2b5a3 -- wait, written as
  produced below: digits tagged left-to-right, i.e. "c6b5a3" for 653
  (hundreds='c' tag on 6, tens='b' tag on 5, units='a' tag on 3).
  b=49 -> "b4a9". c=702 -> "c7b0a2".
  Full line: "c6b5a3+b4a9=c7b0a2"

INDEX_CHARS covers place values 0-26 (single characters), enough for every
digit length used anywhere in this project (far_ood tops out at 26 digits,
needing place value up to 26 for the carry-overflow digit).
"""
import os

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'addition')
OUT_DIR = os.path.dirname(__file__)

SPLITS = ['train', 'val', 'test_indist', 'test_gap', 'test_hint', 'test_ood', 'test_far_ood',
          'test_carry_heavy_gap', 'test_carry_heavy_ood']

INDEX_CHARS = [chr(ord('a') + i) for i in range(26)] + ['!']  # place values 0..26


def tag(number_str):
    L = len(number_str)
    return ''.join(f"{INDEX_CHARS[L - 1 - i]}{ch}" for i, ch in enumerate(number_str))


def reformat_line(line):
    a_str, rest = line.split('+')
    b_str, c_str = rest.split('=')
    a, b = int(a_str), int(b_str)
    c = a + b
    return f"{tag(a_str)}+{tag(b_str)}={tag(str(c))}"


for split in SPLITS:
    src_path = os.path.join(SRC_DIR, f'{split}.txt')
    if not os.path.exists(src_path):
        continue
    with open(src_path) as f:
        lines = [l for l in f.read().splitlines() if l]
    reformatted = [reformat_line(l) for l in lines]
    out_path = os.path.join(OUT_DIR, f'{split}.txt')
    with open(out_path, 'w') as f:
        for l in reformatted:
            f.write(l + '\n')
    max_len = max(len(l) for l in reformatted) if reformatted else 0
    print(f"{split}: {len(reformatted)} examples, max reformatted line length {max_len}")
