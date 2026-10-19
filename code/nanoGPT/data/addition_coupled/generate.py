"""
Reformats the existing (a, b, a+b) triples from data/addition/*.txt into the
position-coupling data format (Cho et al. 2024): both operands zero-padded
to the same length L (so digit i of a and digit i of b share the same
place-value alignment), the sum zero-padded to L+1 digits (room for carry
overflow) and then REVERSED (least-significant digit first).

Example: 653 + 49 = 702  ->  "653+049=2070"
(702 zero-padded to 4 digits = "0702", reversed = "2070")

Position IDs themselves are NOT stored here -- get_batch_coupled (train.py)
computes them on the fly from each row's own content (locations of '+'/'='
directly give the padded operand length L), since the format is fully
self-describing.

Reads every split from the sibling data/addition/ dir (run
data/addition/generate.py first if those don't exist).
"""
import os

SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'addition')
OUT_DIR = os.path.dirname(__file__)

SPLITS = ['train', 'val', 'test_indist', 'test_gap', 'test_hint', 'test_ood', 'test_far_ood',
          'test_carry_heavy_gap', 'test_carry_heavy_ood']


def reformat_line(line):
    a_str, rest = line.split('+')
    b_str, c_str = rest.split('=')
    a, b = int(a_str), int(b_str)
    c = a + b
    L = max(len(a_str), len(b_str))
    padded_a = a_str.zfill(L)
    padded_b = b_str.zfill(L)
    padded_c = str(c).zfill(L + 1)
    reversed_c = padded_c[::-1]
    return f"{padded_a}+{padded_b}={reversed_c}"


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
