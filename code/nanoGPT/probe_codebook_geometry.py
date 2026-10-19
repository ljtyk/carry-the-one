"""
Does the learned digit codebook already have phase/circular structure?

Question being tested: digit_direction lives in codec_dims_per_place dims
(3 for v22/v26) -- more than enough room for a 2D circular arrangement if
that's what training pushed toward. If digit value already maps to a
roughly monotonic rotation around some plane (phase encoding), imposing it
explicitly wouldn't add much. If the codebook is just uniformly separated
with NO respect to numeric adjacency (i.e. nearest-neighbor pairs in angle
space are numerically unrelated, e.g. 2 next to 7), that's evidence the
decode-classification loss (which only rewards separation, not arithmetic
structure) dominated and the combiner's gradient never got a chance to
shape it -- direct motivation for imposing phase structure explicitly.

Usage: python probe_codebook_geometry.py <ckpt_dir>
"""
import sys
import torch
import torch.nn.functional as F

ckpt_dir = sys.argv[1] if len(sys.argv) > 1 else 'out-addition-additive-v22'
ckpt_name = sys.argv[2] if len(sys.argv) > 2 else 'ckpt.pt'
ckpt = torch.load(f'{ckpt_dir}/{ckpt_name}', map_location='cpu', weights_only=False)
sd = ckpt['model']
ma = ckpt['model_args']
print(f"{ckpt_dir}/{ckpt_name}: iter={ckpt.get('iter_num')} dpp={ma.get('codec_dims_per_place')} "
      f"use_magnitude={ma.get('codec_use_magnitude')} max_scale={ma.get('codec_max_scale')}")

direction_raw = sd['digit_direction']  # (10, dpp)
scale_raw = sd['digit_scale_raw']      # (10,)
direction = F.normalize(direction_raw, dim=-1)
scale = ma.get('codec_max_scale', 3.0) * torch.sigmoid(scale_raw)

print("\nmagnitude per digit (0..9):")
print(scale.numpy().round(3))

# pairwise angle matrix (degrees)
cos_sim = direction @ direction.T
cos_sim = cos_sim.clamp(-1, 1)
angles = torch.acos(cos_sim) * 180 / torch.pi
print("\npairwise angle matrix (degrees), rows/cols = digit 0..9:")
for i in range(10):
    print(' '.join(f'{angles[i,j]:5.1f}' for j in range(10)))

# nearest neighbor (by angle) for each digit -- is it numerically adjacent?
print("\nnearest angular neighbor per digit:")
nn_offsets = []
for i in range(10):
    row = angles[i].clone()
    row[i] = 1e9
    j = row.argmin().item()
    offset = min((j - i) % 10, (i - j) % 10)  # circular numeric distance
    nn_offsets.append(offset)
    print(f"  digit {i}: nearest = {j}  (angle {row[j]:.1f} deg, numeric offset {offset})")
print(f"  mean numeric offset of nearest neighbor: {sum(nn_offsets)/10:.2f}  "
      f"(0 = perfectly phase-ordered, ~row of a random permutation's expected offset ~= 3.3 for N=10)")

# PCA: find best-fit 2D plane, project, check if digit order traces a
# monotonic/consistent rotation (phase encoding) vs scattered.
centered = direction - direction.mean(dim=0, keepdim=True)
U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
print(f"\nsingular values of centered directions: {S.numpy().round(3)} "
      f"(if top-2 dominate, codebook is ~coplanar / circle-like)")
proj2d = centered @ Vt[:2].T  # (10, 2)
proj_angles = torch.atan2(proj2d[:, 1], proj2d[:, 0]) * 180 / torch.pi
proj_angles = (proj_angles + 360) % 360
order = torch.argsort(proj_angles)
print("\ndigits sorted by angular position in top-2 PCA plane:")
print("  digit order:", order.tolist())
print("  angles:     ", [f'{proj_angles[d]:.1f}' for d in order.tolist()])
print("  (phase encoding would show this as 0,1,2,...,9 or a rotation/reflection of it)")
