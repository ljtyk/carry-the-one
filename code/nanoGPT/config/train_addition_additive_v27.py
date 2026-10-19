# Stage 21: does the digit codebook already want phase/circular structure,
# and if not, can we get it there?
#
# probe_codebook_geometry.py on v22 found essentially no phase structure:
# mean nearest-angular-neighbor numeric offset 1.9 (1.0 = perfect phase
# order, ~3.3 = random permutation), and the top-3 singular values of the
# centered digit_direction matrix were all close together (1.875/1.856/
# 1.719) -- the codebook genuinely uses all 3 dims, not collapsed to a
# plane. Decode's classification loss only rewards separation, which is
# permutation-invariant, so nothing pushed a random init toward the
# arithmetically meaningful order even if the combiner would benefit from
# digit-sum = angle-addition being learnable as a near-linear function
# instead of an arbitrary 10x10 lookup.
#
# Two changes under test together:
#   codec_dims_per_place: 3 -> 2 -- direction becomes a literal angle (1
#     remaining DOF per digit) instead of a 2-DOF point on a sphere that
#     can dodge committing to a strict cyclic order.
#   codec_seed_numeric_order = True -- digit_direction initialized at
#     consecutive angles (2*pi*d/10) instead of random. NOT frozen --
#     still an ordinary learnable Parameter, free to drift during
#     training. Tests whether numeric order is a stable/attracting
#     arrangement (combiner's own gradient holds or sharpens it) or one
#     nothing anchors (it wanders back toward an arbitrary permutation).
#
# From scratch, not warm-started -- digit_direction's shape itself changes
# (10,3)->(10,2), no clean weight reuse. Everything else held at v22's
# settings so this is a clean, isolated comparison: codec_use_magnitude,
# zero_pad, 4 thinking tokens, repeat_max=24, composed_train_max_digits=20,
# max_answer_digits=30 all unchanged from v22.
#
# snapshot_interval lowered to 1000 (from v22's 5000) specifically so
# probe_codebook_geometry.py can be run against intermediate checkpoints
# to see WHEN/whether the seeded order starts drifting, not just the
# converged end state.

out_dir = 'out-addition-additive-v27'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 1000  # <-- lowered from v22's 5000, for drift tracking

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage21-phase-codebook-2d-seeded'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 4

model_type = 'additive_composed'
combiner_layers = 1
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'

composed_recon_weight = 0.5
composed_consistency_weight = 0.5
max_answer_digits = 30

combiner_repeat_max = 24
combiner_countdown_weight = 0.1
combiner_n_thinking = 4
combiner_use_per_place = True
combiner_explicit_place_tag = False
combiner_magnitude_signal = True

codec_type = 'slot'
codec_dims_per_place = 2  # <-- was 3; forces direction to be a literal angle
codec_use_magnitude = True
codec_max_scale = 3.0
codec_zero_pad = True
codec_seed_numeric_order = True  # <-- the change under test

composed_train_max_digits = 20  # unchanged from v22, deliberately
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
