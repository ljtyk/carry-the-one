# Stage 15: QAM-style slot codec -- codec_use_magnitude=True. Direction
# and magnitude are separate, decoupled parameters (digit_direction +
# bounded digit_scale_raw), magnitude survives into the written slot
# instead of being normalized away, and decode uses the energy-corrected
# rule (dot product minus 0.5*||class||^2, i.e. squared-Euclidean-
# distance ranking) instead of raw dot product -- the correct decision
# rule once class energies vary, and critically it removes the "just
# scale every class vector up uniformly" free lunch a plain dot-product
# softmax would otherwise have.
#
# Motivated directly by two things confirmed empirically this session:
# (1) a pure-direction (unit-norm) codebook only uses the sphere's
# SURFACE, not its volume -- independently-optimized best packing for
# N=11 points in 3D reached ~60.6deg minimum separation, while several
# independently-seeded direction-only training runs all landed short of
# that (45-53deg), in a different, arbitrary configuration each time
# (near-zero structure correlation across seeds) -- a real, avoidable
# local-minimum problem, not fundamental to N=11-in-3D.
# (2) even under the OLD scheme, where magnitude was completely
# functionally inert (erased by per-place normalize every forward pass),
# raw digit_embed norms grew ~100-150x from init with no sign of
# plateauing across 10000 iterations -- meaning simply removing
# normalize without the energy correction and without an explicit bound
# would very likely have produced unbounded blow-up instead of a
# meaningful amplitude code, not a safe change to make on its own.
#
# codec_max_scale=3.0: an arbitrary-but-reasonable ceiling (codewords
# start at 1.5, the sigmoid midpoint) -- worth revisiting once we see
# where scale actually wants to sit under real training pressure.
#
# From scratch -- direction+scale is a different parameterization than
# the plain nn.Embedding used by v20/v20b/v20c/v20d, no compatible
# checkpoint to warm-start from.

out_dir = 'out-addition-additive-v21'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage15-slot-codec-qam'

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

codec_type = 'slot'
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0

composed_train_max_digits = 20
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
