# Stage 23: linear-magnitude codec. Every digit shares the SAME direction
# (codec_dims_per_place=1, no direction freedom at all); only magnitude
# varies with value, seeded roughly linear in d (0..codec_max_scale).
#
# Motivated by working through what "combiner adds the two vectors, then
# does a carry op" actually requires: under phase/circular encoding
# (v27/v28's codec_seed_numeric_order, angle = value), digit-sum mod 10 is
# angle ADDITION, which means the codewords must be COMPLEX-MULTIPLIED,
# not vector-added -- a bilinear operation, and the carry bit (how many
# full wraps happened) isn't recoverable from the resulting angle at all,
# with no spare channel to carry it (magnitude was saturated to a
# constant). Under THIS scheme, vector addition of two operands' codewords
# literally IS digit addition (magnitudes sum), and carry-out is a plain
# magnitude-threshold comparison -- a much more literal match to the
# combiner's presumed mechanism, and no packing/order problem to solve at
# all since order is definitionally linear (0 < 1 < ... < 9), not an
# arbitrary permutation the way circular adjacency was.
#
# v27/v28 also showed that dpp=2 alone (independent of order) roughly 10x'd
# the loss at the same iter count vs dpp=3 (v22) -- a real capacity/
# optimization cost from cutting a dimension, not just an OOD-margin
# story. codec_dims_per_place=1 here cuts even further, so that capacity
# hit is a real risk worth watching for in this run too; if convergence is
# similarly or more degraded, that's evidence the capacity cost dominates
# regardless of which structure replaces the lost dimension.
#
# From scratch (new digit-encoding scheme, no clean weight reuse).
# Everything else at v22-scale settings for a clean three-way comparison:
# codec_zero_pad=True (required -- no blank codeword implemented for this
# scheme), 4 thinking tokens, repeat_max=24, composed_train_max_digits=20,
# max_answer_digits=30. combiner_magnitude_signal=True is especially
# important here since magnitude is the ONLY information channel at all
# (no direction component whatsoever).
#
# snapshot_interval=1000, same as v27/v28, for tracking whether magnitude
# order stays monotonic (probe_codebook_geometry.py would need a small
# magnitude-specific variant, or just print digit_magnitude_raw directly,
# since there's no angle/PCA structure to probe here -- it's a 1D ladder).

out_dir = 'out-addition-additive-v29'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 1000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage23-linear-magnitude-codec'

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
codec_dims_per_place = 1  # <-- was 3 (v22) / 2 (v27/v28); pure scalar-magnitude code
codec_use_magnitude = False  # mutually exclusive with codec_linear_magnitude
codec_max_scale = 3.0
codec_zero_pad = True
codec_linear_magnitude = True
codec_zero_class_weight = 0.082  # <-- the new change under test: measured inverse frequency (digit 0 is ~12.2x overrepresented under padding at these length settings)

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
