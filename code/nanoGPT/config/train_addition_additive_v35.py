# Stage 29: combiner_use_per_place=False under the modern recipe --
# "let's try the combiner first" (see project discussion, the single-
# token encoding idea). Isolates whether the combiner needs per-place
# token resolution to run its carry wavefront on, or can learn carry
# propagation as in-place state evolution on just 2 tokens (whole vec_a,
# whole vec_b) instead.
#
# Encoder and decoder UNCHANGED from v22 -- still the structural,
# non-learned disjoint-slot + zero-pad alignment (deliberately, per
# project discussion: this project's Stage 1-9 arc already showed
# length generalization needs non-learned position alignment; testing
# that again isn't the point here). Only the combiner's INPUT sequence
# changes: combine() feeds it exactly 2 tokens (vec_a, vec_b) + 4
# thinking tokens instead of 30 per-place tokens + 4 thinking tokens.
# decode_from_vector is untouched -- it just slices whatever 384-dim
# vector the combiner hands back into 30 disjoint 3-dim chunks, same as
# always, regardless of how that vector was produced internally.
#
# combiner_magnitude_signal forced False -- the existing assertion
# requires combiner_use_per_place=True for it (mag_bias_place is only
# ever built in the per-place branch; there's no per-place ||qa||,||qb||
# to re-inject when there's only one whole-operand token per side). This
# is a real second variable, not just per-place resolution alone --
# noted, not hidden.
#
# Everything else identical to v22: codec_type='slot', codec_use_magnitude
# =True, codec_zero_pad=True, dpp=3, combiner_n_thinking=4,
# combiner_repeat_max=24, composed_train_max_digits=20, max_answer_digits
# =30. From scratch, not warm-started -- combiner_proj goes entirely
# unused in this branch (the 2 tokens are fed raw, un-projected) and
# combiner_block's learned attention patterns are tuned for nearest-
# neighbour routing among ~30 tokens, not a meaningful init for a
# 6-token sequence.

out_dir = 'out-addition-additive-v35'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 1000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage29-combiner-no-per-place-resolution'

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
combiner_use_per_place = False  # <-- the change under test: 2 tokens (vec_a, vec_b), not 30 per-place tokens
combiner_explicit_place_tag = False
combiner_magnitude_signal = False  # <-- forced off, see comment above (asserted incompatible with combiner_use_per_place=False)

codec_type = 'slot'
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0
codec_zero_pad = True

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
