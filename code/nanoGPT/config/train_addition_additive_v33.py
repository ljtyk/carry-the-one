# Stage 27: v26's full-capacity config (max_answer_digits=128,
# combiner_repeat_max=80, composed_train_max_digits=64) PLUS
# combiner_variable_repeat=True.
#
# Motivation (mech-interp probes A-F on v22, then A/E on v23):
#   - v22 (fixed 24 rounds): the combiner never converges to a fixed point
#     -- monotone state drift, accuracy plateaus r24-48 then decays. It
#     also runs the full 24 rounds on every problem regardless of need.
#   - v23 (= v22 + variable_repeat): the drift is GONE for the trained
#     length range -- indist/gap/hint/ood stay dead flat from r24 to r64
#     -- and it converges ~2x faster (mean round 11 vs 16). BUT far_ood
#     (24-26 digits) REGRESSED 0.77 -> 0.645: converging fast and holding
#     means a carry chain too long to finish by ~round 11-22 locks in a
#     wrong answer. v22's messier "keep churning to 24" was accidentally
#     carrying the longest chains across the line.
#   - The natural carry wavefront moves at ~1.85 rounds/place (probe F4),
#     so a 25-digit maximal carry chain needs ~46 rounds. v23's budget
#     (24) and fast-converge behavior both cap it short.
#
# This run combines both fixes: variable_repeat (stability + efficiency)
# AND a budget/training-range wide enough for genuine long carry chains
# (repeat_max=80 covers ~40-digit worst-case chains; train_max_digits=64
# gives real exposure). The hypothesis: variable_repeat lets the shared
# block hold a converged state indefinitely, so the larger round budget
# is finally safe to use rather than corrupting -- which is exactly what
# v26 (same scale, FIXED repeat) could not do, and plausibly why v26's
# loss plateaued higher than v22 (tons of wasted over-rounding on the
# short problems that dominate the length distribution).
#
# combiner_variable_thinking stays False (v25 dead end; isolate the
# repeat change). Warm-started from v22 exactly as v26/v23 were:
# digit_direction/digit_scale_raw/combiner_proj/combiner_out_slot/
# combiner_mag_proj load exactly; countdown_head (repeat_max+1 = 81 rows
# now) and the combiner block's RoPE/attention buffers (block_size =
# max_answer_digits + n_thinking = 132) are shape-mismatched and left
# fresh -- correct, RoPE is deterministic and countdown_head genuinely
# needs the new size.
#
# Compute: 132-token sequences x up to 80 rounds -- ~600ms/iter on this
# box (same as v26), ~2.5-3h for the full run plus periodic eval.

out_dir = 'out-addition-additive-v33'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v22/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage27-full-capacity-variable-repeat'

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
max_answer_digits = 128

combiner_repeat_max = 80
combiner_countdown_weight = 0.1
combiner_n_thinking = 4
combiner_use_per_place = True
combiner_explicit_place_tag = False
combiner_magnitude_signal = True
combiner_variable_repeat = True   # <-- the change vs v26
combiner_variable_thinking = False

codec_type = 'slot'
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0
codec_zero_pad = True

composed_train_max_digits = 64
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 15000
lr_decay_iters = 15000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
