# Stage 15b: same QAM-style slot codec as v21, warm-started from v21's
# finished checkpoint, with codec_max_scale raised 3.0 -> 8.0. Pure
# config change (max_scale is a multiplier applied at forward time,
# doesn't touch any parameter's SHAPE -- digit_direction/digit_scale_raw/
# etc. are identical shapes regardless of its value), so this warm-starts
# with zero shape mismatches, unlike most other warm-starts this session.
#
# Purpose: test the prediction from project discussion -- all 10 digit
# scales converged to nearly identical, saturated values under
# max_scale=3.0 (raw pre-sigmoid values 6.64-6.90, effectively frozen by
# sigmoid saturation, not because they found a genuinely differentiated
# optimum). Reasoned through the math: for the recon/consistency
# pathways, query IS the class vector being classified against, so
# uniformly rescaling the whole constellation reduces loss "for free"
# (correct-class distance stays exactly 0, wrong-class distances grow)
# without requiring any GENUINE new angular/digit-vs-digit separation --
# predicted this would mostly show up as rescaling (digits still convergge
# to ~the same new, larger magnitude) rather than actual differentiation,
# since nothing about raising the ceiling creates a NEW reason for
# different digits to want different scales. This run tests that
# prediction directly.
#
# Lower learning_rate (2e-4, standard warm-start recipe this session) --
# note the raw scale params (currently ~6.6-6.9) will jump in EFFECTIVE
# magnitude the instant this loads (scale = 8.0*sigmoid(6.8) =~ 7.99
# instead of 3.0*sigmoid(6.8) =~ 3.0), a real discontinuity at warm-start
# time, not a bug -- the lower LR is exactly to let training re-settle
# smoothly through that transient rather than violently.

out_dir = 'out-addition-additive-v21b'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v21/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage15b-slot-codec-qam-max-scale-8'

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
codec_max_scale = 8.0  # <-- the change under test (was 3.0)

composed_train_max_digits = 20
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 15000
lr_decay_iters = 15000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
