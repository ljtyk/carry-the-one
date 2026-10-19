# Stage 16: codec_zero_pad=True -- eliminates the blank class entirely.
# Every operand/sum is represented at the fixed width max_answer_digits
# via ordinary leading-zero decimal padding ("42" -> "000...042"), so
# every place in every training example always has a genuine digit
# target (0-9). No masking, no boundary hints, no externally-computed
# length -- just a different, uniform representation of the same
# numbers, per project discussion.
#
# composed_train_max_digits stays at 20, UNCHANGED from v21c -- this is
# a deliberate, clean isolation: if the L=22 boundary cliff goes away or
# improves, it's attributable to removing blank alone, not to also
# widening the training range (which would confound the comparison).
# combiner_repeat_max also stays at 24, unchanged, same reasoning.
#
# From scratch (not warm-started) -- the representation itself changed
# (every input is now always the full 30-place zero-padded sequence, no
# blank class at all), so training starts fresh rather than carrying
# over weights calibrated for the old blank-padded distribution.

out_dir = 'out-addition-additive-v22'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage16-slot-codec-zero-pad-no-blank'

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
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0
codec_zero_pad = True  # <-- the change under test

composed_train_max_digits = 20  # unchanged from v21c, deliberately
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
