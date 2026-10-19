# Stage 15d: same recipe as v21c, warm-started from it, with
# composed_train_max_digits widened 20 -> 25 so place indices 20-24 get
# real (non-blank) training exposure -- same fix in kind as v13->v17,
# no architectural change, no boundary hints, no masking. Just lets the
# combiner learn the calibration directly from wider data, the same way
# it learned everything else.
#
# max_answer_digits stays 30 (already covers a 25-digit train max's
# sums, up to 26 digits, with headroom left for testing extrapolation
# past the new boundary). combiner_repeat_max 24 -> 28, tracking the
# wider max length (carry propagation depth needs to scale with problem
# size, established earlier this session).

out_dir = 'out-addition-additive-v21d'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v21c/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage15d-slot-codec-qam-wider-range'

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

combiner_repeat_max = 28
combiner_countdown_weight = 0.1
combiner_n_thinking = 4
combiner_use_per_place = True
combiner_explicit_place_tag = False
combiner_magnitude_signal = True

codec_type = 'slot'
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0

composed_train_max_digits = 25  # <-- the change under test (was 20)
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 15000
lr_decay_iters = 15000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
