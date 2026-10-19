# Stage 14b: same slot-codec run as v20, warm-started from v20's own
# checkpoint (iter 12450, robust val loss already 0.041 despite the bug)
# after fixing the blank-position auxiliary-loss bug found by reasoning
# through exactly how digit_embed/blank_embed get trained -- see
# project discussion and model.py's forward() comment at the masking fix.
# No architecture/shape change at all (the fix only changes which
# positions feed the recon loss), so this warm-starts cleanly, no
# shape-mismatch handling needed unlike other warm-starts this session.

out_dir = 'out-addition-additive-v20b'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v20/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage14b-slot-codec-blank-fix'

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

composed_train_max_digits = 20
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 30000
lr_decay_iters = 30000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
