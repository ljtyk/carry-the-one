# Stage 10h: same as v7, weight halved from 0.02 to 0.01.
#
# v7 (0.02) confirmed the sparsity mechanism genuinely works once the
# model_args wiring bug was fixed -- Hoyer ratio crashed from ~12.6 to 2.2
# in just 7000/50000 steps (14%), and bit efficiency at iter 10000 showed a
# real, large effect: place0 191->43, place4 271->85, place9 287->14 dims.
# But that speed/magnitude raised a real concern: place9's extreme
# collapse (14 dims, one dimension carrying variance 9.36) likely reflects
# weak opposing pressure (place9 is undertrained/rarely-seen even under
# skewed sampling, so CE/consistency push back on it far less than on
# place0) rather than a genuinely better, well-supervised representation --
# and the combiner was visibly lagging the encoder's pace (vec_c only
# tightened to 99/206/256 vs vec_a's 43/85/14, and total loss ballooned to
# ~1.0-1.1 driven by the consistency term chasing a fast-moving target,
# even though robust CE stayed low throughout). Dropping to 0.01 to see if
# a gentler pace lets the combiner co-adapt rather than chase.

out_dir = 'out-addition-additive-v8'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v3/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage10h-additive-composed-hoyer-sparsity-0.01'

dataset = 'addition_padded'  # only used for meta.pkl's vocab ('+'/'=' ids for the combiner) -- get_batch_additive generates on the fly
gradient_accumulation_steps = 1
batch_size = 64
block_size = 4  # the combiner's fixed 4-slot sequence length -- encoder/decoder don't use block_size/RoPE at all

model_type = 'additive_composed'
combiner_layers = 4
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'  # combiner only

composed_recon_weight = 0.5
composed_consistency_weight = 0.5
composed_contrib_sparsity_weight = 0.01
max_answer_digits = 30  # comfortably past far_ood's 26 digits

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 50000
lr_decay_iters = 50000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False  # dynamic per-batch operand shapes defeat torch.compile, same as ComposedGPT
