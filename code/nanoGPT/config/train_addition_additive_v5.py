# Stage 10e: same as v4, but composed_contrib_sparsity_weight raised from
# 0.05 to 0.2. v4 ran 20000/50000 steps with 0.05 and showed zero movement
# in bit efficiency (place0/4/9 dim counts stayed within noise of v3's
# pre-sparsity numbers the whole time) -- stopped rather than continue
# waiting, since a weight with no visible effect at 40% through isn't going
# to suddenly matter more as the other losses (already tiny) shrink further.
#
# Warm-started from v3 directly (NOT from v4's partially-trained state) --
# v4's 20k steps under the weak weight didn't move anything, so there's no
# reason to inherit that state over v3's clean, best-performing checkpoint.
# Same lower learning_rate=2e-4 validated for warm-starting in v4's smoke
# tests (1e-3 destabilizes an already-converged checkpoint even through
# full warmup).

out_dir = 'out-addition-additive-v5'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v3/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage10e-additive-composed-sparsity-0.2'

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
composed_contrib_sparsity_weight = 0.2
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
