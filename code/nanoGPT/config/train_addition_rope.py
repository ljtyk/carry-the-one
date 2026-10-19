# Stage 1 baseline re-run with RoPE (rotary position embedding) instead of
# nanoGPT's default learned absolute wpe. Tests whether the persistent 0%
# OOD result across every prior stage is a positional-encoding artifact
# (wpe never learned representations for digit positions beyond the trained
# 1-10 digit range) rather than a genuine depth/adaptive-compute limitation.
# Same architecture and training budget as train_addition.py otherwise, for
# a clean two-way comparison.

out_dir = 'out-addition-rope'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'baseline-rope'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
