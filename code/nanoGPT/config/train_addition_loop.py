# Stage 2: single shared block (attn+FFN together) looped floor(n)+1 times,
# n learned. Same width/heads/training regimen as train_addition.py's
# fixed-depth baseline for a direct, matched-settings comparison -- the
# single block is NOT widened to match total baseline param count (Option A:
# tests whether weight-sharing-across-depth works about as well as distinct
# weights per layer, per the plan's own framing of the "known issue").

out_dir = 'out-addition-loop'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage2-looped'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

model_type = 'looped'
n_layer = 6  # used only for c_proj init scaling in LoopedGPT, matches n_init below
n_head = 6
n_embd = 384
dropout = 0.2
max_loop_iters = 20
n_init = 6.0

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
