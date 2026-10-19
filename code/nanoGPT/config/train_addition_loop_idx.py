# Stage 2 fallback: same as train_addition_loop.py (single shared block,
# looped), but with a learned per-iteration embedding added to the residual
# stream each loop -- tests whether giving the shared block a "which
# iteration am I on" signal fixes the naive-looping underperformance
# (0.33% indist accuracy vs. the fixed-depth baseline's 65-85%).

out_dir = 'out-addition-loop-idx'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage2-looped-idx'

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
loop_index_embed = True

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
