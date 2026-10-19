# Stage 3 pass 1, parameter-matched: same two-block design as
# train_addition_twoblock.py (A loops a fixed number of times, then B runs
# once), but each block is narrowed (n_embd=270 instead of 384) so the two
# blocks TOGETHER match the ~1.78M total params every single-block stage
# (Stage 2, ACT, per-example) used -- isolating whether block specialization
# itself helps, controlled for total parameter budget, rather than
# confounding it with simply having 2x the parameters.

out_dir = 'out-addition-twoblock-matched'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage3-pass1-twoblock-matched'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

model_type = 'two_block'
n_head = 6
n_embd = 270  # 2 blocks at this width = ~1.78M total, matching the single-block stages
dropout = 0.2
max_loop_iters = 20
loop_index_embed = True
n_loop_a = 6

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
