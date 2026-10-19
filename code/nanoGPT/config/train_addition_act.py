# Stage 2.1: same single shared block + loop-index embedding as
# train_addition_loop_idx.py, but the loop count is now decided per-token
# via ACT-style adaptive halting instead of one global learned scalar.

out_dir = 'out-addition-act'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage2.1-act'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

model_type = 'act_looped'
n_layer = 6  # used only for c_proj init scaling convention, not an actual layer count here
n_head = 6
n_embd = 384
dropout = 0.2
max_loop_iters = 20
loop_index_embed = True
act_epsilon = 0.01
ponder_cost_weight = 1e-3

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
