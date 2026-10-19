# Stage 3 pass 1: two distinct blocks -- A loops a FIXED number of times
# (no adaptive halting yet), then B runs once as a terminal readout.
# Isolates whether block specialization helps at all, before pass 2 adds
# per-input adaptive halting on top of this same two-block structure.

out_dir = 'out-addition-twoblock'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage3-pass1-twoblock'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

model_type = 'two_block'
n_head = 6
n_embd = 384
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
