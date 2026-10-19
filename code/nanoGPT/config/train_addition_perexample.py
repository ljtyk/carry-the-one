# Global-per-problem adaptive depth: one shared block, one n predicted per
# training example (not per token), reusing Stage 2's exact floor(n)+1 /
# frac(n) blend. Requires the one-problem-per-sequence padded dataset.

out_dir = 'out-addition-perexample'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage2.1b-per-example'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64

model_type = 'per_example_looped'
n_layer = 6  # used only for c_proj init scaling convention, not an actual layer count here
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
