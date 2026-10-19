# Stage 2.1b recreated on the restarted dataset: global-per-problem adaptive
# depth (one shared block, one n predicted per training example from a
# pooled "look pass" representation, reusing Stage 2's floor(n)+1/frac(n)
# blend), now with RoPE instead of learned absolute position embedding, and
# trained on the regenerated data/addition_padded (mismatched operand
# lengths, four length tiers baked in via the underlying data/addition
# train.txt/val.txt this was re-prepared from). Same hyperparameters as the
# pre-restart train_addition_perexample.py otherwise.

out_dir = 'out-addition-perexample-rope'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage2.1b-per-example-rope'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64

model_type = 'per_example_looped'
n_layer = 6  # used only for c_proj init scaling convention, not an actual layer count here
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
max_loop_iters = 20
n_init = 6.0
loop_index_embed = True

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
