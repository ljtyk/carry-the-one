# Stage 2.2: same as train_addition_loop_rope.py (single shared block,
# looped, RoPE for sequence position), but the loop-index embedding is now
# a fixed sinusoidal encoding instead of a free nn.Embedding lookup table.
# Motivated by Stage 2's learned loop count (loop_n) saturating exactly at
# the max_loop_iters=20 cap by the end of its 50000-iter run -- a free
# lookup table has no representation for iterations beyond that cap and no
# forced relationship between nearby iterations; sinusoidal encoding is
# defined for any iteration index and gives nearby iterations smoothly
# related vectors, the additive-embedding analog of what RoPE does for
# sequence position. Same hyperparameters as train_addition_loop_rope.py
# otherwise.

out_dir = 'out-addition-loop-sinusoidal'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage2.2-loop-sinusoidal'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

model_type = 'looped'
n_layer = 6  # used only for c_proj init scaling in LoopedGPT, matches n_init below
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
max_loop_iters = 20
n_init = 6.0
loop_index_embed = True
loop_encoding = 'sinusoidal'

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
