# Stage 2 recreated on the restarted dataset: single shared block, looped,
# with loop-index embedding (the fix that made naive looping work), now with
# RoPE instead of learned absolute position embedding, and trained on the
# post-restart data/addition (mismatched operand lengths, four length tiers:
# train 1-10, gap 11-13, hint 14-15 sparse, ood 16-18). Same hyperparameters
# as the pre-restart train_addition_loop_idx.py otherwise, for continuity.

out_dir = 'out-addition-loop-rope'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage2-looped-rope'

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

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
