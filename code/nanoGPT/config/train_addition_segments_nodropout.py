# Same as train_addition_segments.py, but dropout=0.0 -- tests whether
# dropout's training-time noise is what's capping the segment-embeddings
# ablation's hint/gap generalization, independent of weight decay (kept at
# 0.3, same as the dropout=0.2 run, for a controlled single-variable diff).

out_dir = 'out-addition-segments-nodropout'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage8-segment-embeddings-nodropout'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 104

model_type = 'gpt'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'
n_segments = 4  # 0=structural/pad, 1=operand a, 2=operand b, 3=answer

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
weight_decay = 0.3
beta2 = 0.99

warmup_iters = 100
