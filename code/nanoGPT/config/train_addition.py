# fixed-depth baseline for the synthetic multi-digit addition task
# same "baby GPT" architecture as train_shakespeare_char.py, for a fair
# matched-parameter-count comparison against later adaptive-depth stages

out_dir = 'out-addition'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'baseline-fixed-depth'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
