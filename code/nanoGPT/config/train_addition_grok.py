# grokking-biased run: same architecture/data as train_addition.py, but
# heavy weight decay, no dropout, constant LR (no decay), and far longer
# training -- testing whether extended training past interpolation induces
# length generalization to the held-out OOD digit counts, or whether OOD
# accuracy stays pinned at 0 regardless of training horizon.

out_dir = 'out-addition-grok'
eval_interval = 2000
eval_iters = 200
log_interval = 50

always_save_checkpoint = True
snapshot_interval = 2000  # numbered ckpt_<iter>.pt for the OOD-accuracy-vs-iter curve

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'grok-attempt'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0  # grokking setups typically rely on weight decay alone, not dropout

learning_rate = 1e-3
max_iters = 50000  # baseline's val loss converged by ~4-5k iters; this gives ~10x runway past that point
weight_decay = 1.0  # much heavier than the 0.1 baseline default -- standard grokking value (Power et al. 2022)
beta2 = 0.99

decay_lr = False  # constant LR: a cosine decay to ~0 would foreclose any late-training dynamics
