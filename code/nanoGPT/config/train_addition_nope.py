# Stage 1 baseline re-run with NoPE (no positional embedding at all -- the
# model relies solely on the asymmetry of causal masking for positional
# information). Cheap control condition alongside train_addition_rope.py:
# if NoPE generalizes to OOD digit lengths roughly as well as RoPE, that's
# evidence the fix is "remove the fixed-length positional artifact" in
# general, not RoPE's relative-rotation mechanism specifically.
# Same architecture and training budget as train_addition.py otherwise.

out_dir = 'out-addition-nope'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'baseline-nope'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'none'

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
