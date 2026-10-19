# Stage 6: RefineGPT -- tests the user's iterative window-refinement idea
# (W=3, stride=1): instead of one-shot left-to-right generation, predict a
# sliding window of 3 answer positions per pass, each position getting
# revised across 3 passes (as the window moves) before locking in. Requires
# bidirectional attention (a position being revised must see later
# positions' current draft) -- a real departure from every causal model
# built this session. See model.py's RefineGPT docstring for the training
# simulation (get_batch_refine) and the full rationale.
#
# Trained from scratch (new architecture, new vocab -- data/addition_refine
# adds a MASK token, kept as a separate dataset dir so the shared
# addition_padded vocab/checkpoints are never put at risk).

out_dir = 'out-addition-refine'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage6-refine'

dataset = 'addition_refine'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 104

model_type = 'refine'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
bidirectional = True
refine_window = 3

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
weight_decay = 0.3
beta2 = 0.99

warmup_iters = 100
