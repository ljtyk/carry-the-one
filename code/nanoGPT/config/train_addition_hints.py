# Ablation: index hints (Zhou et al.-style) -- each digit tagged with a
# letter marking its place value (units=a, tens=b, ...), same letter shared
# across both operands and the answer, digit order left NORMAL (no
# reversal). Tests a mechanism that gives the model the SAME place-value
# alignment information as PositionCoupledGPT, but via literal, visible
# vocabulary tokens (exploitable by ordinary attention/induction heads)
# instead of hidden position ids -- so unlike position coupling, this needs
# ZERO model.py changes, purely a data-format change, using the plain base
# GPT class exactly as-is.

out_dir = 'out-addition-hints'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage7c-index-hints-ablation'

dataset = 'addition_hints'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 165

model_type = 'gpt'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
weight_decay = 0.3
beta2 = 0.99

warmup_iters = 100
