# Ablation: isolates the "reversal" half of position coupling from the
# "coupled position ids" half. Uses the EXACT SAME data as
# out-addition-coupled (data/addition_coupled -- zero-padded operands,
# zero-padded+reversed sum) but with a PLAIN base GPT class and ordinary
# sequential RoPE positions -- no position coupling at all. Only one
# variable changes relative to train_addition_coupled.py: the position
# scheme. If this does much worse than PositionCoupledGPT's 100% (gap/hint/
# ood), that's direct evidence coupling itself (not reversal) is carrying
# most of the effect; if it does comparably well, reversal alone was doing
# more of the work than the literature's "helps but insufficient alone"
# framing suggested.

out_dir = 'out-addition-reversed-only'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage7b-reversed-only-ablation'

dataset = 'addition_coupled'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 104

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
