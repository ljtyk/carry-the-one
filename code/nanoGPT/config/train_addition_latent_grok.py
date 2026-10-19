# Stage 4c: grokking-style extension of the adaptive-latent-steps run.
# Resumes (proper resume -- optimizer state carries over, unlike the
# warm_start used to get to the 15k checkpoint in the first place) from
# out-addition-latent-adaptive/ckpt.pt (test_indist 94.02%, gap/ood 0.00%),
# copied into this out_dir so the original 15k result stays untouched.
#
# Tests whether extended training past convergence, with heavier weight
# decay, induces a grokking-style transition to length generalization --
# per project history (out-addition-grok/, pre-restart, weight_decay=1.0,
# decay_lr=False) this is NOT a sure thing: that run hit a real training
# instability (in-dist accuracy crashed from 97.5% to 0.3% around iter
# 60k before slowly recovering) and OOD accuracy stayed noisy around a
# near-zero baseline the whole 134k-iter run, no convincing trend. Going
# with a more moderate weight_decay=0.3 (not the old run's 1.0) and
# decay_lr=True (not the old run's constant LR) to reduce the risk of
# repeating that instability while still testing the core hypothesis.
#
# max_iters=200000 total (this run picks up from iter_num=15000 in the
# checkpoint, so ~185000 additional iterations). snapshot_interval=5000
# for a full accuracy-vs-iter history, same convention as the old grok run.

out_dir = 'out-addition-latent-grok'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'resume'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage4c-latent-cot-grok'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 90

model_type = 'adaptive_latent_looped'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
n_latent_steps = 10
n_init = 6.0
ponder_cost_weight = 1e-4
countdown_max = 24
countdown_loss_weight = 0.1

learning_rate = 1e-3
max_iters = 200000
lr_decay_iters = 200000
min_lr = 1e-4
weight_decay = 1.0
beta2 = 0.99

warmup_iters = 100
