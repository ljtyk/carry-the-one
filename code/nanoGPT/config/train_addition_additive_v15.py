# Ablation 2 of v13's repeating combiner: combiner_repeat_max=4 (matching
# the OLD fixed-depth combiner's layer count), keeping per-place structure,
# thinking tokens, and the countdown loss. Isolates whether the REPEAT
# COUNT specifically matters (depth must scale with problem size) or
# whether per-place exposure alone, at any reasonable depth, would have
# been enough to unlock v13's L=11 generalization (80.5% sum accuracy).
#
# Note the countdown target (max(len_a,len_b), up to 10 in training) will
# often exceed repeat_max=4 here -- countdown loss still trains fine
# (target gets clamped), it just means the auxiliary signal can't
# distinguish "5 rounds needed" from "10 rounds needed" at this depth,
# which is fine since it's not gating anything regardless.
#
# Reduced max_iters, same rationale as v14.

out_dir = 'out-addition-additive-v15'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage13-ablation2-repeat4'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 4

model_type = 'additive_composed'
combiner_layers = 1
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'

composed_recon_weight = 0.5
composed_consistency_weight = 0.5
max_answer_digits = 30

combiner_repeat_max = 4  # <-- the ablation (was 16 in v13)
combiner_countdown_weight = 0.1
combiner_n_thinking = 4
combiner_use_per_place = True

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
