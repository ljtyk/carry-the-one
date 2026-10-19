# Extends the sparse "hint" exposure range from 14-15 -> 14-18 digits (see
# data/addition/generate.py), moving true zero-exposure OOD out to 19-21.
# Tests the user's hypothesis directly: near-miss errors on long problems
# reflect partial/confused competence, not total absence of the capability,
# and a small amount of correctly-labeled exposure (same sparse density the
# existing 14-15 hint tier already uses) should be enough to fix them --
# without reversing digit order, adding index hints, or any architecture
# change.
#
# Warm-started from out-addition-latent-grok-normfix-cap24-ponder5e3/ckpt.pt
# (the just-completed run: test_indist 96.12%, test_gap 12.83%, test_hint(14-15)
# 31.50%, test_ood(16-18, now folded into hint) 0.00%). loop_n had collapsed to
# ~1.2 avg under ponder_cost_weight=5e-3 by the end of that run -- irrelevant to
# what this experiment is testing (data exposure, not adaptive compute), so left
# unchanged for continuity/comparability.

out_dir = 'out-addition-latent-hint18'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-latent-grok-normfix-cap24-ponder5e3/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage4d-hint18'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 104

model_type = 'adaptive_latent_looped'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
n_latent_steps = 24
n_init = 6.0
ponder_cost_weight = 1e-4  # dropped from 5e-3 (which had crushed loop_n to ~1.2 avg) -- this experiment wants room for the model to use more steps on the harder drilled/hint problems
countdown_max = 24
countdown_loss_weight = 0.1

learning_rate = 1e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
weight_decay = 1.0
beta2 = 0.99

warmup_iters = 100
