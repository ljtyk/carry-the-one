# Seed-variation experiment: identical to v20/v20b's recipe (slot codec,
# blank-masking fix already applied in model.py so this doesn't need the
# v20->v20b restart dance), from scratch, but with a DIFFERENT seed_base
# than the default 1337 v20/v20b used. Purpose: v20b's digit codebook
# was found to be frozen (not wandering, not still separating) by iter
# 5000, with a specific structure (5<->6 and 4<->8 as the tightest pairs)
# that showed no clean value-adjacency pattern -- looks like an arbitrary
# local optimum, not a deliberately-shaped one. This run (and v20d) test
# that directly: if the resulting codebook structure is essentially
# DIFFERENT each time (different closest pairs, no consistent pattern),
# that confirms it's noise/local-minima-dependent, not a real attractor.
# If multiple seeds consistently land on similar structure, that's
# evidence of something more deterministic going on.
#
# 15000 iters, not the full 30000 -- v20b's codebook froze by iter 5000
# and stayed bit-identical through iter 10000, so this only needs enough
# budget to reach the same kind of plateau, not full convergence on
# every other metric.

out_dir = 'out-addition-additive-v20c'
seed_base = 2024
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage14c-slot-codec-seed2024'

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

combiner_repeat_max = 24
combiner_countdown_weight = 0.1
combiner_n_thinking = 4
combiner_use_per_place = True
combiner_explicit_place_tag = False

codec_type = 'slot'
codec_dims_per_place = 3

composed_train_max_digits = 20
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
