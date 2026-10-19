# Stage 22: control for v27. Same forced-2D circular topology
# (codec_dims_per_place=2) but WITHOUT the numeric-order seed --
# digit_direction inits random, same as every run before v27. Isolates
# whether v27's far-OOD regression (77.3%->12.3% at L=24-26 vs v22) came
# from cutting dpp 3->2 (less raw separation capacity) or from imposing
# numeric order specifically. If v28 also collapses at far-OOD, the
# regression is from the dimension cut alone, not the seeding, and this
# also answers "what order does the model pick on its own when confined
# to a circle" -- decode's loss is still permutation-invariant, so
# whatever order it settles on (if any is stable at all) is worth reading
# off with probe_codebook_geometry.py the same way v22's dpp=3 codebook
# was read.
#
# Everything else identical to v27: from scratch, v22-scale settings,
# snapshot_interval=1000 for the same drift-tracking as v27.

out_dir = 'out-addition-additive-v28'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 1000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage22-phase-codebook-2d-unseeded-control'

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
combiner_magnitude_signal = True

codec_type = 'slot'
codec_dims_per_place = 2  # same as v27
codec_use_magnitude = True
codec_max_scale = 3.0
codec_zero_pad = True
codec_seed_numeric_order = False  # <-- the only change from v27: random init, model picks its own order (if any)

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
