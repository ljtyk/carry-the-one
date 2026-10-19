# Stage 10i: adds composed_place_orthogonality_weight on top of the Hoyer
# sparsity penalty (dropped to 0.005, half of v8's 0.01), warm-started from
# v3 (clean, no sparsity pressure yet).
#
# v8 (Hoyer sparsity alone, 0.01) confirmed real, large compression --
# place9 287->4 of 384 dims by iter 30000 -- with in-distribution sum
# accuracy essentially preserved (81.3% vs v3's 81.6%), but ZERO OOD
# improvement (still exactly 0% sum accuracy past 10 digits, same as every
# other variant tried this session). Diagnosis: sparsity only guarantees
# each place uses FEW dimensions, never that DIFFERENT places' few
# dimensions stay separated from each other -- an extreme, unsupervised
# collapse (like place9's) is exactly the kind of thing that could collide
# with other places' content rather than avoid it.
#
# composed_place_orthogonality_weight adds the missing piece directly:
# penalizes mean squared cosine similarity between DIFFERENT places'
# digit-averaged representative contributions (places 0..14), leaving
# digit-to-digit structure at a given place untouched so the combiner's
# carry-arithmetic relational structure isn't disturbed. probe_
# orthogonality_scale.py measured the raw value on v3's checkpoint: 0.155
# (bounded [0,1]) -- weight 0.05 gives an initial contribution (~0.0078)
# comparable to the current tiny CE losses, same conservative-then-verify
# approach used for the Hoyer weight after the wiring bug was fixed.
#
# Sparsity weight lowered to 0.005 (from v8's 0.01) since we're now
# stacking two pressures together -- want to see the orthogonality
# term's own effect clearly rather than have a stronger sparsity pull
# dominate the combination.

out_dir = 'out-addition-additive-v9'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v3/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage10i-additive-composed-sparsity-orthogonality'

dataset = 'addition_padded'  # only used for meta.pkl's vocab ('+'/'=' ids for the combiner) -- get_batch_additive generates on the fly
gradient_accumulation_steps = 1
batch_size = 64
block_size = 4  # the combiner's fixed 4-slot sequence length -- encoder/decoder don't use block_size/RoPE at all

model_type = 'additive_composed'
combiner_layers = 4
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'  # combiner only

composed_recon_weight = 0.5
composed_consistency_weight = 0.5
composed_contrib_sparsity_weight = 0.005
composed_place_orthogonality_weight = 0.05
place_orthogonality_max_place = 15
max_answer_digits = 30  # comfortably past far_ood's 26 digits

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 50000
lr_decay_iters = 50000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False  # dynamic per-batch operand shapes defeat torch.compile, same as ComposedGPT
