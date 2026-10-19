# Stage 13c: same wider-range boundary fix as v17, but warm-started from
# v16 (Ablation 3: combiner_explicit_place_tag=False) instead of v13.
#
# Purpose: v17 confirmed the leading-digit boundary bug is a narrow
# calibration gap fixed by extending composed_train_max_digits, and that
# the SAME failure signature (not a different one) reappears at the new
# boundary. But Ablation 3's original 0%-OOD verdict on "no explicit
# place tag" was measured back when the boundary bug was still
# confounding every ablation -- we flagged that we couldn't rule out
# Ablation 3 being masked by the same bug rather than genuinely needing
# the tag. This run applies the identical fix to the no-tag variant to
# see directly whether it recovers to comparable accuracy (tag was
# never necessary, just correlated with a fixed boundary) or still
# meaningfully lags v17 (tag has real, independent value for the
# network discovering positional structure in the combiner).
#
# combiner_explicit_place_tag stays False (inherited from v16's
# architecture choice). Everything else mirrors v17 exactly so the two
# runs are a clean A/B: same composed_train_max_digits=20,
# combiner_repeat_max=24, learning_rate=2e-4, max_iters=30000.
#
# v16's checkpoint stopped at iter 9000/20000 (not fully iter-converged,
# but val loss 0.0005 is already at a comparable plateau to other stage
# checkpoints at warm-start time) -- same situation v13 was in when v17
# warm-started from it.

out_dir = 'out-addition-additive-v18'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v16/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage13c-ablation3-wider-range'

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
combiner_explicit_place_tag = False  # inherited ablation from v16

composed_train_max_digits = 20
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 30000
lr_decay_iters = 30000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
