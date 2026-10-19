# Stage 15c: combiner_magnitude_signal=True, warm-started from v21's
# finished checkpoint (max_scale=3.0, NOT v21b's max_scale=8.0 side
# experiment, which already answered its own question and isn't the
# baseline this builds on).
#
# Motivated directly by tracing through Block.forward() in project
# discussion: combiner_block is a standard Pre-LN transformer block
# (ln_1/ln_2 BEFORE attn/mlp), which means attention and the FFN are
# structurally blind to each place-token's magnitude every round -- only
# the additive residual skip carries any trace of it, never anything
# attention/mlp actually compute with. This explains (on top of "no loss
# incentive") why all 10 digit magnitudes converged to nearly identical
# values regardless of codec_max_scale (v21 vs v21b: pure rescaling, zero
# differentiation) -- the combiner's core computation couldn't have used
# per-digit magnitude even if the encoder produced it.
#
# combiner_magnitude_signal explicitly re-injects a learned embedding of
# (||qa||, ||qb||) -- the ORIGINAL per-place input magnitudes -- right
# after ln_1/ln_2 at EVERY repeat round (see model.py's combine(),
# run_block()), so attention/mlp can actually condition on it instead of
# only ever seeing it erased.
#
# No architecture change to the codec itself (digit_direction/
# digit_scale_raw/etc. all identical shapes) -- only combiner_mag_proj is
# new (fresh init, small: 2 -> n_embd). Warm start handles this the
# standard way (shape-mismatch popping only matters for changed shapes;
# combiner_mag_proj is simply ABSENT from the source checkpoint, so it's
# left fresh automatically).
#
# Lower learning_rate (2e-4), standard warm-start recipe this session.

out_dir = 'out-addition-additive-v21c'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v21/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage15c-slot-codec-qam-magnitude-signal'

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
combiner_magnitude_signal = True  # <-- the change under test

codec_type = 'slot'
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0

composed_train_max_digits = 20
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 15000
lr_decay_iters = 15000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
