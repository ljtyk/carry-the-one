# Stage 19: expand BOTH ceilings and make BOTH variable simultaneously --
# combiner_repeat_max 24->40, combiner_n_thinking 4->8,
# combiner_variable_repeat=True, combiner_variable_thinking=True.
#
# Rationale (see project discussion): repeat_max=24 was already a real
# capacity bottleneck past L~24 (not just a calibration artifact --
# genuine, scattered errors past that point on the no-blank codec).
# Sampling only WITHIN the old, already-too-small ceiling (as v23 did)
# couldn't fix that, only reallocate around it -- hence v23's real
# length-extrapolation cost. Expanding the ceiling AND making usage of
# it variable at the same time targets both problems together: more raw
# capacity for genuinely long carry chains, plus (per the variable-
# repeat result) training exposure to a wide range of round/thinking
# budgets so the shared block doesn't overfit to one specific trajectory.
#
# n_thinking gets the same treatment given it was confirmed load-bearing
# (v24's ablation to 0 was a clear regression, not neutral) -- varying
# the COUNT instead of fixing it at one value (or removing it) tests
# whether that's also a "brittle to a fixed budget" story rather than a
# "more is strictly better" or "any amount is fine" one.
#
# Warm-started from v22 (clean no-blank baseline, not v23/v24 which
# already carry their own individual trade-offs). Real shape mismatches
# expected and fine: thinking_tokens (4,384)->(8,384), countdown_head
# (25,384)->(41,384) -- both left fresh via warm_start's existing
# pop-and-reinit logic. Everything else (digit codebook, combiner_proj,
# combiner_block, combiner_out_slot, combiner_mag_proj) loads exactly.
#
# composed_train_max_digits stays at 20, unchanged -- same isolation
# principle as every fix this session.

out_dir = 'out-addition-additive-v25'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v22/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage19-expanded-variable-repeat-and-thinking'

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

combiner_repeat_max = 40  # <-- expanded, was 24
combiner_countdown_weight = 0.1
combiner_n_thinking = 8  # <-- expanded, was 4
combiner_use_per_place = True
combiner_explicit_place_tag = False
combiner_magnitude_signal = True
combiner_variable_repeat = True
combiner_variable_thinking = True  # <-- new, in addition to variable_repeat

codec_type = 'slot'
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0
codec_zero_pad = True

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
