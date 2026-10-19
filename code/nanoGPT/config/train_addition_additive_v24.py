# Stage 18: combiner_n_thinking=0 -- ablates the thinking tokens (free
# scratch-space tokens not tied to any place, previously 4 of them)
# entirely. Tests whether they're load-bearing or just unused capacity
# -- place-tokens' own residual streams might already have enough spare
# room to hold intermediate carry state temporarily, only needing to
# resolve to clean digit content by the final round, in which case
# thinking tokens were never necessary.
#
# Warm-started from v22 (the no-blank codec baseline), NOT v23
# (variable-repeat) -- same "test one at a time against a fixed
# baseline" methodology as every other ablation this session.
# thinking_tokens goes from shape (4,384) to (0,384), a real shape
# mismatch -- handled automatically by warm_start's existing pop-and-
# leave-fresh logic (trivial here since a 0-size tensor has nothing to
# initialize anyway). Confirmed via smoke test that combiner_n_thinking=0
# works cleanly (torch handles the empty-tensor concat/expand fine, no
# special-casing needed anywhere in combine()).
#
# composed_train_max_digits, combiner_repeat_max, everything else stays
# identical to v22.

out_dir = 'out-addition-additive-v24'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v22/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage18-no-thinking-tokens'

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
combiner_n_thinking = 0  # <-- the change under test (was 4)
combiner_use_per_place = True
combiner_explicit_place_tag = False
combiner_magnitude_signal = True
combiner_variable_repeat = False

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
