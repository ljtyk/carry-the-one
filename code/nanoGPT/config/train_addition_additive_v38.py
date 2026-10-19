# Stage 32: combiner_attn_band=4 applied to v37's RoPE-addressed thinking
# tokens -- "pressure the algorithm into a more refined version" (project
# discussion). probe_v37_mechanism.py found v37 never converged on v22's
# crisp, single-offset carry relay: attention was spread diffusely across
# roughly +-3 offsets (round 1-2) rather than locking onto one clean
# neighbour, and convergence was a smooth gradual ramp rather than v22's
# onset-then-propagate shape -- plausibly why the hard cliff at L=21
# persists even with structural RoPE positions (probe_v37_margin.py).
#
# Rather than hoping training discovers locality on its own (a soft
# preference the diffuse pattern shows it didn't find), this makes it
# structural: each thinking token may only attend to OTHER thinking
# tokens within +-4 of its own (place-like) position, plus full
# connectivity to/from vec_a/vec_b (the global-bus role, symmetric to how
# v34's combiner_attn_band treats the thinking-token bus for per-place
# tokens). band=4 chosen to comfortably cover the range v37 was ALREADY
# spending attention on (+-3, with some tail to +-6) while cutting the
# longer tail that's diffusing focus rather than doing real work -- not
# derived from anything more precise than "matches the measured range
# with a little margin".
#
# Requires combiner_thinking_rope=True (the band restriction needs the
# thinking tokens to already have place-like sequential positions to band
# OVER) -- see the new combiner_attn_band branch in __init__/combine()
# for the combiner_use_per_place=False case (roles swapped from the
# original per-place version: thinking tokens are now the banded-local
# ones, vec_a/vec_b are the global bus).
#
# Warm-started from v37 -- purely an attention-masking change, zero shape
# differences (same combiner_n_thinking=64, same everything else), so
# every parameter transfers directly. The hope is the model mostly
# adapts existing behaviour to the tighter window rather than relearning
# from scratch, since the band comfortably contains what it was already
# using most of its attention mass on.

out_dir = 'out-addition-additive-v38'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v37/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 1000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage32-combiner-thinking-rope-banded-attention'

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
combiner_n_thinking = 64
combiner_thinking_rope = True
combiner_attn_band = 4  # <-- the change under test
combiner_use_per_place = False
combiner_explicit_place_tag = False
combiner_magnitude_signal = False

codec_type = 'slot'
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0
codec_zero_pad = True

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
