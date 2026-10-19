# Stage 28: attack the ~15% length-extrapolation ceiling directly.
#
# v33 (v26-scale + variable_repeat) hit 100% on every defined test tier
# (1-26 digits) and holds a true fixed point r24->r64 -- but the "test to
# the limits" sweep (eval_additive_limits.py) showed it only generalises
# ~12% past its 64-digit training range (clean to ~72, 0.75 @ 80, 0.55
# @ 96), and EXTRA COMBINER ROUNDS DON'T HELP (140 vs 80 rounds:
# identical). So the ceiling is distribution-bound, not compute-bound.
#
# Diagnosis (probes A-F + the limits result): the encode/decode codec is
# length-invariant by construction and generalises perfectly (recon 1.00
# at 124 digits). The carry wavefront is structurally translation-
# invariant (offset -1, same rule every place). What still leaks length
# in is the COMBINER ATTENTION SOFTMAX: it runs over every place token,
# so the mass on the offset-(-1) neighbour dilutes as operand length
# grows -- the same competitive-budget effect that broke the original
# attention encoder, relocated into the combiner. Plus the fixed
# training ceiling teaches "answers stop at <= L_max digits".
#
# Two changes, together (see project discussion "how to get generalized
# methods", options A + B):
#
#   A. combiner_attn_band = 6 -- place token p attends only to places
#      [p-6, p+6] (a fixed-size local neighbourhood, translation-
#      invariant) plus full connectivity to/from the 4 thinking tokens
#      (kept as a global bus). The attention softmax's key set is now
#      length-INDEPENDENT -> the combiner is a genuine local stencil
#      iterated combiner_repeat_max times, which is length-general by
#      construction.
#
#   B. composed_length_random_max = True -- each batch skews operand
#      lengths toward its OWN random L_max ~ uniform[8, 126] instead of
#      the fixed 64. No consistent ceiling for the model to memorise;
#      training exposure ranges continuously up to near the 128-slot
#      capacity.
#
# combiner_repeat_max 80 -> 128 (a local stencil needs more rounds to
# move a carry across the full place range; variable_repeat keeps this
# cheap -- short problems still run few rounds). Warm-started from v33.
# countdown_head (81 -> 129 rows) left fresh; combiner_band_mask is a
# deterministic non-persistent buffer.
#
# Prediction if it works: the cliff moves from "~15% past train max" to
# "flat until the 128-slot capacity" -- a genuinely length-general
# learned method.

out_dir = 'out-addition-additive-v34'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v33/ckpt_15000.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage28-banded-combiner-no-length-ceiling'

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
max_answer_digits = 128

combiner_repeat_max = 128
combiner_countdown_weight = 0.1
combiner_attn_band = 6          # <-- A: local stencil among place tokens
combiner_n_thinking = 4
combiner_use_per_place = True
combiner_explicit_place_tag = False
combiner_magnitude_signal = True
combiner_variable_repeat = True
combiner_variable_thinking = False

codec_type = 'slot'
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0
codec_zero_pad = True

composed_train_max_digits = 64
composed_length_random_max = True   # <-- B: no fixed length ceiling
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
