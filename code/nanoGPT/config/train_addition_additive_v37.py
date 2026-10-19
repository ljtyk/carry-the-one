# Stage 31: combiner_thinking_rope=True + combiner_n_thinking 32->64.
#
# v36 (combiner_use_per_place=False, 32 generic thinking tokens) proved
# the "workspace hypothesis" -- chance-level (v35, 4 tokens) to a real,
# working algorithm on 83% of in-distribution sums. But it degraded
# faster than v22 and hit a sharp CLIFF at exactly L=21 (66.5% -> 55.5%
# -> 0%), not a smooth continuation of the decline -- unlike v22's
# graceful decay. Diagnosis (project discussion): v36's thinking tokens
# get default sequential RoPE positions (0,1,2,...) with no inherent
# correspondence to digit place -- whatever "token k represents place k"
# mapping it found is fully LEARNED, not structural, the same category
# of thing (Stage 1-9) that never extrapolated in this project.
#
# v22's per-place tokens, by contrast, already get RoPE positions equal
# to their place value for free (position_ids=None defaults to sequential
# index, and combine()'s use_per_place branch builds tokens in place
# order) -- that's what made the offset-(-1) nearest-neighbour carry
# wavefront a RoPE-native, translation-invariant pattern to discover.
#
# This gives v36's self-organized workspace the SAME structural
# addressing: sequence reordered to [think_0..think_63, vec_a, vec_b],
# thinking tokens get explicit positions 0..63 (place-like), vec_a/vec_b
# get positions 64,65 (the "global broadcast" role, mirroring where
# v22's actual thinking tokens sit relative to its place tokens). No
# change to the encoder or decoder -- still just 2 dense input tokens,
# still the point of this whole line of experiments.
#
# combiner_n_thinking 32->64: gives the place-like addressing space at
# least as much room as max_answer_digits=30 needs, with margin (64 was
# also the natural next power-of-2 step up from 32, not tied to a
# specific derivation).
#
# Warm-started from v36 -- digit codewords and encoder/decoder transfer
# directly (already at 100% reconstruction), combiner_block's own
# weights are shape-independent of sequence layout/position (RoPE
# lookup happens via cos/sin indexing, not a learned per-position
# parameter) so they load too. thinking_tokens (32,384)->(64,384) left
# fresh via the existing pop-and-reinit logic. Caveat: since the
# ADDRESSING SCHEME itself changes substantially (reordered sequence +
# meaningful positions vs. arbitrary ones), the combiner_block's learned
# position-dependent attention behavior may not transfer as cleanly as a
# typical pure-hyperparam warm start -- worth watching whether it helps
# or just adds noise to the initial iterations.

out_dir = 'out-addition-additive-v37'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v36/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 1000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage31-combiner-thinking-rope-64-tokens'

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
combiner_n_thinking = 64  # <-- was 32 in v36
combiner_thinking_rope = True  # <-- the change under test
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
