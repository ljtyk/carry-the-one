# Stage 24: expand codec_dims_per_place 3->4, testing the opposite
# direction from today's whole dpp-shrinking arc (v27/v28 at dpp=2,
# v29 at dpp=1). Both shrinking attempts imposed cleaner algebraic
# structure (phase/circular, then linear-magnitude) specifically to try
# to give the combiner an easier mechanism -- and both cost separation
# margin, and in v29's case the combiner never learned to add at all even
# after separation was independently fixed via codec_zero_class_weight.
# Conclusion drawn from that: the combiner appears to benefit from raw
# per-place working room across its 24 repeat rounds far more than from
# human-legible structure -- it's apparently fine inventing its own
# mechanism on a wide arbitrary embedding (v22's dpp=3 plain QAM
# codewords, no phase/linear-magnitude constraint, already got ~100%
# in-dist and 77% far-OOD). This run tests that directly: MORE room than
# v22 had, same otherwise-arbitrary QAM codec (codec_use_magnitude=True,
# no seeding, no linear-magnitude, no phase order) -- if the capacity
# hypothesis is right, dpp=4 should match or beat v22, particularly at
# far-OOD where margin seemed to be the actual limiting factor.
#
# Single-variable change against the v22 baseline: everything else
# (codec_zero_pad, combiner_magnitude_signal, repeat_max=24, n_thinking=4,
# composed_train_max_digits=20, max_answer_digits=30) held identical.
# From scratch, not warm-started -- digit_direction's shape (10,3)->(10,4)
# doesn't transfer.
#
# n_embd=384 / dpp=4 = 96-place capacity (down from v22's 128, still >>
# max_answer_digits=30, no issue).

out_dir = 'out-addition-additive-v30'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 1000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage24-expand-dpp4'

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
codec_dims_per_place = 4  # <-- was 3 in v22; the only change under test
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
