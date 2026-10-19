# Stage 14: swap the HRR codec for the disjoint-slot codec (codec_type=
# 'slot') -- place p owns dims [dpp*p : dpp*(p+1)) exclusively, no
# binding, no FFT, zero cross-place interference within capacity by
# construction rather than by statistical argument (see
# probe_slot_capacity.py: exact 1.000 recovery for every L up to
# n_embd//codec_dims_per_place=128, untrained, then a hard predictable
# cliff -- fundamentally different failure mode than HRR's graceful-then-
# compounding decay, which started eroding whole-number accuracy well
# before its own soft wall around L~20-27).
#
# From scratch, not warm-started -- the representation is different in
# kind from HRR (disjoint slices vs superposition), no compatible
# checkpoint to start from. Everything else mirrors the best validated
# recipe so far (v18: per-place tokens, no explicit place tag -- doesn't
# exist as an option here anyway since it needs place_signal, which the
# slot codec doesn't build -- wider training range, deeper repeat) for a
# clean comparison against v17/v18's HRR results.
#
# codec_dims_per_place=3: per project discussion, a middle point between
# capacity (n_embd//dpp places, exact) and angular separation margin
# between the 10 digit directions sharing each slice (measured: 0/17/38/
# 52 deg margin for 1/2/3/4 dims at n_embd=384, random draw). 128-place
# capacity is >> composed_train_max_digits=20 and >> max_answer_digits=30
# here, so there's slack to increase dims_per_place later (fewer places,
# more margin) if 3 turns out to leave the combiner's output-side
# compression (the one new burden this codec adds -- see project
# discussion) short on room.

out_dir = 'out-addition-additive-v20'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage14-slot-codec'

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
combiner_explicit_place_tag = False  # not an option for codec_type='slot' -- asserted False in model.py's __init__

codec_type = 'slot'
codec_dims_per_place = 3

composed_train_max_digits = 20
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
