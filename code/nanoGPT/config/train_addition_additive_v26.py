# Stage 20: use the codec's actual full capacity. max_answer_digits
# 30->128 (n_embd // codec_dims_per_place, the true hard cap confirmed
# exact all the way out via probe_slot_capacity.py and the trained
# round-trip test), composed_train_max_digits 20->64 (genuine long
# carry-chain training, not just calibration -- zero-padding already
# fixed the calibration-boundary problem, but the real capacity-driven
# decay past composed_train_max_digits still needs actual training
# exposure to move), combiner_repeat_max 24->80 (scaling depth with the
# new training range, same ~1.2x margin ratio as before). No explicit
# place hints -- codec_type='slot' doesn't build the place_signal this
# would need, and the HRR-line precedent (v16->v18) found it wasn't
# load-bearing anyway. combiner_n_thinking stays 4 (v22's proven value,
# not v25's abandoned expansion-to-8). combiner_variable_repeat/
# combiner_variable_thinking both False -- back to fixed, v25's
# variable-and-expanded combination was called a dead end and stopped.
#
# Warm-started from v22 -- digit_direction/digit_scale_raw/
# combiner_proj/combiner_out_slot/combiner_mag_proj are all identical
# shapes and load exactly; countdown_head (repeat_max-dependent) and
# combiner_block's RoPE/attention buffers (block_size-dependent, now
# max_answer_digits+combiner_n_thinking=132 instead of 34) will show as
# shape mismatches and get left fresh/recomputed -- correct behavior,
# RoPE frequencies are deterministic not learned, and countdown_head
# genuinely needs the new size.
#
# Real compute-cost jump: 132-token sequences instead of 34, x 80 repeat
# rounds instead of 24 -- substantially slower per step than anything
# run so far. Novelty test, not a refinement of an already-tuned recipe.

out_dir = 'out-addition-additive-v26'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v22/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage20-full-capacity-128'

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
max_answer_digits = 128  # <-- full codec capacity, was 30

combiner_repeat_max = 80  # <-- was 24
combiner_countdown_weight = 0.1
combiner_n_thinking = 4
combiner_use_per_place = True
combiner_explicit_place_tag = False  # no place hints
combiner_magnitude_signal = True
combiner_variable_repeat = False
combiner_variable_thinking = False

codec_type = 'slot'
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0
codec_zero_pad = True

composed_train_max_digits = 64  # <-- was 20
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 15000
lr_decay_iters = 15000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
