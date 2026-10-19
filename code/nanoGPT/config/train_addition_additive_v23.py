# Stage 17: combiner_variable_repeat=True -- tests whether training with
# a per-batch VARYING actual round count (uniformly sampled between
# "just enough" and the full combiner_repeat_max budget, deliberately
# including genuine excess every batch) fixes the repeat-count
# brittleness found on v22: running MORE rounds than the fixed-24
# trained model ever saw didn't just fail to help, it actively corrupted
# already-correct answers (83.5%->21.5% at L=24 going from repeat=24 to
# repeat=60). Hypothesis: the shared block was only ever given gradient
# from its state at EXACTLY 24 rounds, so it had no reason to learn a
# stable fixed point -- exposure to real excess during training should
# push it toward one.
#
# Warm-started from v22 (the no-blank codec) -- no new parameters
# introduced by this flag (n_rounds is just a per-batch python int
# controlling the existing loop's range, no new weight tensors), so
# every key should match by shape exactly, zero fresh-init gaps.
#
# composed_train_max_digits and combiner_repeat_max both stay unchanged
# (20 and 24) -- same isolation principle as every fix this session:
# only the variable-repeat mechanism is under test here, nothing else
# moves at the same time.

out_dir = 'out-addition-additive-v23'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v22/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage17-variable-repeat'

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
combiner_variable_repeat = True  # <-- the change under test

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
