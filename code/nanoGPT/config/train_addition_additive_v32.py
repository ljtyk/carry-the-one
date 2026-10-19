# Stage 26: does the RNS combiner generalize to lengths it never trained
# on, within its own representational capacity -- or does it hit the same
# "vec_c drifts off the genuine-codeword manifold" problem every other
# codec showed past its training exposure, just in a new (compositional,
# not carry-chain-depth) form?
#
# v31's perfect 1-21 digit sweep didn't actually test this: with
# composed_train_max_digits=20, almost that entire range WAS training
# distribution, and the capacity wall (~21 digits, 17 moduli) sits right
# next to the training ceiling -- there was no real "trained short, tested
# long but still within capacity" gap to expose a generalization problem
# in. The far-OOD collapse at 22+ is a hard CRT capacity wall, not
# evidence about combiner generalization either way.
#
# Single change from v31: composed_train_max_digits 20->10, same 17
# moduli / ~21-digit capacity unchanged. This deliberately opens an
# 11-21-digit window that's within capacity but genuinely unseen in
# training -- existing test files (test_indist 1-10, test_gap 11-13,
# test_hint 14-18, test_ood 19-21) already give granular resolution
# across exactly that window without needing new data. If accuracy holds
# clean through it, the "combiner generalizes fine once there's no carry
# chain" story survives; if it degrades before the capacity wall, that's
# the same population-mismatch problem in a compositional (unseen
# residue-pair) form rather than a carry-chain-depth form.
#
# combiner_repeat_max stays at 24, unchanged -- isolate this one variable.

out_dir = 'out-addition-additive-v32'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 1000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage26-rns-train-short-test-long'

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
max_answer_digits = 30  # only sizes the rns_place_weights buffer's width; real capacity is codec_rns_moduli's product

combiner_repeat_max = 24
combiner_countdown_weight = 0.1
combiner_n_thinking = 4
combiner_use_per_place = True
combiner_explicit_place_tag = False
combiner_magnitude_signal = True

codec_type = 'rns'
codec_rns_moduli = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)  # unchanged from v31, product ~1.9e21
codec_dims_per_place = 3
codec_max_scale = 3.0

composed_train_max_digits = 10  # <-- was 20 in v31; the only change under test
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
