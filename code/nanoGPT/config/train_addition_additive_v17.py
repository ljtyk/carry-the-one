# Stage 13b: fixes the leading-digit boundary issue diagnosed in v13.
# probe_combiner_errors.py showed EVERY wrong OOD example had errors
# confined to just the leading digit(s) -- deep multi-digit carry chains
# (10-12 places) were resolved correctly, only the very top digit was
# wrong (predicted blank, or the wrong value). Mechanism: place index
# >= composed_train_max_digits (10) was NEVER non-blank during training
# (operands never exceeded 10 digits), so the model learned a strong
# "anything past here is blank" prior that's exactly wrong for genuinely
# longer OOD sums. This is a narrow calibration gap, not a capacity
# ceiling -- unlike earlier (less successful) "extend training range"
# attempts this session, which were fighting the encoder's genuine
# information-capacity limit.
#
# composed_train_max_digits: 10 -> 20, so place indices up to ~20 get
# real (non-blank) exposure -- covers the L=11-14 zone where v13's carry
# logic already worked but the boundary decision didn't, with margin.
# combiner_repeat_max: 16 -> 24, since worst-case carry chains now scale
# with operands up to 20 digits (the "depth must scale with problem size"
# lesson from the ablations, applied to the new range).
#
# Warm-started from v13 (not scratch): the carry logic itself already
# works, this is recalibrating the boundary decision for a wider range,
# not relearning arithmetic. countdown_head can't warm-start cleanly
# (its shape depends on combiner_repeat_max, which changed) -- reinits
# fresh via warm_start's non-strict load, fine since it's auxiliary only.
# Lower learning_rate (2e-4, not 1e-3) -- same reasoning as every other
# warm-start this session: the from-scratch LR destabilizes an
# already-converged model.

out_dir = 'out-addition-additive-v17'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v13/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage13b-additive-composed-wider-range'

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
combiner_explicit_place_tag = True

composed_train_max_digits = 20
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 30000
lr_decay_iters = 30000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
