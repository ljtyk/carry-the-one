# Stage 10c: same as v2 (out-addition-additive-v2: tying + skewed length
# sampling + ported consistency loss), except weight_decay dropped from 0.3
# to 0.1 (nanoGPT's own standard default for this model size).
#
# Motivation: v2's total loss spiked into the tens of thousands around step
# 8000 (probe_loss_components.py confirmed the consistency MSE term was
# >99% of the total, driven by encode_operand's weights blowing up in
# scale -- ||vec_a|| mean=1524 for a 10-digit operand, vs ~2-7 per digit
# term on a healthy checkpoint). It's been slowly recovering since via
# weight_decay pulling the scale back down, but "val loss (robust)" has
# stayed flat (~0.70-0.75) the entire run with no visible improvement,
# suggesting the optimizer is spending its effort fighting an oscillation
# between the consistency loss's growth pressure and weight_decay's shrink
# pressure, rather than making steady progress -- classic grokking-adjacent
# territory (weight_decay is the mechanism believed to drive the late
# generalizing transition on algorithmic tasks like this one), except no
# such transition is showing up, which raised the question of whether 0.3
# is too aggressive given the consistency loss's badly-scaled gradients.
#
# This is a single-variable test of that hypothesis, run in parallel with
# just letting v2 finish rather than fixing the consistency loss's scale
# directly (raw MSE vs a KL-divergence-on-decoded-distributions
# reformulation, discussed but not yet implemented) -- if v3 is still
# unstable at 0.1, that points back to the loss formulation itself as the
# real fix, not a weight_decay tuning issue.

out_dir = 'out-addition-additive-v3'
eval_interval = 500
eval_iters = 200
log_interval = 50

# warm-start from v2's finished weights (same architecture, only weight_decay
# differs) rather than training from scratch -- fresh optimizer/iter_num/LR
# schedule (unlike init_from='resume'), so this gets a full new 50k-step
# cosine schedule under the new weight_decay, without inheriting Adam
# moments shaped by v2's unstable middle stretch.
init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v2/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage10c-additive-composed-lower-wd'

dataset = 'addition_padded'  # only used for meta.pkl's vocab ('+'/'=' ids for the combiner) -- get_batch_additive generates on the fly
gradient_accumulation_steps = 1
batch_size = 64
block_size = 4  # the combiner's fixed 4-slot sequence length -- encoder/decoder don't use block_size/RoPE at all

model_type = 'additive_composed'
combiner_layers = 4
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'  # combiner only

composed_recon_weight = 0.5
composed_consistency_weight = 0.5
max_answer_digits = 30  # comfortably past far_ood's 26 digits

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
weight_decay = 0.1  # was 0.3 in v2 -- see module docstring
beta2 = 0.99

warmup_iters = 200

compile = False  # dynamic per-batch operand shapes defeat torch.compile, same as ComposedGPT
