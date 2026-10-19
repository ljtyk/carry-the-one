# Stage 13: repeating combiner. Replaces the old fixed-4-layer attention
# combiner (which only ever saw 2 pre-summed vectors -- no per-place
# resolution at all, so no way to represent "move information between
# places" regardless of training) with an explicit per-place token
# sequence (built by unbinding vec_a/vec_b at every place, the same fixed
# HRR operation the decoder uses) processed by a SHARED (same weights
# every round) attention+FFN block applied combiner_repeat_max times, plus
# a few learned "thinking token" scratch slots. Depth now scales with
# repetition count rather than a fixed number of distinct layers -- carry
# propagation's worst case (adding 1 to a run of 9s) needs a dependency
# chain as long as the number of places, which no fixed-depth pass can
# implement regardless of how much data it sees.
#
# Motivation: v12 (HRR encoder alone) proved the encoder/decoder pair now
# generalizes gracefully well past the trained range (reconstruct accuracy
# 98-100% at L=11, graceful decline through L=14 -- a first for this
# project), but sum accuracy through the OLD combiner was still exactly 0%
# past L=10, and direct diagnosis (probe_vec_family_ood.py) confirmed the
# decoder and encoder both still worked correctly on OOD-length inputs --
# the combiner alone was the bottleneck, failing via a sharp immediate
# cliff rather than the encoder's now-graceful decline.
#
# A dense (not gated) auxiliary countdown loss supervises "how many rounds
# are probably still needed", targeting max(len_a,len_b) as a cheap proxy
# for potential carry-chain length -- deliberately simpler than full
# adaptive halting (see model.py's GPTConfig.combiner_repeat_max comment):
# combiner_repeat_max rounds always run regardless, no gating/early-exit
# in this version.
#
# From scratch -- architecture shape changed completely (no more combine_fn/
# extract_fn, no more fixed 4-layer combiner list), no checkpoint to warm-
# start from.

out_dir = 'out-addition-additive-v13'
eval_interval = 500
eval_iters = 100  # lowered from 200: the repeating combiner is much more expensive per forward pass (16 rounds vs 4 fixed layers), eval_iters=100 keeps eval time reasonable
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage13-additive-composed-repeating-combiner'

dataset = 'addition_padded'  # only used for meta.pkl's vocab ('+'/'=' ids -- unused now by the new combiner, but meta.pkl is still read for it) -- get_batch_additive generates on the fly
gradient_accumulation_steps = 1
batch_size = 64
block_size = 4  # legacy field, unused by AdditiveComposedGPT's encoder/decoder/combiner (comb_config overrides its own block_size internally)

model_type = 'additive_composed'
combiner_layers = 1  # unused by the new repeating combiner (kept only because GPTConfig still has the field) -- see combiner_repeat_max
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'  # combiner block only

composed_recon_weight = 0.5
composed_consistency_weight = 0.5
max_answer_digits = 30  # comfortably past far_ood's 26 digits -- also the repeating combiner's per-place token count

combiner_repeat_max = 16
combiner_countdown_weight = 0.1
combiner_n_thinking = 4

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False  # dynamic per-batch operand shapes defeat torch.compile, same as ComposedGPT
