# Stage 12: HRR/VSA binding encoder. encode_operand now binds each digit to
# its place via circular convolution with a UNITARY role vector (magnitude
# 1 at every frequency, phase = p * a fixed random base frequency) and
# sums; decode_from_vector reads back out via circular correlation, the
# exact mathematical inverse of binding for the matching place, for ANY p
# -- not just ones seen in training. combine_fn and extract_fn are both
# gone entirely: the decoder is now genuinely parameter-free (unbinding IS
# the query), and the encoder has zero learned parameters beyond
# digit_embed itself (place_signal is fixed/unitary, not learned; there is
# no combine_fn to learn a packing function at all).
#
# Why: every previous attempt (sampling fix, sparsity, cross-place
# orthogonality, GRU) tried to get gradient descent to DISCOVER good
# packing/separation through training pressure, with no guarantee it would
# generalize the discovered pattern past the trained range -- and none of
# them moved the OOD wall at all (still exactly 0% sum accuracy past 10
# digits in every prior variant). HRR binding replaces "hope training
# discovers this" with a property that's true by construction at every
# place, trained or not. probe_hrr_capacity.py validated the actual
# capacity/degradation curve with random (untrained) vectors at dim=384:
# near-perfect through L~15-20, with WHOLE-sequence accuracy compounding
# away at much longer L due to per-place error multiplication, not due to
# interference collapsing catastrophically the way the old mechanisms did.
#
# The combiner is UNCHANGED (4-layer attention) -- per project discussion,
# all remaining nonlinear work (carry propagation) is deliberately left to
# it; the encoder/decoder are designed for provable generalization first,
# not to also do arithmetic.
#
# From scratch -- architecture shape changed (no combine_fn/extract_fn
# parameters at all now), no checkpoint to warm-start from. Sparsity/
# orthogonality penalties left off (0, the default) to get an unconfounded
# read on the HRR mechanism alone.

out_dir = 'out-addition-additive-v12'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage12-additive-composed-hrr-binding'

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
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False  # dynamic per-batch operand shapes defeat torch.compile, same as ComposedGPT
