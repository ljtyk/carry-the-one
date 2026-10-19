# Stage 10d: same as v3 (out-addition-additive-v3: tying + skewed length
# sampling + consistency loss + weight_decay=0.1, which converged well --
# in-dist sum accuracy 81.6%, perfect reconstruction 1-10 digits) plus a new
# L1 sparsity penalty on encode_operand's per-digit contributions (before
# the sum), composed_contrib_sparsity_weight.
#
# Motivation: v3 fixed in-distribution training quality but did NOT move the
# OOD wall at all (still exactly 0% sum accuracy past 10 digits, and the
# per-position interference probe showed the cliff at L=11 starts a
# position earlier and is worse everywhere from k6 onward than the original
# 150k baseline). Traced the old attention encoder's much tighter code
# (~2-3 dims/digit vs this encoder's ~190-290/digit, probe_bit_efficiency.py)
# to softmax attention's SELECTIVE, near-one-hot routing -- but that
# selectivity is exactly what caused the original hard-cliff failure (a
# fixed, forcibly-renegotiated competitive budget). An L1 penalty on each
# digit's own contribution (independent per digit, no cross-digit budget)
# aims to recover some of that selective/low-overlap property without
# reintroducing the competitive mechanism that broke length generalization.
#
# probe_sparsity_scale.py checked the raw magnitude of contrib.abs().mean()
# against v3's checkpoint before picking a weight (0.08-0.14 across lengths
# 1/5/10) -- small and bounded, unlike the consistency loss's unbounded raw
# MSE that blew up in v2. 0.05 gives a contribution roughly on the same
# order as the current (very small, ~0.001-0.01) CE losses: a real nudge,
# not expected to dominate.

out_dir = 'out-addition-additive-v4'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v3/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage10d-additive-composed-sparsity'

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
composed_contrib_sparsity_weight = 0.05
max_answer_digits = 30  # comfortably past far_ood's 26 digits

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

# learning_rate lowered from the from-scratch runs' 1e-3: a smoke test
# confirmed 1e-3 destabilizes this already-sharply-converged warm-started
# checkpoint even through the full 200-step warmup (robust loss climbing
# from 0.002 to ~0.2), while 2e-4 stays smoothly in the 0.001-0.02 range --
# standard practice for fine-tuning/warm-starting rather than the original
# from-scratch LR.
learning_rate = 2e-4
max_iters = 50000
lr_decay_iters = 50000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False  # dynamic per-batch operand shapes defeat torch.compile, same as ComposedGPT
