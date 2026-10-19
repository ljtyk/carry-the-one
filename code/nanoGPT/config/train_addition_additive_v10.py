# Stage 11: linear codec. combine_fn and extract_fn changed from 2-layer
# MLPs (Linear->GELU->Linear) to single Linear layers -- combiner stays at
# 4 layers, unchanged (see project discussion: addition-with-carry is a
# genuinely relational, multi-step computation that benefits from depth,
# unlike the codec's job which is inherently additive/compositional; once
# encoder/decoder go linear the combiner is the ONLY nonlinearity left in
# the whole model, which is an argument for keeping its depth, not cutting
# it).
#
# Motivation: Linear(concat(x_d, x_p)) == W_d@x_d + W_p@x_p, so a linear
# combine_fn is architecturally FORCED to combine digit-identity and place
# additively -- exactly the compositional structure the sparsity (Hoyer
# ratio) and cross-place-orthogonality penalties spent several runs trying
# to coax out indirectly, with real compression achieved (v9: 287->4 dims
# at the extreme) but ZERO OOD improvement (still exactly 0% sum accuracy
# past 10 digits, every variant). A separate diagnostic
# (probe_encoder_vs_decoder.py) also showed a real decoder-generalization
# gap at some positions -- a fresh linear probe fit on the same training
# data already out-generalized the trained nonlinear extract_fn (e.g. k=9:
# 0.328 vs 0.157) -- suggesting the nonlinear MLP readout itself may be
# part of the problem, not just the encoder's raw information content.
#
# This is a from-scratch run (architecture shape changed -- no warm start
# possible) but keeps every other validated-good setting from v3: length
# sampling skewed toward composed_train_max_digits, weight_decay=0.1 (0.3
# caused a real instability episode in v2), consistency loss on. Sparsity/
# orthogonality penalties deliberately left OFF this run to get an
# unconfounded read on whether linearity alone moves the OOD wall.

out_dir = 'out-addition-additive-v10'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage11-additive-composed-linear-codec'

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
