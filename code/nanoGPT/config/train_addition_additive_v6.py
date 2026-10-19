# Stage 10f: same as v5, but the sparsity penalty is now the Hoyer ratio
# (||contrib||_1/||contrib||_2 per digit, scale-invariant) instead of raw
# L1 mean. v4/v5 (weights 0.05 and 0.2) confirmed empirically that plain L1
# has a trivial escape hatch: raw contribution magnitude dropped ~13-16%
# but "dims needed for 95% of variance" didn't move at all -- the network
# was satisfying the penalty by uniformly shrinking everything by a
# constant factor, which plain L1 can't tell apart from genuine
# concentration but the scale-invariant Hoyer ratio can (uniform scaling
# leaves ||x||_1/||x||_2 completely unchanged).
#
# probe_sparsity_scale.py measured the raw Hoyer ratio against v3's clean
# checkpoint: ~7.5-13.7 across lengths 1/5/10 (max dense=sqrt(384)=19.6,
# max sparse=1.0) -- so meaningfully far from either extreme already.
# Weight chosen conservatively (0.002) given CE losses are currently tiny
# (~0.001-0.01); unlike the unbounded MSE consistency-loss blowup, this
# term is bounded in [1, 19.6] so there's no runaway-divergence risk, but
# still worth checking via smoke test that it doesn't dominate or wreck
# accuracy before committing to the full run.

out_dir = 'out-addition-additive-v6'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v3/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage10f-additive-composed-hoyer-sparsity'

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
composed_contrib_sparsity_weight = 0.002
max_answer_digits = 30  # comfortably past far_ood's 26 digits

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 50000
lr_decay_iters = 50000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False  # dynamic per-batch operand shapes defeat torch.compile, same as ComposedGPT
