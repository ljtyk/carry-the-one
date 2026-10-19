# Stage 10g: same as v6 (Hoyer ratio sparsity, the scale-invariant fix for
# plain L1's trivial-shrink escape hatch), but weight raised 10x from 0.002
# to 0.02. v6 showed zero movement in the Hoyer ratio at 20% through
# training (7.475->7.710, 12.224->12.372, 13.740->13.799 for L=1/5/10 --
# actually flat/slightly worse, not a slow-but-real trend), despite the
# raw per-step contribution already being comparable in scale to the
# (tiny, ~0.001-0.01) CE losses -- so this doesn't look like a pure
# magnitude problem, more like the gradient direction being too weak/
# diffuse relative to how sharply-tuned the other losses are. Escalating
# the weight to see if a stronger, more decisive pressure actually moves
# it, same escalation pattern used for plain L1 (0.05 -> 0.2) before that
# was diagnosed as the wrong mechanism entirely.
#
# Warm-started from v3 directly, same as v4/v5/v6 -- none of the sparsity
# attempts so far have shown any real effect worth inheriting.

out_dir = 'out-addition-additive-v7'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v3/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage10g-additive-composed-hoyer-sparsity-0.02'

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
composed_contrib_sparsity_weight = 0.02
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
