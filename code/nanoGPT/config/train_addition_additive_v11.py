# Stage 11b: same as v10 (linear codec, combiner still 4 layers), plus a
# LayerNorm on encode_operand's summed output (enc_ln) -- v10 never
# converged (robust CE stuck at ~0.70-0.77 from step 500 through 33500,
# 67% through, total loss wildly unstable the whole time, e.g. spiking to
# 300-450 repeatedly, never self-correcting the way v2's analogous
# blowup did). Root cause: nothing bounded encode_operand's own output
# scale, and composed_consistency_weight's MSE target is computed by the
# SAME weights, creating an unbounded feedback loop -- flagged as the
# right fix back during the original v2 investigation but never
# implemented, since v2's nonlinear (GELU) codec happened to self-correct
# via weight_decay. The linear codec evidently removed whatever weak
# self-regulation GELU provided, so this run adds the LayerNorm directly
# rather than hoping a hyperparameter tweak papers over it again.
#
# From scratch (enc_ln is new parameters, no checkpoint to warm-start from).

out_dir = 'out-addition-additive-v11'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage11b-additive-composed-linear-codec-encln'

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
