# Stage 4b: AdaptiveLatentLoopedGPT -- per-example adaptive latent-step
# count, via PerExampleLoopedGPT's exact floor(n)+frac blend mechanism
# ported to the sequence-extension axis (see model.py docstring). Keeps
# LatentLoopedGPT's countdown_head unchanged.
#
# n_latent_steps is now a CAP (not a fixed count) -- raised from the fixed
# run's 6 to 10 to give the per-example decision real room to work with.
# block_size bumped accordingly (needs padded_max_len(64) + cap(10) +
# headroom).
#
# Warm-started from the completed fixed-K + countdown-head run
# (out-addition-latent-rope/ckpt.pt, test_indist 92.96%) -- shares
# transformer.wte/h.*/ln_f/lm_head/countdown_head exactly, only n_probe
# (new in this class) starts fresh. Separate out_dir so the source
# checkpoint isn't touched. Much shorter max_iters than the from-scratch
# runs (15000 vs 50000) since the transformer body is already trained;
# extend if 15000 isn't enough for n_probe/countdown_head to adapt.

out_dir = 'out-addition-latent-adaptive'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage4b-latent-cot-adaptive'

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-latent-rope/ckpt.pt'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 90

model_type = 'adaptive_latent_looped'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
n_latent_steps = 10
n_init = 6.0
ponder_cost_weight = 1e-3
countdown_max = 24
countdown_loss_weight = 0.1

learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
