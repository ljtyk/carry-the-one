# Stage 4c-fix: same as train_addition_latent_grok.py (adaptive latent
# steps, weight_decay=1.0, ponder_cost_weight=1e-4), but with the new
# latent_ln fix in AdaptiveLatentLoopedGPT -- the latent gets normalized
# right before being re-fed as the next position's embedding, instead of
# handing back a raw, unboundedly-growing-norm residual-stream vector as
# if it were a normal (roughly-fixed-scale) token embedding. See model.py
# docstring for the full rationale.
#
# Warm-started from the most recent surviving snapshot of the grok run
# (out-addition-latent-grok/ckpt_230000.pt, right before the GPU-crash
# interruption -- test_gap 12.00%, test_hint 20.75%, still climbing, true
# ood still 0.00%) -- only latent_ln itself (768 params) is new/fresh,
# everything else (transformer body, lm_head, countdown_head, n_probe)
# transfers directly. Separate out_dir so the pre-fix run stays intact for
# comparison. Expect a real regression at first (the whole downstream
# model has to adapt to what its own latents now look like normalized,
# not raw), per user's explicit acknowledgement -- the question is
# whether it recovers to the same or better place, and/or fixes the norm
# growth without hurting the interpolation-radius growth this run had
# just found.

out_dir = 'out-addition-latent-grok-normfix'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-latent-grok/ckpt_230000.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage4c-latent-cot-grok-normfix'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 104

model_type = 'adaptive_latent_looped'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
n_latent_steps = 24
n_init = 6.0
ponder_cost_weight = 5e-3
countdown_max = 24
countdown_loss_weight = 0.1

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
weight_decay = 1.0
beta2 = 0.99

warmup_iters = 100
