# Stage 4: fixed-count latent chain-of-thought (LatentLoopedGPT, "Coconut"-
# style). Same n_layer/n_head/n_embd as the Stage 1 RoPE+masked baseline
# (train_addition_rope_masked.py) for direct comparability -- the only
# architectural difference is n_latent_steps=6 continuous latent-thought
# positions inserted between the prompt and the real answer, never decoded
# through the vocabulary. No adaptive halting yet (n_latent_steps is a
# fixed hyperparameter) -- this run is purely to test whether the mechanism
# helps at all before adding a halting probe on top.
#
# Requires addition_padded (needs pad_id, and per-row prompt/answer split);
# train.py groups batches by real prompt length for this model_type (see
# get_batch_latent) since the model needs every row in a batch to share the
# same prompt length -- avoids needing a padding-aware attention mask.
#
# block_size must cover padded_max_len (64) + n_latent_steps -- set with
# headroom below.
#
# First full run (this config, learning_rate=1e-3): test_indist 96.34%,
# gap/hint/ood/carry-heavy-transfer all 0.00% -- same length-generalization
# failure as every other stage. Latent-probe analysis (eval_latent_probe.py)
# found the model reliably computes the correct FIRST digit even on unseen
# lengths (confirmed via logit-lens on the latents AND in real generated
# output), but the autoregressive continuation still stops one digit short
# past the trained length range -- i.e. the STOP decision, not digit
# computation, is what fails to generalize. Two changes for this run:
#  1. countdown_head (model.py): auxiliary head predicting tokens-remaining-
#     until-'\n' at every answer position, directly supervised from ground
#     truth -- dense signal for tracking "how much further to go" instead
#     of the main loss's single-position "is this the '\n' yet" judgment.
#  2. learning_rate briefly tried at 3e-3 (up from 1e-3) to test whether the
#     slow-but-stable convergence was LR-bound -- reverted back to 1e-3
#     before that test finished, to isolate the countdown-head change
#     without also confounding it with an LR change at the same time.

out_dir = 'out-addition-latent-rope'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage4-latent-cot-rope'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 80

model_type = 'latent_looped'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
n_latent_steps = 6
countdown_max = 24
countdown_loss_weight = 0.1

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
