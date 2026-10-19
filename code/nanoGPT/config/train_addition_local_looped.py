# Stage 5: LocalLoopedGPT -- classical transformer (same attention/MLP/
# LayerNorm/residual primitives, same final-answer-only cross-entropy, no
# injected per-step labels) with ONE architectural constraint: attention
# restricted to a small local window instead of full causal history. See
# model.py's LocalLoopedGPT docstring for the full rationale -- this removes
# the "attend to the whole sequence, infer its length" shortcut that this
# project's grokking/drilling track (Stage 4) never got past, by making that
# shortcut structurally unrepresentable rather than trying to train it away.
#
# Loop count is DETERMINISTIC (ceil(loop_steps_per_token * prompt_len)), not
# learned -- Stage 4's adaptive step-count mechanisms all settled on a
# roughly length-independent count regardless of real difficulty, exactly
# the failure this is designed to avoid.
#
# Trained from scratch (no warm start -- this is a different architecture,
# not a variant of anything with a compatible checkpoint).
#
# Caveat worth remembering if this underperforms: "a+b=" doesn't align
# same-place-value digits at nearby positions (unlike Neural GPU's tape
# format), so this is a strictly harder setting for local attention than
# the architecture's original proof of concept -- a negative result here
# doesn't cleanly falsify the locality hypothesis by itself.

out_dir = 'out-addition-local-looped'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage5-local-looped'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 104

model_type = 'local_looped'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
attn_window = 6
loop_steps_per_token = 2.0
max_loop_iters = 100

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
weight_decay = 0.3
beta2 = 0.99

warmup_iters = 100
