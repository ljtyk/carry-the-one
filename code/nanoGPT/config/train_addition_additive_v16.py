# Ablation 3 of v13's repeating combiner: combiner_explicit_place_tag=False
# -- removes the hand-added "+ role" positional tag from each per-place
# token, leaving ONLY the combiner block's own RoPE (a standard, generic
# sequence-position signal) for the network to work out positional
# structure from. Per-place unbinding (the actual data-extraction step)
# and combiner_repeat_max=16 stay unchanged from v13.
#
# Motivation: per project discussion, unbinding is a general numeral-codec
# operation (not addition-specific, reusable by any future -/* combiner),
# but explicitly injecting the same place-value vectors used elsewhere
# into the combiner's attention is closer to addition-specific hand-
# holding. Tests whether the network can discover positional structure
# from RoPE alone (standard, not hand-crafted) rather than being handed
# an explicit "this is place p" tag.
#
# Reduced max_iters, same rationale as v14/v15.

out_dir = 'out-addition-additive-v16'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage13-ablation3-no-explicit-place-tag'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 4

model_type = 'additive_composed'
combiner_layers = 1
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'

composed_recon_weight = 0.5
composed_consistency_weight = 0.5
max_answer_digits = 30

combiner_repeat_max = 16
combiner_countdown_weight = 0.1
combiner_n_thinking = 4
combiner_use_per_place = True
combiner_explicit_place_tag = False  # <-- the ablation

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
