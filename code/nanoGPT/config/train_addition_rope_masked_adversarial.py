# Same as train_addition_rope_masked.py (Stage 1, RoPE, masked), but
# data/addition/generate.py now mixes in ADVERSARIAL_FRAC=0.3 carry-heavy
# examples per TRAIN/HINT bucket (see generate.py docstring and
# sample_carry_heavy) -- so a meaningful fraction of training data can't be
# solved by the "output digit count == longer operand's digit count"
# shortcut, forcing genuine carry-cascade tracking instead. test_indist
# stays natural-distribution (unchanged), so its accuracy number is still
# directly comparable to train_addition_rope_masked.py's; test_gap/hint/ood
# are also unchanged (natural distribution) so they measure the same thing
# as before, while the new test_carry_heavy_gap/ood splits specifically
# test whether the carry-cascade rule transferred to unseen lengths.
# Separate out_dir for direct comparison against the non-adversarial run.

out_dir = 'out-addition-rope-masked-adversarial'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'baseline-rope-masked-adversarial'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
