# Same as train_addition_rope.py (Stage 1, RoPE), but data/addition now has
# train_mask.bin/val_mask.bin (see data/addition/prepare.py) -- get_batch
# picks this up automatically and masks operand-digit loss targets to
# ignore_index=-1. Operand digits are random and genuinely unpredictable
# from context, so training loss on them was pure irreducible-entropy noise
# diluting the gradient signal; masking should make loss a cleaner signal
# and concentrate training entirely on the parts of the task that are
# actually learnable (predicting the answer). Separate out_dir from
# train_addition_rope.py's unmasked baseline for a direct comparison.

out_dir = 'out-addition-rope-masked'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'baseline-rope-masked'

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
