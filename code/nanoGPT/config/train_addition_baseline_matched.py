# Matched-duration control for the paper's headline comparison: Stage 1's
# EXACT original baseline config (nanoGPT, unmodified GPT class, character-
# level tokenization -- "0123456789+=\n", one token per digit), just with
# max_iters/lr_decay_iters extended 5000->15000 to match v32's training
# duration. Everything else identical to the original nanoGPT/config/
# train_addition.py (commit 4140a95): n_layer=6, n_head=6, n_embd=384,
# dropout=0.2, batch_size=64, block_size=256, same learning_rate/warmup/
# beta2. Rules out "the original baseline was just undertrained" as an
# alternative explanation for its 0% OOD result before using that
# comparison in a writeup.

out_dir = 'out-addition-baseline-matched'
eval_interval = 250
eval_iters = 200
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'baseline-fixed-depth-matched-duration'

dataset = 'addition_stage1_clean'  # plain copy of data/addition's train.bin/val.bin/meta.pkl WITHOUT train_mask.bin/val_mask.bin -- the operand-digit loss mask was added later in the project and would silently apply to the plain 'gpt' path too (confirmed in train.py's get_batch), confounding "duration" with an unrelated training improvement Stage 1 never had
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 15000  # <-- was 5000 in the original Stage 1 config -- matches v32's duration
lr_decay_iters = 15000  # <-- was 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

compile = False
