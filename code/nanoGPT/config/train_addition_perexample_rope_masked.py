# Same as train_addition_perexample_rope.py (Stage 2.1b, RoPE), but with
# three refinements added mid-session:
#  1. data/addition_padded now has train_mask.bin/val_mask.bin (see
#     data/addition_padded/prepare.py) -- get_batch masks operand-digit and
#     '+'/'=' loss targets to ignore_index=-1 (same rationale as the
#     packed-format Stage 1 masked run).
#  2. Density gap vs the packed format (used by every other stage): it packs
#     ~13.7 problems per 256-token window, so its batch_size=64 gradient
#     step is effectively averaged over ~874 problems; the padded
#     (one-problem-per-row) format only sees 64 per step at the same
#     nominal batch_size. First attempt matched this by raising batch_size
#     to 874 directly, but at batch_size=874 the shared/synchronized loop
#     (block() runs on the whole batch every iteration, bounded by the
#     batch's worst-case straggler) made eval (200 iters x 874 x growing
#     loop_n) stall for 20+ minutes per checkpoint -- impractical. Reverted
#     batch_size to 64 and instead scaled max_iters by ~13.7x (5000 -> 68500)
#     to match total problem exposure over the run while keeping each step
#     (and each eval batch) cheap and straggler-safe.
#  3. PerExampleLoopedGPT now has a ponder-cost penalty (model.py), so
#     max_iters/eval numbers here reflect a run where loop count has an
#     actual restoring force instead of only being able to grow.
# Separate out_dir for direct comparison against the unmasked baseline.

out_dir = 'out-addition-perexample-rope-masked'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage2.1b-per-example-rope-masked'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 64

model_type = 'per_example_looped'
n_layer = 6  # used only for c_proj init scaling convention, not an actual layer count here
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
max_loop_iters = 20
n_init = 6.0
loop_index_embed = True

learning_rate = 1e-3
max_iters = 68500
lr_decay_iters = 68500
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 1370  # keep proportional to max_iters (was 100 @ 5000 iters)
