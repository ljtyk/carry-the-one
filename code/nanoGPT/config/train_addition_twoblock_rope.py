# Stage 3 recreated on the restarted dataset: parameter-matched two-block
# design (block A loops a fixed number of times, then B runs once as a
# terminal readout), each block narrowed so the two blocks TOGETHER
# match the single-block stages' total param budget (isolates whether
# block specialization helps, controlled for total parameter budget).
# RoPE instead of learned absolute position embedding, trained on the
# post-restart data/addition (mismatched operand lengths, four length
# tiers). n_embd=276 (not the pre-restart 270) -- RoPE requires an even
# head_dim (n_embd/n_head), and 270/6=45 is odd; 276/6=46 is the nearest
# even-head_dim width to the ~1.78M single-block-RoPE reference (+3.2%,
# vs 264's -5.5%). Otherwise same hyperparameters as the pre-restart
# train_addition_twoblock_matched.py.

out_dir = 'out-addition-twoblock-rope'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage3-twoblock-rope'

dataset = 'addition'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

model_type = 'two_block'
n_head = 6
n_embd = 276  # 2 blocks at this width = ~1.85M total, closest even-head_dim match to the single-block RoPE budget
dropout = 0.2
pos_encoding = 'rope'
max_loop_iters = 20
loop_index_embed = True
n_loop_a = 6

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
