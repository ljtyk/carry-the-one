# Ablation: BERT-style segment embeddings on an otherwise completely
# classical transformer -- no position coupling, no reversal, no operand
# padding. Tests specifically whether explicit "which group do I belong to"
# information (operand a / operand b / answer, computed deterministically
# from '+'/'=' positions, not learned) fixes the operand-GROUNDING failure
# diagnosed in RefineGPT (wrong predictions that didn't track operand A's
# actual value at all), as distinct from the place-value ALIGNMENT problem
# position coupling solves. Natural (non-padded) digit order, same dataset
# every earlier non-coupled/non-reversed model used.

out_dir = 'out-addition-segments'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage8-segment-embeddings-ablation'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 104

model_type = 'gpt'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
n_segments = 4  # 0=structural/pad, 1=operand a, 2=operand b, 3=answer

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
weight_decay = 0.3
beta2 = 0.99

warmup_iters = 100
