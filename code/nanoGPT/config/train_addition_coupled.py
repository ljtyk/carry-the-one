# Stage 7: PositionCoupledGPT -- direct implementation of the strongest
# proven literature technique for this exact problem (Cho et al. 2024,
# "Position Coupling", NeurIPS): digits of the same place value across both
# operands and the (zero-padded, reversed) sum share the same RoPE position
# id. Plain classical transformer otherwise -- full causal attention, single
# multi-layer stack, no loops/locality/bidirectional tricks, no injected
# per-step supervision. The paper reports >95% exact match extrapolating
# from <=30-digit training to 200-digit addition (6.67x).
#
# Used after this project's from-scratch/self-discovery attempts this
# session (adaptive latent CoT + drilling, LocalLoopedGPT windowed
# attention, RefineGPT masked diffusion v1/v2) all failed to produce
# genuine length generalization -- this tests whether the task is
# achievable at all with a known-correct recipe, isolating "is this
# possible" from "can a model discover it unaided."
#
# Trained from scratch (new data format -- zero-padded, reversed sum -- and
# new position-id scheme, not compatible with any existing checkpoint).

out_dir = 'out-addition-coupled'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage7-position-coupling'

dataset = 'addition_coupled'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 104

model_type = 'coupled'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
weight_decay = 0.3
beta2 = 0.99

warmup_iters = 100
