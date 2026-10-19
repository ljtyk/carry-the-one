# Stage 6b: RefineGPT v2 -- adds self-correction training on top of v1's
# masked-window refinement. v1's diagnostic (eval_refine_diagnose.py) showed
# near-zero per-window loss across the ENTIRE trained digit range (1-10),
# yet real generation accuracy stayed poor at the longer end (digit 8-10
# near 0%) -- because v1 only ever trained on a PERFECT ground-truth
# "already committed" prefix, so real generation (which builds the prefix
# from the model's OWN, sometimes-wrong predictions) hits an off-distribution
# input the moment it makes one mistake, and the error cascades with no way
# to recover. See model.py's RefineGPT docstring and get_batch_refine2's
# docstring (train.py) for the fix: random-fraction corruption training
# (some already-"committed" answer positions are deliberately wrong, model
# must learn to predict correctly anyway) -- closer to real absorbing-state
# discrete diffusion than v1's fixed sliding window.
#
# Paired with confidence-based iterative unmasking at inference (eval_refine2.py)
# instead of v1's fixed left-to-right schedule -- lets easy/certain positions
# get committed first regardless of location, providing scaffolding for hard
# ones, rather than being forced to decide the hardest (earliest, least-
# context) position first every time.
#
# Trained from scratch (new training recipe on the same architecture/vocab
# as v1 -- data/addition_refine, unchanged).

out_dir = 'out-addition-refine2'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage6b-refine-v2'

dataset = 'addition_refine'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 104

model_type = 'refine'
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
pos_encoding = 'rope'
bidirectional = True
refine_v2 = True

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
weight_decay = 0.3
beta2 = 0.99

warmup_iters = 100
