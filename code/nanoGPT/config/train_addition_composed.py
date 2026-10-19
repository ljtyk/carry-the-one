# Stage 9: ComposedGPT -- encode each operand into a single fixed-size
# vector (small bidirectional mini-transformer per operand, shared
# weights), combine the two vectors in a tiny ALWAYS-4-token transformer
# (zero length exposure by construction), decode the answer digit-by-digit
# from the combined vector. Auxiliary loss: decode(encode(a)) == a and
# decode(encode(b)) == b, isolating encoder/decoder round-trip fidelity
# from the combiner's addition function. See model.py's ComposedGPT
# docstring and project discussion.

out_dir = 'out-addition-composed'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage9-composed-encode-add-decode'

dataset = 'addition_padded'  # only used for meta.pkl's vocab (digits/+/=/\n) -- get_batch_composed generates on the fly
gradient_accumulation_steps = 1
batch_size = 64
block_size = 40  # generous headroom: encoder needs digit_count+1, decoder needs answer_len+1, up to far_ood's 26 digits

model_type = 'composed'
n_layer = 4        # decoder depth
enc_layers = 2      # operand-encoder depth (shared for a and b)
combiner_layers = 1  # always exactly 4 slots
n_head = 6
n_embd = 384
dropout = 0.0  # see project discussion: dropout=0.2 measurably capped the segment-embeddings ablation's precision
pos_encoding = 'rope'
composed_recon_weight = 1.0

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
weight_decay = 0.3
beta2 = 0.99

warmup_iters = 100
