# Stage 10: AdditiveComposedGPT -- replaces attention entirely in the
# encoder and decoder (kept only in the combiner), after directly tracing
# the encoder's OOD failure to a specific mechanism: summary-token
# attention concentrates on a small "privileged" subset of digit
# positions, and extending operand length by one digit forces exactly one
# position to be demoted from privileged to ordinary -- which is precisely
# where information gets lost (89% of wrong 11-digit reconstructions drop
# exactly the position that gets demoted). Softmax attention is
# structurally competitive (a fixed budget renegotiated whenever item
# count changes); summation is not. See model.py's AdditiveComposedGPT
# docstring for the full design (shared digit-identity table + shared
# fixed sinusoidal place signal, but separately-learned combine/extract
# FFNs -- encoder and decoder are free to co-discover their own packing
# with the combiner, the only hard constraint is sum-not-attention).
#
# No warm start possible -- this is a structurally different architecture
# from ComposedGPT, not a variant of it.

out_dir = 'out-addition-additive'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage10-additive-composed-no-attention-encoder-decoder'

dataset = 'addition_padded'  # only used for meta.pkl's vocab ('+'/'=' ids for the combiner) -- get_batch_additive generates on the fly
gradient_accumulation_steps = 1
batch_size = 64
block_size = 4  # the combiner's fixed 4-slot sequence length -- encoder/decoder don't use block_size/RoPE at all

model_type = 'additive_composed'
combiner_layers = 4  # matches the reengineered ComposedGPT finding: combiner needs real depth to resolve carries
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'  # combiner only

composed_recon_weight = 0.5
max_answer_digits = 30  # comfortably past far_ood's 26 digits

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 50000
lr_decay_iters = 50000
min_lr = 1e-4
weight_decay = 0.3
beta2 = 0.99

warmup_iters = 200

compile = False  # dynamic per-batch operand shapes defeat torch.compile, same as ComposedGPT
