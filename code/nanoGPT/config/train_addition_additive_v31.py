# Stage 25: Residue Number System (RNS) codec. Instead of positional
# decimal places, each of K=17 slots represents value mod a fixed prime
# (2,3,5,7,...,59 -- product ~1.9e21, covers composed_train_max_digits=20
# with margin). Addition becomes fully independent per-channel: (a+b) mod
# m_k = ((a mod m_k)+(b mod m_k)) mod m_k, no carry propagation between
# channels AT ALL -- structurally different from every codec tried this
# session, all of which inherited decimal's sequential carry-chain
# dependency regardless of digit representation.
#
# residue extraction (encode) and CRT reconstruction (eval-time decode)
# are deterministic closed-form math, NOT learned -- same category as
# HRR's fixed role-vector binding, validated exactly via probe_rns_math.py
# (5000 random trials + boundary check, zero failures) before this run.
# Per-channel codewords are QAM-style (direction+bounded magnitude),
# matching v22's still-best-performing recipe -- no phase encoding, no
# linear-magnitude, given today's consistent evidence against imposed
# per-symbol structure. combiner_repeat_max stays at 24 (v22's value) for
# a clean single-variable comparison -- NOT reduced yet, even though "no
# carry chain" suggests it could be, per the discussion before starting
# this run (isolate the codec's effect first, test repeat reduction as an
# explicit follow-up).
#
# codec_dims_per_place=3 (matches v22, n_embd=384/3=128 capacity, only 17
# channels needed here so far more headroom than max_answer_digits=30 slot
# ever had). No codec_zero_pad (RNS has no blank/padding concept -- every
# channel always has a genuine residue regardless of operand length, by
# construction). No warm start -- entirely new representation.

out_dir = 'out-addition-additive-v31'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 1000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage25-rns-codec'

dataset = 'addition_padded'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 4

model_type = 'additive_composed'
combiner_layers = 1
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'

composed_recon_weight = 0.5
composed_consistency_weight = 0.5
max_answer_digits = 30  # only used to size the rns_place_weights buffer's width; RNS's real capacity is set by codec_rns_moduli's product

combiner_repeat_max = 24
combiner_countdown_weight = 0.1
combiner_n_thinking = 4
combiner_use_per_place = True
combiner_explicit_place_tag = False
combiner_magnitude_signal = True

codec_type = 'rns'
codec_rns_moduli = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59)  # product ~1.9e21, covers composed_train_max_digits=20 with margin
codec_dims_per_place = 3
codec_max_scale = 3.0

composed_train_max_digits = 20
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 15000
lr_decay_iters = 15000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
