# Stage 30: does the combiner self-organize per-place structure out of
# generic thinking-token scratch space, given enough slots? Follow-up to
# v35 (combiner_use_per_place=False -> chance-level accuracy, confirmed
# flat regardless of inference-time round count up to 500).
#
# v35's failure could mean two different things: (a) attention over just
# 2 content tokens fundamentally cannot express carry propagation, no
# matter how much scratch space exists alongside it, or (b) it CAN, but
# 4 thinking tokens (v35's count, inherited from v22) is nowhere near
# enough separate slots to reconstruct anything like 30 independent
# place registers. v22's mech-interp already showed thinking tokens
# spontaneously specialize into distinct functional roles with no
# explicit pressure to -- and each place's codeword sits at a fixed,
# deterministic, length-independent offset within vec_a/vec_b (dims
# [3p:3p+3), same for every example since codec_zero_pad writes every
# operand at the full fixed width) -- a genuinely learnable, consistent
# target for a thinking token's attention pattern to lock onto. This
# tests (b) directly: same 2 content tokens, but combiner_n_thinking
# 4->32, matching v22's total token budget (2+32=34, same as v22's
# 30 place + 4 thinking).
#
# combiner_use_per_place stays False, combiner_magnitude_signal stays
# False (still asserted incompatible). Everything else unchanged from
# v35/v22: codec_type='slot', codec_use_magnitude=True, codec_zero_pad=
# True, dpp=3, combiner_repeat_max=24, composed_train_max_digits=20,
# max_answer_digits=30.
#
# Warm-started from v35 -- this is genuinely just a hyperparam change
# (combiner_n_thinking), same architecture branch, same encoder/decoder;
# combiner_block's own weights are shape-independent of n_thinking (it
# operates generically over whatever sequence it's given) so they
# transfer directly, and the digit codewords (already well-separated --
# v35 hit 100% reconstruct-A/B) give a real head start. thinking_tokens
# (4,384)->(32,384) is a real shape mismatch, left fresh via the
# existing pop-and-reinit warm-start logic. Caveat: v35's combiner_block
# never learned anything CORRECT (chance-level accuracy throughout), so
# warm start here isn't inheriting a proven-useful prior, just avoiding
# re-deriving attention/FFN mechanics and the digit codebook from
# scratch -- worth watching whether it's actually neutral-to-helpful or
# whether it's starting from a bad basin.

out_dir = 'out-addition-additive-v36'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v35/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 1000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage30-combiner-no-per-place-more-thinking-tokens'

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
max_answer_digits = 30

combiner_repeat_max = 24
combiner_countdown_weight = 0.1
combiner_n_thinking = 32  # <-- was 4 in v35; the change under test
combiner_use_per_place = False
combiner_explicit_place_tag = False
combiner_magnitude_signal = False

codec_type = 'slot'
codec_dims_per_place = 3
codec_use_magnitude = True
codec_max_scale = 3.0
codec_zero_pad = True

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
