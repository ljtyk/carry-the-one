# Stage 10b: AdditiveComposedGPT, rebalanced. Three changes vs the original
# stage-10 run (out-addition-additive / out-addition-additive-150k), bundled
# together rather than isolated -- see project discussion:
#
# 1. digit_embed/blank_embed tying (already in model.py's AdditiveComposedGPT
#    since stage 10, but the only real attempt to train it was interrupted
#    at ~900 steps -- this is the first full run of it).
# 2. Length sampling fixed. probe_place_interference.py showed the OOD
#    failure isn't diffuse (uniform decay as more digits are summed) -- it's
#    a sharp cliff concentrated at high place indices (place 8: ~71% acc at
#    L=11, place 9-10: ~40-54%) while low place indices stay ~100% correct
#    even at L=15. Root cause: the old d~Uniform[1,10], other_len~Uniform[1,d]
#    scheme gives place p only P(length>p) training exposure, decaying from
#    100% (place 0) to ~5.5% (place 9). train.py's get_batch_additive now
#    samples len_a/len_b independently via _sample_skewed_length, skewed
#    toward composed_train_max_digits -- every long example trains all its
#    lower places too, and the low end is already saturated at ~100% under
#    the old scheme, so skewing toward the top is a strictly more efficient
#    use of training exposure, not just a bias correction.
# 3. composed_consistency_weight enabled: the "combine(vec_a,vec_b) ==
#    encode(true_sum)" auxiliary loss, ported from ComposedGPT (where it
#    moved vec_c's cosine similarity to encode(true_sum) from ~0.13 to
#    0.978). Implemented to encode each example's true sum at ITS OWN
#    natural digit width (grouped internally in AdditiveComposedGPT.forward),
#    never zero-padded -- encode_operand was never trained with a leading
#    zero digit, so padding would target an out-of-distribution vector.
#
# These three are bundled in one run rather than tested one at a time, so
# this won't cleanly isolate which change did what -- but each is either
# already-decided (tying) or independently well-motivated (sampling fix,
# consistency loss), and re-running three separate 50k-step experiments
# just to isolate them wasn't judged worth the cost here.

out_dir = 'out-addition-additive-v2'
eval_interval = 500
eval_iters = 200
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 10000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage10b-additive-composed-rebalanced'

dataset = 'addition_padded'  # only used for meta.pkl's vocab ('+'/'=' ids for the combiner) -- get_batch_additive generates on the fly
gradient_accumulation_steps = 1
batch_size = 64
block_size = 4  # the combiner's fixed 4-slot sequence length -- encoder/decoder don't use block_size/RoPE at all

model_type = 'additive_composed'
combiner_layers = 4
n_head = 6
n_embd = 384
dropout = 0.0
pos_encoding = 'rope'  # combiner only

composed_recon_weight = 0.5
composed_consistency_weight = 0.5
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
