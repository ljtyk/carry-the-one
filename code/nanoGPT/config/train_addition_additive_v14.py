# Ablation 1 of v13's repeating combiner: combiner_use_per_place=False --
# reverts the combiner's input to the old 2-token [vec_a, vec_b] sequence
# (no per-place unbinding at all), but STILL repeats the shared block
# combiner_repeat_max=16 times, keeps thinking tokens and the countdown
# loss. Isolates whether per-place resolution specifically was necessary
# for v13's L=11 generalization (80.5% sum accuracy, first time ever this
# session), or whether repetition alone -- looking at the same
# impoverished 2-vector input more times -- would have been enough.
#
# Reduced max_iters: v13 already showed its key signal (80.5% at L=11) by
# 30% through a 50k schedule, so a shorter budget should be enough to see
# whether this ablation reproduces or kills that signal.

out_dir = 'out-addition-additive-v14'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'scratch'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage13-ablation1-no-per-place'

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

combiner_repeat_max = 16
combiner_countdown_weight = 0.1
combiner_n_thinking = 4
combiner_use_per_place = False  # <-- the ablation

composed_train_max_digits = 10
composed_adversarial_frac = 0.3

learning_rate = 1e-3
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
