# Stage 13d: same wider-range boundary fix as v17/v18, but warm-started
# from v14 (Ablation 1: combiner_use_per_place=False -- no per-place
# unbind tokens at all, combiner only ever sees vec_a and vec_b as two
# whole-vector tokens) instead of v13 or v16.
#
# Purpose: closes the gap flagged in discussion -- v18 proved the
# explicit place tag was never load-bearing (Ablation 3's original 0%
# verdict was a boundary-bug artifact), by re-running the exact wider-
# range fix and getting v17-equivalent accuracy back. We have NOT run
# the same confirmatory test for Ablation 1 (per-place unbinding
# itself). The only evidence Ablation 1 is genuinely necessary, rather
# than another boundary-bug artifact, is a matched-OOD-distance error
# probe (v14 at L=11/12, one-two digits past the ORIGINAL boundary --
# the same margin where v13 itself scored 80.5% cleanly) showing
# pervasive, non-boundary-shaped errors rather than v13's clean
# leading-digit-only signature. Suggestive, but not proven to the same
# standard as v18. This run closes that gap directly: if per-place
# unbinding is truly load-bearing, this should still fail broadly even
# with the wider range; if it was ALSO just a boundary-bug artifact,
# accuracy should recover to something comparable to v17/v18.
#
# combiner_use_per_place stays False (inherited from v14's architecture
# choice -- combiner sees only the two whole-vector tokens, no per-place
# unbind, no explicit place tag path since that requires per-place
# tokens to attach to). Everything else mirrors v17/v18 exactly so all
# three are a clean three-way comparison: composed_train_max_digits=20,
# combiner_repeat_max=24, learning_rate=2e-4, max_iters=30000.
#
# v14's checkpoint stopped at iter 17000/20000, val loss 0.0006 --
# further converged than v16 was at warm-start time, comparable
# situation to v13/v16 when v17/v18 warm-started from them.

out_dir = 'out-addition-additive-v19'
eval_interval = 500
eval_iters = 100
log_interval = 50

init_from = 'warm_start'
warm_start_ckpt = 'out-addition-additive-v14/ckpt.pt'

always_save_checkpoint = False
snapshot_interval = 5000

wandb_log = False
wandb_project = 'addition'
wandb_run_name = 'stage13d-ablation1-wider-range'

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
combiner_n_thinking = 4
combiner_use_per_place = False  # inherited ablation from v14
combiner_explicit_place_tag = True  # matches v14: still applied as an a/b symmetry-breaking tag on the 2-token [vec_a, vec_b] stack (see model.py combine())

composed_train_max_digits = 20
composed_adversarial_frac = 0.3

learning_rate = 2e-4
max_iters = 30000
lr_decay_iters = 30000
min_lr = 2e-5
weight_decay = 0.1
beta2 = 0.99

warmup_iters = 200

compile = False
