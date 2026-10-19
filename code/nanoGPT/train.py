"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT, LoopedGPT, ACTLoopedGPT, PerExampleLoopedGPT, TwoBlockGPT, LatentLoopedGPT, AdaptiveLatentLoopedGPT, LocalLoopedGPT, RefineGPT, PositionCoupledGPT, ComposedGPT, AdditiveComposedGPT

LATENT_MODEL_TYPES = ('latent_looped', 'adaptive_latent_looped')  # both use get_batch_latent's prompt-length-grouped batching
# LocalLoopedGPT and RefineGPT also need one-prompt-length-per-batch (their
# batch construction depends on knowing where the prompt ends), but each uses
# its own simpler get_batch_* (no K-latent-position shift, no countdown head)
# -- kept as a separate tuple so the LATENT_MODEL_TYPES-gated latent-specific
# logic (get_batch_latent, drill, countdown targets) doesn't apply to them.
PROMPT_GROUPED_MODEL_TYPES = LATENT_MODEL_TYPES + ('local_looped', 'refine')

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
seed_base = 1337 # torch.manual_seed(seed_base + seed_offset) -- was hardcoded to 1337; exposed so different runs can use different init/data-order seeds while keeping everything else identical (e.g. to test whether a learned codebook's structure is seed-dependent noise or a consistent attractor)
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
snapshot_interval = 0 # if > 0, also save a numbered ckpt_<iter_num>.pt every this many iters (for tracking a metric over the course of training, independent of the single rolling ckpt.pt)
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*' or 'warm_start'
resume_ckpt_name = 'ckpt.pt' # init_from='resume' only: override to a numbered snapshot_interval file (e.g. 'ckpt_30000.pt') to recover training state when the rolling "best" ckpt.pt has gone stale (e.g. a harsh val-loss metric stopped it from updating for a long stretch even though real progress kept happening)
warm_start_ckpt = '' # init_from='warm_start' only: path to a checkpoint (.pt file) from a DIFFERENT (but overlapping) architecture -- loads whatever state_dict keys match by name/shape (strict=False, so e.g. a missing n_probe/countdown_head in the source just stays freshly-initialized), always starts a fresh optimizer and iter_num=0/best_val_loss=1e9 (old training state isn't meaningfully transferable across an architecture change)
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
model_type = 'gpt' # 'gpt' (Stage 1), 'looped' (Stage 2), 'act_looped' (Stage 2.1, per-token ACT), 'per_example_looped' (Stage 2.1b, global-per-problem adaptive n, needs padded data), 'two_block' (Stage 3 pass 1, see TwoBlockGPT), 'latent_looped' (fixed-count latent CoT, see LatentLoopedGPT, needs padded data), or 'adaptive_latent_looped' (per-example adaptive latent-step count, see AdaptiveLatentLoopedGPT, needs padded data)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
pos_encoding = 'learned' # 'learned' (default additive absolute wpe), 'rope' (rotary, relative), 'none' (NoPE)
max_loop_iters = 20 # LoopedGPT/ACTLoopedGPT/PerExampleLoopedGPT/TwoBlockGPT: hard cap on floor(n)+1 / max ponder steps
n_init = 6.0 # LoopedGPT/PerExampleLoopedGPT: initial value of the loop-count scalar n
loop_index_embed = False # LoopedGPT/ACTLoopedGPT/PerExampleLoopedGPT/TwoBlockGPT: add a per-iteration embedding each loop (fallback for naive weight-sharing underperformance)
loop_encoding = 'learned' # LoopedGPT only, when loop_index_embed=True: 'learned' (free lookup) or 'sinusoidal' (fixed formula, extrapolates past max_loop_iters)
act_epsilon = 0.01 # ACTLoopedGPT only: halt once cumulative halting probability exceeds 1 - act_epsilon
ponder_cost_weight = 1e-3 # ACTLoopedGPT only: tau, weight of the ponder-cost term in the loss
act_halt_source = 'post_block' # ACTLoopedGPT only: 'post_block' (original) or 'post_attn' (halt-probe reads post-attention state, before mlp)
n_loop_a = 6 # TwoBlockGPT only: fixed number of times block A loops before block B runs once
n_latent_steps = 6 # LatentLoopedGPT: fixed number of latent-thought positions. AdaptiveLatentLoopedGPT: CAP on latent-thought positions (per-example real count is learned, see n_init/max_loop_iters/ponder_cost_weight)
countdown_max = 24 # LatentLoopedGPT/AdaptiveLatentLoopedGPT only: auxiliary countdown head classifies tokens-remaining-until-'\n' into {0..countdown_max}, clamped
countdown_loss_weight = 0.1 # LatentLoopedGPT/AdaptiveLatentLoopedGPT only: weight of the auxiliary countdown loss added to the main cross-entropy
attn_window = None # LocalLoopedGPT only (required, not optional there): restricts attention to the causal-local band [i-attn_window+1, i]
loop_steps_per_token = 2.0 # LocalLoopedGPT only: deterministic loop count = ceil(loop_steps_per_token * prompt_len), NOT learned
bidirectional = False # RefineGPT only (required, not optional there): disables all attention masking
refine_window = 3 # RefineGPT only, v1 (refine_v2=False): sliding refinement window width W (stride is always 1, see get_batch_refine)
refine_v2 = False # RefineGPT only: use get_batch_refine2 (random-fraction corruption, self-correction training) instead of get_batch_refine (v1, fixed sliding window, ground-truth-only prefixes)
n_segments = 0 # GPT only: >0 enables BERT-style segment embeddings (see model.py's GPTConfig.n_segments) and switches to get_batch_segments
# ComposedGPT-only fields (ignored by other classes); see model.py's ComposedGPT
enc_layers = 2 # operand-encoder depth (shared weights, run once each for operand a and b)
combiner_layers = 1 # combiner depth -- always operates on exactly 4 slots (vecA, '+', vecB, '='), regardless of operand length
composed_train_max_digits = 10 # on-the-fly training generation: digit-count range 1..this, matching data/addition/generate.py's TRAIN range
composed_adversarial_frac = 0.3 # fraction of batches built by _sample_carry_heavy_pair instead of _sample_pair, matching generate.py's ADVERSARIAL_FRAC
composed_length_random_max = False # get_batch_additive (AdditiveComposedGPT) only. False: every batch's operand lengths are skewed toward the FIXED composed_train_max_digits -- the model learns "answers are <= that many digits", the diagnosed cause of the ~15% length-extrapolation ceiling (v33 limits test). True: each batch samples its own L_max ~ uniform[8, max_answer_digits-2] and skews toward THAT, so there is no consistent ceiling to memorise -- exposure ranges continuously up to near the slot capacity. Pairs with combiner_attn_band (banded local combiner attention) -- see project discussion "how to get generalized methods".
composed_recon_weight = 1.0 # see model.py's GPTConfig.composed_recon_weight
composed_sum_ce_weight = 1.0 # see model.py's GPTConfig.composed_sum_ce_weight
composed_consistency_weight = 0.0 # see model.py's GPTConfig.composed_consistency_weight
encoder_position_scheme = 'sequential' # see model.py's GPTConfig.encoder_position_scheme
max_answer_digits = 30 # see model.py's GPTConfig.max_answer_digits (AdditiveComposedGPT only)
composed_contrib_sparsity_weight = 0.0 # see model.py's GPTConfig.composed_contrib_sparsity_weight (AdditiveComposedGPT only)
composed_place_orthogonality_weight = 0.0 # see model.py's GPTConfig.composed_place_orthogonality_weight (AdditiveComposedGPT only)
place_orthogonality_max_place = 15 # see model.py's GPTConfig.place_orthogonality_max_place (AdditiveComposedGPT only)
combiner_repeat_max = 16 # see model.py's GPTConfig.combiner_repeat_max (AdditiveComposedGPT only)
combiner_countdown_weight = 0.1 # see model.py's GPTConfig.combiner_countdown_weight (AdditiveComposedGPT only)
combiner_attn_band = 0 # see model.py's GPTConfig.combiner_attn_band (AdditiveComposedGPT only)
combiner_thinking_rope = False # see model.py's GPTConfig.combiner_thinking_rope (AdditiveComposedGPT, combiner_use_per_place=False only)
combiner_n_thinking = 4 # see model.py's GPTConfig.combiner_n_thinking (AdditiveComposedGPT only)
combiner_use_per_place = True # see model.py's GPTConfig.combiner_use_per_place (AdditiveComposedGPT only)
combiner_explicit_place_tag = True # see model.py's GPTConfig.combiner_explicit_place_tag (AdditiveComposedGPT only)
codec_type = 'hrr' # see model.py's GPTConfig.codec_type (AdditiveComposedGPT only)
codec_dims_per_place = 3 # see model.py's GPTConfig.codec_dims_per_place (AdditiveComposedGPT only)
codec_use_magnitude = False # see model.py's GPTConfig.codec_use_magnitude (AdditiveComposedGPT, codec_type='slot' only)
codec_max_scale = 3.0 # see model.py's GPTConfig.codec_max_scale (AdditiveComposedGPT, codec_type='slot' + codec_use_magnitude only)
combiner_magnitude_signal = False # see model.py's GPTConfig.combiner_magnitude_signal (AdditiveComposedGPT, codec_type='slot' + codec_use_magnitude only)
codec_zero_pad = False # see model.py's GPTConfig.codec_zero_pad (AdditiveComposedGPT, codec_type='slot' only)
combiner_variable_repeat = False # see model.py's GPTConfig.combiner_variable_repeat (AdditiveComposedGPT only)
combiner_variable_thinking = False # see model.py's GPTConfig.combiner_variable_thinking (AdditiveComposedGPT only)
codec_seed_numeric_order = False # see model.py's GPTConfig.codec_seed_numeric_order (AdditiveComposedGPT, codec_type='slot' + codec_use_magnitude only)
codec_linear_magnitude = False # see model.py's GPTConfig.codec_linear_magnitude (AdditiveComposedGPT, codec_type='slot' only)
codec_zero_class_weight = 1.0 # see model.py's GPTConfig.codec_zero_class_weight (AdditiveComposedGPT, codec_type='slot' + codec_zero_pad only)
codec_rns_moduli = () # see model.py's GPTConfig.codec_rns_moduli (AdditiveComposedGPT, codec_type='rns' only)
# online error-driven drilling (LATENT_MODEL_TYPES only): every drill_interval
# iters, probe the CURRENT model on fresh random problems in
# [drill_digit_lo, drill_digit_hi] digits (both operands forced to the same
# digit count so the whole probe batch shares one prompt_len), and take
# drill_steps extra gradient steps on whichever ones it gets wrong right now
# -- per user's explicit ask: not a fixed pre-mixed data ratio decided in
# advance, but a live "catch today's mistakes, correct them immediately"
# loop. These probed lengths stop being untouched holdouts once drilled, by
# design -- see project discussion.
drill_enable = False
drill_interval = 250
drill_digit_lo = 11
drill_digit_hi = 21
drill_n_probe = 128
drill_steps = 5
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed_base + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

# attempt to derive vocab_size (and, for padded datasets, pad_id/max_len) from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
padded_pad_id = None
padded_max_len = None
padded_mask_id = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    padded_pad_id = meta.get('pad_id')
    padded_max_len = meta.get('max_len')
    padded_mask_id = meta.get('mask_id')  # RefineGPT only (data/addition_refine)
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# optional per-token loss mask (see data/addition/prepare.py and
# data/addition_padded/prepare.py): 0 at operand-digit target positions
# (random, genuinely unpredictable -- pure irreducible-entropy noise in the
# loss), 1 everywhere else. Works for either the packed-stream or the
# padded per-example format, whichever dataset points to.
mask_train_path = os.path.join(data_dir, 'train_mask.bin')
mask_val_path = os.path.join(data_dir, 'val_mask.bin')
mask_available = os.path.exists(mask_train_path) and os.path.exists(mask_val_path)
if mask_available:
    print(f"found operand-digit loss mask (train_mask.bin/val_mask.bin) in {data_dir}")

# LatentLoopedGPT needs every row in a batch to share the same real prompt
# length (see model.py docstring), so rows are grouped by prompt length
# (position of '=' + 1) instead of sampled i.i.d. across the whole dataset.
# Reuses the existing padded train.bin/val.bin/mask.bin as-is -- no new data
# prep needed, just a different row-selection + target-construction scheme.
prompt_len_groups = {}  # split -> {prompt_len: np.array of row indices}
equals_id = None
if model_type in PROMPT_GROUPED_MODEL_TYPES and padded_max_len is not None:
    with open(meta_path, 'rb') as f:
        meta_full = pickle.load(f)
    equals_id = meta_full['stoi']['=']
    for split in ['train', 'val']:
        bin_path = os.path.join(data_dir, f'{split}.bin')
        data = np.memmap(bin_path, dtype=np.uint16, mode='r').reshape(-1, padded_max_len)
        prompt_lens = (data == equals_id).argmax(axis=1) + 1  # first (only) '=' per row
        groups = {}
        for plen in np.unique(prompt_lens):
            groups[int(plen)] = np.nonzero(prompt_lens == plen)[0]
        prompt_len_groups[split] = groups
        print(f"{split}: grouped into {len(groups)} prompt-length buckets for latent_looped batching")


def get_batch_latent(split):
    groups = prompt_len_groups[split]
    plens = list(groups.keys())
    weights = np.array([len(groups[p]) for p in plens], dtype=np.float64)
    weights /= weights.sum()
    plen = int(np.random.choice(plens, p=weights))  # weighted so this approximates i.i.d. row sampling
    rows_for_plen = groups[plen]
    ix = np.random.choice(rows_for_plen, size=batch_size, replace=True)

    bin_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(bin_path, dtype=np.uint16, mode='r').reshape(-1, padded_max_len)
    rows = torch.from_numpy(data[ix].astype(np.int64))  # (batch_size, padded_max_len)
    K = n_latent_steps
    R = padded_max_len

    # standard one-before/one-after shift (same convention as every other
    # branch): x = rows[:-1], y_orig = rows[1:], same length R-1, elementwise
    # y_orig[:,j] is "the correct token after x[:,j]". forward() splits x at
    # prompt_len and inserts K latents in between, giving an internal
    # sequence of length (R-1)+K; y must be re-indexed to match.
    x = rows[:, :-1]
    y_orig = rows[:, 1:].clone()
    y_orig[y_orig == padded_pad_id] = -1
    if mask_available:
        mask_path = mask_train_path if split == 'train' else mask_val_path
        mask_data = np.memmap(mask_path, dtype=np.uint8, mode='r').reshape(-1, padded_max_len)
        keep = torch.from_numpy(mask_data[ix, 1:].astype(np.bool_))
        y_orig[~keep] = -1

    # internal position m in [0, prompt_len+K-1) has no real target (prompt
    # positions are masked anyway; positions covering the FIRST K-1 latents
    # predict another latent, no ground-truth token exists) -- ignore_index.
    # Internal position (prompt_len+K-1) is the LAST latent -- its "next"
    # position is the first real answer token, i.e. x-column prompt_len,
    # whose correct next token is y_orig[:, prompt_len-1]. From there every
    # subsequent internal position (prompt_len+K+i) corresponds to x-column
    # (prompt_len+i), target y_orig[:, prompt_len+i] -- so the whole tail of
    # y_orig from column prompt_len-1 onward slots in unchanged, shifted
    # right by K. (Previous version started this fill at plen+K instead of
    # plen+K-1, silently leaving the last-latent-to-first-answer-digit
    # transition permanently ignore_index -- the model was never trained to
    # turn its accumulated latent thoughts into the first output digit at
    # all, which produced exactly the "first digit looks like noise, rest
    # compounds from there" failure pattern seen in eval.)
    y = torch.full((batch_size, (R - 1) + K), -1, dtype=torch.int64)
    y[:, plen + K - 1:] = y_orig[:, plen - 1:]

    # auxiliary countdown target: at the same internal position as each real
    # y target, how many tokens (inclusive) remain until and including the
    # row's final '\n'. lengths_data[i] is the real content length (incl.
    # leading '\n'), so the final '\n' sits at column end_pos=lengths_data[i]-1;
    # internal position m's target token sits at original column (m-K+1), so
    # countdown at m = end_pos - (m-K+1) + 1 = end_pos - m + K (count of
    # columns from (m-K+1) through end_pos, inclusive).
    lengths_path = os.path.join(data_dir, f'{split}_lengths.bin')
    lengths_data = np.memmap(lengths_path, dtype=np.int32, mode='r')
    end_pos = torch.from_numpy(lengths_data[ix].astype(np.int64) - 1)  # (batch_size,)
    m_idx = torch.arange((R - 1) + K, dtype=torch.int64)
    countdown = end_pos.view(-1, 1) - m_idx.view(1, -1) + K
    countdown = countdown.clamp(min=0, max=countdown_max)
    countdown[y == -1] = -1  # only meaningful wherever the main target is meaningful

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        countdown = countdown.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        countdown = countdown.to(device)
    return x, y, plen, countdown


def get_batch_local(split):
    """LocalLoopedGPT: same prompt-length-grouped row selection as
    get_batch_latent (reuses prompt_len_groups), but no K-latent-position
    shift and no countdown target -- LocalLoopedGPT loops the SAME sequence
    positions in place (depth-recurrence), it never grows the sequence, so
    plain one-before/one-after next-token targets apply unchanged."""
    groups = prompt_len_groups[split]
    plens = list(groups.keys())
    weights = np.array([len(groups[p]) for p in plens], dtype=np.float64)
    weights /= weights.sum()
    plen = int(np.random.choice(plens, p=weights))
    rows_for_plen = groups[plen]
    ix = np.random.choice(rows_for_plen, size=batch_size, replace=True)

    bin_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(bin_path, dtype=np.uint16, mode='r').reshape(-1, padded_max_len)
    rows = torch.from_numpy(data[ix].astype(np.int64))

    x = rows[:, :-1]
    y = rows[:, 1:].clone()
    y[y == padded_pad_id] = -1
    if mask_available:
        mask_path = mask_train_path if split == 'train' else mask_val_path
        mask_data = np.memmap(mask_path, dtype=np.uint8, mode='r').reshape(-1, padded_max_len)
        keep = torch.from_numpy(mask_data[ix, 1:].astype(np.bool_))
        y[~keep] = -1

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y, plen


def get_batch_refine(split):
    """RefineGPT: simulates one random snapshot of the sliding-window
    refinement schedule (see model.py's RefineGPT docstring). Everything
    before the sampled window start is ground truth ("already committed"),
    the window (width refine_window) plus everything after it is MASK (not
    yet visited, matching real inference), and loss targets are exactly the
    window's true values -- rows whose real content ends before the window
    reaches them naturally get ignore_index there (pad positions), same
    convention as every other get_batch variant here."""
    groups = prompt_len_groups[split]
    plens = list(groups.keys())
    weights = np.array([len(groups[p]) for p in plens], dtype=np.float64)
    weights /= weights.sum()
    plen = int(np.random.choice(plens, p=weights))
    rows_for_plen = groups[plen]
    ix = np.random.choice(rows_for_plen, size=batch_size, replace=True)

    bin_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(bin_path, dtype=np.uint16, mode='r').reshape(-1, padded_max_len)
    rows = torch.from_numpy(data[ix].astype(np.int64))  # (batch_size, padded_max_len)

    # bound t so the window START stays within at least the shortest row's
    # real content -- sampling uniformly over the full padded span (up to
    # padded_max_len - plen) let the window land entirely in pad territory
    # for every row at once on short-answer buckets (all real answers in a
    # prompt_len bucket are within 1 digit of each other), producing an
    # all-ignore_index batch and a NaN loss (observed directly: nan on most
    # smoke-test steps, valid loss only when a bucket's real length happened
    # to cover most of the sampled range by chance)
    lengths_path = os.path.join(data_dir, f'{split}_lengths.bin')
    lengths_data = np.memmap(lengths_path, dtype=np.int32, mode='r')
    min_real_len = int(lengths_data[ix].min())  # real content length (incl. leading/trailing \n)
    max_t = max(0, min_real_len - plen - 1)
    t = np.random.randint(0, max_t + 1)  # window start, relative to column `plen`
    win_start = plen + t
    win_end = min(win_start + refine_window, padded_max_len)

    x = rows.clone()
    y = torch.full((batch_size, padded_max_len), -1, dtype=torch.int64)
    y[:, win_start:win_end] = rows[:, win_start:win_end]
    y[y == padded_pad_id] = -1  # window positions past a row's real content: no target
    x[:, win_start:] = padded_mask_id  # window and everything after: not yet visited

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


_refine2_digit_ids = None  # cached lazily on first use (needs meta_full, loaded above)


def get_batch_refine2(split):
    """RefineGPT v2: random-fraction corruption training, closer to real
    (absorbing-state) discrete diffusion than v1's fixed left-to-right
    sliding window. v1's diagnostic showed near-zero per-window training
    loss across the whole trained digit range, yet real generation accuracy
    stayed poor at the longer end (test_indist digit 8-10 near 0%) --
    v1 only ever trained on a PERFECT ground-truth "locked" prefix, so once
    real (autoregressive-like) generation commits one wrong digit, every
    later pass sees an off-distribution prefix it never learned to handle,
    and the mistake cascades. This targets that directly: each row
    independently corrupts every real answer position with probability p
    (p resampled per batch, matching diffusion training's noise-level
    sampling), where a corrupted position is set to MASK half the time and
    a RANDOM WRONG digit the other half -- so the model must learn to
    predict the true value even when the "given" context around it is
    partly wrong, not just fill in blanks against a trustworthy prefix.
    Loss applies at every corrupted position (both kinds)."""
    global _refine2_digit_ids
    if _refine2_digit_ids is None:
        _refine2_digit_ids = torch.tensor([meta_full['stoi'][str(d)] for d in range(10)])

    groups = prompt_len_groups[split]
    plens = list(groups.keys())
    weights = np.array([len(groups[p]) for p in plens], dtype=np.float64)
    weights /= weights.sum()
    plen = int(np.random.choice(plens, p=weights))
    rows_for_plen = groups[plen]
    ix = np.random.choice(rows_for_plen, size=batch_size, replace=True)

    bin_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(bin_path, dtype=np.uint16, mode='r').reshape(-1, padded_max_len)
    rows = torch.from_numpy(data[ix].astype(np.int64))

    # p ~ Uniform(0,1) made the fully-masked case (p=1, the ONLY condition
    # real generation ever actually starts from) rare in the training
    # sampling distribution -- only ~10% of batches landed above p=0.9 --
    # so the aggregate loss stayed dominated by the much easier low-p
    # majority, hiding a real, confirmed failure at p=1 (direct testing:
    # confident/degenerate wrong answers, ignoring the always-correct
    # prompt) behind a low averaged number. Force p=1 on half of all
    # batches so that exact condition gets undiluted, regular exposure
    # instead of getting averaged away.
    p = 1.0 if np.random.random() < 0.5 else np.random.uniform(0.0, 1.0)
    is_answer = torch.zeros((batch_size, padded_max_len), dtype=torch.bool)
    is_answer[:, plen:] = True
    is_real = rows != padded_pad_id
    corrupt = is_answer & is_real & (torch.rand((batch_size, padded_max_len)) < p)

    # at low sampled p (and/or a short-answer bucket), Bernoulli draws can
    # corrupt NOTHING in the whole batch -- an all-ignore_index target and a
    # NaN loss (observed directly: nan at smoke-test step 50), same failure
    # shape as get_batch_refine's original t-sampling bug. Force at least
    # one valid position corrupted as a fallback.
    eligible = is_answer & is_real
    if not corrupt.any() and eligible.any():
        elig_idx = eligible.nonzero(as_tuple=False)
        choice = elig_idx[np.random.randint(0, elig_idx.size(0))]
        corrupt[choice[0], choice[1]] = True

    x = rows.clone()
    y = torch.full((batch_size, padded_max_len), -1, dtype=torch.int64)
    y[corrupt] = rows[corrupt]

    use_mask = torch.rand((batch_size, padded_max_len)) < 0.5
    mask_positions = corrupt & use_mask
    wrong_positions = corrupt & ~use_mask
    x[mask_positions] = padded_mask_id
    random_digits = torch.randint(0, 10, (batch_size, padded_max_len))
    x[wrong_positions] = _refine2_digit_ids[random_digits[wrong_positions]]

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


_coupled_meta = None  # cached lazily: PositionCoupledGPT reads its own dataset's meta.pkl (data/addition_coupled), independent of whichever `dataset` config points to elsewhere
_COUPLED_BASE = 10  # smallest id used is BASE-2 (newlines); keeps every id >= 0 for any operand length seen this project (up to far_ood's 26 digits)


def get_batch_coupled(split):
    """PositionCoupledGPT: plain random row sampling (no prompt-length
    grouping needed -- position ids are computed per EXAMPLE, not shared
    across a batch), standard next-token x/y shift, plus an explicit
    position_ids tensor built from each row's own content (locations of
    '+'/'=' fully determine the padded operand length L for that row).
    See model.py's PositionCoupledGPT docstring for the id scheme:
    operand digits (both operands, left to right = most-to-least
    significant) get ids BASE..BASE+L-1; the reversed, one-digit-longer sum
    gets the SAME ids in reverse (units first) plus one more (BASE-1) for
    the carry-overflow digit; '+'/'=' both get BASE+L; newlines get BASE-2;
    pad positions get 0 (unused, no loss there)."""
    global _coupled_meta
    if _coupled_meta is None:
        with open(os.path.join('data', 'addition_coupled', 'meta.pkl'), 'rb') as f:
            _coupled_meta = pickle.load(f)
    c_stoi, c_pad_id = _coupled_meta['stoi'], _coupled_meta['pad_id']
    plus_id, eq_id, nl_id = c_stoi['+'], c_stoi['='], c_stoi['\n']

    data_dir_c = os.path.join('data', 'addition_coupled')
    bin_path = os.path.join(data_dir_c, f'{split}.bin')
    max_len_c = _coupled_meta['max_len']
    data = np.memmap(bin_path, dtype=np.uint16, mode='r').reshape(-1, max_len_c)
    ix = np.random.randint(0, data.shape[0], size=batch_size)
    rows = torch.from_numpy(data[ix].astype(np.int64))  # (batch_size, max_len_c)

    mask_path = os.path.join(data_dir_c, f'{split}_mask.bin')
    mask_available_c = os.path.exists(mask_path)
    if mask_available_c:
        mask_data = np.memmap(mask_path, dtype=np.uint8, mode='r').reshape(-1, max_len_c)

    x = rows[:, :-1].clone()
    y = rows[:, 1:].clone()
    y[y == c_pad_id] = -1
    if mask_available_c:
        keep = torch.from_numpy(mask_data[ix, 1:].astype(np.bool_))
        y[~keep] = -1

    T = x.size(1)
    position_ids = torch.zeros((batch_size, T), dtype=torch.long)
    for i in range(batch_size):
        row = rows[i].tolist()
        plus_pos = row.index(plus_id)
        eq_pos = row.index(eq_id)
        L = plus_pos - 1  # leading '\n' occupies index 0
        pid = position_ids[i]
        pid[0] = _COUPLED_BASE - 2  # leading '\n'
        pid[1:1 + L] = torch.arange(_COUPLED_BASE, _COUPLED_BASE + L)  # operand a, MSD->LSD
        pid[plus_pos] = _COUPLED_BASE + L
        pid[plus_pos + 1:plus_pos + 1 + L] = torch.arange(_COUPLED_BASE, _COUPLED_BASE + L)  # operand b, coupled to a
        pid[eq_pos] = _COUPLED_BASE + L
        # reversed sum: L+1 digits, units first -> ids descend from BASE+L-1 to BASE-1
        sum_start = eq_pos + 1
        pid[sum_start:sum_start + L + 1] = torch.arange(_COUPLED_BASE + L - 1, _COUPLED_BASE - 2, -1)
        trailing_nl = sum_start + L + 1
        if trailing_nl < T:
            pid[trailing_nl] = _COUPLED_BASE - 2
        # x is rows[:, :-1] -- position_ids index into x's own positions (0..T-1), matching row[0:T]

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        position_ids = position_ids.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        position_ids = position_ids.to(device)
    return x, y, position_ids


def _composed_digit_range(d):
    if d == 1:
        return 0, 9
    return 10 ** (d - 1), 10 ** d - 1


def _composed_sample_pair(len_a, len_b):
    # python's random, not np.random.randint: operand values run up to
    # 10**26 for far_ood eval, way past np.random.randint's int32/int64
    # bounded-integer path
    lo_a, hi_a = _composed_digit_range(len_a)
    lo_b, hi_b = _composed_digit_range(len_b)
    return random.randint(lo_a, hi_a), random.randint(lo_b, hi_b)


def _composed_sample_carry_heavy_pair(len_a, len_b):
    """Single-pair version of data/addition/generate.py's sample_carry_heavy:
    the longer operand gets a run of trailing 9s (a carry chain), the shorter
    gets a nonzero units digit to trigger a carry into it -- guarantees the
    sum can't be solved by the naive "output length == input length"
    shortcut. See generate.py's module docstring for the full rationale."""
    long_len, short_len = max(len_a, len_b), min(len_a, len_b)
    chain_len = np.random.randint(1, long_len + 1)
    if chain_len == long_len:
        long_digits = '9' * long_len
    else:
        head = str(np.random.randint(1, 10))
        mid = ''.join(str(np.random.randint(0, 10)) for _ in range(long_len - chain_len - 1))
        long_digits = head + mid + '9' * chain_len
    long_val = int(long_digits)
    if short_len == 1:
        short_digits = str(np.random.randint(1, 10))
    else:
        head = str(np.random.randint(1, 10))
        mid = ''.join(str(np.random.randint(0, 10)) for _ in range(short_len - 2))
        tail = str(np.random.randint(1, 10))
        short_digits = head + mid + tail
    short_val = int(short_digits)
    return (long_val, short_val) if len_a == long_len else (short_val, long_val)


_composed_stoi = None  # cached lazily: digit/'+'/'='/'\n' ids, shared with whatever `dataset` config points to


def get_batch_composed(split):
    """ComposedGPT: on-the-fly generation (matches data/addition/generate.py's
    TRAIN-range distribution -- digit-count d in 1..composed_train_max_digits,
    other operand's length uniform in 1..d, composed_adversarial_frac
    carry-heavy mix) instead of reading pre-baked .bin files, because each
    batch must fix a SINGLE (len_a, len_b) pair so the operand encoder can
    tensorize a clean (batch, len) input with no padding/masking -- see
    model.py's ComposedGPT docstring. `split` is accepted for interface
    parity with the other get_batch_* functions but doesn't change anything:
    with this much combinatorial space, train/val collisions are negligible.
    """
    global _composed_stoi
    if _composed_stoi is None:
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            m = pickle.load(f)
        _composed_stoi = m['stoi']
    stoi = _composed_stoi
    digit_ids = [stoi[str(i)] for i in range(10)]
    nl_id = stoi['\n']
    plus_id, eq_id = stoi['+'], stoi['=']
    pad_id_local = digit_ids[0]  # filler for dec_x past a shorter example's real answer; masked out of the loss

    d = np.random.randint(1, composed_train_max_digits + 1)
    other_len = np.random.randint(1, d + 1)
    len_a, len_b = (d, other_len) if np.random.random() < 0.5 else (other_len, d)
    adversarial = np.random.random() < composed_adversarial_frac

    a_vals, b_vals = [], []
    for _ in range(batch_size):
        if adversarial:
            a, b = _composed_sample_carry_heavy_pair(len_a, len_b)
        else:
            a, b = _composed_sample_pair(len_a, len_b)
        a_vals.append(a)
        b_vals.append(b)

    def to_ids(v, width):
        return [digit_ids[int(ch)] for ch in str(v).zfill(width)]

    a_ids = torch.tensor([to_ids(a, len_a) for a in a_vals], dtype=torch.long)
    b_ids = torch.tensor([to_ids(b, len_b) for b in b_vals], dtype=torch.long)
    nl_col = torch.full((batch_size, 1), nl_id, dtype=torch.long)
    a_targets = torch.cat([a_ids, nl_col], dim=1)
    b_targets = torch.cat([b_ids, nl_col], dim=1)

    c_vals = [a + b for a, b in zip(a_vals, b_vals)]
    c_strs = [str(c) for c in c_vals]
    max_L = max(len(s) for s in c_strs)
    dec_x = torch.full((batch_size, max_L), pad_id_local, dtype=torch.long)
    dec_y = torch.full((batch_size, max_L + 1), -1, dtype=torch.long)
    for i, s in enumerate(c_strs):
        ds = torch.tensor([digit_ids[int(ch)] for ch in s], dtype=torch.long)
        L = ds.size(0)
        dec_x[i, :L] = ds
        dec_y[i, :L] = ds
        dec_y[i, L] = nl_id
    # LEFT-zero-padded (zfill), unlike dec_x's right-padding -- leading
    # zeros preserve value, so this is a numerically correct encode_operand
    # input for the true sum, used by the composed_consistency_weight loss
    sum_ids = torch.tensor([to_ids(c, max_L) for c in c_vals], dtype=torch.long)

    tensors = [a_ids, b_ids, dec_x, dec_y, a_targets, b_targets, sum_ids]
    if device_type == 'cuda':
        tensors = [t.pin_memory().to(device, non_blocking=True) for t in tensors]
    else:
        tensors = [t.to(device) for t in tensors]
    a_ids, b_ids, dec_x, dec_y, a_targets, b_targets, sum_ids = tensors
    return a_ids, b_ids, dec_x, dec_y, a_targets, b_targets, sum_ids, plus_id, eq_id


_additive_plus_eq = None  # cached lazily: '+'/'=' vocab ids, needed only for the combiner's structural-token embeddings


def _sample_skewed_length(max_digits, power=2.0):
    """Independent per-operand length draw, skewed toward max_digits (not
    uniform). Diagnosed via probe_place_interference.py: the original
    d~Uniform[1,max], other_len~Uniform[1,d] scheme gives place index p
    only P(length>p) exposure, which decays sharply (e.g. only ~5.5% of
    examples ever populate the highest place at max_digits=10) --
    directly matching the observed accuracy cliff location. Every long
    example trains ALL of its lower places too, while a short example
    only ever trains its own few low places, and the low end is already
    at ~100% accuracy even under the old skewed-short scheme -- so
    skewing toward the top is a more efficient use of training exposure
    than uniform, not just a bias correction."""
    u = random.random()
    return max_digits - int((max_digits - 1) * (u ** power))


def get_batch_additive(split):
    """AdditiveComposedGPT: produces raw digit VALUES (0-9) directly -- the
    encoder no longer uses the shared token vocabulary at all -- and
    right-justified, FIXED-width (max_answer_digits) decode targets with an
    11th "blank" class, since the decoder predicts every place in parallel
    rather than a growing autoregressive sequence, so there is no
    variable-length padding to track. len_a/len_b are sampled independently
    via _sample_skewed_length (see its docstring), not the composed_frac
    coupled scheme get_batch_composed still uses."""
    global _additive_plus_eq
    if _additive_plus_eq is None:
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            m = pickle.load(f)
        _additive_plus_eq = (m['stoi']['+'], m['stoi']['='])
    plus_id, eq_id = _additive_plus_eq

    if composed_length_random_max:
        # no fixed ceiling: each batch skews toward its own random L_max, so
        # the model never sees a consistent "answers stop here" boundary
        lmax = random.randint(8, max_answer_digits - 2)
    else:
        lmax = composed_train_max_digits
    len_a = _sample_skewed_length(lmax)
    len_b = _sample_skewed_length(lmax)
    adversarial = random.random() < composed_adversarial_frac

    a_vals, b_vals = [], []
    for _ in range(batch_size):
        if adversarial:
            a, b = _composed_sample_carry_heavy_pair(len_a, len_b)
        else:
            a, b = _composed_sample_pair(len_a, len_b)
        a_vals.append(a)
        b_vals.append(b)

    def digit_values(v, width):
        return [int(ch) for ch in str(v).zfill(width)]

    def right_justified_targets(vals, width):
        t = torch.full((batch_size, width), 10, dtype=torch.long)  # 10 = blank
        for i, v in enumerate(vals):
            s = str(v)
            L = len(s)
            t[i, width - L:] = torch.tensor([int(ch) for ch in s], dtype=torch.long)
        return t

    def zero_padded_targets(vals, width):
        # codec_zero_pad only: ordinary leading-zero decimal padding, no
        # blank class -- "42" at width 5 is target [0,0,0,4,2], not
        # [10,10,10,4,2]. See model.py's GPTConfig.codec_zero_pad.
        t = torch.zeros((batch_size, width), dtype=torch.long)
        for i, v in enumerate(vals):
            s = str(v)
            L = len(s)
            t[i, width - L:] = torch.tensor([int(ch) for ch in s], dtype=torch.long)
        return t

    def rns_targets(vals, moduli):
        # codec_type == 'rns' only: residue targets, (batch_size, K) --
        # computed directly from the raw (arbitrary-precision Python int)
        # VALUE via plain '%', not from the digit string. No blank/padding
        # concept at all -- every channel always has a genuine residue
        # regardless of the operand's digit length. See model.py's
        # GPTConfig.codec_rns_moduli.
        t = torch.zeros((batch_size, len(moduli)), dtype=torch.long)
        for i, v in enumerate(vals):
            for k, m in enumerate(moduli):
                t[i, k] = v % m
        return t

    a_ids = torch.tensor([digit_values(a, len_a) for a in a_vals], dtype=torch.long)
    b_ids = torch.tensor([digit_values(b, len_b) for b in b_vals], dtype=torch.long)
    c_vals = [a + b for a, b in zip(a_vals, b_vals)]
    if codec_type == 'rns':
        a_targets = rns_targets(a_vals, codec_rns_moduli)
        b_targets = rns_targets(b_vals, codec_rns_moduli)
        dec_y = rns_targets(c_vals, codec_rns_moduli)
    else:
        targets_fn = zero_padded_targets if codec_zero_pad else right_justified_targets
        a_targets = targets_fn(a_vals, max_answer_digits)
        b_targets = targets_fn(b_vals, max_answer_digits)
        dec_y = targets_fn(c_vals, max_answer_digits)

    tensors = [a_ids, b_ids, dec_y, a_targets, b_targets]
    if device_type == 'cuda':
        tensors = [t.pin_memory().to(device, non_blocking=True) for t in tensors]
    else:
        tensors = [t.to(device) for t in tensors]
    a_ids, b_ids, dec_y, a_targets, b_targets = tensors
    return a_ids, b_ids, dec_y, a_targets, b_targets, c_vals, plus_id, eq_id


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    bin_path = os.path.join(data_dir, f'{"train" if split == "train" else "val"}.bin')
    if padded_max_len is not None:
        # one-problem-per-sequence, fixed-length, right-padded rows (see data/addition_padded/prepare.py)
        data = np.memmap(bin_path, dtype=np.uint16, mode='r').reshape(-1, padded_max_len)
        ix = torch.randint(data.shape[0], (batch_size,))
        rows = torch.from_numpy(data[ix].astype(np.int64))
        x = rows[:, :-1]
        y = rows[:, 1:].clone()
        y[y == padded_pad_id] = -1  # ignore_index: no loss on predicting padding
        if mask_available:
            mask_path = mask_train_path if split == 'train' else mask_val_path
            mask_data = np.memmap(mask_path, dtype=np.uint8, mode='r').reshape(-1, padded_max_len)
            keep_rows = torch.from_numpy(mask_data[ix, 1:].astype(np.bool_))
            y[~keep_rows] = -1  # ignore_index: no loss on predicting a random operand digit
    else:
        data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if mask_available:
            mask_path = mask_train_path if split == 'train' else mask_val_path
            mask_data = np.memmap(mask_path, dtype=np.uint8, mode='r')
            keep = torch.stack([torch.from_numpy((mask_data[i+1:i+1+block_size]).astype(np.bool_)) for i in ix])
            y[~keep] = -1  # ignore_index: no loss on predicting a random operand digit
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


_segments_meta = None  # cached lazily: '+'/'='/pad token ids for whatever `dataset` points to


def get_batch_segments(split):
    """Plain GPT + BERT-style segment embeddings: same padded-format batch
    as get_batch, plus a DETERMINISTIC segment_ids tensor (0=structural/pad,
    1=operand a, 2=operand b, 3=answer) computed from each row's own
    content (locations of '+'/'=' -- simple parsing, nothing learned).
    Tests whether explicit "which group do I belong to" information alone
    (no position coupling, no reversal, no padding-to-match-length) fixes
    the operand-grounding failure diagnosed in RefineGPT -- see model.py's
    GPTConfig.n_segments and project discussion."""
    global _segments_meta
    if _segments_meta is None:
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            m = pickle.load(f)
        _segments_meta = (m['stoi']['+'], m['stoi']['='], m['pad_id'])
    plus_id, eq_id, pad_id_local = _segments_meta

    bin_path = os.path.join(data_dir, f'{"train" if split == "train" else "val"}.bin')
    data = np.memmap(bin_path, dtype=np.uint16, mode='r').reshape(-1, padded_max_len)
    ix = torch.randint(data.shape[0], (batch_size,))
    rows = torch.from_numpy(data[ix].astype(np.int64))
    x = rows[:, :-1].clone()
    y = rows[:, 1:].clone()
    y[y == padded_pad_id] = -1
    if mask_available:
        mask_path = mask_train_path if split == 'train' else mask_val_path
        mask_data = np.memmap(mask_path, dtype=np.uint8, mode='r').reshape(-1, padded_max_len)
        keep_rows = torch.from_numpy(mask_data[ix, 1:].astype(np.bool_))
        y[~keep_rows] = -1

    T = x.size(1)
    segment_ids = torch.zeros((batch_size, T), dtype=torch.long)
    for i in range(batch_size):
        row = rows[i].tolist()
        plus_pos = row.index(plus_id)
        eq_pos = row.index(eq_id)
        segment_ids[i, :plus_pos] = 1   # operand a (and its leading '\n', harmless)
        segment_ids[i, plus_pos:eq_pos] = 2  # '+' and operand b
        segment_ids[i, eq_pos:] = 3  # '=' and the answer (trailing pad stays 3 too, harmless: no loss there)
        # x is rows[:, :-1], so segment_ids (built over the full row) is one longer than x;
        # slice to match after the loop

    segment_ids = segment_ids[:, :T]

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        segment_ids = segment_ids.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        segment_ids = segment_ids.to(device)
    return x, y, segment_ids


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, pos_encoding=pos_encoding,
                  max_loop_iters=max_loop_iters, n_init=n_init,
                  loop_index_embed=loop_index_embed, loop_encoding=loop_encoding,
                  act_epsilon=act_epsilon, ponder_cost_weight=ponder_cost_weight, act_halt_source=act_halt_source,
                  pad_id=padded_pad_id, n_loop_a=n_loop_a, n_latent_steps=n_latent_steps,
                  countdown_max=countdown_max, countdown_loss_weight=countdown_loss_weight,
                  attn_window=attn_window, loop_steps_per_token=loop_steps_per_token, bidirectional=bidirectional, n_segments=n_segments,
                  enc_layers=enc_layers, combiner_layers=combiner_layers, composed_recon_weight=composed_recon_weight,
                  composed_sum_ce_weight=composed_sum_ce_weight, composed_consistency_weight=composed_consistency_weight,
                  encoder_position_scheme=encoder_position_scheme, max_answer_digits=max_answer_digits,
                  composed_contrib_sparsity_weight=composed_contrib_sparsity_weight,
                  composed_place_orthogonality_weight=composed_place_orthogonality_weight,
                  place_orthogonality_max_place=place_orthogonality_max_place,
                  combiner_repeat_max=combiner_repeat_max, combiner_countdown_weight=combiner_countdown_weight,
                  combiner_attn_band=combiner_attn_band, combiner_thinking_rope=combiner_thinking_rope,
                  combiner_n_thinking=combiner_n_thinking, combiner_use_per_place=combiner_use_per_place,
                  combiner_explicit_place_tag=combiner_explicit_place_tag,
                  codec_type=codec_type, codec_dims_per_place=codec_dims_per_place,
                  codec_use_magnitude=codec_use_magnitude, codec_max_scale=codec_max_scale,
                  combiner_magnitude_signal=combiner_magnitude_signal,
                  codec_zero_pad=codec_zero_pad, combiner_variable_repeat=combiner_variable_repeat,
                  combiner_variable_thinking=combiner_variable_thinking,
                  codec_seed_numeric_order=codec_seed_numeric_order,
                  codec_linear_magnitude=codec_linear_magnitude,
                  codec_zero_class_weight=codec_zero_class_weight,
                  codec_rns_moduli=codec_rns_moduli) # start with model_args from command line
MODEL_CLASSES = {'gpt': GPT, 'looped': LoopedGPT, 'act_looped': ACTLoopedGPT, 'per_example_looped': PerExampleLoopedGPT, 'two_block': TwoBlockGPT, 'latent_looped': LatentLoopedGPT, 'adaptive_latent_looped': AdaptiveLatentLoopedGPT, 'local_looped': LocalLoopedGPT, 'refine': RefineGPT, 'coupled': PositionCoupledGPT, 'composed': ComposedGPT, 'additive_composed': AdditiveComposedGPT}
model_cls = MODEL_CLASSES[model_type]
if init_from == 'scratch':
    # init a new model from scratch
    print(f"Initializing a new {model_type} model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = model_cls(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir} ({resume_ckpt_name})")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, resume_ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    model_type = checkpoint.get('model_type', model_type)
    model_cls = MODEL_CLASSES[model_type]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'pos_encoding', 'max_loop_iters', 'loop_index_embed', 'loop_encoding', 'pad_id', 'n_loop_a', 'n_latent_steps', 'countdown_max']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = model_cls(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
elif init_from == 'warm_start':
    # Load whatever state_dict keys match by name+shape from a DIFFERENT
    # (but overlapping) architecture -- e.g. LatentLoopedGPT -> Adaptive-
    # LatentLoopedGPT, which share transformer.wte/h.*/ln_f/lm_head/
    # countdown_head exactly but not n_probe (new in the adaptive class).
    # Skips resume's exact-architecture-match machinery entirely: always a
    # fresh optimizer, iter_num=0, best_val_loss=1e9 -- old training state
    # (Adam moments, loss-gated checkpoint threshold) isn't meaningfully
    # transferable across an actual architecture change.
    assert warm_start_ckpt, "init_from='warm_start' requires warm_start_ckpt to point at a source checkpoint"
    print(f"Warm-starting {model_type} from {warm_start_ckpt} (non-strict weight load, fresh optimizer)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = model_cls(gptconf)
    source_checkpoint = torch.load(warm_start_ckpt, map_location=device)
    source_state_dict = source_checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(source_state_dict.items()):
        if k.startswith(unwanted_prefix):
            source_state_dict[k[len(unwanted_prefix):]] = source_state_dict.pop(k)
    # strict=False still raises on a SHAPE mismatch for a key present in
    # both dicts (only skips keys missing from one side entirely) -- pop
    # any such key first so it's just "missing" (left fresh) instead of a
    # hard error. Expected whenever an architecture tweak resizes an
    # existing param (e.g. countdown_head's shape depends on
    # combiner_repeat_max) without changing its name.
    target_state_dict = model.state_dict()
    for k in list(source_state_dict.keys()):
        if k in target_state_dict and source_state_dict[k].shape != target_state_dict[k].shape:
            print(f"warm start: shape mismatch on {k} ({tuple(source_state_dict[k].shape)} -> {tuple(target_state_dict[k].shape)}), left fresh")
            source_state_dict.pop(k)
    missing, unexpected = model.load_state_dict(source_state_dict, strict=False)
    print(f"warm start: {len(missing)} params left fresh (not in source): {missing}")
    print(f"warm start: {len(unexpected)} source params unused (not in target): {unexpected}")
    source_checkpoint = None  # free up memory
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    if 'optimizer' in checkpoint:
        fresh_weight_decays = [g['weight_decay'] for g in optimizer.param_groups]
        optimizer.load_state_dict(checkpoint['optimizer'])
        # load_state_dict restores param_groups (including weight_decay) from the
        # checkpoint verbatim, silently discarding this run's config value -- so a
        # different weight_decay passed on resume would otherwise have no effect
        for g, wd in zip(optimizer.param_groups, fresh_weight_decays):
            g['weight_decay'] = wd
    else:
        # snapshot_interval files intentionally omit optimizer state (they're
        # eval-only, not resume points) -- resuming from one starts Adam's
        # moment estimates fresh. Weights are preserved exactly; only the
        # optimizer's momentum/variance history is lost, a mild perturbation.
        print(f"note: {resume_ckpt_name} has no optimizer state -- starting with a fresh optimizer")
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    # alongside the standard mean cross-entropy, also compute a "robust" val
    # loss: each token's loss clipped at log(vocab_size) -- i.e. no worse than
    # a uniform-random guess -- before averaging. Diagnosed (see project
    # memory) that raw val loss can rise for a long stretch purely because a
    # shrinking-but-increasingly-catastrophic tail of already-wrong
    # predictions gets more confidently wrong (-log(p) is unbounded as
    # p->0), while accuracy and the correctly-predicted majority's own
    # confidence don't move at all. Clipping bounds any single token's
    # contribution so the aggregate tracks real progress instead of outliers.
    out = {}
    model.eval()
    clip_ceiling = math.log(model.config.vocab_size)
    track_hoyer = model_type == 'additive_composed' and getattr(model.config, 'composed_contrib_sparsity_weight', 0) > 0
    if model_type == 'additive_composed' and getattr(model.config, 'composed_place_orthogonality_weight', 0) > 0:
        with torch.no_grad():
            out['place_ortho'] = raw_model.place_orthogonality_loss().item()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        robust_losses = torch.zeros(eval_iters)
        hoyer_ratios = torch.zeros(eval_iters) if track_hoyer else None
        for k in range(eval_iters):
            if model_type in LATENT_MODEL_TYPES:
                X, Y, plen, countdown = get_batch_latent(split)
                with ctx:
                    logits, loss = model(X, targets=Y, prompt_len=plen, countdown_targets=countdown)
            elif model_type == 'local_looped':
                X, Y, plen = get_batch_local(split)
                with ctx:
                    logits, loss = model(X, targets=Y, prompt_len=plen)
            elif model_type == 'refine':
                X, Y = get_batch_refine2(split) if refine_v2 else get_batch_refine(split)
                with ctx:
                    logits, loss = model(X, targets=Y)
            elif model_type == 'coupled':
                X, Y, position_ids = get_batch_coupled(split)
                with ctx:
                    logits, loss = model(X, targets=Y, position_ids=position_ids)
            elif n_segments > 0:
                X, Y, segment_ids = get_batch_segments(split)
                with ctx:
                    logits, loss = model(X, targets=Y, segment_ids=segment_ids)
            elif model_type == 'composed':
                a_ids, b_ids, X, Y, a_tgt, b_tgt, sum_ids, pid, eid = get_batch_composed(split)
                with ctx:
                    logits, loss = model(a_ids, b_ids, X, targets=Y, plus_id=pid, eq_id=eid, a_targets=a_tgt, b_targets=b_tgt, sum_ids=sum_ids)
            elif model_type == 'additive_composed':
                a_ids, b_ids, Y, a_tgt, b_tgt, c_vals, pid, eid = get_batch_additive(split)
                with ctx:
                    logits, loss = model(a_ids, b_ids, targets=Y, plus_id=pid, eq_id=eid, a_targets=a_tgt, b_targets=b_tgt, sum_vals=c_vals)
                if track_hoyer:
                    _, contrib_a = raw_model.encode_operand(a_ids, return_contributions=True)
                    _, contrib_b = raw_model.encode_operand(b_ids, return_contributions=True)
                    r_a = (contrib_a.abs().sum(dim=-1) / contrib_a.norm(dim=-1).clamp(min=1e-8)).mean()
                    r_b = (contrib_b.abs().sum(dim=-1) / contrib_b.norm(dim=-1).clamp(min=1e-8)).mean()
                    hoyer_ratios[k] = (r_a + r_b) / 2
            else:
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
            losses[k] = loss.item()
            per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1),
                                         ignore_index=-1, reduction='none')
            valid = Y.view(-1) != -1
            robust_losses[k] = per_token[valid].clamp(max=clip_ceiling).mean()
        out[split] = losses.mean()
        out[split + '_robust'] = robust_losses.mean()
        if track_hoyer:
            out[split + '_hoyer'] = hoyer_ratios.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
if model_type in LATENT_MODEL_TYPES:
    X, Y, plen, countdown = get_batch_latent('train') # fetch the very first batch
elif model_type == 'local_looped':
    X, Y, plen = get_batch_local('train')
elif model_type == 'refine':
    X, Y = get_batch_refine2('train') if refine_v2 else get_batch_refine('train')
elif model_type == 'coupled':
    X, Y, position_ids = get_batch_coupled('train')
elif n_segments > 0:
    X, Y, segment_ids = get_batch_segments('train')
elif model_type == 'composed':
    a_ids, b_ids, X, Y, a_tgt, b_tgt, sum_ids, plus_id, eq_id = get_batch_composed('train')
elif model_type == 'additive_composed':
    a_ids, b_ids, Y, a_tgt, b_tgt, c_vals, plus_id, eq_id = get_batch_additive('train')
else:
    X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0


def compute_keep_mask_str(text):
    """Same state machine as data/addition_padded/prepare.py's
    compute_keep_mask: 1 = valid loss target (answer digit, final newline),
    0 = operand digit / '+' / '='. Ported here (rather than imported) since
    drilling builds rows on the fly instead of reading pre-baked mask bins."""
    keep = [1] * len(text)
    in_operand = True
    for j, ch in enumerate(text):
        if in_operand and ch.isdigit():
            keep[j] = 0
        if ch in ('+', '='):
            keep[j] = 0
        if ch == '=':
            in_operand = False
        elif ch == '\n':
            in_operand = True
    return keep


def sample_drill_pairs(digit_lo, digit_hi, n):
    # both operands forced to exactly d digits (no mismatched lengths) so
    # every row in the probe batch shares one prompt_len, matching the
    # latent models' single-prompt_len-per-batch requirement
    d = random.randint(digit_lo, digit_hi)
    lo, hi = (0, 9) if d == 1 else (10 ** (d - 1), 10 ** d - 1)
    pairs = [(random.randint(lo, hi), random.randint(lo, hi)) for _ in range(n)]
    return pairs, d


@torch.no_grad()
def greedy_decode_for_drill(prompt_ids, max_new_tokens, prompt_len):
    idx = prompt_ids
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= raw_model.config.block_size else idx[:, -raw_model.config.block_size:]
        with ctx:
            logits, _ = model(idx_cond, prompt_len=prompt_len)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx[:, prompt_len:]


def drill_on_failures():
    """Online hard-example mining: probe the current model on fresh random
    problems outside its dense training range, and immediately take extra
    gradient steps on whichever ones it gets wrong right now. Returns
    (n_wrong, n_probed)."""
    stoi, itos = meta_full['stoi'], meta_full['itos']
    pairs, d = sample_drill_pairs(drill_digit_lo, drill_digit_hi, drill_n_probe)
    prompt_str_len = len(f"{pairs[0][0]}+{pairs[0][1]}=")  # constant: both operands are exactly d digits
    prompt_len = prompt_str_len + 1  # + leading '\n', same convention as get_batch_latent/eval_addition.py
    prompt_ids = torch.tensor([[stoi[c] for c in f"\n{a}+{b}="] for a, b in pairs],
                               dtype=torch.long, device=device)
    max_new_tokens = d + 3  # up to d+1 digit sum, plus newline, plus slack

    model.eval()
    gen = greedy_decode_for_drill(prompt_ids, max_new_tokens, prompt_len)
    model.train()

    wrong_rows = []
    for i, (a, b) in enumerate(pairs):
        text = ''.join(itos[t] for t in gen[i].tolist())
        pred_str = text.split('\n')[0]
        try:
            pred_val = int(pred_str) if pred_str else None
        except ValueError:
            pred_val = None
        if pred_val != a + b:
            wrong_rows.append((a, b))

    n_wrong = len(wrong_rows)
    if n_wrong == 0:
        return 0, drill_n_probe

    # build a training-format batch from exactly the wrong examples, repeated
    # to fill batch_size, same padding/target convention as get_batch_latent
    lines = [f"{a}+{b}={a + b}\n" for a, b in wrong_rows]
    reps = (batch_size + len(lines) - 1) // len(lines)
    lines = (lines * reps)[:batch_size]

    K = n_latent_steps
    # size the buffer off the actual content -- padded_max_len (64) is sized
    # for the dataset's own hint range (up to 18 digits); drilling can reach
    # further (drill_digit_hi up to 21+), whose "\na+b=c\n" strings can run
    # longer than that, so reusing that constant would silently truncate
    R = max(len(f"\n{line}") for line in lines)
    rows = torch.full((batch_size, R), padded_pad_id, dtype=torch.int64)
    lengths = torch.zeros(batch_size, dtype=torch.int64)
    keep_masks = torch.ones((batch_size, R), dtype=torch.bool)
    for i, line in enumerate(lines):
        full = f"\n{line}"
        ids = [stoi[c] for c in full]
        rows[i, :len(ids)] = torch.tensor(ids, dtype=torch.int64)
        lengths[i] = len(ids)
        km = compute_keep_mask_str(full)
        keep_masks[i, :len(km)] = torch.tensor(km, dtype=torch.bool)

    x_d = rows[:, :-1].to(device)
    y_orig_d = rows[:, 1:].clone()
    y_orig_d[y_orig_d == padded_pad_id] = -1
    y_orig_d[~keep_masks[:, 1:]] = -1

    y_d = torch.full((batch_size, (R - 1) + K), -1, dtype=torch.int64)
    y_d[:, prompt_len + K - 1:] = y_orig_d[:, prompt_len - 1:]

    end_pos = (lengths - 1)
    m_idx = torch.arange((R - 1) + K, dtype=torch.int64)
    countdown_d = end_pos.view(-1, 1) - m_idx.view(1, -1) + K
    countdown_d = countdown_d.clamp(min=0, max=countdown_max)
    countdown_d[y_d == -1] = -1

    y_d = y_d.to(device)
    countdown_d = countdown_d.to(device)

    for _ in range(drill_steps):
        with ctx:
            logits_d, loss_d = model(x_d, targets=y_d, prompt_len=prompt_len, countdown_targets=countdown_d)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss_d).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

    return n_wrong, drill_n_probe


def loop_n_log_str():
    # per-example min/avg/max when available (AdaptiveLatentLoopedGPT,
    # PerExampleLoopedGPT) -- much more informative than the mean alone for
    # spotting saturation/collapse at either end. Falls back to the single
    # get_loop_n() value for classes without per-example granularity
    # (LoopedGPT's n is one global scalar, no spread to report).
    per_example = getattr(raw_model, '_last_n_per_example', None)
    if per_example is not None:
        return (f", loop_n min/avg/max {per_example.min().item():.3f}"
                f"/{per_example.mean().item():.3f}/{per_example.max().item():.3f}")
    if hasattr(raw_model, 'get_loop_n'):
        return f", loop_n {raw_model.get_loop_n():.3f}"
    return ""
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        loop_n_str = loop_n_log_str()
        hoyer_str = f", val hoyer {losses['val_hoyer']:.4f} (max dense={model.config.n_embd**0.5:.2f}, max sparse=1.0)" if 'val_hoyer' in losses else ""
        ortho_str = f", place_ortho {losses['place_ortho']:.4f} (max overlap=1.0, min=0.0)" if 'place_ortho' in losses else ""
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
              f"val loss (robust) {losses['val_robust']:.4f}{hoyer_str}{ortho_str}{loop_n_str}")
        if wandb_log:
            wandb_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "val/loss_robust": losses['val_robust'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            if 'val_hoyer' in losses:
                wandb_dict["val/hoyer_ratio"] = losses['val_hoyer']
            if 'place_ortho' in losses:
                wandb_dict["val/place_ortho"] = losses['place_ortho']
            wandb.log(wandb_dict)
        # checkpoint selection uses the robust metric, not raw val loss --
        # raw val loss can keep rising for a long stretch purely from a
        # shrinking-but-increasingly-catastrophic tail of already-wrong
        # predictions (see estimate_loss's comment), which would otherwise
        # starve the rolling "best" checkpoint of updates despite the model
        # not actually getting worse where it matters
        if losses['val_robust'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val_robust']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'model_type': model_type,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        if snapshot_interval > 0 and iter_num > 0 and iter_num % snapshot_interval == 0:
            # no optimizer state: these are eval-only snapshots for tracking a
            # metric over training, not resume points, so skip the ~2x model-size
            # optimizer buffers
            snapshot = {
                'model': raw_model.state_dict(),
                'model_args': model_args,
                'model_type': model_type,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            torch.save(snapshot, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
    if (drill_enable and model_type in LATENT_MODEL_TYPES and iter_num > 0
            and iter_num % drill_interval == 0 and master_process):
        n_wrong, n_probed = drill_on_failures()
        print(f"drill @ iter {iter_num}: {n_wrong}/{n_probed} wrong, {drill_steps} correction steps taken")
        if wandb_log:
            wandb.log({"iter": iter_num, "drill/n_wrong": n_wrong, "drill/n_probed": n_probed})
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if model_type in LATENT_MODEL_TYPES:
                logits, loss = model(X, targets=Y, prompt_len=plen, countdown_targets=countdown)
            elif model_type == 'local_looped':
                logits, loss = model(X, targets=Y, prompt_len=plen)
            elif model_type == 'coupled':
                logits, loss = model(X, targets=Y, position_ids=position_ids)
            elif n_segments > 0:
                logits, loss = model(X, targets=Y, segment_ids=segment_ids)
            elif model_type == 'composed':
                logits, loss = model(a_ids, b_ids, X, targets=Y, plus_id=plus_id, eq_id=eq_id, a_targets=a_tgt, b_targets=b_tgt, sum_ids=sum_ids)
            elif model_type == 'additive_composed':
                logits, loss = model(a_ids, b_ids, targets=Y, plus_id=plus_id, eq_id=eq_id, a_targets=a_tgt, b_targets=b_tgt, sum_vals=c_vals)
            else:
                logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        if model_type in LATENT_MODEL_TYPES:
            X, Y, plen, countdown = get_batch_latent('train')
        elif model_type == 'local_looped':
            X, Y, plen = get_batch_local('train')
        elif model_type == 'refine':
            X, Y = get_batch_refine2('train') if refine_v2 else get_batch_refine('train')
        elif model_type == 'coupled':
            X, Y, position_ids = get_batch_coupled('train')
        elif n_segments > 0:
            X, Y, segment_ids = get_batch_segments('train')
        elif model_type == 'composed':
            a_ids, b_ids, X, Y, a_tgt, b_tgt, sum_ids, plus_id, eq_id = get_batch_composed('train')
        elif model_type == 'additive_composed':
            a_ids, b_ids, Y, a_tgt, b_tgt, c_vals, plus_id, eq_id = get_batch_additive('train')
        else:
            X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        loop_n_str = loop_n_log_str()
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%{loop_n_str}")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
