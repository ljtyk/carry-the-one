"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

def _precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    """cos/sin tables for rotary position embedding, shape (max_seq_len, head_dim) each
    (the head_dim/2 frequencies are duplicated so they line up with the rotate-half convention)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    """x: (B, nh, T, hd); cos/sin: (T, hd), broadcast over batch and heads."""
    return x * cos + _rotate_half(x) * sin


def _sinusoidal_loop_embedding(i, n_embd, base=10000.0, device=None):
    """Fixed (non-learned) sinusoidal encoding for loop iteration i -- same
    formula as the original Transformer's positional encoding / diffusion
    timestep embeddings. Unlike a free nn.Embedding lookup (capped at
    max_loop_iters rows, with no necessary relationship between iterations
    -- iteration 7's vector has zero forced connection to 6's or 8's), this
    is defined for ANY i and gives nearby iterations smoothly related
    vectors: the additive-embedding analog of what RoPE does for sequence
    position, motivated by LoopedGPT's learned loop count saturating
    exactly at max_loop_iters in practice (see project notes)."""
    half = n_embd // 2
    freqs = torch.exp(-math.log(base) * torch.arange(0, half, dtype=torch.float32, device=device) / half)
    args = i * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)])
    if n_embd % 2 == 1:
        emb = torch.cat([emb, torch.zeros(1, device=device)])
    return emb


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.pos_encoding = config.pos_encoding
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        # attn_window: LocalLoopedGPT only. Restricts each position to attending
        # only to itself and the (attn_window - 1) positions immediately before
        # it, instead of the full causal history -- see LocalLoopedGPT docstring
        # for why (removes the "attend to the whole sequence, infer its length"
        # shortcut by construction, forcing genuine position-by-position
        # propagation across loop iterations instead).
        self.attn_window = getattr(config, 'attn_window', None)
        # bidirectional: RefineGPT only. No masking at all -- every position
        # can attend to every other position, including ones "after" it in
        # the raw token order. Needed for iterative refinement: revising an
        # earlier answer digit must be able to see a later digit's current
        # draft, which a causal mask forbids by construction. Mutually
        # exclusive with attn_window (checked in RefineGPT.__init__).
        self.bidirectional = getattr(config, 'bidirectional', False)
        # Interpretability hook (off by default): if a caller sets this to a
        # list, forward() takes the manual (non-flash) attention path and
        # appends the post-softmax weights (B, nh, T, T) every call. Inert
        # otherwise. Used by probe_combiner_attn.py.
        self._attn_store = None
        if self.attn_window is not None:
            idx = torch.arange(config.block_size)
            rel = idx.view(-1, 1) - idx.view(1, -1)  # rel[i,j] = i - j
            window_mask = (rel >= 0) & (rel < self.attn_window)  # causal AND within window
            self.register_buffer("window_mask", window_mask.view(1, 1, config.block_size, config.block_size),
                                  persistent=False)
        if self.pos_encoding == 'rope':
            head_dim = config.n_embd // config.n_head
            assert head_dim % 2 == 0, "RoPE requires an even head_dim (n_embd / n_head)"
            cos, sin = _precompute_rope_freqs(head_dim, config.block_size)
            # not persistent: deterministic from config, no need to bloat checkpoints
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, position_ids=None, attn_mask=None):
        # attn_mask: optional bool tensor broadcastable to (B, nh, T, T) -- True =
        # "this query may attend to this key". When given it OVERRIDES the
        # class's own bidirectional/attn_window/causal masking. Used by
        # AdditiveComposedGPT's banded combiner (combiner_attn_band): a fixed-
        # size local neighbourhood among place tokens plus full connectivity for
        # the thinking tokens, so the attention softmax is over a length-
        # independent key set -- see GPTConfig.combiner_attn_band.
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.pos_encoding == 'rope':
            # rotates q,k by an angle proportional to absolute position, so their dot
            # product (the attention score) depends only on relative offset -- unlike
            # learned absolute wpe, this generalizes to positions beyond block_size
            # seen in a *shorter* training distribution, since the rotation formula
            # itself doesn't need per-position parameters to have been trained
            if position_ids is not None:
                # PositionCoupledGPT only: explicit, non-sequential position ids
                # (digits of the same place value across operands/answer share an
                # id) instead of raw token order -- gather rather than slice.
                # position_ids: (B, T) -> cos/sin: (B, T, head_dim), broadcast over heads.
                cos = self.rope_cos[position_ids].unsqueeze(1)
                sin = self.rope_sin[position_ids].unsqueeze(1)
            else:
                cos = self.rope_cos[:T]
                sin = self.rope_sin[:T]
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash and self._attn_store is None:
            # efficient attention using Flash Attention CUDA kernels
            if attn_mask is not None:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[..., :T, :T], dropout_p=self.dropout if self.training else 0, is_causal=False)
            elif self.bidirectional:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
            elif self.attn_window is not None:
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=self.window_mask[:, :, :T, :T],
                    dropout_p=self.dropout if self.training else 0, is_causal=False)
            else:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attn_mask is not None:
                att = att.masked_fill(~attn_mask[..., :T, :T], float('-inf'))
            elif self.bidirectional:
                pass  # no masking at all
            elif self.attn_window is not None:
                att = att.masked_fill(~self.window_mask[:, :, :T, :T], float('-inf'))
            else:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            if self._attn_store is not None:
                self._attn_store.append(att.detach().clone())
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, position_ids=None, attn_mask=None):
        x = x + self.attn(self.ln_1(x), position_ids=position_ids, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    pos_encoding: str = 'learned' # 'learned' (nanoGPT default: additive absolute wpe), 'rope' (rotary, relative -- applied inside CausalSelfAttention), 'none' (NoPE: no explicit positional signal at all, relies on causal masking alone)
    # LoopedGPT-only fields (ignored by GPT); see LoopedGPT below
    max_loop_iters: int = 20 # hard cap on floor(n)+1, so a runaway n can't blow up memory/compute
    n_init: float = 6.0 # initial value of the learned loop-count scalar n (softplus-parameterized, so always >= 0)
    loop_index_embed: bool = False # add a per-iteration embedding to the residual stream each loop (fallback for when naive weight-sharing underperforms)
    loop_encoding: str = 'learned' # LoopedGPT only, when loop_index_embed=True: 'learned' (free nn.Embedding lookup, capped at max_loop_iters) or 'sinusoidal' (fixed formula, defined for any iteration index, smoothly relates nearby iterations -- see _sinusoidal_loop_embedding)
    # ACTLoopedGPT-only fields (ignored by GPT/LoopedGPT); see ACTLoopedGPT below
    act_epsilon: float = 0.01 # halt once cumulative halting probability exceeds 1 - act_epsilon
    ponder_cost_weight: float = 1e-3 # tau: weight of the ponder-cost term (encourages fewer steps) in the loss
    act_halt_source: str = 'post_block' # 'post_block' (halt-probe reads the state after attn+mlp, original design) or 'post_attn' (reads the state right after attention only, before mlp -- tests whether the cross-token info-routing step alone carries the "am I done" signal). The per-step output used for the residual update and output_accum is always the full post-block state either way.
    # PerExampleLoopedGPT-only field (ignored by other classes); see PerExampleLoopedGPT below
    pad_id: int = None # token id used for right-padding; needed to pool only real (non-pad) positions
    # TwoBlockGPT-only field (ignored by other classes); see TwoBlockGPT below
    n_loop_a: int = 6 # fixed number of times block A loops before block B runs once (Stage 3 pass 1: no adaptive halting yet)
    # LatentLoopedGPT-only field (ignored by other classes); see LatentLoopedGPT below
    n_latent_steps: int = 6 # fixed number of continuous latent-thought positions appended before the real answer (no adaptive halting yet)
    countdown_max: int = 24 # LatentLoopedGPT only: countdown head classifies into {0..countdown_max}, clamped -- generously covers the longest answer+newline in this dataset
    countdown_loss_weight: float = 0.1 # LatentLoopedGPT only: weight of the auxiliary "tokens remaining until '\n'" loss, added to the main cross-entropy
    # LocalLoopedGPT-only fields (ignored by other classes); see LocalLoopedGPT below
    attn_window: int = None # restricts every attention op (in every class, but only ever set for LocalLoopedGPT) to the causal-local band [i-attn_window+1, i] instead of full causal history
    loop_steps_per_token: float = 2.0 # LocalLoopedGPT: deterministic loop count = ceil(loop_steps_per_token * prompt_len), NOT learned -- see class docstring for why
    # RefineGPT-only field (ignored by other classes); see RefineGPT below
    bidirectional: bool = False # disables ALL attention masking (every position can see every other position) -- required for iterative answer-window refinement, see RefineGPT
    # GPT-only field (ignored by other classes): optional BERT-style segment
    # embedding, added to the token embedding at the input, looked up from a
    # DETERMINISTICALLY computed segment id per token (which operand/answer
    # a token belongs to -- simple string parsing, not learned). Tests
    # whether explicit "which group do I belong to" information fixes
    # operand-grounding on an otherwise completely classical transformer (no
    # position coupling, no reversal, no padding) -- see get_batch_segments
    # (train.py) and project discussion. 0 disables it (default GPT, no
    # behavior change).
    n_segments: int = 0
    # ComposedGPT-only fields (ignored by other classes); see ComposedGPT below
    enc_layers: int = 2 # operand-encoder depth (shared weights for operand a and b)
    combiner_layers: int = 1 # combiner depth -- always exactly 4 slots, no length exposure
    composed_recon_weight: float = 1.0 # weight of the auxiliary "decode(encode(operand)) == operand itself" loss (bypasses the combiner: vec_a/vec_b fed directly as the decoder's memory vector), added to the main sum-prediction loss. Isolates encoder/decoder fidelity from the combiner's addition function.
    encoder_position_scheme: str = 'sequential' # ComposedGPT only: 'sequential' (original: digit at index k gets position k+1, offset from summary grows with operand length L) or 'place_value' (digit at index k gets position L-k, so offset from summary is anchored to place value and independent of L -- see encode_operand). Defaults to 'sequential' so checkpoints saved before this flag existed (whose model_args dict has no such key) keep matching the scheme they were actually trained under.
    composed_sum_ce_weight: float = 1.0 # weight of the sum-decode cross-entropy loss (decode_from_vector(vec_c,...) vs the true digits) in the TRAINING loss -- 0 removes it entirely, so the combiner receives NO gradient from ever being decoded; it is trained purely by composed_consistency_weight to produce a vector indistinguishable from a genuine encode(true_sum) output, and decoding is treated as an already-solved capability of the encoder/decoder pair, applied afterward rather than trained for directly. logits are still computed and returned regardless (for eval/monitoring), only the loss contribution is gated.
    composed_consistency_weight: float = 0.0 # weight of the auxiliary "combine(vec_a, vec_b) == encode(true_sum)" loss (MSE, target detached -- the encoder/decoder pair is the fixed universal numeral codec; only the combiner moves toward it). 0 disables it (default: combine()'s output is shaped only by the decode-loss gradient, with no constraint on which basis it lands in). See project discussion: without this, the combiner is free to land anywhere the decoder happens to accommodate, which empirically turned out to be a huge, arbitrary affine transform, not anything resembling the encoder's own coordinate system.
    # AdditiveComposedGPT-only fields (ignored by other classes); see class below
    max_answer_digits: int = 30 # decoder always predicts exactly this many place-value slots in parallel (an 11th "blank" class fills places beyond the true answer length), right-justified -- must comfortably exceed the longest answer ever evaluated (far_ood goes to 26 digits)
    composed_contrib_sparsity_weight: float = 0.0 # weight of a Hoyer-ratio penalty (||contrib||_1/||contrib||_2 per digit, scale-invariant -- NOT plain L1, which turned out to have a trivial escape hatch: the network can satisfy raw L1 by uniformly shrinking magnitude without concentrating onto fewer dims at all, confirmed empirically) on encode_operand's raw per-digit contributions (combine_fn's output, BEFORE the sum) -- pushes each digit's contribution toward activating few dimensions rather than spreading densely across all 384, the same "selective, low-overlap" property that made the old attention encoder's code far more compact (~2-3 dims/digit) than this one's (~100+/digit), but via an independent per-digit penalty instead of softmax's fixed, renegotiated competitive budget (the thing that caused the attention encoder's length-generalization failure in the first place). 0 disables it. NOTE: confirmed to genuinely compress (e.g. one run drove place 9 from 287 to 4 of 384 dims) WITHOUT improving OOD generalization at all -- sparsity alone only guarantees each place uses FEW dimensions, not that DIFFERENT places' few dimensions stay separated from each other; see composed_place_orthogonality_weight for the complementary piece this was missing.
    composed_place_orthogonality_weight: float = 0.0 # weight of a penalty pushing DIFFERENT places' typical contributions apart from each other (mean squared cosine similarity between combine_fn(*, p)'s digit-averaged representative vector, for every pair of places p != p' in 0..place_orthogonality_max_place -- NOT between different digit values at the SAME place, which stays free to organize however's useful for the combiner's carry arithmetic). Motivated by finding that composed_contrib_sparsity_weight alone compresses each place's footprint but does nothing to prevent different places' footprints from colliding with each other -- this term targets the cross-place collision directly, sparsity's missing complement, not a replacement for it. 0 disables it.
    place_orthogonality_max_place: int = 15 # number of places (0..this-1) to compute pairwise cross-place orthogonality over -- covers the trained range (composed_train_max_digits=10) with some margin; doesn't need real digit data, just combine_fn/digit_embed/place_signal at their current weights, so it's cheap regardless of this value.
    # repeating combiner (replaces the fixed-depth attention combiner): a
    # SHARED (same weights every round) attention+FFN Block applied
    # repeatedly over an explicit per-place token sequence (unbind vec_a/
    # vec_b at every place first -- the current combiner never had per-place
    # resolution at all, just 2 pre-summed vectors, which can't represent
    # "move information between places" no matter how it's trained). Fixed
    # depth cannot correctly implement carry propagation in general (worst
    # case: adding 1 to a run of 9s needs the carry to ripple through every
    # place), so depth needs to scale with problem size -- repetition, not
    # a wider/deeper single pass, is the fix. combiner_repeat_max rounds
    # ALWAYS run (no gating/early-exit in this version -- deliberately
    # simpler than full adaptive halting); combiner_countdown_weight adds a
    # dense auxiliary "how many rounds are probably still needed" loss,
    # supervised by max(len_a,len_b) as a cheap, directly-computable proxy
    # for how long a carry chain COULD need to be, not the exact minimal
    # chain length for that specific example.
    combiner_repeat_max: int = 16
    combiner_variable_repeat: bool = False # False (default): every training example runs the full combiner_repeat_max rounds, unconditionally -- the shared block only ever gets gradient from its state at EXACTLY combiner_repeat_max rounds, so it's only ever trained to be correct at that one specific round count. True: forward() samples an actual round count per BATCH, uniformly between target_len (= min(max_len, combiner_repeat_max), the minimum this problem plausibly needs) and combiner_repeat_max (the full budget) -- deliberately including "more rounds than this problem needs" every batch, not just "exactly enough". Confirmed empirically (see project discussion) that running MORE rounds than a fixed-round-trained model ever saw doesn't just fail to help, it actively corrupts an already-correct answer -- the block was never given a reason to leave a converged state alone. This directly targets that: exposure to genuine excess rounds during training should push the block toward a stable fixed point instead of a round-count-specific trajectory. combiner_repeat_max itself stays the upper bound (countdown_head's size, and what inference always uses by default -- see combine()'s n_rounds parameter).
    combiner_variable_thinking: bool = False # mirrors combiner_variable_repeat, for thinking-token COUNT instead of round count. False (default): every training example uses all combiner_n_thinking scratch tokens. True: forward() samples an actual thinking-token count per BATCH, uniformly between 0 and combiner_n_thinking (the ceiling), using thinking_tokens[:n_think] -- see combine()'s n_think parameter. Ablating thinking tokens to 0 outright (fixed, not sampled) was confirmed to genuinely hurt length-extrapolation (not just unused capacity, see project discussion) -- this tests whether VARYING the count instead of removing it, the same fix that worked for repeat-count brittleness, makes the model robust across different scratch-space budgets rather than just picking one bad extreme.
    combiner_countdown_weight: float = 0.1
    combiner_thinking_rope: bool = False # only meaningful when combiner_use_per_place == False. False (default): the [vec_a, vec_b, think_0, ..., think_{n-1}] sequence gets default sequential RoPE positions (0, 1, 2, ..., n+1) -- arbitrary with respect to place semantics, since these are generic scratch tokens with no inherent correspondence to digit place. True: reorders the sequence to [think_0, ..., think_{n-1}, vec_a, vec_b] and gives the thinking tokens explicit positions 0..n-1 -- deliberately mirroring how codec_type=='slot' per-place tokens already get RoPE positions equal to their place value (see combine()'s use_per_place branch, position_ids=None defaults to sequential index), which is what made the offset-(-1) nearest-neighbour carry wavefront learnable in the first place (a RoPE-native relative-offset pattern). Gives the self-organized thinking-token workspace the same translation-invariant, STRUCTURAL (not learned) addressing scheme v22's encoder provides for free, without requiring the encoder itself to expose per-place tokens -- see project discussion, motivated by v36's sharp cliff at L=21 (a learned, not structural, place-to-token mapping). vec_a/vec_b get positions n, n+1, placed AFTER the place-addressed tokens -- the same relative layout as v22 (place-addressed tokens first, global-content tokens after), just with the roles swapped (workspace tokens are place-addressed here; vec_a/vec_b play the "global broadcast" role v22's actual thinking tokens played).
    combiner_attn_band: int = 0 # AdditiveComposedGPT only. 0 (default): combiner attention is full bidirectional over all place+thinking tokens -- confirmed via probe D that only the offset-(-1) neighbour and the thinking-token bus carry signal, but the softmax still runs over EVERY place token, so its per-neighbour mass dilutes as operand length grows (the same competitive-budget effect that broke the original attention encoder, now the diagnosed cause of v33's ~15% length-extrapolation ceiling). >0: place token p may attend only to places [p-band, p+band] (a fixed-size local neighbourhood, translation-invariant), PLUS full connectivity to/from all combiner_n_thinking thinking tokens (kept as a global bus). Makes the attention softmax's key set length-independent -> the combiner becomes a genuine local stencil, iterated combiner_repeat_max times. Requires combiner_magnitude_signal (only that call path threads the mask). See combine() and the combiner_band_mask buffer.
    combiner_n_thinking: int = 4 # extra learned scratch tokens, alongside the per-place tokens, for the repeated block to use as working memory (e.g. a running carry) separate from the place tokens' own content
    combiner_use_per_place: bool = True # ablation switch: False reverts the combiner's input to the old 2-token [vec_a, vec_b] sequence (still repeated combiner_repeat_max times) -- isolates whether per-place resolution specifically was necessary, or repetition alone would have been enough regardless of what it's looking at
    combiner_explicit_place_tag: bool = True # ablation switch: False removes the hand-added "+ role" positional tag from each per-place token, leaving ONLY the combiner block's own RoPE (a standard, generic sequence-position signal, not a hand-crafted place-semantic one) for the network to work out positional structure from -- per project discussion: the unbind step itself is a general numeral-codec operation (not addition-specific, reusable by any future combiner), but explicitly injecting the same place-value vectors used elsewhere is closer to addition-specific hand-holding than to something a `-`/`*` combiner would equally need. Only meaningful when codec_type == 'hrr' (needs place_signal, which 'slot' doesn't build).
    # alternative codec: instead of HRR superposition (every place sharing
    # the same n_embd dims, crosstalk growing with how many places are
    # superposed at once -- see probe_hrr_capacity.py), give every place
    # its OWN disjoint slice of the vector (place p owns dims
    # [dpp*p : dpp*(p+1)], nobody else ever writes there). Zero cross-place
    # interference within capacity BY CONSTRUCTION -- probe_slot_capacity.py
    # confirmed exact (1.000) per-place and whole-sequence recovery for
    # every L up to n_embd // codec_dims_per_place with untrained random
    # digit directions, then a hard, predictable cliff exactly at that
    # boundary -- a fundamentally different failure mode than HRR's graceful
    # decay-then-compounding-collapse (starts eroding whole-sequence
    # accuracy well before the hard wall). Trades HRR's open-ended (but
    # statistical, decaying) capacity for an exact-but-hard-capped one;
    # dominates HRR whenever max_answer_digits comfortably fits under the
    # cap, which it does here (30 << 384//3=128).
    codec_type: str = 'hrr' # 'hrr' (circular-convolution binding, unbounded place range, graceful-then-compounding degradation), 'slot' (disjoint per-place slices, exact within a hard cap of n_embd // codec_dims_per_place places), or 'rns' (disjoint per-RESIDUE-CHANNEL slices -- same disjoint-slot mechanics as 'slot', but each slot represents value mod codec_rns_moduli[k] instead of a decimal digit -- see codec_rns_moduli)
    codec_rns_moduli: tuple = () # only meaningful when codec_type == 'rns'. Pairwise-coprime moduli (e.g. the first K primes) -- each slot represents one residue channel, not one decimal place. Addition becomes fully independent per-channel ((a+b) mod m_k = ((a mod m_k)+(b mod m_k)) mod m_k) -- NO carry propagation between channels at all, unlike positional/decimal representation where carry chains scale with digit count. Product of moduli must exceed the largest representable value for lossless reconstruction (CRT), though reconstruction is a deterministic eval-time-only step, never needed during training (training is just len(moduli) independent per-channel classification losses). encode_operand computes each channel's residue from digit_values via a closed-form weighted-sum-mod-m formula (precomputed per-place weights, NOT learned -- same category as HRR's fixed role-vector binding, not a "hint" to the combiner, which still has to do all the actual arithmetic itself). Per-channel codewords are QAM-style (direction+bounded magnitude, like codec_use_magnitude) since class count varies per channel (moduli differ) -- there's no "plain embedding" option for codec_type=='rns'.
    codec_dims_per_place: int = 3 # only used when codec_type == 'slot'. Trades capacity (n_embd // this many places, exactly) against angular separation margin between the 10 digit directions sharing that slice -- see project discussion: random-direction margin goes 0deg/17deg/38deg/52deg for 1/2/3/4 dims at n_embd=384. 3 chosen as a reasonable middle point (128-place cap, ~38deg margin); cheap to sweep since it's a pure constant, doesn't change the architecture.
    codec_use_magnitude: bool = False # only meaningful when codec_type == 'slot'. False (default): digit_embed is a plain nn.Embedding, per-place normalized to unit norm at write time (pure direction/PSK-style code) -- matches every slot-codec run before this flag existed. True: QAM-style -- digit_embed is decoupled into separate direction and BOUNDED scale parameters (codec_max_scale), magnitude is written as-is (no per-place normalize) and genuinely carries information, and decode uses the energy-corrected rule (query.class_i - 0.5*||class_i||^2, i.e. squared-Euclidean-distance ranking) instead of raw dot product. Motivated by two confirmed-empirically findings: (1) a plain unit-norm codebook only uses the sphere's surface, not its volume, capping how well 11 symbols can separate in a few dims (see project discussion, direction-only optimum ~60.6deg for N=11,dim=3 vs what training reached); (2) even under the OLD direction-only scheme, where magnitude was completely functionally inert, raw digit_embed norms were STILL growing ~100-150x from init with no sign of plateauing across 10000 iterations -- confirming that letting magnitude matter without an explicit bound and without the energy correction (which removes the "just scale everything up" free-lunch escape hatch for softmax) would very likely run away rather than settle into a meaningful amplitude code.
    codec_max_scale: float = 3.0 # only meaningful when codec_type == 'slot' and codec_use_magnitude == True. Hard ceiling on each codeword's magnitude (scale = codec_max_scale * sigmoid(raw_scale_param)) -- by construction, not left to weight decay to (hopefully) restrain, given the runaway-growth finding above.
    combiner_magnitude_signal: bool = False # only meaningful when codec_type == 'slot' and codec_use_magnitude == True. combiner_block is a standard Pre-LN transformer block (ln_1/ln_2 BEFORE attn/mlp) -- confirmed via project discussion that this makes attention and the FFN structurally blind to each place-token's magnitude, every round: only the additive residual skip carries any trace of it, never anything attention/mlp actually COMPUTE with. True: explicitly re-injects a small learned embedding of (||qa||, ||qb||) -- the ORIGINAL per-place input magnitudes, not the evolving internal state's own norm -- right after ln_1/ln_2 at EVERY repeat round, so attention/mlp can actually condition on it instead of only ever seeing it erased. False (default): behavior identical to before this flag existed.
    codec_zero_pad: bool = False # only meaningful when codec_type == 'slot'. Eliminates the blank class entirely: every operand/sum is represented at the FIXED width max_answer_digits via ordinary leading-zero decimal padding ("42" -> "000...042"), so every place in every training example always has a genuine digit target (0-9) -- never "sometimes populated, sometimes not". Targets the actual root cause of the recurring leading-digit boundary bug (a per-ABSOLUTE-POSITION "usually blank" prior from uneven training exposure across positions, not a capacity limit -- confirmed twice this session, once in the HRR line, once in the slot+QAM line): with no blank class, there's no such prior to miscalibrate in the first place. No hints, no masking, no externally-computed boundary -- just a different, uniform representation of the same numbers. False (default): current right-justify-with-blank-padding behavior, unchanged.
    codec_seed_numeric_order: bool = False # only meaningful when codec_type == 'slot' and codec_use_magnitude == True. Motivated by probe_codebook_geometry.py finding that the trained-from-random-init codebook (v22) shows essentially no phase/circular structure (mean nearest-angular-neighbor numeric offset 1.9, vs 1.0 for perfect phase order and ~3.3 for a random permutation) -- decode's separation-only loss is permutation-invariant, so nothing pushes a random init toward the arithmetically meaningful order even if the combiner would benefit from it. True: digit_direction is initialized (not frozen -- still a normal learnable Parameter, fully free to drift during training) to sit at consecutive angles around the circle, digit_direction[d] = init_scale * [cos(2*pi*d/10), sin(2*pi*d/10), 0, 0, ...] -- only meaningful/expected to matter when combined with codec_dims_per_place==2, since with >=3 dims the codebook can just rotate/tilt off the seeded plane without needing to break numeric order. Tests whether numeric order is a stable/attracting arrangement (combiner gradient holds or sharpens it) vs. one nothing anchors (it drifts back toward an arbitrary permutation). False (default): current torch.randn(10, codec_dim) * 0.02 init, unchanged.
    codec_zero_class_weight: float = 1.0 # only meaningful when codec_type == 'slot' and codec_zero_pad == True. Cross-entropy weight for class 0 in decode_from_vector (applied to sum-decode AND both reconstruction losses, all three share the same zero-padded representation). Padding makes digit 0 far more frequent than any other digit under codec_zero_pad -- measured ~12x at composed_train_max_digits=20/max_answer_digits=30 under _sample_skewed_length (mean operand length ~14.1 of 30 slots, so ~53% of positions are padding). Motivated by codec_linear_magnitude collapsing digits 1-9's magnitude separation while digit 0 claimed a hugely disproportionate share of the (explicitly zero-sum, gap-cumsum-normalized) magnitude budget -- an unweighted mean cross-entropy lets nailing the very-common "is this padding" call dominate the loss at the expense of the other 9 classes' separation. 1.0 (default): unweighted, unchanged from every prior run. ~0.08 (1/12.2, the measured inverse frequency) makes digit 0's expected total gradient contribution per-example roughly match every other digit's.
    codec_linear_magnitude: bool = False # only meaningful when codec_type == 'slot'. A different hypothesis from codec_use_magnitude's QAM scheme (per-digit direction AND magnitude) or codec_seed_numeric_order's phase scheme (per-digit angle, digit-sum = angle addition, a bilinear/multiplicative operation): here EVERY digit shares the SAME direction (codec_dims_per_place must be 1 -- no direction freedom at all, pure scalar code), and only magnitude varies with digit value. Motivated by: under this scheme, vector addition of two operands' codewords IS digit addition (magnitudes literally sum), and carry-out is a simple magnitude-threshold comparison -- not the bilinear complex-multiplication the phase scheme needs, and not requiring the combiner to discover a good packing (there's nothing to pack, order is definitionally linear). Magnitude is built from 9 free gap params (softplus'd, cumsum'd, then the whole ladder linearly rescaled to land exactly on [0, codec_max_scale]) rather than one independent sigmoid-bounded scalar per digit -- see __init__'s comment for why: the original per-digit-sigmoid version collapsed digits 1-9 together over training (unbounded raw params with no weight decay ran into sigmoid's saturating tails, where growing raw-space separation buys ever-less magnitude-space separation). Requires codec_zero_pad=True (blank-codeword construction not implemented for this scheme). combiner_magnitude_signal is especially relevant here since magnitude is the ONLY channel carrying information at all.

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        transformer_modules = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.pos_encoding == 'learned':
            transformer_modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.segment_emb = nn.Embedding(config.n_segments, config.n_embd) if config.n_segments > 0 else None
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, segment_ids=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.segment_emb is not None:
            assert segment_ids is not None, "n_segments > 0 requires segment_ids at every forward call"
            tok_emb = tok_emb + self.segment_emb(segment_ids)
        if self.config.pos_encoding == 'learned':
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            # 'rope': positional info injected inside attention instead, per-block
            # 'none' (NoPE): no explicit positional signal at all, relies on causal masking
            x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
            if hasattr(block.attn, 'rope_cos'):
                block.attn.rope_cos = block.attn.rope_cos[:block_size]
                block.attn.rope_sin = block.attn.rope_sin[:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class LoopedGPT(nn.Module):
    """
    Stage 2: single shared transformer block (attention + FFN together, same
    width as a standard GPT block) applied floor(n)+1 times, where n is a
    learned scalar. The final application is blended by frac(n) so effective
    depth is continuous and differentiable: just below an integer n, the
    blended term contributes ~0 (matching floor(n) full applications); just
    above, it contributes ~all of itself (matching floor(n)+1 applications).

    A standalone class (not a GPT subclass) mirroring GPT's structure, so
    Stage 1's GPT class stays untouched and reproducible.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config)]),  # single shared block, applied in a loop
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.pos_encoding == 'learned':
            transformer_modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # per-iteration embedding added to the residual stream before each loop
        # application, so the shared block weights can behave differently
        # depending on loop depth despite being reused (Universal Transformer-style
        # "timestep embedding" fix for naive weight-sharing underperformance).
        # 'learned': free nn.Embedding lookup (original approach). 'sinusoidal':
        # fixed formula, no parameters needed, defined for any iteration index.
        self.loop_embed = None
        if config.loop_index_embed and config.loop_encoding == 'learned':
            self.loop_embed = nn.Embedding(config.max_loop_iters, config.n_embd)

        # n = softplus(loop_n_raw), so n >= 0 always; init so softplus(raw) == config.n_init
        n_init = torch.tensor(float(config.n_init))
        raw_init = torch.log(torch.expm1(n_init))  # inverse softplus
        self.loop_n_raw = nn.Parameter(raw_init)

        self.apply(self._init_weights)
        # scaled init for residual projections, per GPT-2 paper -- config.n_layer here is
        # repurposed as "expected effective depth" for this scaling only (no separate
        # per-layer blocks exist to iterate over), matching config.n_init by convention
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_loop_n(self):
        """Current value of the learned loop-count scalar (for logging)."""
        with torch.no_grad():
            return F.softplus(self.loop_n_raw).item()

    def _loop_idx_emb(self, i, device):
        if not self.config.loop_index_embed:
            return 0
        if self.config.loop_encoding == 'learned':
            return self.loop_embed.weight[i]
        return _sinusoidal_loop_embedding(i, self.config.n_embd, device=device)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        if self.config.pos_encoding == 'learned':
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(tok_emb)

        block = self.transformer.h[0]
        n = F.softplus(self.loop_n_raw)
        n_floor = int(torch.floor(n).item())
        n_floor = max(0, min(n_floor, self.config.max_loop_iters - 1))  # leave room for the +1
        # n itself is unbounded; without clamping, frac > 1 once n exceeds max_loop_iters
        # would extrapolate past x_last instead of interpolating between x and x_last
        # (never triggered in the Stage 2 run -- n stayed well under the cap -- but latent)
        frac = torch.clamp(n - n_floor, max=1.0)  # differentiable; n_floor itself is not (control flow only)

        for i in range(n_floor):
            x = block(x + self._loop_idx_emb(i, device))
        x_last = block(x + self._loop_idx_emb(n_floor, device))
        x = (1 - frac) * x + frac * x_last

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # loop_n_raw is a scalar (0-dim), same "no decay" treatment as biases/LayerNorm
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ MFU estimate using the *actual* current loop depth (floor(n)+1), not a static n_layer. """
        N = self.get_num_params()
        cfg = self.config
        L = min(int(torch.floor(F.softplus(self.loop_n_raw)).item()), cfg.max_loop_iters - 1) + 1
        H, Q, T = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class LocalLoopedGPT(nn.Module):
    """
    Tests whether a CLASSICAL transformer (same attention + MLP + LayerNorm +
    residual-stream primitives as every other class here, same final-answer-
    only cross-entropy loss, no injected per-step/carry labels) can discover
    a length-generalizing addition algorithm on its own, given one specific
    architectural constraint: attention restricted to a small local window
    (config.attn_window) instead of the full causal history.

    Motivation (see project discussion): a full-attention shared-block loop
    (LoopedGPT) can, in a single attention op, look at the ENTIRE sequence
    and read off cheap global features like "how many digits does this
    problem have" -- a shortcut that fits the training distribution just as
    well as genuine carry propagation, and empirically that's what gets
    found (this project's whole grokking/drilling track never produced
    extrapolation past directly-trained-or-drilled lengths). Restricting
    attention to a small window removes that shortcut from the hypothesis
    space entirely: no single attention operation can see far enough to
    infer sequence length, so the ONLY way to correctly propagate a carry
    from one end of a long operand to the other is to actually propagate it,
    one window-hop at a time, across repeated applications of the shared
    block -- structurally the same "local, weight-shared, repeated update"
    inductive bias behind Neural GPU's (Kaiser & Sutskever 2015) genuine
    length generalization on binary addition, adapted to stay inside the
    transformer family (windowed self-attention, a standard, well-precedented
    transformer variant) rather than switching to a convolutional recurrent
    architecture.

    Loop count is DETERMINISTIC, not learned: n_loops = ceil(loop_steps_per_token
    * prompt_len), clamped to max_loop_iters. This project's adaptive-step
    mechanisms (LoopedGPT's learned n, AdaptiveLatentLoopedGPT's n_probe) were
    both found to settle on a roughly length-INDEPENDENT step count regardless
    of actual difficulty -- exactly the failure this class is designed to
    avoid, since with a WINDOWED attention span, a longer operand genuinely
    needs proportionally more hops for a carry to cross it. Tying step count
    directly to length (rather than hoping the model learns to do so) isolates
    the one variable actually being tested: does locality-forced propagation,
    given enough steps to complete it, produce a general algorithm.

    Requires one-problem-per-sequence, right-padded input (pad_id) --
    prompt_len must be well-defined per batch, same requirement as
    PerExampleLoopedGPT/AdaptiveLatentLoopedGPT.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_id is not None, "LocalLoopedGPT requires config.pad_id (one-problem-per-sequence padded data)"
        assert config.attn_window is not None, "LocalLoopedGPT requires config.attn_window (this is the whole point)"
        self.config = config

        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config)]),  # single shared block, applied in a loop
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.pos_encoding == 'learned':
            transformer_modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self._last_n_loops = None

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_loop_n(self):
        """Most recent forward's deterministic loop count (for logging)."""
        return self._last_n_loops

    def forward(self, idx, targets=None, prompt_len=None):
        assert prompt_len is not None, "LocalLoopedGPT requires prompt_len (same for every row in the batch)"
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        if self.config.pos_encoding == 'learned':
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(tok_emb)

        n_loops = min(self.config.max_loop_iters, max(1, math.ceil(self.config.loop_steps_per_token * prompt_len)))
        self._last_n_loops = n_loops

        block = self.transformer.h[0]
        for _ in range(n_loops):
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
            if hasattr(block.attn, 'window_mask'):
                block.attn.window_mask = block.attn.window_mask[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ MFU estimate using the actual (deterministic, length-dependent) loop count. """
        N = self.get_num_params()
        cfg = self.config
        L = self._last_n_loops if self._last_n_loops is not None else 1
        H, Q, T = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu


class RefineGPT(nn.Module):
    """
    Tests the user's iterative window-refinement idea: instead of one-shot
    left-to-right autoregressive generation, the answer is produced by a
    sliding window (width W, stride 1) that revisits each position multiple
    times before locking it in, using BIDIRECTIONAL attention so a position
    being revised can see later positions' current draft (something a causal
    mask forbids by construction). get_batch_refine (train.py) simulates one
    random pass of this schedule per training example: everything before the
    window is ground truth (already "committed"), the window itself plus
    everything after it is MASK, and the loss targets are exactly the
    window's true values. Structurally the same family as BERT/Mask-Predict
    -style masked-LM training (Ghazvininejad et al. 2019's non-autoregressive
    translation refinement is the closest published analog), applied here to
    the addition answer span specifically.

    Deliberately a PLAIN (non-shared, non-looped) transformer stack -- this
    is meant to isolate whether iterative revision itself helps, as a
    separate question from this project's depth-recurrence/latent-loop axis.
    Forces config.bidirectional=True; every attention op sees the entire
    sequence with no masking at all (prompt AND answer region alike -- the
    prompt is fully known upfront, no reason to hide any of it either).

    Generation (see eval_refine.py) maintains a running canvas initialized to
    all MASK in the answer region, then for each window position
    t = 0..L-1: feeds (prompt + canvas) through the model, reads off
    predictions for window [t, t+W-1], writes them into the canvas
    (revising the draft for t+1..t+W-1, filling in t+W-1 for the first
    time), and locks position t (never touched again) once the window moves
    past it.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.bidirectional, "RefineGPT requires config.bidirectional=True (this is the whole point)"
        assert config.attn_window is None, "bidirectional and attn_window are mutually exclusive"
        self.config = config

        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # plain stack, NOT shared
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.pos_encoding == 'learned':
            transformer_modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        if self.config.pos_encoding == 'learned':
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(tok_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # unlike causal models, inference needs logits at arbitrary window
        # positions (not just the last one), so always compute the full set
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ Standard fixed-depth estimate -- plain stack, no shared/looped block. """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu


class PositionCoupledGPT(nn.Module):
    """
    Plain classical transformer (single multi-layer stack, full causal
    attention, no looping/locality/bidirectional tricks -- identical to the
    base GPT class) implementing position coupling (Cho et al. 2024,
    "Position Coupling: Improving Length Generalization of Arithmetic
    Transformers Using Task Structure", NeurIPS 2024): instead of raw
    sequential position ids, digits of the SAME PLACE VALUE across both
    operands and the answer get the SAME RoPE position id, and the answer is
    zero-padded (to one digit longer than the operands, for carry overflow)
    and REVERSED (least-significant digit first).

    Directly targets the specific failure this whole project kept finding:
    "a+b=" doesn't align same-place-value digits at nearby raw token
    positions, so a plain model (or even a LOCALITY-constrained one, see
    LocalLoopedGPT) has no easy way to know which operand digits combine
    with which -- position coupling makes that alignment explicit via
    position ids, which RoPE (already relative-offset-based) can exploit
    directly. Per the paper: models trained on <=30 digits generalize to
    200-digit addition (>95% exact match) with this scheme; this is the
    literature's strongest demonstrated result on this exact problem, used
    here after this project's from-scratch/self-discovery attempts (loops,
    latent CoT, drilling, locality, masked refinement) all failed to
    produce genuine length generalization.

    get_batch_coupled (train.py) computes position ids on the fly from each
    padded row's structure (locations of '+'/'=' and the fixed per-bucket
    operand length), rather than storing them as separate data -- fully
    determined by the row's own content, no extra prep needed.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pos_encoding == 'rope', "PositionCoupledGPT requires RoPE (coupled ids are gathered into the RoPE table, not a sequential wpe lookup)"
        self.config = config

        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, position_ids=None):
        assert position_ids is not None, "PositionCoupledGPT requires explicit position_ids (see get_batch_coupled)"
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        x = self.transformer.drop(self.transformer.wte(idx))
        for block in self.transformer.h:
            x = block(x, position_ids=position_ids)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu


class ACTLoopedGPT(nn.Module):
    """
    Stage 2.1: same single shared block + loop-index embedding as LoopedGPT,
    but the loop count is no longer one global learned scalar -- it's decided
    per TOKEN POSITION via Adaptive Computation Time (Graves 2016).

    At each ponder step, a small linear+sigmoid halting unit reads each
    position's current hidden state and outputs a halting probability. Once
    a position's cumulative halting probability crosses 1 - act_epsilon, it
    "halts": its representation freezes (but keeps participating in later
    steps' attention for positions still running) and stops accumulating
    further weight. The final per-position representation is a weighted
    average of every intermediate state it passed through, weighted by its
    halting probabilities (all steps get weight = that step's own p, except
    the final/halting step, which gets weight = the leftover remainder, so
    weights sum to exactly 1) -- this is what makes the discrete-looking
    "how many steps did this token take" decision fully differentiable.

    A small ponder-cost penalty (num steps + remainder) is added to the loss
    to stop the model from just always taking the max steps.

    Per-token (not per-example) because training examples are packed many-
    per-sequence (see data/addition/prepare.py) -- there's no clean "one
    forward pass = one problem" boundary to hang a per-example decision on.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config)]),  # single shared block, applied in a loop
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.pos_encoding == 'learned':
            transformer_modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.loop_embed = nn.Embedding(config.max_loop_iters, config.n_embd) if config.loop_index_embed else None

        # halting unit: reads a position's hidden state, outputs P(halt now)
        self.halt_probe = nn.Linear(config.n_embd, 1)

        self.apply(self._init_weights)
        # bias the halting unit toward LOW initial halting probability (favor more
        # pondering at init) so training starts with room to learn when to stop,
        # rather than risking an immediate always-halt-at-step-1 collapse
        torch.nn.init.constant_(self.halt_probe.bias, -1.0)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * 6))  # matches other classes' n_init=6 convention

        self._last_avg_ponder_steps = None
        self._last_ponder_cost = None
        self._last_n_updates = None

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_loop_n(self):
        """Average ponder steps taken in the most recent forward pass (for logging)."""
        return self._last_avg_ponder_steps

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        if self.config.pos_encoding == 'learned':
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(tok_emb)

        block = self.transformer.h[0]
        max_steps = self.config.max_loop_iters
        epsilon = self.config.act_epsilon

        halting_cum = torch.zeros(b, t, device=device)      # sum of p_n for still-running positions
        still_running = torch.ones(b, t, dtype=torch.bool, device=device)
        n_updates = torch.zeros(b, t, device=device)        # ponder steps taken so far
        remainders = torch.zeros(b, t, device=device)       # remainder weight assigned at halting step
        output_accum = torch.zeros_like(x)
        step_weights = []  # mean weight contributed by each step, averaged over all (b,t) positions

        halt_source = self.config.act_halt_source

        for i in range(max_steps):
            idx_emb = self.loop_embed.weight[i] if self.loop_embed is not None else 0
            x_in = x + idx_emb
            if halt_source == 'post_attn':
                # halt-probe reads the state right after attention (before mlp) --
                # the per-step output used everywhere else is still the full
                # post-block (attn+mlp) state, only the halting SIGNAL's source changes
                x_mid = x_in + block.attn(block.ln_1(x_in))
                p = torch.sigmoid(self.halt_probe(x_mid)).squeeze(-1)  # (b, t)
                x_new = x_mid + block.mlp(block.ln_2(x_mid))
            else:
                x_new = block(x_in)
                p = torch.sigmoid(self.halt_probe(x_new)).squeeze(-1)  # (b, t)
            cum_after = halting_cum + p
            is_last_step = (i == max_steps - 1)
            halt_now = still_running & ((cum_after >= 1.0 - epsilon) | is_last_step)

            # weight for this step: remainder if halting now, else the raw p_n;
            # positions that already halted in an earlier step get weight 0
            weight = torch.where(halt_now, 1.0 - halting_cum, p) * still_running.float()

            output_accum = output_accum + weight.unsqueeze(-1) * x_new
            step_weights.append(weight.mean().item())
            remainders = torch.where(halt_now, weight, remainders)
            n_updates = n_updates + still_running.float()
            halting_cum = torch.where(halt_now, halting_cum, cum_after)  # frozen positions' cum no longer matters

            # only still-running positions' representations update; halted
            # positions freeze so later steps' attention sees a stable state
            x = torch.where(still_running.unsqueeze(-1), x_new, x)

            still_running = still_running & ~halt_now
            if not still_running.any():
                break

        self._last_step_weights = step_weights  # list, index i = mean weight of ponder step i across all positions
        self._last_avg_ponder_steps = n_updates.mean().item()
        self._last_ponder_cost = (n_updates + remainders).mean()
        self._last_n_updates = n_updates.detach()  # full (b, t) per-position ponder-step counts, for halting-vs-difficulty analysis

        x = self.transformer.ln_f(output_accum)

        if targets is not None:
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = ce_loss + self.config.ponder_cost_weight * self._last_ponder_cost
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ MFU estimate using the *average* ponder steps from the last forward pass. """
        N = self.get_num_params()
        cfg = self.config
        L = max(1, round(self._last_avg_ponder_steps)) if self._last_avg_ponder_steps else cfg.max_loop_iters
        H, Q, T = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class PerExampleLoopedGPT(nn.Module):
    """
    Global-per-problem adaptive depth: one shared block (attn+FFN), same
    width as LoopedGPT, applied floor(n)+1 times exactly like Stage 2 --
    except n is no longer a single global learned scalar. It's predicted
    per EXAMPLE (one n per training row, not per token) by:
      1. running the block once, unconditionally, on every position (a
         mandatory "look around" pass so the difficulty estimate is based
         on context gathered via attention, not just raw token identity);
      2. mean-pooling that pass's output over the example's real (non-pad)
         positions into a single vector;
      3. a linear+softplus probe turning that vector into one n.
    From there it's exactly Stage 2's floor(n)+1 / frac(n) blend, just
    gated per batch element instead of applying to the whole batch
    uniformly -- so unlike ACTLoopedGPT, the final representation is only
    ever a blend of two states (never a multi-step weighted average), and
    unlike Stage 2, different examples in the same batch can genuinely
    take different numbers of loops.

    Requires one-problem-per-sequence, right-padded input (see
    data/addition_padded/prepare.py) -- a per-example decision isn't
    meaningful on nanoGPT's default packed-stream format, where one
    sequence holds several unrelated problems.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_id is not None, "PerExampleLoopedGPT requires config.pad_id (one-problem-per-sequence padded data)"
        self.config = config

        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config)]),  # single shared block, applied in a loop
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.pos_encoding == 'learned':
            transformer_modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.loop_embed = nn.Embedding(config.max_loop_iters, config.n_embd) if config.loop_index_embed else None
        self.n_probe = nn.Linear(config.n_embd, 1)  # predicts this example's (additional, beyond the look pass) loop budget

        self.apply(self._init_weights)
        # bias n_probe toward Stage 2's n_init convention so training starts near a familiar depth
        n_init = torch.tensor(float(config.n_init))
        torch.nn.init.zeros_(self.n_probe.weight)
        torch.nn.init.constant_(self.n_probe.bias, torch.log(torch.expm1(n_init)).item())
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * 6))

        self._last_avg_n = None

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_loop_n(self):
        """Average (1 look pass + n) across the most recent batch, for logging."""
        return self._last_avg_n

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        if self.config.pos_encoding == 'learned':
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(tok_emb)

        block = self.transformer.h[0]
        max_extra = self.config.max_loop_iters - 1  # budget left after the mandatory look pass

        # step 0: mandatory shared look pass, every position, so the difficulty
        # estimate below is based on attended-to context, not raw embeddings alone
        idx_emb0 = self.loop_embed.weight[0] if self.loop_embed is not None else 0
        x = block(x + idx_emb0)

        pad_mask = (idx != self.config.pad_id).float().unsqueeze(-1)  # (b, t, 1)
        pooled = (x * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1.0)  # (b, C)
        n = F.softplus(self.n_probe(pooled)).squeeze(-1)  # (b,)
        n_floor = torch.floor(n).long().clamp(0, max_extra - 1)  # leave room for the final blended step
        # n itself is unbounded (softplus), but n_floor saturates at max_extra-1 -- without
        # clamping frac too, examples with n > max_extra would get frac > 1, extrapolating
        # past x_new instead of interpolating between x and x_new
        frac = (n - n_floor.float()).clamp(max=1.0)

        self._last_avg_n = (1 + n).mean().item()  # +1 for the mandatory look pass; unclamped, so this can read above what the model actually executes once capped
        self._last_n_per_example = (1 + n).detach()  # per-example version of the above, for analysis (e.g. loop count vs. digit count)

        still_running = torch.ones(b, dtype=torch.bool, device=device)
        for i in range(max_extra):
            idx_emb = self.loop_embed.weight[i + 1] if self.loop_embed is not None else 0
            x_new = block(x + idx_emb)

            is_last_i = (i == max_extra - 1)
            is_blend_step = still_running & ((n_floor == i) | is_last_i)
            is_full_step = still_running & ~is_blend_step

            blend_w = frac.view(b, 1, 1)
            x_if_blend = (1 - blend_w) * x + blend_w * x_new
            x = torch.where(is_blend_step.view(b, 1, 1), x_if_blend,
                             torch.where(is_full_step.view(b, 1, 1), x_new, x))

            still_running = still_running & ~is_blend_step
            if not still_running.any():
                break

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # ponder-cost penalty: a direct, always-differentiable function of the
            # raw continuous n (not mediated by the saturating floor/frac clamp
            # above) -- without this, once n exceeds max_extra, frac pins at 1.0
            # and d(frac)/dn becomes exactly 0, so the task loss has literally no
            # gradient left to pull n back down (grow-only dead zone). This term
            # supplies that missing restoring force, and discourages runaway
            # loop counts the way ACTLoopedGPT's ponder cost already does.
            ponder_cost = (1 + n).mean()  # +1 for the mandatory look pass, matching _last_avg_n's convention
            loss = ce_loss + self.config.ponder_cost_weight * ponder_cost
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L = max(1, round(self._last_avg_n)) if self._last_avg_n else cfg.max_loop_iters
        H, Q, T = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class TwoBlockGPT(nn.Module):
    """
    Stage 3 pass 1: block specialization without adaptive halting yet. Two
    DISTINCT blocks (own weights each, same width as the single-block
    stages) -- block A loops a FIXED number of times (config.n_loop_a, a
    hyperparameter, not learned), then block B runs once as a terminal
    "commit to an answer" readout. Isolates whether specializing into a
    "thinking" block and a "committing" block helps at all, before Stage 3
    pass 2 adds per-input adaptive halting on top of this same structure.

    Uses the original packed-stream data (data/addition), not the padded
    one-problem-per-sequence format -- there's no per-example decision
    here, n_loop_a is a fixed constant for every input, same as Stage 1.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            block_a=Block(config),
            block_b=Block(config),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.pos_encoding == 'learned':
            transformer_modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.loop_embed = nn.Embedding(config.max_loop_iters, config.n_embd) if config.loop_index_embed else None

        self.apply(self._init_weights)
        # scaled init for residual projections, per GPT-2 paper -- effective depth here
        # is n_loop_a applications of A plus 1 of B
        effective_depth = config.n_loop_a + 1
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * effective_depth))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_loop_n(self):
        """Fixed loop count for A (+1 for B), for logging consistency with the other classes."""
        return float(self.config.n_loop_a + 1)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        tok_emb = self.transformer.wte(idx)
        if self.config.pos_encoding == 'learned':
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
        else:
            x = self.transformer.drop(tok_emb)

        block_a = self.transformer.block_a
        for i in range(self.config.n_loop_a):
            idx_emb = self.loop_embed.weight[i] if self.loop_embed is not None else 0
            x = block_a(x + idx_emb)

        x = self.transformer.block_b(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in [self.transformer.block_a, self.transformer.block_b]:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L = cfg.n_loop_a + 1
        H, Q, T = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class LatentLoopedGPT(nn.Module):
    """
    Fixed-count latent chain-of-thought ("Coconut"-style, Hao et al. 2024).
    Unlike every other *LoopedGPT class here, this does NOT reapply one
    shared block to a fixed-length sequence in place -- it runs the FULL
    n_layer stack (same depth/width as Stage 1, not a single shared block)
    repeatedly over a GROWING sequence. Each step's final hidden state
    (post-last-block, pre-ln_f) at the last position is used directly as
    the next position's input embedding -- never decoded through lm_head,
    never touching the vocabulary. After config.n_latent_steps such
    appended positions, the model switches back to real token embeddings
    for the actual answer, which is scored normally.

    This is a genuinely different axis from LoopedGPT/ACTLoopedGPT/
    PerExampleLoopedGPT's depth-recurrence: those reprocess a FIXED set of
    positions with more nonlinear depth (attention among the same
    positions, over and over); this one EXTENDS the sequence, so each new
    latent step can attend back over a growing chain of prior latent
    states -- the sequence-length axis, not the depth axis, which is the
    one relative position encodings (RoPE) are actually built to
    generalize over. See project discussion for the full reasoning.

    n_latent_steps is a fixed hyperparameter here (no adaptive halting yet
    -- that's the natural next step once this base mechanism is validated).

    Also carries an auxiliary countdown_head (see config.countdown_max/
    countdown_loss_weight): predicts tokens-remaining-until-'\n' at every
    answer position, directly supervised from ground truth (trivial to
    compute, no new data needed). Motivated by the OOD failure mode this
    project keeps finding: the model reliably computes correct leading
    digits (confirmed via logit-lens on the latents) but stops one digit
    short past its trained length range, i.e. it looks like the STOP
    decision is the thing that failed to generalize, not digit computation
    -- and the main loss only ever gives one position (wherever '\n'
    happens to land) any signal about that decision. This gives a dense,
    dedicated signal for tracking "how much further to go" instead.

    Batching constraint: requires every row in a batch to share the same
    real PROMPT length (train.py's get_batch groups rows by prompt length
    for this model_type specifically). This is what lets the latent-
    insertion column be a single python int per batch instead of a
    per-row offset -- avoiding the need for a left-padding-aware attention
    mask, which nothing in this codebase currently implements (the
    existing padded format's causal masking only correctly ignores padding
    when padding comes AFTER all real content, i.e. right-padding).
    forward()'s targets must be pre-shifted/masked by the caller to match
    the LATENT-EXTENDED internal sequence layout (see train.py get_batch):
    positions covering the prompt and the K latent steps are ignore_index,
    positions covering the real answer follow the usual masking convention.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_id is not None, "LatentLoopedGPT requires config.pad_id (one-problem-per-sequence padded data)"
        self.config = config

        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.pos_encoding == 'learned':
            transformer_modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        # auxiliary head: predicts tokens-remaining-until-'\n' at every answer
        # position, classified into {0..countdown_max} (clamped) -- dense,
        # directly-supervised signal for tracking "how much further to go",
        # instead of the main loss's only-ever-one-position "is this the '\n'
        # yet" judgment. See project discussion: the OOD failure looks like a
        # mislearned stop condition (stops around the longest length actually
        # seen), not a digit-computation failure -- this targets that directly.
        self.countdown_head = nn.Linear(config.n_embd, config.countdown_max + 1)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed_tokens(self, idx, pos_start):
        tok_emb = self.transformer.wte(idx)
        if self.config.pos_encoding == 'learned':
            t = idx.size(1)
            pos = torch.arange(pos_start, pos_start + t, dtype=torch.long, device=idx.device)
            tok_emb = tok_emb + self.transformer.wpe(pos)
        return tok_emb

    def _embed_latent(self, vec, pos):
        # vec: (b, 1, C) raw hidden state, no wte lookup -- only 'learned'
        # pos_encoding needs an explicit position vector added here; RoPE
        # applies its rotation inside attention purely from position index,
        # so it needs no equivalent treatment at the embedding stage
        if self.config.pos_encoding == 'learned':
            b = vec.size(0)
            pos_idx = torch.full((b, 1), pos, dtype=torch.long, device=vec.device)
            vec = vec + self.transformer.wpe(pos_idx)
        return vec

    def _run_stack(self, emb):
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        return x  # pre-ln_f -- ln_f only applied once, right before lm_head, at the very end

    def forward(self, idx, targets=None, prompt_len=None, countdown_targets=None):
        assert prompt_len is not None, "LatentLoopedGPT requires prompt_len (same for every row in the batch)"
        device = idx.device
        b, t = idx.size()
        K = self.config.n_latent_steps
        assert t + K <= self.config.block_size, \
            f"Cannot forward sequence of length {t} + {K} latent steps, block size is only {self.config.block_size}"

        prompt_idx = idx[:, :prompt_len]
        emb = self._embed_tokens(prompt_idx, pos_start=0)  # (b, prompt_len, C)

        # K sequential latent-recycling steps: each runs the full stack over
        # the CURRENT sequence and appends its last position's hidden state
        # as the next position's raw input embedding -- see class docstring
        latents = []  # (b, C) pre-position-embed hidden state at each step, for analysis (e.g. logit-lens probing)
        for step in range(K):
            h = self._run_stack(emb)
            raw_latent = h[:, -1, :]  # (b, C), before any 'learned' pos_encoding is added
            latents.append(raw_latent.detach())
            last = self._embed_latent(raw_latent.unsqueeze(1), pos=prompt_len + step)
            emb = torch.cat([emb, last], dim=1)
        self._last_latents = torch.stack(latents, dim=1) if latents else None  # (b, K, C)

        # switch back to real token embeddings for the answer (and any
        # trailing padding), appended after the K latents
        answer_idx = idx[:, prompt_len:]
        answer_emb = self._embed_tokens(answer_idx, pos_start=prompt_len + K)
        emb = torch.cat([emb, answer_emb], dim=1)  # (b, t + K, C)

        h = self._run_stack(emb)
        h = self.transformer.ln_f(h)

        if targets is not None:
            logits = self.lm_head(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            if countdown_targets is not None:
                countdown_logits = self.countdown_head(h)
                countdown_loss = F.cross_entropy(countdown_logits.view(-1, countdown_logits.size(-1)),
                                                  countdown_targets.view(-1), ignore_index=-1)
                loss = loss + self.config.countdown_loss_weight * countdown_loss
        else:
            logits = self.lm_head(h[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # NOTE: undercounts true FLOPs -- doesn't account for the K extra
        # full-stack passes during the latent-recycling phase (K+1 total
        # passes per forward, growing in length), only the final full-length
        # pass. Diagnostic printout only, doesn't affect training.
        N = self.get_num_params()
        cfg = self.config
        H, Q, T = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * cfg.n_layer * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu


class AdaptiveLatentLoopedGPT(nn.Module):
    """
    LatentLoopedGPT + per-example adaptive step count, via the exact
    floor(n)+frac blend PerExampleLoopedGPT (Stage 2.1b) uses for
    depth-recurrence -- ported to the sequence-extension axis instead.

    Mechanically: config.n_latent_steps is now a CAP (not a fixed count).
    Step 0 is a mandatory, unconditional latent pass (same rationale as
    PerExampleLoopedGPT's mandatory look pass: the difficulty/step-count
    estimate should be based on attended-to context, not raw embeddings).
    n_probe reads that first latent and predicts, per example, how many
    ADDITIONAL steps (n, softplus-parameterized) are needed. For each
    subsequent step i in range(cap-1): every example in the batch still
    computes a fresh latent (same "whole batch recomputes every step,
    already-decided rows' compute is wasted but zero-gradient via where()"
    tradeoff PerExampleLoopedGPT already has) -- but only the example's OWN
    n_floor-th step gets used as a genuine soft blend between its last
    committed latent and the new one (frac-weighted); steps after that
    just re-append a frozen copy of the already-committed latent, so every
    example ends up with the SAME total sequence length (cap+1 positions
    including latents) regardless of its own effective step count --
    avoids needing ragged/per-example sequence lengths in a batched tensor.
    ponder_cost_weight penalizes the raw (unclamped) n directly, same
    restoring-force rationale as PerExampleLoopedGPT/ACTLoopedGPT.

    Keeps LatentLoopedGPT's countdown_head unchanged (dense tokens-
    remaining-until-'\n' supervision) -- these are orthogonal mechanisms,
    both still active here.

    latent_ln: pre-LN transformers never rescale the residual stream itself
    (LayerNorm only normalizes copies fed into attn/mlp -- see Block), so it
    grows in norm with depth by construction; empirically this ran up ~6x
    across the K latent steps (eval_latent_readout_alignment.py). Re-feeding
    that raw, growing-scale vector as if it were a normal token embedding
    (which wte produces at a roughly fixed scale) is a plausible source of
    drift the model has to compensate for rather than a deliberate signal.
    latent_ln normalizes ONLY the copy hand off to _embed_latent for
    re-insertion into the sequence -- cur_latent itself stays raw throughout
    (blending math, n_probe, and _last_latents storage all unchanged), so
    this is a minimal, surgical change, not a redesign.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pad_id is not None, "AdaptiveLatentLoopedGPT requires config.pad_id (one-problem-per-sequence padded data)"
        self.config = config

        transformer_modules = dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        )
        if config.pos_encoding == 'learned':
            transformer_modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(transformer_modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.countdown_head = nn.Linear(config.n_embd, config.countdown_max + 1)
        self.n_probe = nn.Linear(config.n_embd, 1)  # predicts additional steps beyond the mandatory first one
        self.latent_ln = LayerNorm(config.n_embd, bias=config.bias)  # normalizes the latent right before it's re-fed as the next position's embedding -- see class docstring

        self.apply(self._init_weights)
        n_init = torch.tensor(float(config.n_init))
        torch.nn.init.zeros_(self.n_probe.weight)
        torch.nn.init.constant_(self.n_probe.bias, torch.log(torch.expm1(n_init)).item())
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self._last_avg_n = None

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, 'wpe'):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_loop_n(self):
        """Average (1 mandatory pass + n) across the most recent batch, for logging."""
        return self._last_avg_n

    def _embed_tokens(self, idx, pos_start):
        tok_emb = self.transformer.wte(idx)
        if self.config.pos_encoding == 'learned':
            t = idx.size(1)
            pos = torch.arange(pos_start, pos_start + t, dtype=torch.long, device=idx.device)
            tok_emb = tok_emb + self.transformer.wpe(pos)
        return tok_emb

    def _embed_latent(self, vec, pos):
        if self.config.pos_encoding == 'learned':
            b = vec.size(0)
            pos_idx = torch.full((b, 1), pos, dtype=torch.long, device=vec.device)
            vec = vec + self.transformer.wpe(pos_idx)
        return vec

    def _run_stack(self, emb):
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        return x

    def forward(self, idx, targets=None, prompt_len=None, countdown_targets=None):
        assert prompt_len is not None, "AdaptiveLatentLoopedGPT requires prompt_len (same for every row in the batch)"
        device = idx.device
        b, t = idx.size()
        cap = self.config.n_latent_steps  # total latent positions, at most
        max_extra = cap - 1  # budget left after the mandatory first pass
        assert t + cap <= self.config.block_size, \
            f"Cannot forward sequence of length {t} + {cap} latent steps, block size is only {self.config.block_size}"

        prompt_idx = idx[:, :prompt_len]
        emb = self._embed_tokens(prompt_idx, pos_start=0)

        # step 0: mandatory, unconditional latent pass -- same rationale as
        # PerExampleLoopedGPT's mandatory look pass (difficulty estimate
        # grounded in attended-to context, not raw embeddings)
        h = self._run_stack(emb)
        cur_latent = h[:, -1, :]  # (b, C)
        latents = [cur_latent.detach()]
        emb = torch.cat([emb, self._embed_latent(self.latent_ln(cur_latent).unsqueeze(1), pos=prompt_len)], dim=1)

        # clamp the pre-activation floor before softplus: softplus's gradient
        # (sigmoid of its input) vanishes as the input goes very negative,
        # same saturating mechanism as the high-n ratchet this class's
        # ponder-cost penalty already fixes, just on the low end. Without a
        # floor, an ordinary large gradient step (e.g. from ponder_cost's
        # constant downward pressure combined with a too-high LR) can push
        # the pre-activation deep enough that NEITHER the task loss NOR
        # ponder_cost has any gradient left to pull it back -- a permanent
        # trap at n~0, observed directly during training (loop_n pinned at
        # exactly 1.000 for many consecutive steps). -3 keeps
        # sigmoid(-3)~0.047, a clearly healthy gradient, while still letting
        # n reach softplus(-3)~0.049 -- effectively zero extra steps for
        # genuinely easy examples -- so this doesn't meaningfully restrict
        # the achievable range, only removes the un-recoverable dead zone.
        n_preact = self.n_probe(cur_latent).squeeze(-1).clamp(min=-3.0)
        n = F.softplus(n_preact)  # (b,)
        n_floor = torch.floor(n).long().clamp(0, max_extra - 1)
        frac = (n - n_floor.float()).clamp(max=1.0)
        self._last_avg_n = (1 + n).mean().item()
        self._last_n_per_example = (1 + n).detach()  # per-example version, for analysis (e.g. loop count vs. digit count)

        still_running = torch.ones(b, dtype=torch.bool, device=device)
        for i in range(max_extra):
            h = self._run_stack(emb)
            new_latent = h[:, -1, :]

            is_last_i = (i == max_extra - 1)
            is_blend_step = still_running & ((n_floor == i) | is_last_i)
            is_full_step = still_running & ~is_blend_step  # still deciding, not yet at its own commit point

            blend_w = frac.view(b, 1)
            blended = (1 - blend_w) * cur_latent + blend_w * new_latent
            cur_latent = torch.where(is_blend_step.view(b, 1), blended,
                                      torch.where(is_full_step.view(b, 1), new_latent, cur_latent))
            latents.append(cur_latent.detach())
            emb = torch.cat([emb, self._embed_latent(self.latent_ln(cur_latent).unsqueeze(1), pos=prompt_len + 1 + i)], dim=1)

            still_running = still_running & ~is_blend_step

        self._last_latents = torch.stack(latents, dim=1)  # (b, cap, C)

        answer_idx = idx[:, prompt_len:]
        answer_emb = self._embed_tokens(answer_idx, pos_start=prompt_len + cap)
        emb = torch.cat([emb, answer_emb], dim=1)

        h = self._run_stack(emb)
        h = self.transformer.ln_f(h)

        if targets is not None:
            logits = self.lm_head(h)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            ponder_cost = (1 + n).mean()
            loss = ce_loss + self.config.ponder_cost_weight * ponder_cost
            if countdown_targets is not None:
                countdown_logits = self.countdown_head(h)
                countdown_loss = F.cross_entropy(countdown_logits.view(-1, countdown_logits.size(-1)),
                                                  countdown_targets.view(-1), ignore_index=-1)
                loss = loss + self.config.countdown_loss_weight * countdown_loss
        else:
            logits = self.lm_head(h[:, [-1], :])
            loss = None
        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # NOTE: undercounts true FLOPs, same caveat as LatentLoopedGPT's.
        N = self.get_num_params()
        cfg = self.config
        H, Q, T = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * cfg.n_layer * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu


class ComposedGPT(nn.Module):
    """
    Isolates the "represent each operand as a single fixed-size token"
    hypothesis (user proposal): a small bidirectional operand ENCODER
    (shared weights for operand a and b) reduces a variable-length digit
    string down to ONE vector, read off a prepended learnable summary
    token. A tiny COMBINER transformer mixes the two operand vectors --
    it always sees exactly 4 "tokens" (vecA, '+', vecB, '='), REGARDLESS
    of operand length, so it has zero length-generalization exposure by
    construction: whatever this stage does or doesn't learn can't be
    blamed on variable sequence length or a moving relative-offset target
    (the failure mode diagnosed for plain sequential RoPE). Its output at
    the '=' slot becomes a single memory vector, prepended as the first
    input to a standard causal DECODER, which autoregressively generates
    the answer digit-by-digit exactly like plain GPT.

    Purpose: find out WHERE a hierarchical/compressive architecture
    actually breaks on this task -- if it fails to extrapolate, is it
    because the encoder can't compress an unfamiliar-length number into a
    faithful fixed-size vector, or because the decoder can't read an
    exact value back out of one -- independent of any cross-operand
    place-value alignment machinery (there is none here).

    Batching constraint: operand a and operand b must each have a FIXED
    digit-count within one batch (the encoder processes a (batch, len)
    tensor), so training uses on-the-fly generation with one (len_a,
    len_b) pair per batch instead of the usual pre-baked, mixed-length
    .bin files -- see train.py's get_batch_composed.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.pos_encoding == 'rope', "ComposedGPT requires RoPE (encoder and decoder each need to extrapolate to unseen lengths)"
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.summary_token = nn.Parameter(torch.randn(config.n_embd) * 0.02)

        enc_config = dataclasses.replace(config, n_layer=config.enc_layers, bidirectional=True, attn_window=None)
        comb_config = dataclasses.replace(config, n_layer=config.combiner_layers, bidirectional=False, attn_window=None)

        self.encoder = nn.ModuleList([Block(enc_config) for _ in range(config.enc_layers)])
        self.enc_ln = LayerNorm(config.n_embd, bias=config.bias)
        self.combiner = nn.ModuleList([Block(comb_config) for _ in range(config.combiner_layers)])
        self.comb_ln = LayerNorm(config.n_embd, bias=config.bias)
        self.decoder = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.dec_ln = LayerNorm(config.n_embd, bias=config.bias)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                n_total_layers = config.enc_layers + config.combiner_layers + config.n_layer
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_total_layers))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_operand(self, digit_ids):
        """digit_ids: (B, L) token ids for one operand's digits (fixed L
        across the batch), most-significant digit first. Returns
        (B, n_embd): the summary token's final hidden state after
        BIDIRECTIONAL self-attention over [summary, d_1, ..., d_L] --
        every digit can attend to every other digit in the same operand,
        and the summary token pools all of them into one vector.

        Position ids are PLACE-VALUE-anchored, not sequential: the summary
        token sits at id 0, and the digit at sequence index k (0 = most
        significant) gets id (L - k), so the LEAST significant digit is
        ALWAYS at relative offset 1 from the summary token, the tens digit
        always at offset 2, etc., regardless of the operand's total length
        L. Mirrors PositionCoupledGPT's core mechanism (make the offset the
        model must learn to attend across independent of length) applied
        inside the encoder -- see project discussion on why the plain
        sequential version (offset from summary grows with L) can't
        extrapolate to unseen operand lengths."""
        B, L = digit_ids.size()
        tok = self.wte(digit_ids)
        summary = self.summary_token.view(1, 1, -1).expand(B, 1, -1)
        x = torch.cat([summary, tok], dim=1)
        x = self.drop(x)
        position_ids = None
        if self.config.encoder_position_scheme == 'place_value':
            device = digit_ids.device
            digit_pos = torch.arange(L, 0, -1, device=device)  # [L, L-1, ..., 1]
            position_ids = torch.cat([torch.zeros(1, dtype=torch.long, device=device), digit_pos])
            position_ids = position_ids.unsqueeze(0).expand(B, -1)
        for block in self.encoder:
            x = block(x, position_ids=position_ids)
        x = self.enc_ln(x)
        return x[:, 0, :]

    def combine(self, vec_a, vec_b, plus_id, eq_id):
        """vec_a, vec_b: (B, n_embd). Builds the fixed 4-slot sequence
        [vecA, '+', vecB, '='] and returns the '=' slot's final hidden
        state -- this stage's sequence length is ALWAYS 4, independent of
        either operand's digit count."""
        B = vec_a.size(0)
        plus_emb = self.wte(torch.tensor([plus_id], device=vec_a.device)).view(1, -1).expand(B, -1)
        eq_emb = self.wte(torch.tensor([eq_id], device=vec_a.device)).view(1, -1).expand(B, -1)
        x = torch.stack([vec_a, plus_emb, vec_b, eq_emb], dim=1)
        x = self.drop(x)
        for block in self.combiner:
            x = block(x)
        x = self.comb_ln(x)
        return x[:, -1, :]

    def decode_from_vector(self, memory_vec, dec_x, targets=None):
        """memory_vec: (B, n_embd) -- fed as the decoder's position-0
        input (a continuous embedding, not a vocab lookup). dec_x: (B, L)
        -- teacher-forced digit ids (NOT including the memory slot or the
        trailing '\\n'). targets: (B, L + 1) -- [d_1, ..., d_L, '\\n']
        (or -1 to ignore). Shared by the main sum-decode path (memory_vec
        = combine(vec_a, vec_b)) and the auxiliary reconstruction path
        (memory_vec = vec_a or vec_b directly, bypassing the combiner)."""
        dec_tok = self.wte(dec_x)
        x = torch.cat([memory_vec.unsqueeze(1), dec_tok], dim=1)
        x = self.drop(x)
        for block in self.decoder:
            x = block(x)
        x = self.dec_ln(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def forward(self, a_ids, b_ids, dec_x, targets=None, plus_id=None, eq_id=None,
                a_targets=None, b_targets=None, sum_ids=None):
        """a_ids: (B, L_a), b_ids: (B, L_b) -- fixed-length digit ids per
        operand, ALSO reused directly as the teacher-forced decoder input
        for the auxiliary reconstruction losses (a_ids/b_ids already ARE
        the operand's own digit sequence). dec_x/targets: the main
        sum-prediction decode, as in decode_from_vector. a_targets/
        b_targets, if given: [a_ids, '\\n'] / [b_ids, '\\n'] -- enables
        the auxiliary "can the vector be decoded back into the exact
        operand it came from" loss, weighted by composed_recon_weight.
        sum_ids, if given: (B, L_sum) the TRUE sum's own digit ids,
        LEFT-zero-padded to a common width within the batch (zero-padding
        on the left preserves value, unlike the decoder's right-padding)
        -- enables the auxiliary "combine(vec_a, vec_b) == encode(true
        sum)" consistency loss, weighted by composed_consistency_weight,
        target detached so only the combiner moves toward the encoder's
        own coordinate system (see GPTConfig.composed_consistency_weight).
        Returns the main sum-task logits always; loss is the combined
        loss when targets is given, else None."""
        vec_a = self.encode_operand(a_ids)
        vec_b = self.encode_operand(b_ids)
        vec_c = self.combine(vec_a, vec_b, plus_id, eq_id)

        logits, sum_ce_loss = self.decode_from_vector(vec_c, dec_x, targets)
        loss = self.config.composed_sum_ce_weight * sum_ce_loss if sum_ce_loss is not None else None

        if targets is not None and a_targets is not None:
            _, loss_a = self.decode_from_vector(vec_a, a_ids, a_targets)
            _, loss_b = self.decode_from_vector(vec_b, b_ids, b_targets)
            loss = loss + self.config.composed_recon_weight * (loss_a + loss_b)

        if targets is not None and sum_ids is not None and self.config.composed_consistency_weight > 0:
            with torch.no_grad():
                vec_sum_target = self.encode_operand(sum_ids)
            consistency_loss = F.mse_loss(vec_c, vec_sum_target)
            loss = loss + self.config.composed_consistency_weight * consistency_loss

        return logits, loss

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        for stack in (self.encoder, self.combiner, self.decoder):
            for block in stack:
                if hasattr(block.attn, 'rope_cos'):
                    block.attn.rope_cos = block.attn.rope_cos[:block_size]
                    block.attn.rope_sin = block.attn.rope_sin[:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        H, Q, T = cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        n_total_layers = cfg.enc_layers + cfg.combiner_layers + cfg.n_layer
        flops_per_token = 6 * N + 12 * n_total_layers * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu


def _precompute_place_signals(n_places, dim, base=10000.0):
    """Fixed (non-learned) sinusoidal signal per integer place value, vectorized
    version of _sinusoidal_loop_embedding -- same role here: a smooth,
    continuous function of an integer index, defined identically for ANY
    index including ones never seen in training, unlike a free nn.Embedding
    lookup which has no defined row for an untrained index. Returns
    (n_places, dim)."""
    half = dim // 2
    inv_freq = torch.exp(-math.log(base) * torch.arange(0, half, dtype=torch.float32) / half)
    p = torch.arange(n_places, dtype=torch.float32)
    args = torch.outer(p, inv_freq)  # (n_places, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(n_places, 1)], dim=-1)
    return emb


def _precompute_unitary_roles(n_places, dim, seed=0):
    """HRR/VSA role vectors for circular-convolution binding: constructed so
    every frequency component of each role vector's FFT has magnitude
    EXACTLY 1, with only the phase depending on place p (phase_k(p) =
    p * base_freq[k], base_freq fixed and random per frequency bin). This
    "unitary" property is what makes binding norm-preserving and
    unbinding (circular correlation) an EXACT inverse for the matching
    place, for ANY p -- not just ones seen in training -- and makes
    cross-place interference behave as incoherent (graceful, ~1/sqrt(L))
    noise rather than a learned, unverified-for-untrained-places pattern.
    See probe_hrr_capacity.py for the validated capacity/degradation
    curve this buys (near-perfect through L~15-20 at dim=384, whole-
    sequence accuracy only compounds away at much longer L due to
    per-place errors multiplying, not due to interference collapsing).
    Returns (n_places, dim), real-valued."""
    n_freq = dim // 2 + 1  # matches torch.fft.rfft's output size for a length-`dim` real signal
    g = torch.Generator().manual_seed(seed)
    base_freq = torch.rand(n_freq, generator=g) * 2 * math.pi
    # DC (k=0) and, if dim is even, Nyquist (k=dim//2) bins must be REAL for
    # a real-valued time-domain signal -- a p-dependent phase there isn't
    # meaningful (there's no "positive/negative frequency pair" to rotate
    # against). Force those two bins' phase to 0 for every p; only the
    # interior bins actually encode place via phase.
    base_freq[0] = 0.0
    if dim % 2 == 0:
        base_freq[-1] = 0.0
    p = torch.arange(n_places, dtype=torch.float32)
    phase = torch.outer(p, base_freq)  # (n_places, n_freq)
    spectrum = torch.polar(torch.ones_like(phase), phase)  # magnitude 1 everywhere, phase = p*base_freq
    roles = torch.fft.irfft(spectrum, n=dim)  # (n_places, dim), real; ||roles[p]|| ~= 1 by Parseval given unit-magnitude spectrum -- do NOT rescale this (a bug here previously: rescaling to sqrt(dim) uniformly scaled every frequency's magnitude away from 1, breaking the exact norm-preservation property this whole construction exists for -- confirmed empirically, it compounded to a ~dim-factor error through a full bind+unbind round trip)
    return roles


def _circular_bind(f, r):
    """Circular convolution: binds filler f to role r. f, r: (..., C) real,
    broadcastable. Returns same shape. Norm-preserving in f when r is
    unitary (see _precompute_unitary_roles). .contiguous() on both inputs:
    CPU's MKL FFT backend errors ("Inconsistent configuration parameters")
    on the non-contiguous, zero-stride tensors .expand() produces --
    confirmed this crashes on CPU (harmlessly silent on CUDA's cuFFT,
    which tolerates it) -- so this is enforced once here rather than
    trusting every call site to remember it."""
    f, r = f.contiguous(), r.contiguous()
    C = f.size(-1)
    return torch.fft.irfft(torch.fft.rfft(f, n=C) * torch.fft.rfft(r, n=C), n=C)


def _circular_unbind(vec, r):
    """Circular correlation: the exact inverse of _circular_bind(f, r) w.r.t.
    r, when r is unitary -- i.e. _circular_unbind(_circular_bind(f, r), r) == f
    exactly. vec, r: (..., C) real, broadcastable. See _circular_bind's
    docstring re: the .contiguous() calls."""
    vec, r = vec.contiguous(), r.contiguous()
    C = vec.size(-1)
    return torch.fft.irfft(torch.fft.rfft(vec, n=C) * torch.conj(torch.fft.rfft(r, n=C)), n=C)


class AdditiveComposedGPT(nn.Module):
    """
    A ComposedGPT variant that replaces attention entirely in the encoder
    and decoder, keeping it only in the combiner. Motivated by a specific,
    directly-diagnosed failure mode of the attention-based encoder: its
    summary-token attention concentrates on a small "privileged" subset of
    digit positions (empirically, roughly every other position plus
    whichever is last), and extending operand length by one digit forces
    exactly one position to be demoted from privileged to ordinary --
    which is precisely where information gets lost (89% of wrong
    reconstructions at 11 digits drop exactly the second-to-least-significant
    digit, the position that gets demoted when an 11th digit appears). This
    is a structural consequence of softmax attention being COMPETITIVE (a
    fixed budget that must be renegotiated whenever the item count changes),
    not a training accident.

    Encoder: each digit's value and place are combined by a small, freely
    learned FFN into a contribution vector, and all contributions are
    SUMMED (not attention-averaged) into one vector. Summation has no fixed
    budget -- an 11th digit just adds one more term, nothing else has to
    lose share.

    Decoder: for every place value in parallel (no autoregression, no
    recurrence -- each place is an independent function of the memory
    vector and its own place value, by design: nothing about carries
    should live in the decoder), a small freely learned FFN produces
    logits over 11 classes (digit 0-9, or "blank" for places beyond the
    true answer length). Predicts config.max_answer_digits places every
    time, right-justified (the true digits always occupy the LAST L
    slots; blank fills the rest), so there is no length-prediction step
    separate from the per-place classification itself.

    What IS shared between encoder and decoder (both structurally
    necessary for length generalization, not learned, so nothing is lost
    by sharing): the digit-identity embedding table (10 rows, exactly the
    wte/lm_head tying pattern already used elsewhere in this file) and the
    fixed sinusoidal place-value signal (deterministic, zero parameters,
    used identically both directions so "place value p" means the exact
    same thing whether encoding or decoding). The actual combining FFN
    (encoder) and extracting FFN (decoder) are NOT tied -- each is free to
    discover its own best strategy, and the combiner is free to co-adapt
    with whatever packing the encoder settles on, same as before. The only
    hard, non-negotiable constraint is that aggregation is a sum, not an
    attention-weighted average -- everything else is learned.

    The combiner is unchanged in spirit from ComposedGPT: a small
    attention-based transformer over the fixed 4-slot sequence
    [vec_a, '+', vec_b, '=']. It never faces the competitive-budget problem
    the encoder had (always exactly 4 slots, never variable), so there's
    no diagnosed reason to remove attention there.
    """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # structural tokens only ('+', '=') for the combiner
        self.codec_type = config.codec_type
        self.rns = config.codec_type == 'rns'
        assert not (config.codec_use_magnitude and config.codec_linear_magnitude), \
            "codec_use_magnitude (QAM, per-digit direction+magnitude) and codec_linear_magnitude (shared direction, per-digit magnitude only) are mutually exclusive digit-encoding schemes"
        self.linear_magnitude = config.codec_type == 'slot' and config.codec_linear_magnitude
        self.use_magnitude = self.rns or (config.codec_type == 'slot' and (config.codec_use_magnitude or self.linear_magnitude))  # RNS always uses QAM-style (direction+magnitude) per-channel codewords -- see GPTConfig.codec_rns_moduli
        self.zero_pad = config.codec_type == 'slot' and config.codec_zero_pad  # RNS has no zero_pad/blank concept at all -- every operand always populates all K channels regardless of digit length, by construction
        assert not (config.combiner_magnitude_signal and config.codec_type not in ('slot', 'rns')), \
            "combiner_magnitude_signal's mag_bias is only ever computed in combine()'s codec_type in ('slot','rns') branch"
        codec_dim = config.n_embd if config.codec_type == 'hrr' else config.codec_dims_per_place
        if self.rns:
            # See GPTConfig.codec_rns_moduli. Each of K channels gets its
            # own QAM-style codeword table, padded to the largest modulus
            # (rns_valid_mask marks which classes are real vs. padding).
            # rns_place_weights[k, p] = 10^p mod moduli[k] -- a fixed,
            # precomputed, NON-learned lookup (register_buffer, not a
            # Parameter): encode_operand reads it to compute each channel's
            # residue via a closed-form weighted sum, exactly the same
            # category of fixed construction as HRR's place_signal role
            # vectors, not something training discovers.
            moduli = config.codec_rns_moduli
            assert len(moduli) > 0, "codec_type=='rns' requires codec_rns_moduli to be set"
            K = len(moduli)
            max_m = max(moduli)
            self.rns_num_channels = K
            self.rns_max_modulus = max_m
            self.register_buffer('rns_moduli', torch.tensor(moduli, dtype=torch.long), persistent=False)
            P_max = config.max_answer_digits
            weights = torch.zeros(K, P_max, dtype=torch.long)
            for k, m in enumerate(moduli):
                w = 1
                for p in range(P_max):
                    weights[k, p] = w
                    w = (w * 10) % m
            self.register_buffer('rns_place_weights', weights, persistent=False)
            valid = torch.zeros(K, max_m, dtype=torch.bool)
            for k, m in enumerate(moduli):
                valid[k, :m] = True
            self.register_buffer('rns_valid_mask', valid, persistent=False)
            self.rns_direction = nn.Parameter(torch.randn(K, max_m, codec_dim) * 0.02)
            self.rns_scale_raw = nn.Parameter(torch.zeros(K, max_m))
        elif self.linear_magnitude:
            # Every digit shares the SAME direction (no direction freedom
            # at all -- codec_dims_per_place must be 1); only magnitude
            # varies with value. Under this scheme vector addition of two
            # operands' codewords literally IS digit addition (magnitudes
            # sum), and carry-out is a plain magnitude-threshold test --
            # not the bilinear complex-multiplication codec_seed_numeric_
            # order's phase scheme needs. See GPTConfig.codec_linear_magnitude.
            assert config.codec_dims_per_place == 1, \
                "codec_linear_magnitude is a pure scalar-magnitude code -- codec_dims_per_place should be 1 (no direction freedom to spend extra dims on)"
            assert config.codec_zero_pad, \
                "codec_linear_magnitude doesn't implement a blank codeword -- use codec_zero_pad=True"
            # Gap-cumsum parameterization, not one sigmoid-bounded scalar
            # per digit (the original version of this scheme, replaced
            # after probe_codebook_geometry-style inspection found it
            # collapsing digits 1-9 together over training: cross-entropy
            # never stops rewarding more separation/confidence, raw params
            # have no weight decay (1D, excluded by configure_optimizers),
            # so they ran away unboundedly toward sigmoid's saturating
            # tails -- and sigmoid'(x)->0 there, so growing raw-space
            # separation bought ever-LESS magnitude-space separation, the
            # opposite of what's needed). Here 9 free gap params (one
            # between each consecutive digit pair) go through softplus
            # (always positive, no saturating ceiling of its own --
            # monotonicity guaranteed by construction, not something
            # training discovers or maintains), cumsum to an unnormalized
            # ladder, then the WHOLE ladder is rescaled by one global
            # factor (unnorm[-1]) to land exactly on [0, codec_max_scale].
            # Growing one gap directly reallocates a proportional share of
            # the fixed total via a linear rescale, not an asymptotic
            # squash -- there's no saturating region for raw params to
            # waste effort drifting into. digit 0 lands at exactly 0 and
            # digit 9 at exactly codec_max_scale by construction (both
            # sensible: 0 as the additive identity, 9 as the max value
            # under a magnitude=value scheme), with the 8 interior digits'
            # spacing the only genuinely free/learned part. Init at raw=0
            # for all 9 gaps -- softplus(0)=ln(2) uniformly, so initial
            # spacing is already the perfectly even ladder.
            self.digit_gap_raw = nn.Parameter(torch.zeros(9))
        elif self.use_magnitude:
            # QAM-style: direction and magnitude are SEPARATE parameters,
            # not one raw vector split post-hoc. A plain nn.Embedding's
            # gradient conflates a radial component (changes magnitude)
            # with a tangential component (changes direction) -- measured
            # directly (see project discussion) that under the old
            # direction-only scheme, ~92% of the raw gradient was being
            # spent on the radial part, which per-place normalize then
            # discarded every forward pass, leaving only ~8% actually able
            # to reshape the codebook. Splitting gives each its own clean,
            # undiluted gradient. Magnitude is explicitly bounded (see
            # digit_scale property) rather than left for weight decay to
            # hopefully restrain -- confirmed empirically that raw
            # embedding norms grew ~100-150x from init with NO sign of
            # plateauing across 10000 iterations even when magnitude was
            # completely functionally inert (erased by the old normalize);
            # letting it matter unbounded would very likely run away.
            if config.codec_seed_numeric_order:
                # seed at consecutive angles around a circle in the first two
                # dims (numeric order 0..9), zero in any remaining dims --
                # still a plain learnable Parameter, free to drift or hold.
                init_scale = 0.02
                digit_angles = torch.arange(10, dtype=torch.float32) * (2 * math.pi / 10)
                init_dir = torch.zeros(10, codec_dim)
                init_dir[:, 0] = torch.cos(digit_angles)
                init_dir[:, 1] = torch.sin(digit_angles)
                self.digit_direction = nn.Parameter(init_dir * init_scale)
            else:
                self.digit_direction = nn.Parameter(torch.randn(10, codec_dim) * 0.02)
            self.digit_scale_raw = nn.Parameter(torch.zeros(10))
            if not self.zero_pad:
                self.blank_direction = nn.Parameter(torch.randn(codec_dim) * 0.02)
                self.blank_scale_raw = nn.Parameter(torch.zeros(1))
        else:
            self.digit_embed = nn.Embedding(10, codec_dim)  # shared digit-identity table: encoder lookup AND decoder classification target (via dot-product in decode_from_vector)
            if not self.zero_pad:
                self.blank_embed = nn.Parameter(torch.randn(codec_dim) * 0.02)  # separate, NOT tied -- "there is no digit here" has no encoder-side counterpart to share with
        # codec_zero_pad: eliminates the blank class entirely. Every
        # operand/sum is represented at a FIXED width (max_answer_digits)
        # via ordinary leading-zero decimal padding ("42" -> "000...042"),
        # the same way normal positional notation already permits -- so
        # EVERY place, in EVERY training example, always has a genuine
        # digit target (0-9), never "sometimes populated, sometimes not".
        # This removes the root cause of the recurring leading-digit
        # boundary bug (see project discussion): that bug came from
        # "blank" being a per-ABSOLUTE-POSITION learned prior that
        # different positions get wildly uneven training exposure to --
        # with no blank class at all, there's no such prior to be
        # miscalibrated in the first place. Also makes the earlier
        # blank-position recon-loss masking fix (forward()'s
        # a_targets.masked_fill(a_targets==10, -1)) moot: there's no more
        # class 10, so no more structurally-zero, ungradientable query at
        # any position -- every position has genuine content and genuine
        # gradient, always.
        self.drop = nn.Dropout(config.dropout)

        if config.codec_type == 'hrr':
            # HRR/VSA binding, not a learned combine_fn/extract_fn: encode_operand
            # binds each digit to its place via circular convolution with a
            # UNITARY role vector (magnitude 1 at every frequency, phase = p *
            # a fixed random base frequency) and sums; decode_from_vector reads
            # back out via circular correlation, the EXACT mathematical inverse
            # of binding for the matching place, for ANY p including ones never
            # seen in training. This replaces the earlier hope that gradient
            # descent would discover good packing/separation on its own (tried
            # via sparsity + cross-place-orthogonality penalties, which measurably
            # compressed and separated the representation but produced ZERO OOD
            # improvement) with a mathematical guarantee that doesn't need
            # discovering: probe_hrr_capacity.py validated near-perfect recovery
            # through L~15-20 at dim=384 using RANDOM (untrained) vectors and
            # this exact mechanism, with graceful (not catastrophic) degradation
            # beyond that -- a structurally different failure mode than either
            # the old attention encoder's hard cliff or the sum encoder's
            # trained-but-unverified interference pattern.
            #
            # This also makes the decoder genuinely parameter-free (no extract_fn
            # at all): unbinding IS the query, by construction -- "decoder should
            # be simple, just read off the vector" is satisfied about as
            # literally as possible. All remaining nonlinear work (carry
            # propagation) is left entirely to the combiner, which is unchanged.
            place_table = _precompute_unitary_roles(config.max_answer_digits, config.n_embd)
            self.register_buffer('place_signal', place_table, persistent=False)  # deterministic, zero learned parameters
            self.enc_ln = LayerNorm(config.n_embd, bias=config.bias)  # safety net: binding itself is norm-preserving, but digit_embed is learned and could still drift in scale during training
        else:
            # Disjoint-slot codec: place p owns dims [dpp*p : dpp*(p+1))
            # exclusively -- no binding, no FFT, no role vectors at all, and
            # (unlike HRR) NO cross-place interference within capacity, by
            # construction rather than by statistical argument. See
            # probe_slot_capacity.py and GPTConfig.codec_type's comment.
            # Deliberately no analogue of enc_ln here: LayerNorm computed
            # over the FULL n_embd dims would fold in however many trailing
            # slots are still zero-padded (short operands leave most of the
            # vector at exactly 0), making its mean/std -- and hence every
            # active slot's rescaling -- depend on operand LENGTH. That
            # reintroduces a length-dependent coupling between otherwise-
            # disjoint places, exactly what this scheme exists to avoid.
            # encode_operand instead L2-normalizes each digit's OWN dpp-dim
            # code independently (see encode_operand) -- a per-place, not
            # per-vector, safety net.
            assert config.n_embd % config.codec_dims_per_place == 0, \
                "n_embd must be divisible by codec_dims_per_place (trailing remainder dims would sit permanently unused)"
            if config.codec_type == 'slot':
                assert config.max_answer_digits <= config.n_embd // config.codec_dims_per_place, \
                    f"max_answer_digits ({config.max_answer_digits}) exceeds slot capacity " \
                    f"(n_embd // codec_dims_per_place = {config.n_embd // config.codec_dims_per_place})"
            else:  # 'rns' -- capacity is bounded by channel COUNT, not max_answer_digits (which is irrelevant to RNS's own capacity, only to how wide the place_weights buffer is precomputed)
                assert self.rns_num_channels <= config.n_embd // config.codec_dims_per_place, \
                    f"codec_rns_moduli has {self.rns_num_channels} channels, exceeding slot capacity " \
                    f"(n_embd // codec_dims_per_place = {config.n_embd // config.codec_dims_per_place})"
            assert not config.combiner_explicit_place_tag, "combiner_explicit_place_tag needs place_signal, which codec_type in ('slot','rns') doesn't build (and v18 already showed it's not load-bearing anyway)"
            if config.combiner_magnitude_signal:
                assert config.codec_use_magnitude or config.codec_linear_magnitude or self.rns, "combiner_magnitude_signal only makes sense when magnitude carries real information"
                assert config.combiner_use_per_place, "combiner_magnitude_signal's mag_bias is only ever computed in combine()'s per-place branch"
            if config.codec_seed_numeric_order:
                assert config.codec_use_magnitude, "codec_seed_numeric_order seeds digit_direction, which only exists when codec_use_magnitude=True"
                assert config.codec_dims_per_place >= 2, "numeric-order seeding needs at least 2 dims (cos, sin) to place points on a circle"

        self.num_places = self.rns_num_channels if self.rns else config.max_answer_digits  # P everywhere below: RNS has K residue-channel slots, not max_answer_digits decimal-place slots

        # Repeating combiner: SAME weights every round (Universal-Transformer-
        # style), applied over an explicit per-place token sequence built by
        # unbinding (HRR) or slicing (slot/rns) vec_a/vec_b at every place --
        # see GPTConfig.combiner_repeat_max's comment for the full rationale.
        # Bidirectional (every place/thinking token can attend to every
        # other one, every round): carry information needs to move freely
        # across places, not just in one causal direction, and every
        # place-token's final state (not just a single "last position")
        # feeds the rebind/write-back step.
        comb_config = dataclasses.replace(config, n_layer=1, bidirectional=True, attn_window=None,
                                           block_size=self.num_places + config.combiner_n_thinking)  # sequence is P place-tokens + thinking tokens, not the original 4 slots -- RoPE/mask buffers must cover this
        self.combiner_block = Block(comb_config)
        self.combiner_attn_band = config.combiner_attn_band
        if self.combiner_attn_band > 0:
            if config.combiner_use_per_place:
                assert config.combiner_magnitude_signal, "combiner_attn_band only threads the mask through combine()'s magnitude-signal call path when combiner_use_per_place=True"
                Pn = self.num_places
                Tn = Pn + config.combiner_n_thinking
                idx = torch.arange(Tn)
                is_think = idx >= Pn
                local = (idx.view(-1, 1) - idx.view(1, -1)).abs() <= self.combiner_attn_band   # |p - p'| <= band
                mask = local | is_think.view(-1, 1) | is_think.view(1, -1)                       # + full connectivity for thinking tokens
                self.register_buffer('combiner_band_mask', mask.view(1, 1, Tn, Tn), persistent=False)  # deterministic, zero learned params
            else:
                # combiner_use_per_place=False: roles are swapped from the
                # per-place case -- the THINKING tokens are now the place-
                # like tokens to band locally (only meaningful alongside
                # combiner_thinking_rope, which is what gives them place-
                # like sequential positions to band OVER in the first
                # place); vec_a/vec_b play the global-bus role thinking
                # tokens played above. See GPTConfig.combiner_attn_band and
                # GPTConfig.combiner_thinking_rope.
                assert config.combiner_thinking_rope, "combiner_attn_band with combiner_use_per_place=False requires combiner_thinking_rope -- banding needs the thinking tokens to already have place-like sequential positions"
                Tn = config.combiner_n_thinking + 2
                idx = torch.arange(Tn)
                is_bus = idx >= config.combiner_n_thinking  # vec_a, vec_b = global bus (last 2 indices)
                local = (idx.view(-1, 1) - idx.view(1, -1)).abs() <= self.combiner_attn_band
                mask = local | is_bus.view(-1, 1) | is_bus.view(1, -1)
                self.register_buffer('combiner_band_mask', mask.view(1, 1, Tn, Tn), persistent=False)
        self.combiner_proj = nn.Linear(2 * codec_dim, config.n_embd)  # combines each place's (vec_a, vec_b) content into one full-width token -- codec_dim is n_embd for 'hrr' (unbind output), codec_dims_per_place for 'slot'/'rns' (raw slice)
        self.thinking_tokens = nn.Parameter(torch.randn(config.combiner_n_thinking, config.n_embd) * 0.02)
        self.countdown_head = nn.Linear(config.n_embd, config.combiner_repeat_max + 1)  # dense auxiliary signal only -- doesn't gate anything in this version
        self.comb_ln = LayerNorm(config.n_embd, bias=config.bias)
        if config.codec_type in ('slot', 'rns'):
            self.combiner_out_slot = nn.Linear(config.n_embd, config.codec_dims_per_place)  # compresses each place-token's full-width final state back down to its dpp-dim output code -- the one new burden this codec places on the combiner, see project discussion
        if config.combiner_magnitude_signal:
            self.combiner_mag_proj = nn.Linear(2, config.n_embd)  # (||qa||, ||qb||) per place -> a full-width additive bias, re-injected after ln_1/ln_2 at every repeat round so attn/mlp can actually condition on magnitude -- see GPTConfig.combiner_magnitude_signal

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * max(config.combiner_repeat_max, 1)))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def digit_codewords(self):
        """Returns (10, codec_dim): the digit-identity codebook, read by
        BOTH encode_operand (what gets written into a slot/bound into a
        place) and decode_from_vector (the classification target table).
        Plain learned table unless codec_type == 'slot' and
        codec_use_magnitude == True, in which case direction and
        (bounded) magnitude are separate parameters recombined here --
        see __init__ and GPTConfig.codec_use_magnitude."""
        if self.linear_magnitude:
            gaps = F.softplus(self.digit_gap_raw)  # (9,), always > 0 -- monotonicity by construction
            unnorm = torch.cat([gaps.new_zeros(1), torch.cumsum(gaps, dim=0)])  # (10,), unnorm[0]=0
            scale = self.config.codec_max_scale * unnorm / unnorm[-1]  # (10,) -- linear rescale, no saturation; scale[0]=0, scale[9]=codec_max_scale exactly
            return scale.unsqueeze(-1)  # (10, 1) -- codec_dims_per_place asserted ==1
        if self.use_magnitude:
            direction = F.normalize(self.digit_direction, dim=-1)
            scale = self.config.codec_max_scale * torch.sigmoid(self.digit_scale_raw)
            return direction * scale.unsqueeze(-1)
        return self.digit_embed.weight

    def rns_class_codewords(self):
        """Returns (K, max_modulus, codec_dim): per-channel QAM-style
        codeword tables, padded to the largest modulus -- only the first
        moduli[k] entries of channel k are ever written to or read from
        (see rns_valid_mask, used to mask padding classes out of decode's
        logits). Same direction+bounded-magnitude construction as
        digit_codewords()'s use_magnitude branch, just per-channel instead
        of a single shared table -- see GPTConfig.codec_rns_moduli."""
        direction = F.normalize(self.rns_direction, dim=-1)  # (K, max_m, dpp)
        scale = self.config.codec_max_scale * torch.sigmoid(self.rns_scale_raw)  # (K, max_m)
        return direction * scale.unsqueeze(-1)

    def blank_codeword(self):
        """Returns (codec_dim,): the blank-class codeword, same
        construction logic as digit_codewords()."""
        assert not self.zero_pad, "blank_codeword() shouldn't be called at all when codec_zero_pad=True -- there's no blank class in this mode"
        if self.use_magnitude:
            direction = F.normalize(self.blank_direction, dim=-1)
            scale = self.config.codec_max_scale * torch.sigmoid(self.blank_scale_raw)
            return direction * scale
        return self.blank_embed

    def encode_operand(self, digit_values, return_contributions=False):
        """digit_values: (B, L) integer digit VALUES 0-9 (NOT vocab token
        ids), most-significant digit first. Returns (B, n_embd).

        codec_type == 'hrr': the sum of every digit's (identity, place) HRR
        binding -- circular convolution with a unitary place role vector,
        not a learned combine_fn -- so no attention, no competition, no
        budget renegotiated as L changes, AND no reliance on gradient
        descent having discovered good packing for places beyond what
        training showed it (see place_signal's construction,
        _precompute_unitary_roles). If return_contributions, also returns
        the raw (B, L, C) per-digit bound vectors before the sum -- used
        only by the composed_contrib_sparsity_weight penalty.

        codec_type == 'slot': each digit's dpp-dim code written into its
        OWN disjoint slice, dims [dpp*p : dpp*(p+1)) for place p -- no
        binding, no sum, no crosstalk with any other place at all. Asserts
        L doesn't exceed the hard slot capacity (checked in __init__ for
        max_answer_digits, but individual operands could in principle be
        longer than that -- shouldn't happen given how callers construct
        digit_values, but fail loudly rather than silently truncating)."""
        B, L = digit_values.size()
        if self.codec_type == 'hrr':
            d_emb = self.digit_embed(digit_values)  # (B, L, C)
            places = torch.arange(L - 1, -1, -1, device=digit_values.device)  # [L-1, ..., 0]: units=0
            role = self.place_signal[places].unsqueeze(0).expand(B, -1, -1)  # (B, L, C)
            contributions = _circular_bind(self.drop(d_emb), role)  # (B, L, C)
            vec = contributions.sum(dim=1)  # (B, C)
            vec = self.enc_ln(vec)  # safety net -- see __init__'s comment on enc_ln
            return (vec, contributions) if return_contributions else vec
        elif self.codec_type == 'rns':
            # Each channel's residue is a closed-form, NON-learned function
            # of digit_values -- residue_k = (sum_p digit_p * (10^p mod
            # m_k)) mod m_k, computed via the precomputed rns_place_weights
            # buffer. No carry, no sequential dependency across places (the
            # weighted sum is fully parallel) -- see GPTConfig.
            # codec_rns_moduli. Then each channel's residue value is looked
            # up in that channel's OWN learned QAM codeword table, exactly
            # like the 'slot' codec's digit lookup, just keyed by residue
            # instead of decimal digit.
            dpp = self.config.codec_dims_per_place
            K = self.rns_num_channels
            places = torch.arange(L - 1, -1, -1, device=digit_values.device)  # place value per input position, units=0 last (MSD-first input)
            w = self.rns_place_weights[:, places]  # (K, L)
            # matmul/einsum on integer tensors isn't implemented on CUDA
            # (baddbmm_cuda has no Long kernel) -- route through float32
            # instead, exact here since the largest possible sum (well
            # under max_answer_digits * 9 * max(moduli) ~ 10^5) is far
            # inside float32's exact-integer range (2^24) -- BUT autocast
            # (bf16/fp16) would silently downcast this matmul regardless of
            # the .float() calls (bf16 only has ~8 mantissa bits, exact
            # only up to 256, nowhere near the ~10^4 sums involved), which
            # measurably corrupted residues in practice -- confirmed via
            # direct comparison: identical inputs gave exact residues with
            # autocast disabled, wrong ones (worse for longer L, more terms
            # to round) with it enabled, matching v31's observed flat loss
            # plateau and length-dependent reconstruction failure exactly.
            # Explicitly disabling autocast for just this op is the fix,
            # same category of problem as combiner_out_slot's .float() but
            # needing the stronger tool since that alone doesn't stop
            # autocast from downcasting the matmul's actual computation.
            with torch.autocast(device_type=digit_values.device.type, enabled=False):
                residues_f = torch.einsum('bl,kl->bk', digit_values.float(), w.float())  # (B, K)
            residues = residues_f.round().long() % self.rns_moduli.unsqueeze(0)  # (B, K)
            codewords = self.rns_class_codewords()  # (K, max_m, dpp)
            k_idx = torch.arange(K, device=digit_values.device).unsqueeze(0).expand(B, -1)  # (B, K)
            gathered = codewords[k_idx, residues]  # (B, K, dpp) -- codewords[k, residues[b,k], :]
            gathered = self.drop(gathered)
            flat = gathered.reshape(B, K * dpp)
            vec = flat.new_zeros(B, self.config.n_embd)
            vec[:, :K * dpp] = flat
            return (vec, None) if return_contributions else vec
        else:
            dpp = self.config.codec_dims_per_place
            max_places = self.config.n_embd // dpp
            assert L <= max_places, f"operand length {L} exceeds slot capacity {max_places} (n_embd={self.config.n_embd}, codec_dims_per_place={dpp})"
            d_emb = self.digit_codewords()[digit_values]  # (B, L, dpp) -- place value for input position i is L-1-i (MSD-first)
            d_emb = self.drop(d_emb)
            if not self.use_magnitude:
                d_emb = F.normalize(d_emb, dim=-1)  # per-PLACE safety net against scale drift -- NOT a global LayerNorm, see __init__'s comment on why. Skipped when codec_use_magnitude: magnitude is meant to carry information there, so it must survive to the written slot, not get stripped back to unit norm (digit_codewords() already applies its OWN explicit, bounded scale instead -- see GPTConfig.codec_use_magnitude)
            contributions = d_emb.flip(dims=[1])  # contributions[:, p, :] is now place p's code (flip turns "MSD-first, units last" into "place 0 (units) first, ascending")
            flat = contributions.reshape(B, L * dpp)
            vec = flat.new_zeros(B, self.config.n_embd)
            vec[:, :L * dpp] = flat
            if self.zero_pad:
                # No blank class: extend with ordinary leading-zero decimal
                # padding up to the fixed width max_answer_digits, using
                # the digit-0 codeword -- NOT left as literal zero vectors.
                # "42" is represented identically to "000...00042" -- see
                # GPTConfig.codec_zero_pad.
                W = self.config.max_answer_digits
                assert L <= W, f"operand length {L} exceeds max_answer_digits {W}"
                zero_codeword = self.digit_codewords()[0]  # (dpp,) -- same normalization convention as the real digits above (already-scaled if use_magnitude, else raw -- normalize to match d_emb's own treatment)
                if not self.use_magnitude:
                    zero_codeword = F.normalize(zero_codeword, dim=-1)
                vec[:, L * dpp:W * dpp] = zero_codeword.unsqueeze(0).expand(W - L, dpp).reshape(-1).unsqueeze(0)
            else:
                pass  # places >= L (this operand's own length) simply stay exactly 0 -- decode reads them as blank
            return (vec, contributions) if return_contributions else vec

    def place_orthogonality_loss(self):
        """Mean squared cosine similarity between DIFFERENT places' typical
        bound contributions (each place's representative vector = the HRR
        binding averaged over all 10 digit values, so digit-to-digit
        structure at a given place is untouched -- only cross-place overlap
        is penalized). With unitary role vectors this should already be
        low in expectation regardless of training (that's the whole point
        of the unitary construction), so this is more a diagnostic/
        insurance term than a load-bearing one now. Needs no real digit
        data, just current digit_embed/place_signal, so it's cheap.
        codec_type == 'slot' has zero cross-place overlap by construction
        (disjoint dims) -- not wired up for it, not meaningful there."""
        assert self.codec_type == 'hrr', "place_orthogonality_loss needs place_signal; codec_type='slot' has zero cross-place overlap by construction so this penalty isn't meaningful there"
        P = self.config.place_orthogonality_max_place
        device = self.digit_embed.weight.device
        digits = torch.arange(10, device=device)
        reprs = []
        for p in range(P):
            d_emb = self.digit_embed(digits)  # (10, C)
            role = self.place_signal[p].unsqueeze(0).expand(10, -1)  # (10, C)
            bound = _circular_bind(d_emb, role)  # (10, C)
            reprs.append(bound.mean(dim=0))  # (C,)
        reprs = torch.stack(reprs)  # (P, C)
        reprs = F.normalize(reprs, dim=-1)
        sim = reprs @ reprs.t()  # (P, P)
        mask = ~torch.eye(P, dtype=torch.bool, device=device)
        return (sim[mask] ** 2).mean()

    def combine(self, vec_a, vec_b, plus_id=None, eq_id=None, max_len=None, n_rounds=None, n_think=None):
        """Repeating combiner: builds an explicit per-place token sequence
        by unbinding vec_a/vec_b at every place (the OLD combiner only ever
        saw 2 pre-summed vectors -- no per-place resolution at all, so it
        had no way to represent "move information between places" no
        matter how it was trained), then applies a SHARED (same weights
        every round) attention+FFN block repeatedly. Depth that scales with
        repetition count, not a fixed number of distinct layers: carry
        propagation's worst case (adding 1 to a run of 9s) needs a
        dependency chain as long as the number of places, which a
        fixed-depth pass structurally cannot implement regardless of
        training. combiner_repeat_max rounds always run -- no gating/
        early-exit in this version, deliberately simpler than full adaptive
        halting. max_len, if given (max(len_a,len_b), an easy-to-compute
        proxy for how long a carry chain COULD need to be, not the exact
        minimal chain length for this specific example): supervises a
        dense, per-round auxiliary "how many rounds are probably still
        needed" countdown -- doesn't gate anything, purely an auxiliary
        training signal, same spirit as LatentLoopedGPT's countdown_head.
        Returns (vec_c, countdown_loss); countdown_loss is None if max_len
        wasn't given. plus_id/eq_id accepted for call-site compatibility
        with the old signature, unused now."""
        B = vec_a.size(0)
        P = self.num_places  # max_answer_digits for 'hrr'/'slot', K residue channels for 'rns' -- see __init__
        device = vec_a.device
        use_per_place = self.config.combiner_use_per_place
        role = None  # only assigned (and only meaningful) for codec_type == 'hrr'
        n_rounds = n_rounds if n_rounds is not None else self.config.combiner_repeat_max  # variable-depth training (see GPTConfig.combiner_variable_repeat / forward()): defaults to the full budget, matching every call site that doesn't explicitly pass a shorter count (inference included)
        n_think = n_think if n_think is not None else self.config.combiner_n_thinking  # variable thinking-token count (see GPTConfig.combiner_variable_thinking / forward()): defaults to the full budget, same convention as n_rounds

        if use_per_place and self.codec_type == 'hrr':
            places = torch.arange(P, device=device)
            role = self.place_signal[places].unsqueeze(0).expand(B, -1, -1)  # (B, P, C)
            qa = _circular_unbind(vec_a.unsqueeze(1).expand(-1, P, -1), role)  # (B, P, C)
            qb = _circular_unbind(vec_b.unsqueeze(1).expand(-1, P, -1), role)  # (B, P, C)
            tokens = self.combiner_proj(torch.cat([qa, qb], dim=-1))  # (B, P, C)
            if self.config.combiner_explicit_place_tag:
                tokens = tokens + role  # hand-added positional tag, on top of whatever the combiner block's own RoPE already provides -- see GPTConfig.combiner_explicit_place_tag
        elif use_per_place:  # codec_type == 'slot': per-place resolution is just slicing, no unbind needed
            dpp = self.config.codec_dims_per_place
            qa = vec_a[:, :P * dpp].view(B, P, dpp)  # (B, P, dpp) -- vec_a's own disjoint slots for places 0..P-1; places past vec_a's own length are already exactly 0 from encode_operand
            qb = vec_b[:, :P * dpp].view(B, P, dpp)
            tokens = self.combiner_proj(torch.cat([qa, qb], dim=-1))  # (B, P, C) -- combiner_proj sized 2*dpp -> C for codec_type == 'slot' (see __init__)
            if self.config.combiner_magnitude_signal:
                # ||qa||, ||qb|| computed from the ORIGINAL input slots, not
                # x's own evolving norm (which reflects attn/mlp's own
                # transformations, not "was this a real digit or blank") --
                # see GPTConfig.combiner_magnitude_signal for why this needs
                # to be re-injected after normalization rather than added
                # once at the start.
                mag_feat = torch.stack([qa.norm(dim=-1), qb.norm(dim=-1)], dim=-1)  # (B, P, 2)
                mag_bias_place = self.combiner_mag_proj(mag_feat)  # (B, P, C)
        else:
            # Ablation: no per-place resolution at all -- just the 2
            # pre-summed vectors, like the old combiner, but STILL repeated
            # combiner_repeat_max times. Isolates whether per-place exposure
            # specifically mattered, or whether repetition alone (looking at
            # the same impoverished input more times) would have sufficed.
            tokens = torch.stack([vec_a, vec_b], dim=1)  # (B, 2, C)
            if self.config.combiner_explicit_place_tag:
                tokens = tokens + self.place_signal[:2].unsqueeze(0)  # place_signal[0]/[1] used here only to break the a/b symmetry, not as real place semantics
        position_ids = None
        if not use_per_place and self.config.combiner_thinking_rope:
            # See GPTConfig.combiner_thinking_rope: put the thinking tokens
            # FIRST so they get place-like sequential RoPE positions
            # 0..n_think-1, vec_a/vec_b AFTER at positions n_think, n_think+1.
            think = self.thinking_tokens[:n_think].unsqueeze(0).expand(B, -1, -1)
            x = self.drop(torch.cat([think, tokens], dim=1))  # (B, n_think+2, C)
            position_ids = torch.arange(n_think + 2, device=device).unsqueeze(0).expand(B, -1)
        else:
            think = self.thinking_tokens[:n_think].unsqueeze(0).expand(B, -1, -1)  # (B, n_think, C) -- prefix of the full ceiling, see GPTConfig.combiner_variable_thinking
            x = self.drop(torch.cat([tokens, think], dim=1))  # (B, P_or_2+n_think, C)

        # Interpretability hook (off by default, zero cost when unset): if a
        # caller sets self._combiner_trace = [], every round's full pre-comb_ln
        # state is appended, plus the initial state before round 1 -- used by
        # probe_combiner_dynamics.py (per-round decode + fixed-point analysis).
        trace = getattr(self, '_combiner_trace', None)
        if trace is not None:
            trace.append(x.detach().clone())
        # Interpretability hook (off by default): if a caller sets
        # self._combiner_hook to a callable, it is invoked as hook(t, x) after
        # every round t (0-indexed) and its return value replaces x -- used by
        # probe_combiner_patch.py for activation patching / causal interventions.
        chook = getattr(self, '_combiner_hook', None)
        # banded local attention (see GPTConfig.combiner_attn_band), sliced
        # to the live n_think. use_per_place layout: place tokens 0..P-1
        # (banded), thinking P..P+n_think-1 (bus). combiner_thinking_rope
        # layout: thinking tokens 0..n_think-1 (banded, place-like
        # positions), vec_a/vec_b last (bus) -- the max-sized buffer's bus
        # indices sit at [combiner_n_thinking, combiner_n_thinking+1]
        # regardless of the live n_think, since thinking_tokens[:n_think]
        # is always a PREFIX of the full set.
        band_mask = None
        if getattr(self, 'combiner_attn_band', 0) > 0:
            if use_per_place:
                keep = list(range(P)) + list(range(self.num_places, self.num_places + n_think))
            else:
                n_think_max = self.config.combiner_n_thinking
                keep = list(range(n_think)) + [n_think_max, n_think_max + 1]
            keep_t = torch.tensor(keep, device=vec_a.device)
            band_mask = self.combiner_band_mask[:, :, keep_t][:, :, :, keep_t]

        if self.config.combiner_magnitude_signal:
            mag_bias = torch.cat([mag_bias_place, mag_bias_place.new_zeros(B, n_think, self.config.n_embd)], dim=1)  # (B, P+n_think, C) -- zero for thinking tokens, which have no magnitude to speak of

            def run_block(x):
                # Inlined Block.forward (see Block class) instead of
                # self.combiner_block(x): Pre-LN's ln_1/ln_2 normalize away
                # magnitude every round (see GPTConfig.
                # combiner_magnitude_signal), so mag_bias is re-added AFTER
                # each norm, every round, rather than once at the input
                # (which ln_1 would just erase again on round 2+).
                blk = self.combiner_block
                x = x + blk.attn(blk.ln_1(x) + mag_bias, position_ids=position_ids, attn_mask=band_mask)
                x = x + blk.mlp(blk.ln_2(x) + mag_bias)
                return x
        else:
            def run_block(x):
                return self.combiner_block(x, position_ids=position_ids, attn_mask=band_mask)

        countdown_loss = None
        if max_len is not None:
            target_len = min(max_len, self.config.combiner_repeat_max)
            countdown_losses = []
            for t in range(n_rounds):
                x = run_block(x)
                if chook is not None:
                    x = chook(t, x)
                if trace is not None:
                    trace.append(x.detach().clone())
                logits = self.countdown_head(x.mean(dim=1))  # (B, repeat_max+1) -- simple mean-pool over all tokens
                remaining = torch.full((B,), max(target_len - t, 0), dtype=torch.long, device=device)
                countdown_losses.append(F.cross_entropy(logits, remaining))
            countdown_loss = torch.stack(countdown_losses).mean()
        else:
            for t in range(n_rounds):
                x = run_block(x)
                if chook is not None:
                    x = chook(t, x)
                if trace is not None:
                    trace.append(x.detach().clone())

        x = self.comb_ln(x)
        if use_per_place and self.codec_type == 'hrr':
            place_tokens = x[:, :P, :]  # (B, P, C) -- drop the thinking tokens, keep only the per-place ones
            vec_c = _circular_bind(place_tokens, role).sum(dim=1)  # rebind each place, superpose -- same mechanism encode_operand uses
        elif use_per_place:  # codec_type == 'slot': write each place's own output straight into its own disjoint slot, no rebind/sum at all
            dpp = self.config.codec_dims_per_place
            place_tokens = x[:, :P, :]  # (B, P, C)
            out_codes = self.combiner_out_slot(place_tokens).float()  # (B, P, dpp) -- the combiner's one new burden: compress its full-width per-place state back to a clean dpp-dim code (see __init__). .float(): unlike the 'hrr' path's final step (_circular_bind, an FFT op autocast leaves in fp32, matching encode_operand's fp32 output), this is a plain nn.Linear, which autocast WOULD cast to bf16/fp16 -- forcing fp32 here keeps vec_c's dtype consistent with encode_operand's regardless of codec_type, avoiding a dtype-mismatch crash where vec_c meets encode_operand's output (e.g. composed_consistency_weight's vec_sum_target)
            vec_c = out_codes.new_zeros(B, self.config.n_embd)
            vec_c[:, :P * dpp] = out_codes.reshape(B, P * dpp)
        else:
            # ablation: no place structure to rebind from -- just read off
            # the "vec_b slot" token directly, same convention the old
            # combiner used (last content token). Position depends on
            # combiner_thinking_rope's layout: [think..., vec_a, vec_b]
            # puts vec_b at the very end; the default [vec_a, vec_b,
            # think...] puts it at index 1.
            vec_c = x[:, -1, :] if (self.config.combiner_thinking_rope) else x[:, 1, :]
        return vec_c, countdown_loss

    def decode_from_vector(self, memory_vec, targets=None):
        """memory_vec: (B, n_embd). Predicts config.max_answer_digits
        places in PARALLEL -- no autoregression, no recurrence, every place
        an independent function of memory_vec and its own place value.
        targets, if given: (B, max_answer_digits) class indices 0-9 (digit)
        or 10 (blank), right-justified (true digits occupy the LAST L
        slots).

        codec_type == 'hrr': genuinely parameter-free readout -- unbinding
        (circular correlation with that place's role vector) IS the query,
        the exact mathematical inverse of encode_operand's binding for the
        matching place -- no learned extract_fn at all.

        codec_type == 'slot': equally parameter-free -- the query for a
        place is just that place's own disjoint slice, read directly, no
        unbind needed since there was never anything else superposed into
        those dims to begin with."""
        B = memory_vec.size(0)
        if self.codec_type == 'rns':
            # Separate path, not folded into the shared digit_codewords()/
            # class_vectors logic below: class COUNT varies per channel
            # (moduli differ), so there's no single shared table the way
            # decimal digits (always exactly 10, or 11 with blank) have.
            # targets, if given: (B, K) residue class indices, one per
            # channel -- see GPTConfig.codec_rns_moduli and train.py's
            # rns_targets(). No zero_pad/blank concept here at all.
            dpp = self.config.codec_dims_per_place
            K = self.rns_num_channels
            mem_slots = memory_vec[:, :K * dpp].view(B, K, dpp)  # (B, K, dpp) -- each channel's own disjoint slice
            codewords = self.rns_class_codewords()  # (K, max_m, dpp)
            logits = torch.einsum('bkd,kmd->bkm', mem_slots, codewords)  # (B, K, max_m)
            logits = logits - 0.5 * (codewords ** 2).sum(dim=-1).unsqueeze(0)  # energy-corrected rule, same reasoning as codec_use_magnitude -- see GPTConfig.codec_use_magnitude
            logits = logits.masked_fill(~self.rns_valid_mask.unsqueeze(0), float('-inf'))  # each channel's padding classes (beyond its own modulus) can never be predicted or contribute probability mass
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.reshape(-1, self.rns_max_modulus), targets.reshape(-1), ignore_index=-1)
            return logits, loss
        P = self.config.max_answer_digits
        places = torch.arange(P - 1, -1, -1, device=memory_vec.device)  # slot 0 = highest place value
        if self.codec_type == 'hrr':
            role = self.place_signal[places].unsqueeze(0).expand(B, -1, -1)  # (B, P, C)
            mem_expanded = memory_vec.unsqueeze(1).expand(-1, P, -1)  # (B, P, C) -- re-fed at every place, not just once
            query = _circular_unbind(mem_expanded, role)  # (B, P, C)
        else:
            dpp = self.config.codec_dims_per_place
            mem_slots = memory_vec.view(B, -1, dpp)  # (B, n_embd // dpp, dpp) -- every place's own disjoint slice
            query = mem_slots[:, places, :]  # (B, P, dpp) -- gather into output order (slot 0 = highest place value)
        if self.zero_pad:
            class_vectors = self.digit_codewords()  # (10, C) -- no blank class at all, see GPTConfig.codec_zero_pad
        else:
            class_vectors = torch.cat([self.digit_codewords(), self.blank_codeword().unsqueeze(0)], dim=0)  # (11, C) -- C is n_embd for 'hrr', codec_dims_per_place for 'slot'
        n_classes = class_vectors.size(0)
        logits = query @ class_vectors.t()  # (B, P, n_classes) -- classify by dot-product against the SAME table encode_operand reads from
        if self.use_magnitude:
            # Energy-corrected (squared-Euclidean-distance-equivalent) rule
            # -- plain dot-product ranking is only the right decision rule
            # when every class has equal energy (the direction-only/PSK
            # regime). Once magnitude varies, subtracting each candidate's
            # own energy makes uniformly scaling every class vector up a
            # no-op for the decision boundary, closing the "just grow
            # everything" escape hatch that would otherwise let softmax
            # cheat its way to lower loss via pure magnitude inflation
            # instead of genuine separation (see GPTConfig.codec_use_magnitude).
            logits = logits - 0.5 * (class_vectors ** 2).sum(dim=-1)

        loss = None
        if targets is not None:
            weight = None
            if self.zero_pad and self.config.codec_zero_class_weight != 1.0:
                # Class 0's cross-entropy weight, down from 1.0 -- see
                # GPTConfig.codec_zero_class_weight. Padding makes digit 0
                # far more frequent than any other digit under
                # codec_zero_pad (measured ~12x at composed_train_max_digits
                # =20/max_answer_digits=30 under _sample_skewed_length), so
                # an unweighted mean cross-entropy lets nailing "is this
                # padding" dominate the loss at the expense of the other 9
                # classes -- this applies uniformly to every decode_from_
                # vector call (sum decode AND both reconstruction losses),
                # since all three targets are the same zero-padded
                # representation and share this same imbalance.
                weight = logits.new_ones(n_classes)
                weight[0] = self.config.codec_zero_class_weight
            loss = F.cross_entropy(logits.view(-1, n_classes), targets.view(-1), ignore_index=-1, weight=weight)
        return logits, loss

    def forward(self, a_ids, b_ids, targets=None, plus_id=None, eq_id=None,
                a_targets=None, b_targets=None, sum_vals=None):
        """a_ids: (B, L_a), b_ids: (B, L_b) -- fixed-length digit VALUES
        (0-9) per operand. targets: (B, max_answer_digits) sum-decode
        targets. a_targets/b_targets, if given: same shape, the auxiliary
        "decode(encode(operand)) == operand itself" reconstruction targets,
        weighted by composed_recon_weight. sum_vals, if given: a length-B
        plain Python list of the true sum's integer VALUE per example --
        NOT tensorized/padded, because summed lengths vary within a batch
        even when a_ids/b_ids share one length, and encode_operand was
        never trained with a leading-zero digit, so padding a shorter sum
        up to a common width would encode an out-of-distribution quantity
        rather than a genuine "true sum" reference. Grouped internally by
        each example's own natural digit count and encoded at that width.
        Enables the auxiliary "combine(vec_a, vec_b) == encode(true sum)"
        consistency loss, weighted by composed_consistency_weight, target
        detached (see GPTConfig.composed_consistency_weight)."""
        sparsity_weight = self.config.composed_contrib_sparsity_weight
        if targets is not None and sparsity_weight > 0:
            vec_a, contrib_a = self.encode_operand(a_ids, return_contributions=True)
            vec_b, contrib_b = self.encode_operand(b_ids, return_contributions=True)
        else:
            vec_a = self.encode_operand(a_ids)
            vec_b = self.encode_operand(b_ids)
        max_len = max(a_ids.size(1), b_ids.size(1)) if targets is not None else None
        n_rounds = None
        if max_len is not None and self.config.combiner_variable_repeat:
            # See GPTConfig.combiner_variable_repeat: sample the actual
            # round count ONCE per batch (matches the existing granularity
            # -- one length regime per batch already, via
            # _sample_skewed_length), uniformly between "just enough" and
            # the full budget, so excess rounds get real training exposure
            # every batch, not just occasionally.
            target_len = min(max_len, self.config.combiner_repeat_max)
            n_rounds = torch.randint(target_len, self.config.combiner_repeat_max + 1, (1,)).item()
        n_think = None
        if max_len is not None and self.config.combiner_variable_thinking:
            # Mirrors n_rounds above, independently sampled -- see
            # GPTConfig.combiner_variable_thinking. Range is 0 to the full
            # ceiling (not tied to max_len/target_len the way n_rounds is,
            # since there's no equivalent "minimum thinking tokens this
            # problem needs" proxy the way there's a minimum round count).
            n_think = torch.randint(0, self.config.combiner_n_thinking + 1, (1,)).item()
        vec_c, countdown_loss = self.combine(vec_a, vec_b, plus_id, eq_id, max_len=max_len, n_rounds=n_rounds, n_think=n_think)

        logits, loss = self.decode_from_vector(vec_c, targets)

        if countdown_loss is not None and self.config.combiner_countdown_weight > 0:
            loss = loss + self.config.combiner_countdown_weight * countdown_loss

        if targets is not None and a_targets is not None:
            if self.codec_type == 'slot':
                # codec_type == 'slot' only: a blank-target position's query
                # is EXACTLY the zero vector (encode_operand never writes
                # there), so its logit against EVERY class -- digit or
                # blank -- is exactly 0 regardless of digit_embed/
                # blank_embed's values. That's not just hard to learn, it's
                # structurally unlearnable (gradient w.r.t. blank_embed
                # there is the query itself, i.e. also exactly 0) -- confirmed
                # directly, not just reasoned about. Left unmasked, these
                # positions sit at a permanent ~log(11) floor and, worse,
                # dilute the cross-entropy mean's gradient on the REAL
                # digit positions (which the vast majority of operands leave
                # a minority of, out of max_answer_digits=30 slots) by
                # roughly P/L_real. HRR doesn't have this problem -- its
                # "blank" query is crosstalk noise, not exact zero, so it's
                # genuinely learnable there -- hence gating this to 'slot'
                # only rather than masking for both.
                a_targets = a_targets.masked_fill(a_targets == 10, -1)
                b_targets = b_targets.masked_fill(b_targets == 10, -1)
            _, loss_a = self.decode_from_vector(vec_a, a_targets)
            _, loss_b = self.decode_from_vector(vec_b, b_targets)
            loss = loss + self.config.composed_recon_weight * (loss_a + loss_b)

        if targets is not None and sparsity_weight > 0:
            # Hoyer ratio (||x||_1/||x||_2 per vector, NOT raw L1 mean): plain
            # L1 has a trivial escape hatch -- confirmed empirically (v4/v5,
            # weights 0.05 and 0.2) that the network satisfies it by uniformly
            # shrinking every contribution's magnitude by a constant factor
            # (raw magnitude dropped ~13-16%, dims-needed-for-95%-variance
            # unchanged at all). The L1/L2 ratio is scale-invariant -- both
            # norms shrink by the same factor under uniform scaling, so
            # blanket shrinkage cannot reduce it at all, only genuine
            # concentration onto fewer dimensions can.
            def hoyer_ratio(x):
                return (x.abs().sum(dim=-1) / x.norm(dim=-1).clamp(min=1e-8)).mean()
            sparsity_loss = hoyer_ratio(contrib_a) + hoyer_ratio(contrib_b)
            loss = loss + sparsity_weight * sparsity_loss

        ortho_weight = self.config.composed_place_orthogonality_weight
        if targets is not None and ortho_weight > 0:
            loss = loss + ortho_weight * self.place_orthogonality_loss()

        if targets is not None and sum_vals is not None and self.config.composed_consistency_weight > 0:
            device = vec_c.device
            by_len = {}
            for i, v in enumerate(sum_vals):
                by_len.setdefault(len(str(v)), []).append(i)
            vec_sum_target = torch.zeros_like(vec_c)
            with torch.no_grad():
                for L, idxs in by_len.items():
                    digit_tensor = torch.tensor(
                        [[int(ch) for ch in str(sum_vals[i])] for i in idxs],
                        dtype=torch.long, device=device)
                    vec_sum_target[idxs] = self.encode_operand(digit_tensor)
            consistency_loss = F.mse_loss(vec_c, vec_sum_target)
            loss = loss + self.config.composed_consistency_weight * consistency_loss

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        H, Q = cfg.n_head, cfg.n_embd // cfg.n_head
        T = cfg.max_answer_digits + cfg.combiner_n_thinking  # the repeating combiner's actual sequence length, not the old fixed 4 slots
        flops_per_token = 6 * N + 12 * cfg.combiner_repeat_max * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu
